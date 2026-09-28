"""Agent Run（执行任务）的线程编排、事件广播和控制入口。"""

import logging
import os
import queue
import threading
from dataclasses import dataclass
from typing import Callable

from app.agent.runner import RunControl
from app.memory import repository


logger = logging.getLogger(__name__)
EventQueue = queue.Queue[tuple[str, dict] | None]
WorkerFactory = Callable[[RunControl, Callable[[str, dict], None]], None]
EVENT_QUEUE_MAXSIZE = 256


@dataclass
class _RunSession:
    run_id: str
    owner_id: str
    control: RunControl
    events: EventQueue
    worker_factory: WorkerFactory
    thread: threading.Thread | None = None
    disconnected: bool = False


class RunManager:
    """在进程内管理活动执行；持久化状态仍以 SQLite 任务表为准。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, _RunSession] = {}

    def _timeout_seconds(self) -> float:
        raw = os.getenv("AGENT_RUN_TIMEOUT_SECONDS", os.getenv("MODEL_TIMEOUT_SECONDS", "120"))
        try:
            return max(1.0, float(raw))
        except ValueError:
            return 120.0

    def _emit(self, session: _RunSession, event: str, data: dict) -> None:
        with self._lock:
            if session.disconnected:
                return
        payload = {"run_id": session.run_id, **data}
        try:
            session.events.put_nowait((event, payload))
        except queue.Full:
            # 客户端长期不消费时停止任务，避免 token（令牌）无限堆积占用内存。
            logger.warning("Agent event queue is full, disconnecting run_id=%s", session.run_id)
            with self._lock:
                session.disconnected = True
            session.control.cancel()

    def _start_session(
        self,
        run_id: str,
        owner_id: str,
        worker_factory: WorkerFactory,
        *,
        status: str,
    ) -> EventQueue:
        task = repository.get_agent_task(run_id, owner_id)
        if task is None:
            raise LookupError("任务不存在")
        control = RunControl(self._timeout_seconds())
        session = _RunSession(
            run_id=run_id,
            owner_id=owner_id,
            control=control,
            events=queue.Queue(maxsize=EVENT_QUEUE_MAXSIZE),
            worker_factory=worker_factory,
        )
        with self._lock:
            self._sessions[run_id] = session
        self._emit(
            session,
            "run",
            {
                "conversation_id": task["conversation_id"],
                "status": status,
            },
        )
        self._emit(session, "status", {"status": status})
        thread = threading.Thread(
            target=self._run_worker,
            args=(session,),
            name=f"agent-run-{run_id[:8]}",
            daemon=True,
        )
        session.thread = thread
        thread.start()
        return session.events

    def start(
        self,
        run_id: str,
        owner_id: str,
        worker_factory: WorkerFactory,
    ) -> EventQueue:
        """启动 queued Run；新 Run 的额度由调用方在创建前预占。"""
        task = repository.get_agent_task(run_id, owner_id)
        if task is None:
            raise LookupError("任务不存在")
        if task["status"] != "queued":
            raise ValueError("只有 queued 任务可以启动")
        repository.update_agent_task(
            run_id,
            owner_id,
            expected_statuses={"queued"},
            status="running",
        )
        return self._start_session(
            run_id, owner_id, worker_factory, status="running"
        )

    def _run_worker(self, session: _RunSession) -> None:
        try:
            session.worker_factory(
                session.control,
                lambda event, data: self._emit(session, event, data),
            )
        except Exception:
            # 工作线程只向浏览器发送稳定错误码，详细异常留在调用方日志。
            logger.exception("Agent run failed run_id=%s", session.run_id)
            repository.update_agent_task(
                session.run_id,
                session.owner_id,
                expected_statuses={"running", "pause_requested"},
                status="failed",
                error_code="agent_failed",
                error_message="Agent 执行失败，请检查后端日志",
            )
            self._emit(
                session,
                "error",
                {"status": 500, "code": "agent_failed", "detail": "Agent 执行失败，请检查后端日志"},
            )
            self._emit(session, "status", {"status": "failed"})
        finally:
            task = repository.get_agent_task(session.run_id, session.owner_id)
            if task and task["status"] in {"running", "pause_requested"}:
                # Worker（工作线程）异常返回但没有写入终态时，不能留下永远运行的任务。
                repository.update_agent_task(
                    session.run_id,
                    session.owner_id,
                    expected_statuses={task["status"]},
                    status="failed",
                    error_code="agent_incomplete",
                    error_message="Agent 未正常完成任务",
                )
                self._emit(
                    session,
                    "error",
                    {
                        "status": 500,
                        "code": "agent_incomplete",
                        "detail": "Agent 未正常完成任务，请重试",
                    },
                )
                self._emit(session, "status", {"status": "failed"})
                task = repository.get_agent_task(session.run_id, session.owner_id)
            if not session.disconnected:
                try:
                    session.events.put_nowait(None)
                except queue.Full:
                    logger.warning("Agent event queue is full at shutdown run_id=%s", session.run_id)
            with self._lock:
                if self._sessions.get(session.run_id) is session and (
                    task is None or task["status"] != "paused"
                ):
                    self._sessions.pop(session.run_id, None)

    def request_pause(self, run_id: str, owner_id: str) -> dict:
        task = repository.get_agent_task(run_id, owner_id)
        if task is None:
            raise LookupError("任务不存在")
        if task["status"] in {"queued", "running"}:
            if repository.update_agent_task(
                run_id,
                owner_id,
                expected_statuses={task["status"]},
                status="pause_requested",
            ):
                with self._lock:
                    session = self._sessions.get(run_id)
                if session:
                    session.control.request_pause()
                if session:
                    self._emit(session, "status", {"status": "pause_requested"})
            return repository.get_agent_task(run_id, owner_id) or task
        if task["status"] == "paused":
            return task
        raise ValueError("当前任务不能暂停")

    def resume(
        self,
        run_id: str,
        owner_id: str,
        worker_factory: WorkerFactory | None = None,
    ) -> EventQueue:
        task = repository.get_agent_task(run_id, owner_id)
        if task is None:
            raise LookupError("任务不存在")
        if task["status"] != "paused":
            raise ValueError("只有 paused 任务可以恢复")
        with self._lock:
            session = self._sessions.get(run_id)
            factory = worker_factory or (session.worker_factory if session else None)
        if factory is None:
            raise LookupError("任务恢复信息已过期，请重新提问")
        if not repository.update_agent_task(
            run_id,
            owner_id,
            expected_statuses={"paused"},
            status="running",
        ):
            raise ValueError("任务状态已改变，请刷新后重试")
        return self._start_session(run_id, owner_id, factory, status="running")

    def cancel(self, run_id: str, owner_id: str) -> dict:
        task = repository.get_agent_task(run_id, owner_id)
        if task is None:
            raise LookupError("任务不存在")
        if task["status"] in {"completed", "stopped", "timed_out", "failed"}:
            return task
        with self._lock:
            session = self._sessions.get(run_id)
        if session:
            session.control.cancel()
            self._emit(session, "cancelled", {"status": "stopped"})
        repository.update_agent_task(
            run_id,
            owner_id,
            expected_statuses={task["status"]},
            status="stopped",
            error_code="cancelled",
            error_message="任务已停止",
        )
        if session and task["status"] != "running":
            try:
                session.events.put_nowait(None)
            except queue.Full:
                with self._lock:
                    session.disconnected = True
        return repository.get_agent_task(run_id, owner_id) or task

    def disconnect(self, run_id: str, owner_id: str) -> dict:
        """客户端断开时停止活动任务，但保留已经暂停的任务供之后恢复。"""
        task = repository.get_agent_task(run_id, owner_id)
        if task is None:
            raise LookupError("任务不存在")
        with self._lock:
            session = self._sessions.get(run_id)
            if session:
                session.disconnected = True
                session.control.cancel()
        if task["status"] in {"queued", "running", "pause_requested"}:
            repository.update_agent_task(
                run_id,
                owner_id,
                expected_statuses={task["status"]},
                status="stopped",
                error_code="client_disconnected",
                error_message="客户端已断开，任务已停止",
            )
        return repository.get_agent_task(run_id, owner_id) or task

    def get_events(self, run_id: str, owner_id: str) -> EventQueue:
        with self._lock:
            session = self._sessions.get(run_id)
        if session is None or session.owner_id != owner_id:
            raise LookupError("任务不存在")
        return session.events

    def reset(self) -> None:
        """测试和进程关闭时释放内存中的订阅，不删除持久化任务。"""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.control.cancel()
