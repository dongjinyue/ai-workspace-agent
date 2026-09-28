import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Callable
from uuid import uuid4

from app.agent.service import AgentResult, run_agent
from app.memory import repository
from app.observability.trace import RequestTrace


HISTORY_WINDOW = 20
# 旧会话以 admin 作为所有者保留；所有新服务调用都必须显式指定 owner_id。
ADMIN_OWNER_ID = "admin"
logger = logging.getLogger(__name__)


class ConversationNotFoundError(LookupError):
    """客户端提交了不存在的 Conversation ID（会话标识符）。"""


@dataclass(frozen=True)
class ConversationTurnResult:
    conversation_id: str
    history_messages: int
    agent: AgentResult
    trace: RequestTrace


class ConversationService:
    def __init__(self) -> None:
        # 同一进程内按会话串行执行，避免问题和回答交叉写入。
        self._locks_guard = threading.Lock()
        self._conversation_locks: dict[str, threading.Lock] = {}

    @contextmanager
    def _conversation_lock(self, conversation_id: str):
        with self._locks_guard:
            lock = self._conversation_locks.setdefault(
                conversation_id, threading.Lock()
            )
        with lock:
            yield

    def list_conversations(
        self,
        *,
        owner_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, str | int]]:
        return repository.list_conversations(
            owner_id=owner_id, limit=limit, offset=offset
        )

    def create_conversation(
        self, title: str = "新会话", *, owner_id: str
    ) -> dict[str, str]:
        conversation_id = repository.create_conversation(title, owner_id=owner_id)
        return {"id": conversation_id, "title": title}

    def rename_conversation(
        self, conversation_id: str, title: str, *, owner_id: str
    ) -> dict[str, str]:
        if not repository.rename_conversation(conversation_id, title, owner_id=owner_id):
            raise ConversationNotFoundError("会话不存在")
        return {"id": conversation_id, "title": title}

    def delete_conversation(
        self, conversation_id: str, *, owner_id: str
    ) -> None:
        if not repository.delete_conversation(conversation_id, owner_id=owner_id):
            raise ConversationNotFoundError("会话不存在")

    def resolve_conversation(
        self, conversation_id: str | None, *, owner_id: str
    ) -> str:
        if conversation_id is None:
            return repository.create_conversation(owner_id=owner_id)
        if not repository.conversation_exists(conversation_id, owner_id=owner_id):
            raise ConversationNotFoundError("会话不存在")
        return conversation_id

    def chat(
        self,
        *,
        message: str,
        knowledge_base_id: str | None,
        conversation_id: str | None,
        owner_id: str,
        allowed_tools: frozenset[str] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> ConversationTurnResult:
        # 同一 Conversation（会话）可有多次请求，每次必须有独立 request_id。
        request_id = uuid4().hex
        started_at = datetime.now(timezone.utc).isoformat()
        request_started = perf_counter()
        resolved_id: str | None = None
        resolved_id = self.resolve_conversation(conversation_id, owner_id=owner_id)
        with self._conversation_lock(resolved_id):
            user_message_id: int | None = None
            try:
                # 先保存，再读取。当前用户消息因此只进入模型上下文一次。
                user_message_id = repository.save_message(
                    resolved_id, "user", message, owner_id=owner_id
                )
                repository.use_first_message_as_title(
                    resolved_id, message, owner_id=owner_id
                )
                history = repository.get_messages(
                    resolved_id, limit=HISTORY_WINDOW, owner_id=owner_id
                )
                public_history = [
                    {"role": item["role"], "content": item["content"]}
                    for item in history
                ]
                # 所有会话统一进入 Agent Loop，由模型按需决定是否调用工具。
                agent_result = run_agent(
                    message=message,
                    knowledge_base_id=knowledge_base_id,
                    conversation_id=resolved_id,
                    history_messages=public_history,
                    allowed_tools=allowed_tools,
                    on_token=on_token,
                )
            except Exception as error:
                # 未完成轮次不应污染后续上下文；只回滚本次用户消息。
                if user_message_id is not None:
                    repository.delete_message(user_message_id, owner_id=owner_id)
                logger.error(
                    "Chat request failed request_id=%s conversation_id=%s error_type=%s",
                    request_id,
                    resolved_id,
                    type(error).__name__,
                )
                raise

            duration_ms = round((perf_counter() - request_started) * 1000, 3)
            # 只组装可安全公开的执行元数据，不包含模型内部思维过程。
            rag_trace: dict = {
                "hit": agent_result.matched_chunks > 0,
                "results": agent_result.matched_chunks,
            }
            retrieval_debug = agent_result.retrieval_debug or {}
            for key in ("top_k", "max_distance", "returned", "matches"):
                if key in retrieval_debug:
                    rag_trace[key] = retrieval_debug[key]

            trace = RequestTrace(
                request_id=request_id,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=duration_ms,
                steps=agent_result.steps,
                skill=agent_result.active_skill,
                tools=agent_result.tool_traces or [],
                rag=rag_trace,
                llm_calls=agent_result.llm_calls,
                llm_duration_ms=agent_result.llm_duration_ms,
            )
            repository.save_assistant_message_with_trace(
                resolved_id,
                agent_result.answer,
                trace.to_dict(),
                owner_id=owner_id,
            )
            logger.info(
                "Chat request completed request_id=%s conversation_id=%s duration_ms=%.3f",
                request_id,
                resolved_id,
                duration_ms,
            )
            return ConversationTurnResult(
                conversation_id=resolved_id,
                history_messages=len(history),
                agent=agent_result,
                trace=trace,
            )

    def run_resumable_task(
        self,
        *,
        run_id: str,
        message: str,
        knowledge_base_id: str | None,
        conversation_id: str,
        owner_id: str,
        allowed_tools: frozenset[str] | None,
        control,
        checkpoint_callback,
        status_callback,
        on_token=None,
    ) -> ConversationTurnResult:
        """在会话锁内执行可恢复 Run，避免并发问题交叉写入消息。"""
        from app.agent.checkpoint import deserialize_agent_state, serialize_agent_state
        from app.agent.runner import run_resumable_agent
        from app.skills.registry import select_skill

        task = repository.get_agent_task(run_id, owner_id)
        if task is None or task["conversation_id"] != conversation_id:
            raise ConversationNotFoundError("任务不存在")

        request_started = perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        with self._conversation_lock(conversation_id):
            if task["user_message_id"] is None:
                user_message_id = repository.save_message(
                    conversation_id, "user", message, owner_id=owner_id
                )
                repository.use_first_message_as_title(
                    conversation_id, message, owner_id=owner_id
                )
                repository.update_agent_task(
                    run_id,
                    owner_id,
                    expected_statuses={"running", "pause_requested"},
                    user_message_id=user_message_id,
                )

            if task["checkpoint"]:
                initial_state = deserialize_agent_state(task["checkpoint"])
                initial_state["stream_callback"] = on_token
                initial_state["allowed_tools"] = allowed_tools
            else:
                history = repository.get_messages(
                    conversation_id,
                    limit=HISTORY_WINDOW,
                    owner_id=owner_id,
                )
                skill = select_skill(message)
                initial_state = {
                    "messages": [
                        {"role": item["role"], "content": item["content"]}
                        for item in history
                    ],
                    "conversation_id": conversation_id,
                    "knowledge_base_id": knowledge_base_id,
                    "active_skill": skill.name if skill else None,
                    "steps": 0,
                    "tools_used": [],
                    "matched_chunks": 0,
                    "retrieved_chunks": [],
                    "final_answer": None,
                    "tool_traces": [],
                    "llm_calls": 0,
                    "llm_duration_ms": 0.0,
                    "retrieval_debug": {},
                    "allowed_tools": allowed_tools,
                    "stream_callback": on_token,
                }

            result = run_resumable_agent(
                initial_state,
                control=control,
                checkpoint_callback=checkpoint_callback,
                status_callback=status_callback,
                on_token=on_token,
            )
            duration_ms = round((perf_counter() - request_started) * 1000, 3)
            rag_trace: dict = {
                "hit": result.matched_chunks > 0,
                "results": result.matched_chunks,
            }
            retrieval_debug = result.retrieval_debug or {}
            for key in ("top_k", "max_distance", "returned", "matches"):
                if key in retrieval_debug:
                    rag_trace[key] = retrieval_debug[key]
            trace = RequestTrace(
                request_id=run_id,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=duration_ms,
                steps=result.steps,
                skill=result.active_skill,
                tools=result.tool_traces or [],
                rag=rag_trace,
                llm_calls=result.llm_calls,
                llm_duration_ms=result.llm_duration_ms,
            )
            if result.status == "completed":
                repository.save_assistant_message_with_trace(
                    conversation_id,
                    result.answer,
                    trace.to_dict(),
                    owner_id=owner_id,
                )
                repository.update_agent_task(
                    run_id,
                    owner_id,
                    expected_statuses={"running"},
                    status="completed",
                    checkpoint=serialize_agent_state(initial_state),
                )
            elif result.status == "stopped":
                repository.update_agent_task(
                    run_id,
                    owner_id,
                    expected_statuses={"running", "pause_requested"},
                    status="stopped",
                )
            elif result.status == "timed_out":
                repository.update_agent_task(
                    run_id,
                    owner_id,
                    expected_statuses={"running", "pause_requested"},
                    status="timed_out",
                    error_code="timeout",
                    error_message="任务执行超时，请重试",
                )
            return ConversationTurnResult(
                conversation_id=conversation_id,
                history_messages=len(
                    repository.get_messages(
                        conversation_id, limit=HISTORY_WINDOW, owner_id=owner_id
                    )
                ),
                agent=result,
                trace=trace,
            )

    def get_history(
        self, conversation_id: str, *, owner_id: str
    ) -> list[dict[str, str]]:
        if not repository.conversation_exists(conversation_id, owner_id=owner_id):
            raise ConversationNotFoundError("会话不存在")
        return repository.get_messages(conversation_id, owner_id=owner_id)

    def get_history_with_traces(
        self,
        conversation_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        owner_id: str,
    ) -> list[dict]:
        if not repository.conversation_exists(conversation_id, owner_id=owner_id):
            raise ConversationNotFoundError("会话不存在")
        return repository.get_messages_with_traces(
            conversation_id,
            limit=limit,
            offset=offset,
            owner_id=owner_id,
        )
