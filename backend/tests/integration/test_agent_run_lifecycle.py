import json
import threading
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import main
from app.agent.service import AgentResult
from app.auth import Principal
from app.main import app, get_current_principal
from app.memory import repository
from app.memory.service import ConversationTurnResult
from app.observability.trace import RequestTrace
from app.security import InMemoryRateLimiter


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def act_as_verified_admin():
    app.dependency_overrides.clear()
    app.dependency_overrides[get_current_principal] = lambda: Principal(
        role="admin", owner_id="admin"
    )
    main.rate_limiter = InMemoryRateLimiter(60)
    main.agent_run_manager.reset()
    yield
    main.agent_run_manager.reset()
    main.rate_limiter = InMemoryRateLimiter(60)
    app.dependency_overrides.clear()


def _event_data(response_text: str, event_name: str) -> list[dict]:
    lines = response_text.splitlines()
    values = []
    for index, line in enumerate(lines):
        if line == f"event: {event_name}" and index + 1 < len(lines):
            values.append(json.loads(lines[index + 1].removeprefix("data: ")))
    return values


def _turn(answer: str = "完整回答") -> ConversationTurnResult:
    result = AgentResult(
        answer=answer,
        tool_called=False,
        tool_name=None,
        tools_used=[],
        active_skill=None,
        steps=1,
        matched_chunks=0,
        tool_traces=[],
        llm_calls=1,
        llm_duration_ms=1.0,
    )
    trace = RequestTrace(
        request_id="r" * 32,
        started_at="2026-09-28T00:00:00+00:00",
        completed_at="2026-09-28T00:00:01+00:00",
        duration_ms=1000.0,
        steps=1,
        skill=None,
        tools=[],
        rag={"hit": False, "results": 0},
        llm_calls=1,
        llm_duration_ms=1.0,
    )
    return ConversationTurnResult(
        conversation_id="will-be-replaced",
        history_messages=1,
        agent=result,
        trace=trace,
    )


def test_stream_creates_run_and_routes_events_with_run_id():
    def fake_run_resumable_task(**kwargs):
        kwargs["status_callback"]("running")
        kwargs["on_token"]("完整")
        kwargs["on_token"]("回答")
        kwargs["status_callback"]("completed")
        turn = _turn()
        return ConversationTurnResult(
            conversation_id=kwargs["conversation_id"],
            history_messages=turn.history_messages,
            agent=turn.agent,
            trace=turn.trace,
        )

    with (
        patch.object(
            main.conversation_service,
            "run_resumable_task",
            side_effect=fake_run_resumable_task,
        ),
        TestClient(app) as client,
    ):
        response = client.post(
            "/api/agent/chat/stream",
            json={"message": "请回答"},
            headers={"Origin": "http://localhost:3000"},
        )

        assert response.status_code == 200
        run_events = _event_data(response.text, "run")
        assert len(run_events) == 1
        run_id = run_events[0]["run_id"]
        assert run_events[0]["status"] in {"queued", "running"}
        token_events = _event_data(response.text, "token")
        assert [item["content"] for item in token_events] == ["完整", "回答"]
        assert all(item["run_id"] == run_id for item in token_events)
        assert _event_data(response.text, "done")[0]["run_id"] == run_id

        status = client.get(
            f"/api/agent/runs/{run_id}",
            headers={"Origin": "http://localhost:3000"},
        )

    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert "owner_id" not in status.json()


def test_run_control_rejects_other_owner():
    conversation_id = repository.create_conversation(owner_id="admin")
    message_id = repository.save_message(
        conversation_id, "user", "私有任务", owner_id="admin"
    )
    task = repository.create_agent_task(conversation_id, "admin", message_id)

    app.dependency_overrides[get_current_principal] = lambda: Principal(
        role="guest", owner_id="guest:other"
    )
    with TestClient(app) as client:
        response = client.get(
            f"/api/agent/runs/{task['run_id']}",
            headers={"Origin": "http://localhost:3000"},
        )
        pause = client.post(
            f"/api/agent/runs/{task['run_id']}/pause",
            headers={"Origin": "http://localhost:3000"},
        )

    assert response.status_code == 404
    assert pause.status_code == 404


def _drain_events(events):
    collected = []
    while True:
        item = events.get(timeout=2)
        if item is None:
            return collected
        collected.append(item)


def test_pause_and_resume_reuse_same_run_without_automatic_continuation():
    conversation_id = repository.create_conversation(owner_id="admin")
    message_id = repository.save_message(
        conversation_id, "user", "暂停测试", owner_id="admin"
    )
    task = repository.create_agent_task(conversation_id, "admin", message_id)
    started = threading.Event()
    calls = [0]

    def worker_factory(control, emit):
        calls[0] += 1
        if calls[0] == 1:
            started.set()
            while control.check_boundary() is None:
                time.sleep(0.001)
            assert repository.update_agent_task(
                task["run_id"],
                "admin",
                expected_statuses={"pause_requested"},
                status="paused",
            )
            emit("paused", {"status": "paused"})
            return

        assert repository.update_agent_task(
            task["run_id"],
            "admin",
            expected_statuses={"running"},
            status="completed",
        )
        emit("done", {"answer": "恢复完成"})

    first_events = main.agent_run_manager.start(
        task["run_id"], "admin", worker_factory
    )
    assert started.wait(timeout=2)
    main.agent_run_manager.request_pause(task["run_id"], "admin")
    first = _drain_events(first_events)

    assert repository.get_agent_task(task["run_id"], "admin")["status"] == "paused"
    assert calls[0] == 1
    assert any(event == "paused" for event, _data in first)

    second_events = main.agent_run_manager.resume(task["run_id"], "admin")
    second = _drain_events(second_events)

    assert calls[0] == 2
    assert any(event == "done" for event, _data in second)
    assert all(data["run_id"] == task["run_id"] for _event, data in second)
    assert repository.get_agent_task(task["run_id"], "admin")["status"] == "completed"
    with pytest.raises(LookupError):
        main.agent_run_manager.get_events(task["run_id"], "admin")


def test_disconnect_stops_active_run_and_releases_session():
    conversation_id = repository.create_conversation(owner_id="admin")
    message_id = repository.save_message(
        conversation_id, "user", "断开测试", owner_id="admin"
    )
    task = repository.create_agent_task(conversation_id, "admin", message_id)
    started = threading.Event()

    def worker_factory(control, _emit):
        started.set()
        try:
            while control.check_boundary() is None:
                time.sleep(0.001)
        except Exception:
            return

    events = main.agent_run_manager.start(task["run_id"], "admin", worker_factory)
    assert started.wait(timeout=2)

    main.agent_run_manager.disconnect(task["run_id"], "admin")
    # 断开后订阅者已经离开，不应等待一个只给在线客户端使用的结束事件。
    del events
    deadline = time.time() + 2
    while time.time() < deadline:
        try:
            main.agent_run_manager.get_events(task["run_id"], "admin")
        except LookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail("断开后的 Run session 未释放")

    assert repository.get_agent_task(task["run_id"], "admin")["status"] == "stopped"
    with pytest.raises(LookupError):
        main.agent_run_manager.get_events(task["run_id"], "admin")


def test_resume_rebuilds_worker_after_manager_reset():
    conversation_id = repository.create_conversation(owner_id="admin")
    message_id = repository.save_message(
        conversation_id, "user", "重启后恢复", owner_id="admin"
    )
    task = repository.create_agent_task(conversation_id, "admin", message_id)

    def pause_worker(_control, emit):
        assert repository.update_agent_task(
            task["run_id"],
            "admin",
            expected_statuses={"running"},
            status="pause_requested",
        )
        assert repository.update_agent_task(
            task["run_id"],
            "admin",
            expected_statuses={"pause_requested"},
            status="paused",
        )
        emit("paused", {"status": "paused"})

    first_events = main.agent_run_manager.start(
        task["run_id"], "admin", pause_worker
    )
    _drain_events(first_events)
    assert repository.update_agent_task(
        task["run_id"],
        "admin",
        expected_statuses={"paused"},
        checkpoint={
            "messages": [{"role": "user", "content": "重启后恢复"}],
            "knowledge_base_id": None,
        },
    )
    main.agent_run_manager.reset()

    def completed_turn(**kwargs):
        assert repository.update_agent_task(
            task["run_id"],
            "admin",
            expected_statuses={"running"},
            status="completed",
        )
        turn = _turn("恢复成功")
        return ConversationTurnResult(
            conversation_id=kwargs["conversation_id"],
            history_messages=turn.history_messages,
            agent=turn.agent,
            trace=turn.trace,
        )

    with (
        patch.object(
            main.conversation_service,
            "run_resumable_task",
            side_effect=completed_turn,
        ),
        TestClient(app) as client,
    ):
        response = client.post(
            f"/api/agent/runs/{task['run_id']}/resume",
            headers={"Origin": "http://localhost:3000"},
        )

    assert response.status_code == 200
    assert _event_data(response.text, "done")[0]["answer"] == "恢复成功"


def test_cancel_wins_over_late_worker_completion():
    conversation_id = repository.create_conversation(owner_id="admin")
    message_id = repository.save_message(
        conversation_id, "user", "停止测试", owner_id="admin"
    )
    task = repository.create_agent_task(conversation_id, "admin", message_id)
    started = threading.Event()

    def worker_factory(control, _emit):
        started.set()
        while True:
            try:
                control.check_boundary()
            except Exception:
                return
            time.sleep(0.001)

    main.agent_run_manager.start(task["run_id"], "admin", worker_factory)
    assert started.wait(timeout=2)
    cancelled = main.agent_run_manager.cancel(task["run_id"], "admin")

    assert cancelled["status"] == "stopped"
    time.sleep(0.02)
    assert repository.get_agent_task(task["run_id"], "admin")["status"] == "stopped"


def test_resume_does_not_reserve_guest_quota_again():
    calls = [0]

    def fake_run_resumable_task(**kwargs):
        calls[0] += 1
        status = "paused" if calls[0] == 1 else "completed"
        kwargs["status_callback"](status)
        result = AgentResult(
            answer="恢复完成" if status == "completed" else "",
            tool_called=False,
            tool_name=None,
            tools_used=[],
            active_skill=None,
            steps=1,
            matched_chunks=0,
            status=status,
        )
        turn = _turn(result.answer or "")
        return ConversationTurnResult(
            conversation_id=kwargs["conversation_id"],
            history_messages=1,
            agent=result,
            trace=turn.trace,
        )

    app.dependency_overrides[get_current_principal] = lambda: Principal(
        role="guest", owner_id="guest:quota", guest_session_hash="q" * 64
    )
    with (
        patch.object(main, "_reserve_guest_quota") as reserve,
        patch.object(
            main.conversation_service,
            "run_resumable_task",
            side_effect=fake_run_resumable_task,
        ),
        TestClient(app) as client,
    ):
        first = client.post(
            "/api/agent/chat/stream",
            json={"message": "先暂停"},
            headers={"Origin": "http://localhost:3000"},
        )
        run_id = _event_data(first.text, "run")[0]["run_id"]
        assert _event_data(first.text, "paused")
        resumed = client.post(
            f"/api/agent/runs/{run_id}/resume",
            headers={"Origin": "http://localhost:3000"},
        )

    assert resumed.status_code == 200
    assert _event_data(resumed.text, "done")[0]["answer"] == "恢复完成"
    reserve.assert_called_once()


def test_worker_failure_emits_stable_error_without_exception_details():
    conversation_id = repository.create_conversation(owner_id="admin")
    message_id = repository.save_message(
        conversation_id, "user", "错误测试", owner_id="admin"
    )
    task = repository.create_agent_task(conversation_id, "admin", message_id)

    def failing_worker(_control, _emit):
        raise RuntimeError("secret-provider-stack-trace")

    events = main.agent_run_manager.start(task["run_id"], "admin", failing_worker)
    received = _drain_events(events)
    error_payloads = [data for event, data in received if event == "error"]

    assert error_payloads[0]["code"] == "agent_failed"
    assert "secret-provider" not in str(error_payloads[0])
    assert repository.get_agent_task(task["run_id"], "admin")["status"] == "failed"
