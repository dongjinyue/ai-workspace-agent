from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.auth import Principal
from app.agent.service import AgentResult
from app.main import app, get_current_principal
from app.memory.service import ConversationTurnResult
from app.observability.trace import RequestTrace


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def act_as_verified_admin():
    app.dependency_overrides.clear()
    app.dependency_overrides[get_current_principal] = lambda: Principal(
        role="admin", owner_id="admin"
    )
    yield
    app.dependency_overrides.clear()


def test_agent_stream_emits_tokens_and_final_payload():
    result = AgentResult(
        answer="完整回答",
        tool_called=False,
        tool_name=None,
        tools_used=[],
        active_skill=None,
        steps=1,
        matched_chunks=0,
        tool_traces=[],
        llm_calls=1,
        llm_duration_ms=1.0,
        tool_source=None,
    )

    def fake_run_resumable_task(**kwargs):
        kwargs["status_callback"]("running")
        kwargs["on_token"]("完整")
        kwargs["on_token"]("回答")
        kwargs["status_callback"]("completed")
        return ConversationTurnResult(
            conversation_id=kwargs["conversation_id"],
            history_messages=1,
            agent=result,
            trace=RequestTrace(
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
            ),
        )

    with patch(
        "app.memory.service.ConversationService.run_resumable_task",
        side_effect=fake_run_resumable_task,
    ):
        response = TestClient(
            app, headers={"Origin": "http://localhost:3000"}
        ).post(
            "/api/agent/chat/stream",
            json={"message": "请回答"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: token' in response.text
    assert '"content": "完整"' in response.text
    assert '"content": "回答"' in response.text
    assert 'event: done\ndata:' in response.text
    assert '"answer": "完整回答"' in response.text
