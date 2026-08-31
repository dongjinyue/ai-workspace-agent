from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.agent.service import AgentResult
from app.main import app


pytestmark = pytest.mark.integration


def test_chat_returns_safe_structured_trace():
    result = AgentResult(
        answer="3973",
        tool_called=True,
        tool_name="calculator",
        tools_used=["calculator"],
        active_skill=None,
        steps=3,
        matched_chunks=0,
        tool_traces=[
            {"name": "calculator", "source": "local", "duration_ms": 1.25}
        ],
        llm_calls=2,
        llm_duration_ms=0.1,
        tool_source="local",
    )
    with patch("app.memory.service.run_agent", return_value=result):
        response = TestClient(app).post(
            "/api/chat", json={"message": "137 乘 29 是多少？"}
        )

    assert response.status_code == 200
    trace = response.json()["trace"]
    assert len(trace["request_id"]) == 32
    assert trace["completed_at"] >= trace["started_at"]
    assert trace["duration_ms"] >= trace["llm_duration_ms"]
    assert trace["steps"] == 3
    assert trace["tools"][0]["name"] == "calculator"
    assert trace["rag"] == {"hit": False, "results": 0}
    assert "message" not in trace
    assert "api_key" not in str(trace).lower()

    # 执行轨迹存入独立审计表，刷新页面后仍可随助手消息恢复。
    conversation_id = response.json()["conversation_id"]
    restored = TestClient(app).get(
        f"/api/conversations/{conversation_id}/messages"
    )
    assistant_message = restored.json()["messages"][-1]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["trace"]["request_id"] == trace["request_id"]
    assert assistant_message["trace"]["tools"][0]["name"] == "calculator"
