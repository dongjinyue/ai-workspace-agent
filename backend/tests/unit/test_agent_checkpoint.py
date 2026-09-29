import json
from types import SimpleNamespace

import pytest

from app.agent.checkpoint import (
    deserialize_agent_state,
    merge_agent_update,
    serialize_agent_state,
)


pytestmark = pytest.mark.unit


def _state():
    tool_call = SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(
            name="search_knowledge_base",
            arguments='{"query":"安全问题"}',
        ),
    )
    assistant_message = SimpleNamespace(
        role="assistant",
        content=None,
        tool_calls=[tool_call],
    )
    return {
        "messages": [
            {"role": "user", "content": "安全问题"},
            assistant_message,
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "完整检索正文：不应进入检查点",
            },
        ],
        "knowledge_base_id": "kb-safe",
        "conversation_id": "conversation-safe",
        "active_skill": None,
        "steps": 2,
        "tools_used": ["search_knowledge_base"],
        "matched_chunks": 1,
        "retrieved_chunks": ["完整检索正文：不应进入检查点"],
        "final_answer": None,
        "tool_traces": [{"name": "search_knowledge_base", "duration_ms": 12.3}],
        "llm_calls": 1,
        "llm_duration_ms": 21.0,
        "stream_callback": lambda _token: None,
        "untrusted_debug": "sk-secret-key",
    }


def test_checkpoint_round_trip_keeps_tool_call_but_excludes_sensitive_runtime_data():
    state = _state()
    state["next_node"] = "tools"
    serialized = serialize_agent_state(state)

    assert "完整检索正文：不应进入检查点" not in serialized
    assert "sk-secret-key" not in serialized
    assert "stream_callback" not in serialized

    restored = deserialize_agent_state(serialized)
    tool_call = restored["messages"][1].tool_calls[0]
    assert tool_call.function.name == "search_knowledge_base"
    assert restored["retrieved_chunks"] == []
    assert "stream_callback" not in restored
    assert restored["next_node"] == "tools"


def test_deserialize_agent_state_accepts_repository_checkpoint_dict():
    serialized = serialize_agent_state(_state())
    checkpoint = json.loads(serialized)

    restored = deserialize_agent_state(checkpoint)

    assert restored["conversation_id"] == "conversation-safe"
    assert restored["retrieved_chunks"] == []


def test_merge_agent_update_does_not_mutate_original_state():
    state = _state()
    updated = merge_agent_update(
        state,
        {"steps": 3, "final_answer": "已完成", "tools_used": ["calculator"]},
    )

    assert state["steps"] == 2
    assert state["final_answer"] is None
    assert updated["steps"] == 3
    assert updated["final_answer"] == "已完成"
    assert updated["tools_used"] == ["calculator"]
