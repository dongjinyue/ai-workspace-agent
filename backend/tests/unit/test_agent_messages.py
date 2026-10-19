import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import checkpoint


def test_checkpoint_restores_legacy_tool_calls_as_standard_messages():
    state = checkpoint.deserialize_agent_state({"messages": [
        {"role": "user", "content": "计算"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {
            "name": "calculator", "arguments": '{"a":1,"b":2,"operation":"add"}'}}]},
    ]})
    assert isinstance(state["messages"][0], HumanMessage)
    assert isinstance(state["messages"][1], AIMessage)
    assert state["messages"][1].tool_calls[0]["args"]["a"] == 1


def test_checkpoint_versioned_messages_hide_tool_contents_and_artifacts():
    state = {"messages": [HumanMessage(content="问题"), ToolMessage(
        content="私有文档", tool_call_id="call-1", artifact={"secret": "私有文档"})],
        "retrieved_chunks": ["私有文档"], "steps": 2}
    raw = checkpoint.serialize_agent_state(state)
    assert json.loads(raw)["checkpoint_version"] == 1
    assert "私有文档" not in raw
    restored = checkpoint.deserialize_agent_state(raw)
    assert isinstance(restored["messages"][1], ToolMessage)
    assert restored["omitted_tool_result_ids"] == ["call-1"]


def test_checkpoint_rejects_unknown_version():
    with pytest.raises(ValueError, match="版本"):
        checkpoint.deserialize_agent_state({"checkpoint_version": 999, "messages": []})
