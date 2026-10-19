from unittest.mock import patch
import pytest
from langchain_core.messages import AIMessage, SystemMessage
from app.agent.runner import _result_from_state
from app.agent.service import run_agent


def summary_state():
    return {"messages": [AIMessage(content="政策名称：补贴规定\n关键规定：凭空增加十万元")],
            "active_skill": "policy_summary", "tools_used": ["search_knowledge_base"],
            "steps": 3, "matched_chunks": 1, "retrieved_chunks": ["政策名称：补贴规定"],
            "final_answer": None}


def test_both_entrypoints_ground_policy_fields_identically():
    state = summary_state()
    resumed = _result_from_state(state, "completed")
    with patch("app.agent.graph.agent_graph.invoke", return_value=state):
        sync = run_agent("总结公司政策", "kb")
    assert resumed.answer == sync.answer
    assert "凭空" not in resumed.answer
    assert "关键规定：知识库未说明" in resumed.answer


def test_empty_completed_answer_is_rejected_but_pause_is_valid():
    with pytest.raises(RuntimeError, match="最终回答"):
        _result_from_state({"messages": [AIMessage(content="...")]}, "completed")
    assert _result_from_state({"messages": []}, "paused").answer == ""


def test_budget_ending_on_tool_never_returns_internal_prompt():
    from app.agent.nodes import tool_node
    state = {"messages": [AIMessage(content="", tool_calls=[{"name": "calculator", "args": {
        "a": 1, "b": 2, "operation": "add"}, "id": "last"}])], "steps": 4,
        "tools_used": [], "active_skill": None, "knowledge_base_id": None}
    updated = {**state, **tool_node(state)}
    result = _result_from_state(updated, "completed")
    assert "最大执行步数" in result.answer
    assert "现在只能根据" not in result.answer


def test_completed_result_requires_assistant_not_system_or_tool():
    with pytest.raises(RuntimeError, match="最终回答"):
        _result_from_state({"messages": [SystemMessage(content="内部指令")]}, "completed")
