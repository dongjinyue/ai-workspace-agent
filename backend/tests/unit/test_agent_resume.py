from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from app.agent import runner


def restored(tool, steps=2):
    return {"messages": [HumanMessage(content="问题"), AIMessage(content="", tool_calls=[
        {"name": tool, "id": "done", "args": {"query": "问题"} if tool == "search_knowledge_base" else {}}
    ]), ToolMessage(content="[工具结果已省略，恢复时重新检索]", tool_call_id="done")],
        "omitted_tool_result_ids": ["done"], "steps": steps, "llm_calls": 1,
        "next_node": "agent", "control_status": "paused"}


def test_resume_completed_tool_is_not_executed_twice():
    assert hasattr(runner, "prepare_resumed_state")
    state = runner.prepare_resumed_state(restored("calculator"))
    assert state["next_node"] == "end"
    assert "重新发起" in state["final_answer"]
    assert state["steps"] == 2


def test_resume_retrieval_reacquires_within_remaining_budget():
    assert hasattr(runner, "prepare_resumed_state")
    state = runner.prepare_resumed_state(restored("search_knowledge_base"))
    assert state["next_node"] == "tools"
    assert state["messages"][-1].tool_calls[0]["name"] == "search_knowledge_base"
    assert not state.get("omitted_tool_result_ids")
    assert state["steps"] == 2
    assert state.get("control_status") is None


def test_resume_retrieval_never_resets_exhausted_budget():
    assert hasattr(runner, "prepare_resumed_state")
    state = runner.prepare_resumed_state(restored("search_knowledge_base", 4))
    assert state["next_node"] == "end"
    assert "步数" in state["final_answer"]
    assert state["steps"] == 4
