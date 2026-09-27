import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from app.agent.nodes import agent_node, tool_node
from app.agent.registry import TOOLS
from app.agent.service import run_agent


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_run_agent_threads_guest_allowlist_into_graph_state():
    expected = frozenset({"calculator", "search_knowledge_base"})
    graph_result = {
        "final_answer": "完成",
        "messages": [],
        "active_skill": None,
        "tools_used": [],
        "steps": 1,
        "matched_chunks": 0,
        "tool_traces": [],
    }
    with patch(
        "app.agent.graph.agent_graph.invoke", return_value=graph_result
    ) as invoke:
        result = run_agent("你好", None, allowed_tools=expected)

    assert result.answer == "完成"
    assert invoke.call_args.args[0]["allowed_tools"] == expected


def test_guest_model_schema_contains_only_allowlisted_tools():
    allowed = frozenset({"calculator", "search_knowledge_base"})
    captured = {}
    message = SimpleNamespace(content="安全回答", tool_calls=None)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: (
                    captured.update(kwargs)
                    or SimpleNamespace(choices=[SimpleNamespace(message=message)])
                )
            )
        )
    )
    state = {
        "messages": [{"role": "user", "content": "你好"}],
        "knowledge_base_id": None,
        "conversation_id": None,
        "active_skill": None,
        "steps": 0,
        "tools_used": [],
        "matched_chunks": 0,
        "retrieved_chunks": [],
        "final_answer": None,
        "tool_traces": [],
        "llm_calls": 0,
        "llm_duration_ms": 0,
        "allowed_tools": allowed,
    }
    with (
        patch("app.agent.nodes._client", return_value=fake_client),
        patch(
            "app.agent.nodes.get_agent_tool_schemas",
            return_value=[
                _schema("calculator"),
                _schema("search_knowledge_base"),
                _schema("get_knowledge_base_info"),
                _schema("dangerous_admin_tool"),
            ],
        ),
    ):
        agent_node(state)

    assert [item["function"]["name"] for item in captured["tools"]] == [
        "calculator",
        "search_knowledge_base",
    ]


def test_fake_llm_call_cannot_execute_blocked_registered_tool(monkeypatch):
    called = []
    registration = TOOLS["get_knowledge_base_info"]
    monkeypatch.setitem(
        TOOLS,
        registration.name,
        replace(registration, handler=lambda **kwargs: called.append(kwargs) or {}),
    )
    tool_call = SimpleNamespace(
        id="fake-call",
        function=SimpleNamespace(name="get_knowledge_base_info", arguments="{}"),
    )
    state = {
        "messages": [SimpleNamespace(tool_calls=[tool_call])],
        "knowledge_base_id": "private-base",
        "conversation_id": "conversation",
        "active_skill": None,
        "steps": 1,
        "tools_used": [],
        "matched_chunks": 0,
        "retrieved_chunks": [],
        "final_answer": None,
        "tool_traces": [],
        "allowed_tools": frozenset({"calculator", "search_knowledge_base"}),
    }

    result = tool_node(state)
    tool_result = json.loads(result["messages"][-2]["content"])

    assert called == []
    assert "error" in tool_result
    assert result["matched_chunks"] == 0
