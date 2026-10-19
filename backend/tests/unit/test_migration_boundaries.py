"""验收清单中的异常与预算边界，均不访问供应商网络。"""
import json
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.agent import nodes, tracing
from app.agent.checkpoint import deserialize_agent_state, serialize_agent_state
from app.agent.model_response import consume_model_stream
from app.agent.registry import TOOLS
from app.agent.runner import RunControl, run_resumable_agent
from app.agent.service import invoke_registered_tool, run_agent
from tests.unit.test_langchain_streaming import SequenceModel, state
from tests.unit.test_langsmith_tracing import RecordingClient


@pytest.fixture(autouse=True)
def static_schemas(monkeypatch):
    monkeypatch.setattr(nodes, "get_agent_tool_schemas", lambda: [
        {"type": "function", "function": {"name": name, "description": name,
         "parameters": {"type": "object", "properties": {}}}}
        for name in ("calculator", "get_current_time", "calculate_text_stats", "search_knowledge_base")])


def test_fifth_model_step_cannot_execute_another_tool(monkeypatch):
    message = AIMessage(content="", tool_calls=[{"name": "calculator", "args": {}, "id": "over-budget"}])
    monkeypatch.setattr(nodes, "_client", lambda: SequenceModel([message]))
    initial = state(steps=4, tools_used=["calculator"])
    update = nodes.agent_node(initial)
    assert "最大执行步数" in update["final_answer"]
    assert nodes.route_after_agent({**initial, **update}) == "end"


def test_multiple_tool_calls_are_rejected_before_any_execution(monkeypatch):
    calls = []
    monkeypatch.setattr("app.agent.service.execute_tool", lambda *args, **kwargs: calls.append(args))
    message = AIMessage(content="", tool_calls=[
        {"name": "calculator", "args": {}, "id": "one"},
        {"name": "get_current_time", "args": {}, "id": "two"}])
    with pytest.raises(RuntimeError, match="一个工具"):
        nodes.tool_node(state(messages=[message]))
    assert calls == []


def test_unfinished_json_is_not_treated_as_valid_completed_arguments():
    turn = consume_model_stream(iter([AIMessageChunk(content="", tool_call_chunks=[{
        "name": "calculator", "id": "bad", "args": '{"a":1', "index": 0}])]),
        on_token=None, control_poll=None, allow_tools=True, buffer_text=True)
    assert turn.message.invalid_tool_calls
    updated = nodes.tool_node(state(messages=[turn.message]))
    assert "error" in json.loads(updated["messages"][-2].content)


def test_empty_stream_rejected_and_empty_usage_chunk_ignored():
    with pytest.raises(RuntimeError, match="没有返回"):
        consume_model_stream(iter([]), on_token=None, control_poll=None, allow_tools=True, buffer_text=True)
    parts = []
    turn = consume_model_stream(iter([AIMessageChunk(content=""), AIMessageChunk(content="完成")]),
        on_token=parts.append, control_poll=None, allow_tools=False, buffer_text=False)
    assert parts == ["完成"]
    assert turn.message.content == "完成"


@pytest.mark.parametrize("name,question", [
    ("calculator", "计算2+3"), ("get_current_time", "当前时间"), ("calculate_text_stats", "统计字符数")])
def test_stream_without_knowledge_base_can_use_each_special_tool(monkeypatch, name, question):
    model = SequenceModel([AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": "one"}]), AIMessage(content="完成")])
    monkeypatch.setattr(nodes, "_client", lambda: model)
    monkeypatch.setattr("app.agent.service.execute_tool", lambda *args, **kwargs: {"result": 5})
    result = run_agent(question, None, on_token=lambda _: None)
    assert result.tools_used == [name]
    assert result.steps == 3


def test_knowledge_tool_receives_only_trusted_backend_resource(monkeypatch):
    captured = []
    registration = TOOLS["search_knowledge_base"]
    monkeypatch.setitem(TOOLS, registration.name, replace(registration, handler=lambda **kwargs: captured.append(kwargs) or {"chunks": []}))
    invoke_registered_tool(registration.name, {"query": "问题"}, "trusted-backend")
    assert captured == [{"query": "问题", "knowledge_base_id": "trusted-backend"}]


@pytest.mark.parametrize("version", [0, 1])
def test_resume_pending_call_preserves_id_and_budget(monkeypatch, version):
    count = []
    registration = TOOLS["calculator"]
    monkeypatch.setitem(TOOLS, "calculator", replace(registration, handler=lambda **kwargs: count.append(kwargs) or {"result": 3}))
    legacy = state(messages=[{"role": "user", "content": "计算1+2"}, {"role": "assistant", "content": None,
        "tool_calls": [{"id": "pending-id", "type": "function", "function": {"name": "calculator",
            "arguments": '{"a":1,"b":2,"operation":"add"}'}}]}], next_node="tools", steps=1, llm_calls=1)
    restored = deserialize_agent_state(legacy if version == 0 else serialize_agent_state(legacy))
    assert restored["messages"][-1].tool_calls[0]["id"] == "pending-id"
    monkeypatch.setattr(nodes, "_client", lambda: SequenceModel([AIMessage(content="结果为3")]))
    result = run_resumable_agent(restored, control=RunControl(), checkpoint_callback=lambda _: None, status_callback=lambda _: None)
    assert len(count) == 1
    assert result.steps == 3
    assert result.llm_calls == 2


@pytest.mark.parametrize("operation", ["create_run", "update_run"])
def test_trace_send_failure_preserves_actual_graph_answer(monkeypatch, operation):
    class FailingClient(RecordingClient):
        def create_run(self, **kwargs):
            if operation == "create_run":
                raise RuntimeError("private-transport")
            super().create_run(**kwargs)
        def update_run(self, **kwargs):
            if operation == "update_run":
                raise RuntimeError("private-transport")
            super().update_run(**kwargs)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    monkeypatch.setattr(tracing, "Client", lambda **kwargs: FailingClient())
    monkeypatch.setattr(nodes, "_client", lambda: SequenceModel([AIMessage(content="你好")]))
    assert run_agent("你好", None).answer == "你好"
