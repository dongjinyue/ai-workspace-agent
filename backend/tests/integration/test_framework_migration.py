"""真实模型适配器、图与 MCP；只有供应商 HTTP 响应由测试控制。"""
import json
from uuid import uuid4

import httpx
import pytest

from app.agent import nodes, tracing
from app.agent.llm import create_chat_model
from app.agent.registry import discover_mcp_tools
from app.agent.service import run_agent
from tests.unit.test_langsmith_tracing import RecordingClient


@pytest.mark.parametrize("streaming", [False, True])
def test_real_framework_flow_and_metadata_only_tracing(monkeypatch, streaming):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://model.example/v1")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "not-a-real-key")
    trace_client = RecordingClient()
    monkeypatch.setattr(tracing, "Client", lambda **kwargs: trace_client)
    requests = []
    discover_mcp_tools.cache_clear()
    discover_mcp_tools()

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        count = len(requests)
        if count <= 2:
            name = "calculator" if count == 1 else "get_current_time"
            arguments = {"a": 2, "b": 3, "operation": "add"} if count == 1 else {}
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"call-{count}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }]}
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": "计算为5，时间查询完成。"}
            finish = "stop"
        payload = {"id": "test", "created": 1, "model": "qwen-test", "object": "chat.completion",
                   "choices": [{"index": 0, "message": message, "finish_reason": finish}]}
        if not body.get("stream"):
            return httpx.Response(200, json=payload)
        delta = dict(message)
        if delta.get("tool_calls"):
            delta["tool_calls"][0]["index"] = 0
        event = {**payload, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n")

    tokens = []
    correlation = uuid4().hex
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(nodes, "_client", lambda: create_chat_model(http_client=client))
        result = run_agent("计算2+3再查时间，敏感问题XYZ", None,
                           on_token=tokens.append if streaming else None, trace_correlation_id=correlation)
    assert result.steps == 5
    assert result.llm_calls == 3
    assert result.tools_used == ["calculator", "get_current_time"]
    assert result.tool_source == "mcp"
    assert json.loads(requests[1]["messages"][-2]["content"])["result"] == 5
    assert json.loads(requests[2]["messages"][-2]["content"])["source"] == "mcp"
    if streaming:
        assert "".join(tokens) == result.answer
    starts = [event for event in trace_client.events if "run_type" in event]
    assert len([event for event in starts if event["run_type"] == "llm"]) == 3
    assert len([event for event in starts if event["run_type"] == "tool"]) == 2
    assert len({str(event["trace_id"]) for event in starts}) == 1
    assert all(event["extra"]["metadata"]["correlation_id"] == correlation for event in starts)
    assert "敏感问题XYZ" not in json.dumps(trace_client.events, default=str)


def test_resumable_completed_graph_finishes_trace_without_error(monkeypatch):
    from app.agent.runner import run_resumable_agent, RunControl
    from langchain_core.messages import AIMessage
    from tests.unit.test_langchain_streaming import SequenceModel, state
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    client = RecordingClient()
    monkeypatch.setattr(tracing, "Client", lambda **kwargs: client)
    monkeypatch.setattr(nodes, "_client", lambda: SequenceModel([AIMessage(content="你好")]))
    result = run_resumable_agent(state(messages=[{"role": "user", "content": "你好"}]),
        control=RunControl(), checkpoint_callback=lambda _: None, status_callback=lambda _: None,
        on_token=lambda _: None)
    assert result.status == "completed"
    started = {event["id"] for event in client.events if "id" in event}
    ended = {event["run_id"] for event in client.events if "run_id" in event}
    assert started == ended
    assert not any(event.get("error") for event in client.events)


def test_policy_resume_persists_only_validated_answer(monkeypatch, tmp_path):
    """实际后台工厂、仓库、检查点、技能和图共同验证恢复后的最终答案。"""
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace
    from langchain_core.messages import AIMessage
    from app import main
    from app.auth import Principal
    from app.agent.registry import TOOLS
    from app.agent.runner import RunControl
    from app.memory import repository
    from tests.unit.test_langchain_streaming import SequenceModel

    monkeypatch.setenv("APP_DATABASE_PATH", str(tmp_path / "policy.db"))
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    trace_client = RecordingClient()
    monkeypatch.setattr(tracing, "Client", lambda **kwargs: trace_client)
    source = "政策名称：补贴制度\n关键规定：凭有效发票报销"
    registration = TOOLS["search_knowledge_base"]
    monkeypatch.setitem(TOOLS, "search_knowledge_base", replace(registration, handler=lambda **kwargs: {
        "matched": True, "chunks": [source]}))
    model = SequenceModel([
        AIMessage(content="", tool_calls=[{"name": "search_knowledge_base", "args": {"query": "补贴制度"}, "id": "policy-call"}]),
        AIMessage(content="政策名称：补贴制度\n关键规定：每人十万元"),
    ])
    monkeypatch.setattr(nodes, "_client", lambda: model)
    principal = Principal(role="admin", owner_id="policy-owner")
    conversation = repository.create_conversation(owner_id=principal.owner_id)
    task = repository.create_agent_task(conversation, principal.owner_id, None)
    run_id = task["run_id"]
    repository.update_agent_task(run_id, principal.owner_id, expected_statuses={"queued"}, status="running")
    control = RunControl()
    events = []

    def emit(event, data):
        events.append((event, data))
        if event == "trace" and data["trace"]["steps"] == 1 and not control.poll():
            repository.update_agent_task(run_id, principal.owner_id, expected_statuses={"running"}, status="pause_requested")
            control.request_pause()

    worker = main._build_agent_worker_factory(task, principal, message="总结公司补贴政策", knowledge_base_id="kb-test")
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(worker, control, emit).result(timeout=20)
    paused = repository.get_agent_task(run_id, principal.owner_id)
    assert paused["status"] == "paused"
    assert paused["checkpoint"]["checkpoint_version"] == 1
    repository.update_agent_task(run_id, principal.owner_id, expected_statuses={"paused"}, status="running")
    resumed_worker = main._build_resumable_worker_factory(paused, principal)
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(resumed_worker, RunControl(), lambda event, data: events.append((event, data))).result(timeout=20)
    messages = repository.get_messages(conversation, owner_id=principal.owner_id)
    assert [item["role"] for item in messages] == ["user", "assistant"]
    answer = messages[-1]["content"]
    assert "十万元" not in answer
    assert "关键规定：知识库未说明" in answer
    done = [data for event, data in events if event == "done"]
    assert len(done) == 1
    assert done[0]["answer"] == answer
    assert "".join(data["content"] for event, data in events if event == "token") == answer
    starts = [event for event in trace_client.events if "run_type" in event]
    roots = [event for event in starts if not event.get("parent_run_id")]
    assert len(roots) == 2
    assert roots[0]["id"] != roots[1]["id"]
    assert all(event["extra"]["metadata"]["correlation_id"] == run_id for event in roots)
    assert any(event["run_type"] == "retriever" for event in starts)
