import importlib
import importlib.util
import json
from uuid import uuid4

import pytest


def tracing_module():
    assert importlib.util.find_spec("app.agent.tracing") is not None
    return importlib.import_module("app.agent.tracing")


class RecordingClient:
    def __init__(self, **kwargs):
        self.events = []
    def create_run(self, **kwargs):
        self.events.append(kwargs)
    def update_run(self, **kwargs):
        self.events.append(kwargs)
    def close(self):
        pass


def test_disabled_tracing_never_constructs_client(monkeypatch):
    tracing = tracing_module()
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.setenv("LANGSMITH_API_KEY", "secret-key")
    monkeypatch.setattr(tracing, "Client", lambda **kwargs: pytest.fail("不应创建客户端"))
    with tracing.agent_trace_config("a" * 32) as config:
        assert not config.get("callbacks")


def test_enabled_without_key_preserves_execution(monkeypatch):
    tracing = tracing_module()
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setattr(tracing, "Client", lambda **kwargs: pytest.fail("不应创建客户端"))
    with tracing.agent_trace_config("a" * 32) as config:
        assert not config.get("callbacks")


def test_trace_payload_excludes_contents_metadata_and_exceptions(monkeypatch):
    tracing = tracing_module()
    client = RecordingClient()
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "key-not-uploaded")
    monkeypatch.setattr(tracing, "Client", lambda **kwargs: client)
    with tracing.agent_trace_config("a" * 32) as config:
        handler = config["callbacks"][0]
        run = uuid4()
        handler.on_chain_start({"id": ["private-source"]}, {"messages": "private-question"},
                               run_id=run, name="private-name", metadata={"secret": "private-metadata"})
        handler.on_chain_error(RuntimeError("private-exception"), run_id=run)
    raw = json.dumps(client.events, default=str)
    assert "private-" not in raw
    assert "key-not-uploaded" not in raw
    assert len(client.events) == 2
    assert client.events[1]["error"] == "execution_failed"


def test_tracing_transport_failure_does_not_fail_business(monkeypatch):
    tracing = tracing_module()
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    def fail(**kwargs):
        raise RuntimeError("private-transport")
    monkeypatch.setattr(tracing, "Client", fail)
    with tracing.agent_trace_config("a" * 32) as config:
        assert not config.get("callbacks")


def test_real_client_does_not_add_runtime_or_environment_data(monkeypatch):
    """检查真实 SDK 最后的请求体，避免仅测试发送前的自定义字典。"""
    import requests
    from langsmith import Client
    from langsmith import env
    tracing = tracing_module()
    requests_out = []
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    monkeypatch.setattr(env, "get_runtime_environment", lambda: {"private-runtime": "private-value"})
    monkeypatch.setattr(env, "get_langchain_env_var_metadata", lambda: {"private-env": "private-value"})

    def capture(self, method, pathname, **kwargs):
        requests_out.append(kwargs.get("request_kwargs", kwargs).get("data", b""))
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        return response

    monkeypatch.setattr(Client, "request_with_retries", capture)
    with tracing.agent_trace_config("b" * 32) as config:
        handler = config["callbacks"][0]
        run = uuid4()
        handler.on_chain_start({}, {"content": "private-question"}, run_id=run, name="LangGraph")
        handler.on_chain_end({"answer": "private-answer"}, run_id=run)
    assert len(requests_out) == 2
    serialized = b"".join(value.encode() if isinstance(value, str) else value for value in requests_out)
    assert b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" in serialized
    assert b"private-" not in serialized


def test_direct_tool_entry_disables_automatic_content_tracing(monkeypatch):
    from langchain_core.tracers.langchain import LangChainTracer, wait_for_all_tracers
    from app.agent.service import execute_tool
    captured = []
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    # 即使回归重新启用了自动追踪，也不能真的发送测试请求。
    from langsmith import Client
    import requests
    def no_network(*args, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        return response
    monkeypatch.setattr(Client, "request_with_retries", no_network)
    monkeypatch.setattr(LangChainTracer, "_persist_run_single", lambda self, run: captured.append(run))
    result = execute_tool("calculator", '{"a":12345,"b":67890,"operation":"add"}', None)
    wait_for_all_tracers()
    assert result["result"] == 80235
    assert captured == []
