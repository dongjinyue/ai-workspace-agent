"""通过受控传输验证真实 LangChain 模型适配，而非只模拟模型对象。"""
import json

import httpx
import pytest
from langchain_core.messages import AIMessage

from app.agent import llm


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)


def test_chat_model_uses_original_qwen_config(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("QWEN_MODEL", "qwen-test")
    monkeypatch.setenv("QWEN_BASE_URL", "https://model.example/v1")
    assert hasattr(llm, "create_chat_model")
    model = llm.create_chat_model()
    assert model.model_name == "qwen-test"
    assert model.openai_api_base == "https://model.example/v1"
    assert model.request_timeout == 45
    assert model.max_retries == 2


def test_bind_tools_uses_chat_completions_and_disables_thinking(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://model.example/v1")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 1,
            "model": "qwen-test", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "完成"},
                "finish_reason": "stop"}],
        })

    assert hasattr(llm, "create_chat_model")
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = llm.create_chat_model(http_client=client).bind_tools([
            {"name": "calculator", "description": "计算", "parameters": {"type": "object", "properties": {}}}
        ], parallel_tool_calls=False)
        answer = model.invoke("你好")
    assert isinstance(answer, AIMessage)
    assert answer.content == "完成"
    assert requests[0].url.path == "/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert body["tools"][0]["function"]["name"] == "calculator"
    assert body["parallel_tool_calls"] is False
    assert body["enable_thinking"] is False


@pytest.mark.parametrize("timeout", ["0", "301", "nan", "bad"])
def test_chat_model_rejects_invalid_timeout(monkeypatch, timeout):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", timeout)
    assert hasattr(llm, "create_chat_model")
    with pytest.raises(RuntimeError, match="MODEL_TIMEOUT_SECONDS"):
        llm.create_chat_model()
