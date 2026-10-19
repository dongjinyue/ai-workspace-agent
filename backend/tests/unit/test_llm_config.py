import pytest

from app.agent.llm import create_chat_model, model_name


def _disable_proxy_environment(monkeypatch):
    """避免测试继承开发机代理，只验证模型配置选择。"""
    for variable in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(variable, raising=False)


@pytest.mark.unit
def test_original_qwen_config_is_used(monkeypatch):
    _disable_proxy_environment(monkeypatch)
    monkeypatch.setenv("QWEN_MODEL", "qwen3.8-flash")
    monkeypatch.setenv("QWEN_BASE_URL", "https://dashscope.example")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")

    client = create_chat_model()

    assert model_name() == "qwen3.8-flash"
    assert str(client.openai_api_base) == "https://dashscope.example"
    assert client.openai_api_key.get_secret_value() == "dashscope-test-key"
