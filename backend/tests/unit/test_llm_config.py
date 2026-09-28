import pytest

from app.agent.llm import create_llm_client, model_name, model_request_options


def _disable_proxy_environment(monkeypatch):
    """避免测试继承开发机代理，确保只验证配置选择逻辑。"""
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
def test_generic_model_config_is_preferred_over_legacy_qwen_config(monkeypatch):
    _disable_proxy_environment(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "deepseek-flash")
    monkeypatch.setenv("QWEN_MODEL", "qwen3.8-flash")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("QWEN_BASE_URL", "https://dashscope.example")
    monkeypatch.setenv("LLM_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")

    client = create_llm_client()

    assert model_name() == "deepseek-flash"
    assert str(client.base_url) == "https://api.deepseek.com"
    assert client.api_key == "deepseek-test-key"
    assert model_request_options() == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }


@pytest.mark.unit
def test_legacy_qwen_config_remains_supported(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("QWEN_MODEL", "qwen3.8-flash")
    monkeypatch.setenv("QWEN_BASE_URL", "https://dashscope.example")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")

    assert model_name() == "qwen3.8-flash"
    assert model_request_options() == {
        "extra_body": {"enable_thinking": False}
    }
