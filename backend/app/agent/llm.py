import os

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)


class ModelServiceUnavailableError(RuntimeError):
    """模型供应商不可用；对外只暴露安全、可操作的错误提示。"""


def translate_model_error(error: Exception) -> None:
    """把供应商异常转换为稳定的应用异常，避免泄漏原始响应。"""
    if isinstance(
        error,
        (
            AuthenticationError,
            PermissionDeniedError,
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
        ),
    ):
        raise ModelServiceUnavailableError(
            "模型服务暂时不可用，请检查 API Key、模型额度或稍后重试"
        ) from error


def create_llm_client() -> OpenAI:
    """创建带明确超时和有限重试的模型客户端。"""
    # 对话模型与知识库向量模型可以来自不同供应商，不能共用同一个密钥。
    # 保留旧变量回退，避免已有的通义千问部署立即失效。
    api_key = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("服务器没有配置 LLM_API_KEY")

    timeout_seconds = float(os.getenv("MODEL_TIMEOUT_SECONDS", "45"))
    if timeout_seconds <= 0 or timeout_seconds > 300:
        raise RuntimeError("MODEL_TIMEOUT_SECONDS 必须在 0 到 300 秒之间")

    return OpenAI(
        api_key=api_key,
        base_url=(
            os.getenv("LLM_BASE_URL")
            or os.getenv("QWEN_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        timeout=timeout_seconds,
        max_retries=2,
    )


def model_name() -> str:
    return os.getenv("LLM_MODEL") or os.getenv("QWEN_MODEL", "qwen-max")


def model_request_options() -> dict[str, object]:
    """返回当前模型需要的额外请求参数。"""
    # DeepSeek 的 Chat Completions 接口使用 thinking 参数；
    # 通义千问兼容接口使用 enable_thinking，二者不能混用。
    if model_name().startswith("deepseek-"):
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {"extra_body": {"enable_thinking": False}}
