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
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("服务器没有配置 DASHSCOPE_API_KEY")

    timeout_seconds = float(os.getenv("MODEL_TIMEOUT_SECONDS", "45"))
    if timeout_seconds <= 0 or timeout_seconds > 300:
        raise RuntimeError("MODEL_TIMEOUT_SECONDS 必须在 0 到 300 秒之间")

    return OpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        timeout=timeout_seconds,
        max_retries=2,
    )


def model_name() -> str:
    return os.getenv("QWEN_MODEL", "qwen-max")
