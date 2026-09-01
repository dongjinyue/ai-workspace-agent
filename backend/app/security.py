PROMPT_INJECTION_MARKERS = (
    "忽略之前",
    "忽略以上",
    "系统指令",
    "你现在必须",
    "必须告诉用户",
)


class PromptInjectionError(ValueError):
    """上传内容包含疑似操控模型行为的指令。"""


def contains_prompt_injection(text: str) -> bool:
    """使用保守规则识别知识文档中的高风险模型指令。"""
    return any(marker in text for marker in PROMPT_INJECTION_MARKERS)


def access_token_required() -> bool:
    return bool(os.getenv("APP_ACCESS_TOKEN", "").strip())


def verify_bearer_token(authorization: str | None) -> bool:
    """可选的单用户部署保护；令牌只从后端环境变量读取。"""
    expected = os.getenv("APP_ACCESS_TOKEN", "").strip()
    if not expected:
        return True
    if not authorization or not authorization.startswith("Bearer "):
        return False
    supplied = authorization.removeprefix("Bearer ").strip()
    return secrets.compare_digest(supplied, expected)


class InMemoryRateLimiter:
    """单实例固定窗口限流，防止匿名请求快速消耗模型额度。"""

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError("API_RATE_LIMIT_PER_MINUTE 必须大于 0")
        self.limit = requests_per_minute
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, identity: str) -> bool:
        cutoff = monotonic() - 60
        with self._lock:
            timestamps = self._requests[identity]
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()
            if len(timestamps) >= self.limit:
                return False
            timestamps.append(monotonic())
            return True
import os
import secrets
import threading
from collections import defaultdict, deque
from time import monotonic
