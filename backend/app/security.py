import os
import secrets
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from time import monotonic
from typing import Iterable, Sequence


PROMPT_INJECTION_MARKERS = (
    "忽略之前",
    "忽略以上",
    "系统指令",
    "你现在必须",
    "必须告诉用户",
)


@dataclass(frozen=True)
class TextRegion:
    """文档正文中的可定位区域，例如 PDF 的某一页。"""

    start: int
    end: int
    label: str


@dataclass(frozen=True)
class PromptInjectionFinding:
    """一次命中结果，只保留向用户解释所需的安全元数据。"""

    filename: str
    marker: str
    location: str


class PromptInjectionError(ValueError):
    """上传内容包含疑似操控模型行为的指令。"""

    def __init__(
        self,
        findings: str | Iterable[PromptInjectionFinding],
    ) -> None:
        # 保留字符串入参兼容性，避免旧的内部调用改变行为。
        if isinstance(findings, str):
            self.findings: tuple[PromptInjectionFinding, ...] = ()
            message = findings
        else:
            self.findings = tuple(findings)
            details = [
                f"文件《{finding.filename}》{finding.location}发现疑似提示词注入，"
                f"命中规则“{finding.marker}”"
                for finding in self.findings[:5]
            ]
            if len(self.findings) > 5:
                details.append(f"另有 {len(self.findings) - 5} 处命中")
            message = "；".join(details) + "。上传已拒绝。"
        super().__init__(message)


def find_prompt_injections(
    text: str,
    *,
    source_filename: str | None = None,
    regions: Sequence[TextRegion] = (),
) -> list[PromptInjectionFinding]:
    """查找命中规则并返回文件、页码/行号等安全定位信息。

    这里仍然是保守的规则检测，不调用模型，也不把整行文档内容返回给前端。
    PDF 由解析器提供页区域；其他格式至少提供提取文本中的行号。
    """
    if not text:
        return []

    filename = source_filename or "文档"
    candidates: list[tuple[int, int, str, TextRegion | None, int]] = []
    for marker_index, marker in enumerate(PROMPT_INJECTION_MARKERS):
        offset = text.find(marker)
        while offset >= 0:
            region = next(
                (
                    item
                    for item in regions
                    if item.start <= offset < item.end
                ),
                None,
            )
            candidates.append(
                (
                    offset,
                    marker_index,
                    marker,
                    region,
                    text.count("\n", 0, offset) + 1,
                )
            )
            offset = text.find(marker, offset + len(marker))

    # 同一行命中多个关键词时只提示最先出现的一个，避免错误信息过于冗长。
    candidates.sort(key=lambda item: (item[0], item[1]))
    findings: list[PromptInjectionFinding] = []
    seen_locations: set[tuple[str, int]] = set()
    for offset, _marker_index, marker, region, line_number in candidates:
        if region is None:
            location = f"第 {line_number} 行"
            location_key = (location, line_number)
        else:
            page_line = text.count("\n", region.start, offset) + 1
            location = f"{region.label}第 {page_line} 行"
            location_key = (region.label, page_line)
        if location_key in seen_locations:
            continue
        seen_locations.add(location_key)
        findings.append(
            PromptInjectionFinding(
                filename=filename,
                marker=marker,
                location=location,
            )
        )
    return findings


def contains_prompt_injection(text: str) -> bool:
    """使用保守规则识别知识文档中的高风险模型指令。"""
    return bool(find_prompt_injections(text))


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
