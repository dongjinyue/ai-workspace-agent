import json
from typing import Any


def format_sse(event: str, data: dict[str, Any]) -> str:
    """把一个事件编码为浏览器可识别的 SSE 文本帧。"""
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )
