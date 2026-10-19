"""累积标准模型分片，在完整调用形成前绝不执行工具。"""
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from langchain_core.messages import AIMessage, AIMessageChunk, message_chunk_to_message


@dataclass
class ModelTurn:
    message: AIMessage | None
    control_status: str | None = None
    answer_prefix: str = ""


def consume_model_stream(chunks: Iterator[AIMessageChunk], *, on_token: Callable[[str], None] | None,
                         control_poll: Callable[[], str | None] | None,
                         allow_tools: bool, buffer_text: bool) -> ModelTurn:
    """工具阶段缓冲正文；纯文本阶段可立即发送，控制信号不保存半成品。"""
    from app.agent.messages import message_text
    accumulated = None
    emitted = []
    try:
        for chunk in chunks:
            status = control_poll() if control_poll else None
            if status:
                return ModelTurn(None, status, "".join(emitted))
            accumulated = chunk if accumulated is None else accumulated + chunk
            text = message_text(chunk)
            if text and not buffer_text and not allow_tools and on_token:
                on_token(text)
                emitted.append(text)
            status = control_poll() if control_poll else None
            if status:
                return ModelTurn(None, status, "".join(emitted))
        if accumulated is None:
            raise RuntimeError("模型没有返回回答或工具调用")
        message = message_chunk_to_message(accumulated)
        # LangChain 的分片解析允许不完整 JSON；执行前必须再按完整 JSON 解析。
        if accumulated.tool_call_chunks:
            valid, invalid = [], []
            for call in accumulated.tool_call_chunks:
                try:
                    args = json.loads(call.get("args") or "")
                    if not isinstance(args, dict) or not call.get("name") or not call.get("id"):
                        raise ValueError("无效工具调用")
                    valid.append({"name": call["name"], "id": call["id"], "args": args})
                except (ValueError, TypeError):
                    invalid.append({"name": call.get("name"), "id": call.get("id"), "args": call.get("args"), "error": "无效工具参数"})
            message = AIMessage(content=message.content, tool_calls=valid, invalid_tool_calls=invalid,
                                usage_metadata=message.usage_metadata)
        if not message.tool_calls and not message.invalid_tool_calls and buffer_text and on_token:
            text = message_text(message)
            if text:
                on_token(text)
        return ModelTurn(message)
    finally:
        close = getattr(chunks, "close", None)
        if callable(close):
            close()
