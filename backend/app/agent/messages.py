"""隔离旧 SDK 消息、标准消息与持久化格式的差异。"""
import json
from typing import Any

from langchain.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


def _mapping(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return vars(value)


def normalize_messages(messages: list[Any]) -> list[BaseMessage]:
    """旧消息只在输入边界转换，运行态统一为 LangChain 类型。"""
    result = []
    for value in messages:
        if isinstance(value, BaseMessage):
            result.append(value)
            continue
        data = _mapping(value)
        role, content = data.get("role"), data.get("content") or ""
        if role == "user":
            message = HumanMessage(content=content)
        elif role == "system":
            message = SystemMessage(content=content)
        elif role == "tool":
            message = ToolMessage(content=content, tool_call_id=data["tool_call_id"])
        elif role == "assistant":
            calls, invalid = [], []
            for item in data.get("tool_calls") or []:
                call = _mapping(item)
                if "function" not in call:
                    calls.append(call)
                    continue
                function = _mapping(call["function"])
                raw = function.get("arguments", "{}")
                try:
                    args = json.loads(raw)
                    if not isinstance(args, dict):
                        raise ValueError("工具参数必须为对象")
                    calls.append({"name": function["name"], "args": args, "id": call["id"]})
                except (ValueError, TypeError):
                    invalid.append({"name": function["name"], "args": raw, "id": call["id"], "error": "无效工具参数"})
            message = AIMessage(content=content, tool_calls=calls, invalid_tool_calls=invalid)
        else:
            raise ValueError("不支持的消息类型")
        result.append(message)
    return result


def message_text(message: Any) -> str:
    content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
    if isinstance(content, str):
        return content
    # 不把推理块、工具参数或非文本数据混入公开回答。
    return "".join(block.get("text", "") for block in content or [] if isinstance(block, dict) and block.get("type") == "text")


def dump_messages(messages: list[BaseMessage]) -> list[dict]:
    return messages_to_dict(messages)


def load_messages(raw: list[dict], *, version: int) -> list[BaseMessage]:
    if version == 0:
        return normalize_messages(raw)
    if version == 1:
        return messages_from_dict(raw)
    raise ValueError("不支持的检查点版本")
