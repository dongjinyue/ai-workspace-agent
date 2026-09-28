"""Agent 状态的安全检查点序列化。"""

import json
import re
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from openai.types.chat import ChatCompletionMessage

from app.agent.state import AgentState


_STATE_FIELDS = frozenset(
    {
        "messages",
        "knowledge_base_id",
        "conversation_id",
        "active_skill",
        "steps",
        "tools_used",
        "matched_chunks",
        "retrieved_chunks",
        "final_answer",
        "tool_traces",
        "llm_calls",
        "llm_duration_ms",
        "retrieval_debug",
        "allowed_tools",
    }
)
_SECRET_PATTERN = re.compile(
    r"(?:sk-|api[_-]?key\s*[:=]\s*|token\s*[:=]\s*|secret\s*[:=]\s*)[^\s,;]+",
    re.IGNORECASE,
)
_OMIT = object()


def _redact_text(value: str) -> str:
    return _SECRET_PATTERN.sub("[已脱敏]", value)


def _to_json_safe(value: Any, *, key: str | None = None) -> Any:
    """把模型对象转换为普通 JSON，并删除运行时回调和文档正文。"""
    if key == "stream_callback":
        return _OMIT
    if key == "retrieved_chunks":
        # 检查点只记录检索数量，恢复时由 Agent 重新取得可信上下文。
        return []
    if key == "content" and isinstance(value, str) and key == "content":
        # 工具结果可能包含完整知识库 chunk，不能进入持久化检查点。
        return _redact_text(value)
    if callable(value):
        return _OMIT
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        result = {}
        for child_key, child_value in value.items():
            if child_key == "stream_callback":
                continue
            if child_key == "retrieved_chunks":
                result[child_key] = []
                continue
            safe_value = _to_json_safe(child_value, key=str(child_key))
            if safe_value is not _OMIT:
                result[str(child_key)] = safe_value
        if result.get("role") == "tool":
            result["content"] = "[工具结果已省略，恢复时重新检索]"
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        result = []
        for item in value:
            safe_value = _to_json_safe(item)
            if safe_value is not _OMIT:
                result.append(safe_value)
        return result
    if hasattr(value, "model_dump"):
        return _to_json_safe(value.model_dump(mode="json", exclude_none=True), key=key)
    if isinstance(value, SimpleNamespace) or hasattr(value, "__dict__"):
        return _to_json_safe(vars(value), key=key)
    return str(value)


def serialize_agent_state(state: AgentState) -> str:
    """只序列化 AgentState 的公开可恢复字段，不持久化回调或未知字段。"""
    safe_state = {}
    for key, value in state.items():
        if key not in _STATE_FIELDS:
            continue
        safe_value = _to_json_safe(value, key=key)
        if safe_value is not _OMIT:
            safe_state[key] = safe_value
    return json.dumps(safe_state, ensure_ascii=False, separators=(",", ":"))


def _restore_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return message
    if message.get("role") == "assistant" and message.get("tool_calls"):
        return ChatCompletionMessage.model_validate(message)
    return message


def deserialize_agent_state(raw: str) -> AgentState:
    """从检查点恢复状态，并把带工具调用的助手消息还原为模型对象。"""
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError("检查点必须是 JSON 对象")
    decoded["messages"] = [
        _restore_message(message) for message in decoded.get("messages", [])
    ]
    if isinstance(decoded.get("allowed_tools"), list):
        decoded["allowed_tools"] = frozenset(decoded["allowed_tools"])
    decoded.setdefault("retrieved_chunks", [])
    decoded.setdefault("tool_traces", [])
    decoded.setdefault("tools_used", [])
    decoded.setdefault("matched_chunks", 0)
    decoded.setdefault("steps", 0)
    decoded.setdefault("llm_calls", 0)
    decoded.setdefault("llm_duration_ms", 0.0)
    decoded.setdefault("retrieval_debug", {})
    return decoded


def merge_agent_update(state: AgentState, update: dict[str, Any]) -> AgentState:
    """合并 LangGraph 节点增量，避免修改检查点使用的旧状态对象。"""
    merged = deepcopy(dict(state))
    merged.update(deepcopy(update))
    return merged
