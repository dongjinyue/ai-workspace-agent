"""Agent 状态的安全检查点序列化。"""

import json
import re
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from app.agent.messages import normalize_messages, dump_messages, load_messages

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
        "next_node",
        "paused_answer_prefix",
        "omitted_tool_result_ids",
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
    safe_state = {"checkpoint_version": 1}
    safe_messages = []
    omitted = set(state.get("omitted_tool_result_ids", []))
    for message in normalize_messages(state.get("messages", [])):
        # 重新构造最小消息，剥离 artifact、响应元数据和 SDK 原始字段。
        if isinstance(message, ToolMessage):
            omitted.add(message.tool_call_id)
            safe_messages.append(ToolMessage(content="[工具结果已省略，恢复时重新检索]", tool_call_id=message.tool_call_id))
        elif isinstance(message, AIMessage):
            safe_messages.append(AIMessage(content=message.content, tool_calls=message.tool_calls, invalid_tool_calls=message.invalid_tool_calls))
        elif isinstance(message, HumanMessage):
            safe_messages.append(HumanMessage(content=message.content))
        elif isinstance(message, SystemMessage):
            safe_messages.append(SystemMessage(content=message.content))
    safe_state["messages"] = _to_json_safe(dump_messages(safe_messages))
    safe_state["omitted_tool_result_ids"] = sorted(omitted)
    for key, value in state.items():
        if key not in _STATE_FIELDS or key in {"messages", "omitted_tool_result_ids"}:
            continue
        safe_value = _to_json_safe(value, key=key)
        if safe_value is not _OMIT:
            safe_state[key] = safe_value
    return json.dumps(safe_state, ensure_ascii=False, separators=(",", ":"))


def deserialize_agent_state(raw: str | dict) -> AgentState:
    """从字符串或仓库已解析的字典恢复状态，并还原模型消息对象。"""
    # repository（仓库层）读取 SQLite（数据库）时已经把 JSON 解析成字典，
    # 直接复制后再处理，避免恢复流程对同一份检查点重复 json.loads（JSON 解析）。
    decoded = deepcopy(raw) if isinstance(raw, dict) else json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError("检查点必须是 JSON 对象")
    version = decoded.pop("checkpoint_version", 0)
    if type(version) is not int or version not in {0, 1}:
        raise ValueError("不支持的检查点版本")
    decoded["messages"] = load_messages(decoded.get("messages", []), version=version)
    if version == 0:
        decoded["omitted_tool_result_ids"] = [m.tool_call_id for m in decoded["messages"] if isinstance(m, ToolMessage) and m.content.startswith("[工具结果已省略")]
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
    decoded.setdefault("next_node", None)
    decoded.setdefault("paused_answer_prefix", "")
    return decoded


def merge_agent_update(state: AgentState, update: dict[str, Any]) -> AgentState:
    """合并 LangGraph 节点增量，避免修改检查点使用的旧状态对象。"""
    merged = deepcopy(dict(state))
    merged.update(deepcopy(update))
    return merged
