"""用户可见执行结果与基于原文的政策摘要校验。"""
from dataclasses import dataclass
from typing import Any
from app.agent.state import AgentState
from app.agent.registry import TOOLS
from app.agent.messages import message_text
from app.agent.messages import normalize_messages
from langchain_core.messages import AIMessage
from app.security import PROMPT_INJECTION_MARKERS

@dataclass(frozen=True)
class AgentResult:
    answer: str
    tool_called: bool
    tool_name: str | None
    tools_used: list[str]
    active_skill: str | None
    steps: int
    matched_chunks: int
    tool_traces: list[dict[str, Any]] | None = None
    llm_calls: int = 0
    llm_duration_ms: float = 0.0
    tool_source: str | None = None
    mcp_server: str | None = None
    llm_called: bool = True
    # 可暂停执行器使用；普通同步调用始终为 completed。
    status: str = "completed"
    retrieval_debug: dict[str, Any] | None = None


POLICY_SUMMARY_FIELDS = (
    "政策名称",
    "适用条件",
    "关键规定",
    "所需材料",
    "例外情况",
)
def _trusted_source_text(chunks: list[str]) -> str:
    """丢弃包含注入标记的整个检索块，避免分行载荷绕过。"""
    return "\n\n".join(
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
        and not any(marker in chunk for marker in PROMPT_INJECTION_MARKERS)
    )


def _ground_policy_summary(answer: str, chunks: list[str]) -> str:
    """只保留能够逐字追溯到知识库、且不像注入指令的字段值。"""
    source_text = _trusted_source_text(chunks)
    values: dict[str, list[str]] = {field: [] for field in POLICY_SUMMARY_FIELDS}
    current_field: str | None = None

    for raw_line in answer.splitlines():
        line = raw_line.strip()
        matched_field = next(
            (field for field in POLICY_SUMMARY_FIELDS if line.startswith(f"{field}：")),
            None,
        )
        if matched_field:
            current_field = matched_field
            inline_value = line.removeprefix(f"{matched_field}：").strip()
            if inline_value:
                values[matched_field].append(inline_value)
        elif current_field and line:
            values[current_field].append(line)

    output = []
    for field in POLICY_SUMMARY_FIELDS:
        candidates = values[field]
        grounded_values = []
        for candidate in candidates:
            value = candidate.removeprefix("- ").strip()
            looks_like_injection = any(
                marker in value for marker in PROMPT_INJECTION_MARKERS
            )
            if (
                value == "知识库未说明"
                or (value and value in source_text and not looks_like_injection)
            ):
                grounded_values.append(candidate)
        field_value = "\n".join(grounded_values) or "知识库未说明"
        output.append(f"{field}：{field_value}")
    return "\n".join(output)


def result_from_state(state: AgentState, *, status: str = "completed") -> AgentResult:
    """所有执行入口共用答案校验；未完成任务允许没有正文。"""
    messages = normalize_messages(state.get("messages", []))
    last = messages[-1] if messages else None
    # 系统指令、工具结果和尚待执行的工具调用都不是用户可见的最终回答。
    text = message_text(last) if isinstance(last, AIMessage) and not last.tool_calls and not last.invalid_tool_calls else ""
    answer = state.get("final_answer") or text
    if status == "completed" and not any(character.isalnum() for character in answer):
        raise RuntimeError("工作流没有生成最终回答")
    tools = list(state.get("tools_used", []))
    if status == "completed" and not state.get("final_answer") and state.get("active_skill") == "policy_summary":
        answer = _ground_policy_summary(answer, state.get("retrieved_chunks", []))
    registration = TOOLS.get(tools[-1]) if tools else None
    return AgentResult(
        answer=answer, tool_called=bool(tools), tool_name=tools[-1] if tools else None,
        tools_used=tools, active_skill=state.get("active_skill"), steps=state.get("steps", 0),
        matched_chunks=state.get("matched_chunks", 0), tool_traces=state.get("tool_traces", []),
        llm_calls=state.get("llm_calls", 0), llm_duration_ms=state.get("llm_duration_ms", 0),
        retrieval_debug=state.get("retrieval_debug", {}), tool_source=registration.source if registration else None,
        mcp_server=registration.server if registration and registration.source == "mcp" else None, status=status,
    )
