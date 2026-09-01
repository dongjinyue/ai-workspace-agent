import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from jsonschema import ValidationError, validate
from app.agent.registry import (
    ALLOWED_MCP_TOOLS,
    TOOLS,
    discover_mcp_tools,
    get_agent_tool_schemas,
)
from app.mcp.client import MCPClient, MCPClientError
from app.security import PROMPT_INJECTION_MARKERS
from app.agent.llm import create_llm_client, model_name


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


def execute_tool(
    tool_name: str,
    raw_arguments: str,
    knowledge_base_id: str | None,
    *,
    allowed_tools: tuple[str, ...] | None = None,
) -> dict:
    """执行白名单中的工具，并严格校验模型提供的参数。"""
    registration = TOOLS.get(tool_name)
    if registration is None:
        raise ValueError(f"不允许调用工具：{tool_name}")
    if allowed_tools is not None and tool_name not in allowed_tools:
        raise ValueError(f"当前技能不允许调用工具：{tool_name}")
    try:
        decoded = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        raise ValueError("工具参数不是有效 JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("工具参数必须是 JSON 对象")

    if registration.source == "mcp":
        if tool_name not in ALLOWED_MCP_TOOLS or registration.input_schema is None:
            raise ValueError(f"不允许调用 MCP 工具：{tool_name}")
        try:
            validate(instance=decoded, schema=registration.input_schema)
        except ValidationError as error:
            raise ValueError("MCP 工具参数不符合 JSON Schema") from error
        try:
            value = MCPClient().call_tool_sync(tool_name, decoded)
        except MCPClientError as error:
            raise RuntimeError("MCP 工具当前不可用") from error
        return {
            "source": "mcp",
            "server": registration.server,
            "untrusted_data": value,
        }

    if registration.arguments_model is None or registration.handler is None:
        raise RuntimeError("本地工具注册不完整")
    validated = registration.arguments_model.model_validate(decoded)
    arguments: dict[str, Any] = validated.model_dump()
    if registration.needs_knowledge_base:
        arguments["knowledge_base_id"] = knowledge_base_id
    return registration.handler(**arguments)


def _message_content(message: Any) -> str | None:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


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


def run_agent(
    message: str,
    knowledge_base_id: str | None,
    *,
    conversation_id: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
) -> AgentResult:
    """从 START 开始调用已编译的 LangGraph 工作流。"""
    from app.agent.graph import agent_graph
    from app.skills.registry import select_skill

    skill = select_skill(message)

    result = agent_graph.invoke(
        {
            "messages": history_messages
            if history_messages is not None
            else [{"role": "user", "content": message}],
            "conversation_id": conversation_id,
            "knowledge_base_id": knowledge_base_id,
            "active_skill": skill.name if skill else None,
            "steps": 0,
            "tools_used": [],
            "matched_chunks": 0,
            "retrieved_chunks": [],
            "final_answer": None,
            "tool_traces": [],
            "llm_calls": 0,
            "llm_duration_ms": 0.0,
        }
    )
    answer = result.get("final_answer") or _message_content(result["messages"][-1])
    # 兼容模型偶发返回省略号或纯标点；这种内容不能作为成功回答保存。
    meaningful_answer = "".join(
        character for character in str(answer or "") if character.isalnum()
    )
    if not meaningful_answer:
        raise RuntimeError("工作流没有生成最终回答")

    tools_used = result["tools_used"]
    last_registration = TOOLS.get(tools_used[-1]) if tools_used else None
    if tools_used and tools_used[-1] == "search_knowledge_base":
        source_text = "\n\n".join(result["retrieved_chunks"])
        if result["active_skill"] == "policy_summary" and source_text:
            answer = _ground_policy_summary(answer, result["retrieved_chunks"])
    return AgentResult(
        answer=answer,
        tool_called=bool(tools_used),
        tool_name=tools_used[-1] if tools_used else None,
        tools_used=tools_used,
        active_skill=result["active_skill"],
        steps=result["steps"],
        matched_chunks=result["matched_chunks"],
        tool_traces=result.get("tool_traces", []),
        llm_calls=result.get("llm_calls", 0),
        llm_duration_ms=result.get("llm_duration_ms", 0.0),
        tool_source=last_registration.source if last_registration else None,
        mcp_server=(
            last_registration.server
            if last_registration and last_registration.source == "mcp"
            else None
        ),
    )
