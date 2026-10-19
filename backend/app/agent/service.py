from app.agent.result import AgentResult, result_from_state, _ground_policy_summary
from app.agent.messages import normalize_messages
from app.agent.tracing import traced_agent
import json
from typing import Any, Callable
from langchain_core.runnables import RunnableConfig

# 保留旧模块的注册表和客户端导出，兼容既有检查及调用方。
from app.agent.registry import (
    ALLOWED_MCP_TOOLS,
    TOOLS,
    discover_mcp_tools,
    get_agent_tool_schemas,
)
from app.mcp.client import MCPClient, MCPClientError
from app.agent.tool_adapter import invoke_registered_tool


# 访客只获得无副作用的计算、公开检索和已验证的只读 MCP 工具。
GUEST_ALLOWED_TOOLS = frozenset(
    {
        "calculator",
        "search_knowledge_base",
        "get_current_time",
        "calculate_text_stats",
    }
)


def execute_tool(
    tool_name: str,
    raw_arguments: str,
    knowledge_base_id: str | None,
    *,
    allowed_tools: frozenset[str] | tuple[str, ...] | None = None,
    config: RunnableConfig | None = None,
) -> dict:
    """执行白名单中的工具，并严格校验模型提供的参数。"""
    try:
        decoded = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        raise ValueError("工具参数不是有效 JSON") from error
    return invoke_registered_tool(tool_name, decoded, knowledge_base_id, allowed_tools=allowed_tools, config=config)


@traced_agent
def run_agent(
    message: str,
    knowledge_base_id: str | None,
    *,
    conversation_id: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    allowed_tools: frozenset[str] | None = None,
    on_token: Callable[[str], None] | None = None,
    config: RunnableConfig | None = None,
) -> AgentResult:
    """从 START 开始调用已编译的 LangGraph 工作流。"""
    from app.agent.graph import agent_graph
    from app.skills.registry import select_skill

    skill = select_skill(message)

    result = agent_graph.invoke(
        {
            "messages": normalize_messages(history_messages
            if history_messages is not None
            else [{"role": "user", "content": message}]),
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
            "retrieval_debug": {},
            "allowed_tools": allowed_tools,
            "stream_callback": on_token,
        }, config=config,
    )
    public_result = result_from_state(result)
    if on_token and public_result.active_skill == 'policy_summary':
        on_token(public_result.answer)
    return public_result
