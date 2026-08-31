from functools import lru_cache
from typing import Any

from app.agent.skill import Skill, ToolDefinition
from app.agent.skills.calculator_skill import CALCULATOR_SKILL
from app.agent.skills.knowledge_skill import KNOWLEDGE_SKILL
from app.agent.skills.mcp_skill import MCP_SKILL
from app.agent.skills.time_skill import TIME_SKILL
from app.mcp.client import MCPClient, MCPClientError


SKILLS: tuple[Skill, ...] = (
    TIME_SKILL,
    CALCULATOR_SKILL,
    KNOWLEDGE_SKILL,
    MCP_SKILL,
)

# MCP 工具即使被 Server 暴露，也必须再次通过 Host Allowlist（宿主允许列表）。
ALLOWED_MCP_TOOLS = frozenset({"get_current_time", "calculate_text_stats"})
DYNAMIC_TOOLS: dict[str, ToolDefinition] = {}


def _static_tools() -> dict[str, ToolDefinition]:
    return {tool.name: tool for skill in SKILLS for tool in skill.tools}


# 保持一个稳定字典对象，便于执行层和测试观察动态注册结果。
TOOLS: dict[str, ToolDefinition] = _static_tools()


def register_dynamic_tool(tool: ToolDefinition) -> None:
    """注册已经过宿主安全检查的运行时工具，拒绝覆盖静态工具。"""
    if tool.name in _static_tools():
        raise ValueError(f"动态工具不能覆盖静态工具：{tool.name}")
    DYNAMIC_TOOLS[tool.name] = tool
    TOOLS[tool.name] = tool


def unregister_mcp_tools(server: str) -> None:
    """移除指定 Server 的旧发现结果，避免故障后继续向模型暴露失效工具。"""
    names = [
        name
        for name, tool in DYNAMIC_TOOLS.items()
        if tool.source == "mcp" and tool.server == server
    ]
    for name in names:
        DYNAMIC_TOOLS.pop(name, None)
        TOOLS.pop(name, None)


def get_tool_definitions() -> dict[str, ToolDefinition]:
    return TOOLS


def get_tool_schemas() -> list[dict[str, Any]]:
    return [tool.schema for tool in get_tool_definitions().values()]


def get_tool_handlers() -> dict[str, Any]:
    """为教学和检查提供处理函数视图；MCP 工具由协议客户端执行。"""
    return {
        name: tool.handler
        for name, tool in get_tool_definitions().items()
        if tool.handler is not None
    }


@lru_cache(maxsize=1)
def discover_mcp_tools() -> tuple[dict[str, Any], ...]:
    """通过真实 MCP list_tools 动态发现并注册允许的外部工具。"""
    # 每次重新发现前清理旧快照；若 Server 已宕机，模型不会继续看到过期 Schema。
    unregister_mcp_tools("workspace")
    schemas: list[dict[str, Any]] = []
    for item in MCPClient().list_tools_sync():
        name = item["name"]
        if name not in ALLOWED_MCP_TOOLS:
            continue
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": item["description"],
                "parameters": item["input_schema"],
            },
        }
        register_dynamic_tool(
            ToolDefinition(
                name=name,
                schema=schema,
                handler=None,
                source="mcp",
                server="workspace",
                input_schema=item["input_schema"],
            )
        )
        schemas.append(schema)
    return tuple(schemas)


def get_agent_tool_schemas() -> list[dict[str, Any]]:
    """MCP 不可用时保留所有静态 Skill，不拖垮 Agent。"""
    try:
        discover_mcp_tools()
    except MCPClientError:
        pass
    return get_tool_schemas()
