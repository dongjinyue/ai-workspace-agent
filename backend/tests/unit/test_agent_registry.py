import pytest

from app.agent.registry import (
    DYNAMIC_TOOLS,
    SKILLS,
    TOOLS,
    get_tool_handlers,
    get_tool_schemas,
    register_dynamic_tool,
)
from app.agent.skill import ToolDefinition


pytestmark = pytest.mark.unit


def test_static_skills_automatically_supply_schemas_and_handlers():
    """新增静态 Skill 后，注册表应自动汇总结构和处理函数。"""
    skill_names = {skill.name for skill in SKILLS}
    schema_names = {item["function"]["name"] for item in get_tool_schemas()}
    handlers = get_tool_handlers()

    assert {"time", "calculator", "knowledge", "mcp"} <= skill_names
    assert {
        "calculator",
        "search_knowledge_base",
        "get_knowledge_base_info",
    } <= schema_names
    assert {
        "calculator",
        "search_knowledge_base",
        "get_knowledge_base_info",
    } <= handlers.keys()


def test_registry_supports_dynamic_tool_registration():
    """模拟 MCP 工具发现，确认 Schema 与执行元数据同时进入注册表。"""
    name = "test_dynamic_weather"
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": "测试天气工具",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    definition = ToolDefinition(
        name=name,
        schema=schema,
        handler=None,
        source="mcp",
        server="test",
        input_schema=schema["function"]["parameters"],
    )

    try:
        register_dynamic_tool(definition)
        assert TOOLS[name] is definition
        assert schema in get_tool_schemas()
    finally:
        # 测试完成后清理全局注册状态，避免影响其他用例。
        DYNAMIC_TOOLS.pop(name, None)
        TOOLS.pop(name, None)


def test_dynamic_tool_cannot_override_static_tool():
    with pytest.raises(ValueError, match="不能覆盖静态工具"):
        register_dynamic_tool(TOOLS["calculator"])
