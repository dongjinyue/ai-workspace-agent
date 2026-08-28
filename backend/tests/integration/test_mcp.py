from unittest.mock import patch

import pytest

from app.agent.service import (
    ALLOWED_MCP_TOOLS,
    TOOLS,
    discover_mcp_tools,
    execute_tool,
)
from app.mcp.client import MCPClient, MCPClientError


pytestmark = pytest.mark.integration


def test_mcp_server_environment_does_not_inherit_backend_secrets():
    with patch.dict(
        "os.environ",
        {
            "DASHSCOPE_API_KEY": "should-never-reach-mcp",
            "QWEN_BASE_URL": "https://secret.example",
            "PATH": "safe-path",
            "SYSTEMROOT": r"C:\Windows",
        },
        clear=True,
    ):
        environment = MCPClient().server.env

    assert environment["PATH"] == "safe-path"
    assert environment["SYSTEMROOT"] == r"C:\Windows"
    assert environment["PYTHONUTF8"] == "1"
    assert environment["PYTHONIOENCODING"] == "utf-8"
    assert "DASHSCOPE_API_KEY" not in environment
    assert "QWEN_BASE_URL" not in environment


def test_real_mcp_tool_discovery_over_stdio():
    tools = MCPClient().list_tools_sync()
    names = {tool["name"] for tool in tools}
    assert names == {"get_current_time", "calculate_text_stats"}


def test_real_mcp_tool_call_over_stdio():
    result = MCPClient().call_tool_sync(
        "calculate_text_stats", {"text": "Hello\nWorld"}
    )
    assert result == {"characters": 11, "lines": 2}


def test_discovery_registers_only_allowlisted_tools():
    discover_mcp_tools.cache_clear()
    discovered = [
        {
            "name": "get_current_time",
            "description": "time",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "delete_all_files",
            "description": "unsafe",
            "input_schema": {"type": "object"},
        },
    ]
    with patch("app.agent.service.MCPClient.list_tools_sync", return_value=discovered):
        schemas = discover_mcp_tools()
    assert {item["function"]["name"] for item in schemas} <= ALLOWED_MCP_TOOLS
    assert "delete_all_files" not in TOOLS
    discover_mcp_tools.cache_clear()


def test_mcp_arguments_are_validated_against_discovered_schema():
    discover_mcp_tools.cache_clear()
    discover_mcp_tools()
    with pytest.raises(ValueError, match="JSON Schema"):
        execute_tool(
            "calculate_text_stats", '{"text": 123}', None
        )


def test_mcp_result_is_marked_as_untrusted_data():
    discover_mcp_tools.cache_clear()
    discover_mcp_tools()
    result = execute_tool(
        "calculate_text_stats", '{"text": "Hello\\nWorld"}', None
    )
    assert result == {
        "source": "mcp",
        "server": "workspace",
        "untrusted_data": {"characters": 11, "lines": 2},
    }


def test_mcp_failure_does_not_expose_internal_exception():
    discover_mcp_tools.cache_clear()
    discover_mcp_tools()
    with patch(
        "app.agent.service.MCPClient.call_tool_sync",
        side_effect=MCPClientError("secret stack and API key"),
    ):
        with pytest.raises(RuntimeError, match="MCP 工具当前不可用") as error:
            execute_tool("get_current_time", "{}", None)
    assert "secret" not in str(error.value)
