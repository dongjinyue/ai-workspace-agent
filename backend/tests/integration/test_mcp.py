from unittest.mock import patch
from types import SimpleNamespace

import pytest

from app.agent.service import (
    ALLOWED_MCP_TOOLS,
    TOOLS,
    discover_mcp_tools,
    execute_tool,
    run_agent,
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


class _MCPSelectingLLM:
    """只模拟模型决策；工具发现和调用仍走真实 MCP 协议。"""

    def __init__(self, final_answer: str = "统计完成") -> None:
        self.calls = 0
        self.final_answer = final_answer

    def create(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            tool_call = SimpleNamespace(
                id="mcp-call",
                function=SimpleNamespace(
                    name="calculate_text_stats",
                    arguments='{"text": "Hello\\nWorld"}',
                ),
            )
            message = SimpleNamespace(content=None, tool_calls=[tool_call])
        else:
            message = SimpleNamespace(
                content=self.final_answer,
                tool_calls=None,
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_agent_selects_and_really_calls_mcp_tool():
    """验证 Agent 路由、动态注册和真实 call_tool 的完整生产路径。"""
    discover_mcp_tools.cache_clear()
    completions = _MCPSelectingLLM()
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    with patch("app.agent.nodes._client", return_value=fake_client):
        result = run_agent("统计 Hello 和 World 的字符数与行数", None)

    assert result.answer == "统计完成"
    assert result.tool_name == "calculate_text_stats"
    assert result.tool_source == "mcp"
    assert result.mcp_server == "workspace"


def test_agent_safely_degrades_when_mcp_server_becomes_unavailable():
    """工具调用阶段故障时，Agent 应返回通用说明且不泄露内部异常。"""
    discover_mcp_tools.cache_clear()
    discover_mcp_tools()
    completions = _MCPSelectingLLM("工具暂时不可用，请稍后再试。")
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    with (
        patch("app.agent.nodes._client", return_value=fake_client),
        patch(
            "app.agent.service.MCPClient.call_tool_sync",
            side_effect=MCPClientError("secret internal crash"),
        ),
    ):
        result = run_agent("统计文本", None)

    assert result.answer == "工具暂时不可用，请稍后再试。"
    assert "secret" not in result.answer


def test_discovery_failure_removes_stale_mcp_schemas_but_keeps_local_tools():
    discover_mcp_tools.cache_clear()
    discover_mcp_tools()
    assert "get_current_time" in TOOLS

    discover_mcp_tools.cache_clear()
    with patch(
        "app.agent.registry.MCPClient.list_tools_sync",
        side_effect=MCPClientError("server down"),
    ):
        from app.agent.registry import get_agent_tool_schemas

        schemas = get_agent_tool_schemas()

    names = {item["function"]["name"] for item in schemas}
    assert "get_current_time" not in names
    assert {"calculator", "search_knowledge_base"} <= names
