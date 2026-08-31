import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Coroutine, TypeVar

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


BACKEND_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_SERVER = BACKEND_DIR / "mcp_servers" / "workspace_server.py"
T = TypeVar("T")
logger = logging.getLogger(__name__)

MCP_ENV_ALLOWLIST = (
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
)


def _server_environment() -> dict[str, str]:
    """只向 MCP 子进程传递启动所需变量，避免泄露后端密钥。"""
    environment = {
        name: os.environ[name]
        for name in MCP_ENV_ALLOWLIST
        if os.environ.get(name)
    }
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    return environment


class MCPClientError(RuntimeError):
    """MCP Server 不可用或返回了无效响应。"""


class MCPClient:
    """通过 stdio（标准输入输出）启动并访问 workspace MCP Server。"""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        if timeout_seconds <= 0 or timeout_seconds > 60:
            raise ValueError("MCP 超时时间必须大于 0 且不超过 60 秒")
        if not WORKSPACE_SERVER.is_file():
            raise MCPClientError("workspace MCP Server 当前不可用")
        self.timeout_seconds = timeout_seconds
        self.server = StdioServerParameters(
            # 使用当前 Python 解释器和参数数组启动，不经过 Shell（命令解释器）。
            command=sys.executable,
            args=[str(WORKSPACE_SERVER)],
            cwd=BACKEND_DIR,
            env=_server_environment(),
            encoding="utf-8",
            encoding_error_handler="replace",
        )

    async def _with_session(self, operation: str, **kwargs: Any) -> Any:
        try:
            async with asyncio.timeout(self.timeout_seconds):
                async with stdio_client(self.server) as (read, write):
                    async with ClientSession(
                        read,
                        write,
                        read_timeout_seconds=self.timeout_seconds,
                    ) as session:
                        await session.initialize()
                        if operation == "list_tools":
                            return await session.list_tools()
                        if operation == "call_tool":
                            return await session.call_tool(
                                kwargs["name"], kwargs.get("arguments", {})
                            )
                        raise ValueError("不支持的 MCP 操作")
        except Exception as error:
            # 日志只记录异常类型，不记录工具参数、文档内容或密钥。
            logger.warning(
                "MCP operation failed operation=%s error_type=%s",
                operation,
                type(error).__name__,
            )
            raise MCPClientError("workspace MCP Server 当前不可用") from error

    async def list_tools(self) -> list[dict[str, Any]]:
        """真实执行 Tool Discovery（工具发现），并返回可序列化的工具描述。"""
        response = await self._with_session("list_tools")
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema,
            }
            for tool in response.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """通过 MCP 协议调用工具，不直接导入 Server 中的 Python 函数。"""
        response = await self._with_session(
            "call_tool", name=name, arguments=arguments
        )
        if getattr(response, "isError", False):
            raise MCPClientError("MCP Tool 执行失败")
        structured = getattr(response, "structured_content", None)
        if structured is not None:
            return structured
        texts = [
            item.text for item in response.content if getattr(item, "type", None) == "text"
        ]
        if len(texts) == 1:
            try:
                return json.loads(texts[0])
            except json.JSONDecodeError:
                return texts[0]
        return texts

    @staticmethod
    def _run(coroutine: Coroutine[Any, Any, T]) -> T:
        """为当前同步 LangGraph 节点运行异步 MCP API。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        raise MCPClientError("不能在已有异步事件循环中使用同步 MCP Client")

    def list_tools_sync(self) -> list[dict[str, Any]]:
        return self._run(self.list_tools())

    def call_tool_sync(self, name: str, arguments: dict[str, Any]) -> Any:
        return self._run(self.call_tool(name, arguments))
