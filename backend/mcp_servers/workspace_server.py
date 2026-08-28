from datetime import datetime

from mcp.server.mcpserver import MCPServer


mcp = MCPServer("workspace")


@mcp.tool()
def get_current_time() -> str:
    """获取服务器当前本地时间，返回 ISO 8601 格式的时间字符串。"""
    return datetime.now().astimezone().isoformat()


@mcp.tool()
def calculate_text_stats(text: str) -> dict[str, int]:
    """统计文本的 Unicode 字符数和行数；空文本也按一行计算。"""
    return {
        "characters": len(text),
        "lines": len(text.splitlines()) or 1,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
