"""旧工具模块的兼容层。

新代码应从 app.agent.registry 获取工具，或从 app.agent.skills 导入具体实现。
"""

from app.agent.registry import get_tool_schemas
from app.agent.skills.calculator_skill import calculator as calculator_tool
from app.agent.skills.knowledge_skill import knowledge_search as search_knowledge_base_tool


AGENT_TOOL_SCHEMAS = get_tool_schemas()

__all__ = [
    "AGENT_TOOL_SCHEMAS",
    "calculator_tool",
    "search_knowledge_base_tool",
]
