from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel


@dataclass(frozen=True)
class ToolDefinition:
    """描述一个工具的 Schema（结构）、处理函数和安全边界。"""

    name: str
    schema: dict[str, Any]
    handler: Callable[..., dict] | None
    arguments_model: type[BaseModel] | None = None
    needs_knowledge_base: bool = False
    source: Literal["local", "mcp"] = "local"
    server: str | None = None
    input_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class Skill:
    """把一组相关工具组织成可注册的 Agent（智能代理）能力。"""

    name: str
    description: str
    tools: list[ToolDefinition] = field(default_factory=list)
