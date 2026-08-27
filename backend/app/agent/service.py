import json
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent.tools import calculator_tool, search_knowledge_base_tool


class KnowledgeSearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=4000)


class CalculatorArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: float
    b: float
    operation: Literal["add", "subtract", "multiply", "divide"]


@dataclass(frozen=True)
class ToolRegistration:
    function: Callable[..., dict]
    arguments_model: type[BaseModel]
    needs_knowledge_base: bool = False


TOOLS = {
    "search_knowledge_base": ToolRegistration(
        search_knowledge_base_tool,
        KnowledgeSearchArguments,
        needs_knowledge_base=True,
    ),
    "calculator": ToolRegistration(calculator_tool, CalculatorArguments),
}


@dataclass(frozen=True)
class AgentResult:
    answer: str
    tool_called: bool
    tool_name: str | None
    tools_used: list[str]
    steps: int
    matched_chunks: int
    llm_called: bool = True


def execute_tool(
    tool_name: str,
    raw_arguments: str,
    knowledge_base_id: str | None,
) -> dict:
    """执行白名单中的工具，并严格校验模型提供的参数。"""
    registration = TOOLS.get(tool_name)
    if registration is None:
        raise ValueError(f"不允许调用工具：{tool_name}")
    try:
        decoded = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        raise ValueError("工具参数不是有效 JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("工具参数必须是 JSON 对象")

    validated = registration.arguments_model.model_validate(decoded)
    arguments: dict[str, Any] = validated.model_dump()
    if registration.needs_knowledge_base:
        arguments["knowledge_base_id"] = knowledge_base_id
    return registration.function(**arguments)


def _message_content(message: Any) -> str | None:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def run_agent(message: str, knowledge_base_id: str | None) -> AgentResult:
    """从 START 开始调用已编译的 LangGraph 工作流。"""
    from app.agent.graph import agent_graph
    from app.agent.nodes import SYSTEM_PROMPT

    result = agent_graph.invoke(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            "knowledge_base_id": knowledge_base_id,
            "steps": 0,
            "tools_used": [],
            "matched_chunks": 0,
            "retrieved_chunks": [],
            "final_answer": None,
        }
    )
    answer = result.get("final_answer") or _message_content(result["messages"][-1])
    if not answer:
        raise RuntimeError("工作流没有生成最终回答")

    tools_used = result["tools_used"]
    if tools_used and tools_used[-1] == "search_knowledge_base":
        source_text = "\n\n".join(result["retrieved_chunks"])
        if source_text and answer.strip() not in source_text:
            answer = source_text
    return AgentResult(
        answer=answer,
        tool_called=bool(tools_used),
        tool_name=tools_used[-1] if tools_used else None,
        tools_used=tools_used,
        steps=result["steps"],
        matched_chunks=result["matched_chunks"],
    )
