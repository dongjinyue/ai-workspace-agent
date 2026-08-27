from typing import Any, TypedDict


class AgentState(TypedDict):
    """LangGraph 各节点共享的运行状态。"""

    messages: list[Any]
    knowledge_base_id: str | None
    active_skill: str | None
    steps: int
    tools_used: list[str]
    matched_chunks: int
    retrieved_chunks: list[str]
    final_answer: str | None
