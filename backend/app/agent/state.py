from typing import Any, Callable, NotRequired, TypedDict


class AgentState(TypedDict):
    """LangGraph 各节点共享的运行状态。"""

    messages: list[Any]
    knowledge_base_id: str | None
    conversation_id: str | None
    active_skill: str | None
    steps: int
    tools_used: list[str]
    matched_chunks: int
    retrieved_chunks: list[str]
    final_answer: str | None
    tool_traces: list[dict[str, Any]]
    llm_calls: int
    llm_duration_ms: float
    retrieval_debug: NotRequired[dict[str, Any]]
    allowed_tools: NotRequired[frozenset[str] | None]
    # 最近安全边界之后要进入的节点；检查点恢复时由执行器使用。
    next_node: NotRequired[str | None]
    # 模型流式输出期间的控制轮询回调，不会写入持久化检查点。
    control_poll: NotRequired[Callable[[], str | None] | None]
    # 暂停前已经输出的回答前缀，用于恢复时合并后续内容。
    paused_answer_prefix: NotRequired[str]
    # 节点内部检测到的控制状态，由执行器统一转换成 Run 终态。
    control_status: NotRequired[str | None]
    # 仅用于流式接口传递文本片段，不会写入会话或返回给模型。
    stream_callback: NotRequired[Callable[[str], None] | None]
