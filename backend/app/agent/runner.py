"""支持安全边界暂停、恢复、停止和超时的 Agent 执行器。"""

from time import monotonic
from typing import Any, Callable

from app.agent.checkpoint import merge_agent_update
from app.agent.graph import agent_graph
from app.agent.state import AgentState


class RunCancelled(Exception):
    """主动停止信号，只在节点边界被处理。"""


class RunTimedOut(Exception):
    """总执行时限信号，只在节点边界被处理。"""


class RunControl:
    """Run（执行任务）的线程安全控制标记。"""

    def __init__(self, timeout_seconds: float = 120.0, *, clock=monotonic):
        if timeout_seconds <= 0:
            raise ValueError("任务超时时间必须大于 0 秒")
        self._clock = clock
        self._deadline = clock() + timeout_seconds
        self._pause_requested = False
        self._cancel_requested = False

    def request_pause(self) -> None:
        self._pause_requested = True

    def cancel(self) -> None:
        self._cancel_requested = True

    def check_boundary(self) -> str | None:
        """在模型或工具节点结束后检查控制信号。"""
        if self._cancel_requested:
            raise RunCancelled
        if self._clock() >= self._deadline:
            raise RunTimedOut
        return "pause_requested" if self._pause_requested else None


def _message_content(message: Any) -> str | None:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _result_from_state(state: AgentState, status: str):
    """将公开状态转换成原有 AgentResult，保持普通调用方兼容。"""
    from app.agent.registry import TOOLS
    from app.agent.service import AgentResult

    messages = state.get("messages", [])
    answer = state.get("final_answer")
    if not answer and messages:
        answer = _message_content(messages[-1])
    tools_used = list(state.get("tools_used", []))
    last_registration = TOOLS.get(tools_used[-1]) if tools_used else None
    return AgentResult(
        answer=str(answer or ""),
        tool_called=bool(tools_used),
        tool_name=tools_used[-1] if tools_used else None,
        tools_used=tools_used,
        active_skill=state.get("active_skill"),
        steps=state.get("steps", 0),
        matched_chunks=state.get("matched_chunks", 0),
        tool_traces=state.get("tool_traces", []),
        llm_calls=state.get("llm_calls", 0),
        llm_duration_ms=state.get("llm_duration_ms", 0.0),
        tool_source=last_registration.source if last_registration else None,
        mcp_server=(
            last_registration.server
            if last_registration and last_registration.source == "mcp"
            else None
        ),
        status=status,
    )


def _next_node(node_name: str, state: AgentState, node_update: dict[str, Any]) -> str:
    requested = node_update.get("next_node")
    if requested in {"agent", "tools", "end"}:
        return requested
    from app.agent.nodes import route_after_agent, route_after_tools

    if node_name == "agent":
        return route_after_agent(state)
    if node_name == "tools":
        return route_after_tools(state)
    raise RuntimeError(f"未知的 Agent 节点：{node_name}")


def run_resumable_agent(
    initial_state: AgentState,
    *,
    control: RunControl,
    checkpoint_callback: Callable[[AgentState], None],
    status_callback: Callable[[str], None],
    on_token: Callable[[str], None] | None = None,
):
    """逐节点消费 LangGraph 更新，并在下一节点开始前处理控制信号。"""
    state = merge_agent_update(initial_state, {})
    state["stream_callback"] = on_token
    state["next_node"] = (
        state.get("next_node")
        if state.get("next_node") in {"agent", "tools", "end"}
        else "agent"
    )
    status_callback("running")

    def controlled_result(status: str):
        checkpoint_callback(state)
        status_callback(status)
        return _result_from_state(state, status)

    try:
        control.check_boundary()
        while state.get("next_node") != "end":
            stream = agent_graph.stream(state, stream_mode="updates")
            yielded = False
            for output in stream:
                for node_name, node_update in output.items():
                    if not isinstance(node_update, dict):
                        raise RuntimeError("Agent 节点返回了无效状态")
                    yielded = True
                    state = merge_agent_update(state, node_update)
                    state["next_node"] = _next_node(node_name, state, node_update)
                    checkpoint_callback(state)
                    if state["next_node"] == "end":
                        status_callback("completed")
                        return _result_from_state(state, "completed")
                    boundary = control.check_boundary()
                    if boundary == "pause_requested":
                        return controlled_result("paused")
            if not yielded:
                raise RuntimeError("Agent 工作流没有产生节点更新")
        status_callback("completed")
        return _result_from_state(state, "completed")
    except RunCancelled:
        return controlled_result("stopped")
    except RunTimedOut:
        return controlled_result("timed_out")
