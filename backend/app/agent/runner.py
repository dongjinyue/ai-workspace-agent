"""支持安全边界暂停、恢复、停止和超时的 Agent 执行器。"""

from contextlib import closing
from time import monotonic
from typing import Any, Callable
from uuid import uuid4
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from app.agent.messages import normalize_messages
from app.agent.tracing import traced_agent

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

    def poll(self) -> str | None:
        """返回模型流式输出期间可立即处理的控制状态。"""
        if self._cancel_requested:
            return "stopped"
        if self._clock() >= self._deadline:
            return "timed_out"
        return "pause_requested" if self._pause_requested else None

    def check_boundary(self) -> str | None:
        """在模型或工具节点结束后检查控制信号。"""
        status = self.poll()
        if status == "stopped":
            raise RunCancelled
        if status == "timed_out":
            raise RunTimedOut
        return status


def _result_from_state(state: AgentState, status: str):
    from app.agent.result import result_from_state
    return result_from_state(state, status=status)


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


def prepare_resumed_state(state: AgentState) -> AgentState:
    """结果未持久化时只重取只读检索，不重复计算或外部工具执行。"""
    state = merge_agent_update(state, {})
    state.pop("control_status", None)
    state["messages"] = normalize_messages(state.get("messages", []))
    omitted = set(state.get("omitted_tool_result_ids", []))
    if not omitted or state.get("next_node") == "end" or state.get("final_answer"):
        return state
    # 答案前缀不是检索依据；恢复后仍需完整工具结果才能继续生成事实。
    calls = [(index, call) for index, message in enumerate(state["messages"]) if isinstance(message, AIMessage)
             for call in message.tool_calls if call.get("id") in omitted]
    if calls and len(calls) == len(omitted) and all(call["name"] == "search_knowledge_base" for _, call in calls):
        if state["steps"] <= 3:
            index, call = calls[-1]
            state["messages"] = [*state["messages"][:index], AIMessage(content="", tool_calls=[{
                "name": call["name"], "args": call["args"], "id": uuid4().hex}])]
            state.update(next_node="tools", omitted_tool_result_ids=[], retrieved_chunks=[], paused_answer_prefix="")
            return state
        answer = "恢复所需检索依据未保存，剩余执行步数不足，请重新发起请求。"
    else:
        answer = "任务恢复所需的工具结果未保存，请重新发起请求。"
    state.update(next_node="end", final_answer=answer)
    return state


@traced_agent
def run_resumable_agent(
    initial_state: AgentState,
    *,
    control: RunControl,
    checkpoint_callback: Callable[[AgentState], None],
    status_callback: Callable[[str], None],
    on_token: Callable[[str], None] | None = None,
    config: RunnableConfig | None = None,
):
    """逐节点消费 LangGraph 更新，并在下一节点开始前处理控制信号。"""
    state = prepare_resumed_state(initial_state)
    state["stream_callback"] = on_token
    # 运行时回调只存在于内存中，checkpoint（检查点）序列化会自动排除它。
    state["control_poll"] = control.poll
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
        if control.check_boundary() == "pause_requested":
            return controlled_result("paused")
        while state.get("next_node") != "end":
            stream = agent_graph.stream(state, stream_mode="updates", **({"config": config} if config else {}))
            yielded = False
            with closing(stream):
                for output in stream:
                    for node_name, node_update in output.items():
                        if not isinstance(node_update, dict):
                            raise RuntimeError("Agent 节点返回了无效状态")
                        yielded = True
                        state = merge_agent_update(state, node_update)
                        control_status = state.get("control_status")
                        if control_status in {"paused", "stopped", "timed_out"}:
                            state["next_node"] = "agent" if control_status == "paused" else "end"
                            return controlled_result(control_status)
                        state["next_node"] = _next_node(node_name, state, node_update)
                        checkpoint_callback(state)
                        if state["next_node"] == "end":
                            # 自然耗尽图流，使根追踪正常结束后再返回业务结果。
                            continue
                        boundary = control.check_boundary()
                        if boundary == "pause_requested":
                            return controlled_result("paused")
            if not yielded:
                raise RuntimeError("Agent 工作流没有产生节点更新")
        status_callback("completed")
        result = _result_from_state(state, 'completed')
        if on_token and result.active_skill == 'policy_summary':
            on_token(result.answer)
        return result
    except RunCancelled:
        return controlled_result("stopped")
    except RunTimedOut:
        return controlled_result("timed_out")
