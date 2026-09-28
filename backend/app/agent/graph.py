from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    agent_node,
    route_after_agent,
    route_after_tools,
    tool_node,
)
from app.agent.state import AgentState


def route_from_start(state: AgentState) -> str:
    """从检查点恢复时跳过已经完成的节点。"""
    next_node = state.get("next_node")
    return next_node if next_node in {"agent", "tools", "end"} else "agent"


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_conditional_edges(
    START, route_from_start, {"agent": "agent", "tools": "tools", "end": END}
)
builder.add_conditional_edges(
    "agent", route_after_agent, {"tools": "tools", "end": END}
)
builder.add_conditional_edges(
    "tools", route_after_tools, {"agent": "agent", "end": END}
)

agent_graph = builder.compile()
