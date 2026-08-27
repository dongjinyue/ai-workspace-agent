from langgraph.graph import END, START, StateGraph

from app.agent.nodes import agent_node, route_after_agent, route_after_tools, tool_node
from app.agent.state import AgentState


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", route_after_agent, {"tools": "tools", "end": END}
)
builder.add_conditional_edges(
    "tools", route_after_tools, {"agent": "agent", "end": END}
)

agent_graph = builder.compile()

