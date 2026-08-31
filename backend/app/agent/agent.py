"""Agent 公共入口。

实际循环由 LangGraph 负责；保留这个薄入口让调用方无需了解图的内部结构。
"""

from app.agent.service import AgentResult, run_agent

__all__ = ["AgentResult", "run_agent"]
