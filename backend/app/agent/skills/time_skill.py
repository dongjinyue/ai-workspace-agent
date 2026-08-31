from app.agent.skill import Skill


# 时间工具由 MCP Server 在运行时提供，这里只声明能力归属，不直接导入 Server 函数。
TIME_SKILL = Skill(
    name="time",
    description="通过 workspace MCP Server 提供当前日期和时间能力。",
)
