import math
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.agent.skill import Skill, ToolDefinition


class CalculatorArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: float
    b: float
    operation: Literal["add", "subtract", "multiply", "divide"]


def calculator(
    a: float,
    b: float,
    operation: Literal["add", "subtract", "multiply", "divide"],
) -> dict:
    """执行有限数字的基础运算，不使用存在代码执行风险的 eval。"""
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("计算器只接受有限数字")
    if operation == "divide" and b == 0:
        raise ValueError("除数不能为 0")

    operations = {
        "add": lambda: a + b,
        "subtract": lambda: a - b,
        "multiply": lambda: a * b,
        "divide": lambda: a / b,
    }
    result = operations[operation]()
    if not math.isfinite(result):
        raise ValueError("计算结果超出支持范围")
    return {"a": a, "b": b, "operation": operation, "result": result}


CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "执行确定性的加、减、乘、除运算。遇到算术问题时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "第一个数字"},
                "b": {"type": "number", "description": "第二个数字"},
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "运算类型：加、减、乘、除",
                },
            },
            "required": ["a", "b", "operation"],
            "additionalProperties": False,
        },
    },
}

CALCULATOR_SKILL = Skill(
    name="calculator",
    description="提供经过参数校验的安全数学计算能力。",
    tools=[ToolDefinition("calculator", CALCULATOR_SCHEMA, calculator, CalculatorArguments)],
)
