import math
from typing import Literal

from app.rag.service import semantic_search


KNOWLEDGE_BASE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "搜索当前用户的企业知识库。"
            "当问题涉及公司政策、产品规则、员工制度或上传资料中的事实时使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "需要在企业知识库中搜索的问题",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


CALCULATOR_TOOL = {
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


AGENT_TOOL_SCHEMAS = [KNOWLEDGE_BASE_TOOL, CALCULATOR_TOOL]


def search_knowledge_base_tool(
    knowledge_base_id: str | None,
    query: str,
) -> dict:
    """返回检索事实，不生成最终自然语言答案。"""
    if not knowledge_base_id:
        return {"matched": False, "chunks": []}

    matches = semantic_search(knowledge_base_id, query)
    return {
        "matched": bool(matches),
        "chunks": [match.document for match in matches],
        "similarities": [round(match.similarity, 4) for match in matches],
    }


def calculator_tool(
    a: float,
    b: float,
    operation: Literal["add", "subtract", "multiply", "divide"],
) -> dict:
    """执行有限浮点数的确定性算术运算。"""
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("计算器只接受有限数字")

    operations = {
        "add": lambda: a + b,
        "subtract": lambda: a - b,
        "multiply": lambda: a * b,
        "divide": lambda: a / b,
    }
    if operation == "divide" and b == 0:
        raise ValueError("除数不能为 0")

    result = operations[operation]()
    if not math.isfinite(result):
        raise ValueError("计算结果超出支持范围")

    return {
        "a": a,
        "b": b,
        "operation": operation,
        "result": result,
    }
