import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.agent.service import run_agent


pytestmark = pytest.mark.unit


class SequentialToolLLM:
    """模拟模型连续选择计算器和时间工具，最后输出答案。"""

    def __init__(self):
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            name = "calculator"
            arguments = '{"a": 125, "b": 36, "operation": "multiply"}'
        elif self.calls == 2:
            name = "get_current_time"
            arguments = "{}"
        else:
            message = SimpleNamespace(content="计算和时间查询均已完成", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        tool_call = SimpleNamespace(
            id=f"call-{self.calls}",
            function=SimpleNamespace(name=name, arguments=arguments),
        )
        message = SimpleNamespace(content=None, tool_calls=[tool_call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_agent_loop_can_call_multiple_tools_sequentially():
    completions = SequentialToolLLM()
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    def fake_execute(name, raw_arguments, _knowledge_base_id, **_kwargs):
        arguments = json.loads(raw_arguments)
        if name == "calculator":
            return {"result": arguments["a"] * arguments["b"]}
        return {"source": "mcp", "untrusted_data": "2026-08-31 12:00:00"}

    schemas = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in ("calculator", "get_current_time")
    ]
    with (
        patch("app.agent.nodes._client", return_value=fake_client),
        patch("app.agent.nodes.get_agent_tool_schemas", return_value=schemas),
        patch("app.agent.service.execute_tool", side_effect=fake_execute),
    ):
        result = run_agent("计算 125 * 36，然后告诉我现在几点", None)

    assert result.answer == "计算和时间查询均已完成"
    assert result.tools_used == ["calculator", "get_current_time"]
    assert result.steps == 5


def test_invalid_tool_arguments_are_safely_returned_to_model():
    """工具参数错误不泄露异常，也不应让整个 Agent 请求崩溃。"""
    bad_call = SimpleNamespace(
        id="bad-call",
        function=SimpleNamespace(name="calculator", arguments='{"a": "abc"}'),
    )
    first = SimpleNamespace(content=None, tool_calls=[bad_call])
    final = SimpleNamespace(content="这个表达式无法安全计算。", tool_calls=None)
    responses = [first, final]

    class FakeCompletions:
        def create(self, **_kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=responses.pop(0))])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    with (
        patch("app.agent.nodes._client", return_value=fake_client),
        patch("app.agent.nodes.get_agent_tool_schemas", return_value=[]),
    ):
        result = run_agent("计算 abc + 1", None)

    assert result.answer == "这个表达式无法安全计算。"
    assert result.tool_name == "calculator"
