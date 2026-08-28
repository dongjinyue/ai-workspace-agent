import pytest

from app.agent.service import execute_tool


pytestmark = pytest.mark.unit


def test_calculator_happy_path():
    result = execute_tool(
        "calculator",
        '{"a": 137, "b": 29, "operation": "multiply"}',
        None,
    )
    assert result["result"] == 3973


def test_calculator_rejects_division_by_zero():
    with pytest.raises(ValueError, match="除数不能为 0"):
        execute_tool(
            "calculator",
            '{"a": 1, "b": 0, "operation": "divide"}',
            None,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        '{"a": 1, "b": 2, "operation": "power"}',
        '{"a": 1, "b": 2, "operation": "add", "code": "rm -rf /"}',
        '"137 * 29"',
    ],
)
def test_calculator_rejects_invalid_or_dangerous_arguments(arguments):
    with pytest.raises(ValueError):
        execute_tool("calculator", arguments, None)
