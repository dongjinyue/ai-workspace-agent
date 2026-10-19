import pytest
from langchain_core.tools import BaseTool
from app.agent import service


def test_calculator_uses_real_structured_tool():
    assert hasattr(service, "invoke_registered_tool")
    from app.agent.tool_adapter import build_tool
    tool = build_tool("calculator", knowledge_base_id=None, allowed_tools=None)
    assert isinstance(tool, BaseTool)
    assert tool.invoke({"a": 1, "b": 2, "operation": "add"})["result"] == 3
    assert service.invoke_registered_tool("calculator", {"a": 1, "b": 2, "operation": "add"}, None)["result"] == 3


def test_tool_allowlist_is_enforced_at_execution():
    assert hasattr(service, "invoke_registered_tool")
    with pytest.raises(ValueError, match="不允许"):
        service.invoke_registered_tool("calculator", {"a": 1, "b": 2, "operation": "add"}, None, allowed_tools=frozenset())


def test_model_cannot_override_knowledge_context():
    assert hasattr(service, "invoke_registered_tool")
    with pytest.raises(ValueError):
        service.invoke_registered_tool("search_knowledge_base", {"query": "问题", "knowledge_base_id": "other"}, "trusted")
