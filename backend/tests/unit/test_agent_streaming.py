from types import SimpleNamespace
from unittest.mock import patch

from app.agent.nodes import agent_node


def test_agent_node_forwards_final_text_chunks_to_stream_callback():
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="第一"))]
        ),
        # 兼容模型在流结束或返回用量信息时发送的空 choices 数据块。
        SimpleNamespace(choices=[]),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="部分"))]
        ),
    ]
    captured = {}
    tokens = []

    def create(**kwargs):
        captured.update(kwargs)
        return iter(chunks)

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    state = {
        "messages": [{"role": "user", "content": "请继续回答"}],
        "knowledge_base_id": "trusted-kb",
        "conversation_id": None,
        "active_skill": None,
        "steps": 2,
        "tools_used": ["search_knowledge_base"],
        "matched_chunks": 1,
        "retrieved_chunks": ["资料"],
        "final_answer": None,
        "tool_traces": [],
        "llm_calls": 1,
        "llm_duration_ms": 0,
        "allowed_tools": None,
        "stream_callback": tokens.append,
    }

    with patch("app.agent.nodes._client", return_value=fake_client):
        result = agent_node(state)

    assert tokens == ["第一", "部分"]
    assert result["messages"][-1].content == "第一部分"
    assert captured["stream"] is True
    assert captured["tools"] == []
    assert captured["tool_choice"] == "none"
