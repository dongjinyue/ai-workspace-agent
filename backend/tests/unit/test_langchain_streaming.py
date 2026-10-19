"""同一张真实图在普通、流式和中断条件下的行为。"""
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from app.agent import nodes
from app.agent.service import run_agent


class SequenceModel:
    def __init__(self, turns):
        self.turns = iter(turns)

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None):
        return next(self.turns)

    def stream(self, messages, config=None):
        message = next(self.turns)
        yield AIMessageChunk(content=message.content, tool_calls=message.tool_calls)


def state(**overrides):
    result = dict(messages=[{"role": "user", "content": "计算后查时间"}], knowledge_base_id=None,
                  active_skill=None, steps=0, tools_used=[], retrieved_chunks=[], matched_chunks=0,
                  tool_traces=[], llm_calls=0, llm_duration_ms=0, final_answer=None)
    return {**result, **overrides}


@pytest.mark.parametrize("streaming", [False, True])
def test_two_tools_finish_in_five_steps(monkeypatch, streaming):
    model = SequenceModel([
        AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"a": 2, "b": 3, "operation": "add"}, "id": "one"}]),
        AIMessage(content="", tool_calls=[{"name": "get_current_time", "args": {}, "id": "two"}]),
        AIMessage(content="计算与时间查询完成"),
    ])
    monkeypatch.setattr(nodes, "_client", lambda: model)
    monkeypatch.setattr("app.agent.service.execute_tool", lambda *a, **k: {"result": 5})
    tokens = []
    result = run_agent("计算后查时间", None, on_token=tokens.append if streaming else None)
    assert result.steps == 5
    assert result.llm_calls == 3
    assert result.tools_used == ["calculator", "get_current_time"]
    if streaming:
        assert "".join(tokens) == result.answer


def test_stream_buffers_text_followed_by_tool_chunks(monkeypatch):
    class MixedModel(SequenceModel):
        def stream(self, messages, config=None):
            yield AIMessageChunk(content="这是中间决策")
            yield AIMessageChunk(content="", tool_call_chunks=[{"name": "calculator", "args": '{"a":2,', "id": "one", "index": 0}])
            yield AIMessageChunk(content="", tool_call_chunks=[{"name": None, "args": '"b":3,"operation":"add"}', "id": None, "index": 0}])
    monkeypatch.setattr(nodes, "_client", lambda: MixedModel([]))
    tokens = []
    result = nodes.agent_node(state(stream_callback=tokens.append))
    assert tokens == []
    assert result["messages"][-1].tool_calls[0]["args"] == {"a": 2, "b": 3, "operation": "add"}


def test_pause_partial_tool_call_does_not_persist_or_execute(monkeypatch):
    closed = []
    class PartialModel(SequenceModel):
        def stream(self, messages, config=None):
            try:
                yield AIMessageChunk(content="", tool_call_chunks=[{"name": "calculator", "args": '{"a":', "id": "one", "index": 0}])
                yield AIMessageChunk(content="", tool_call_chunks=[{"name": None, "args": "2}", "id": None, "index": 0}])
            finally:
                closed.append(True)
    monkeypatch.setattr(nodes, "_client", lambda: PartialModel([]))
    polls = iter([None, "pause_requested"])
    result = nodes.agent_node(state(stream_callback=lambda _: None, control_poll=lambda: next(polls, "pause_requested")))
    assert result["control_status"] == "paused"
    assert result["llm_calls"] == 1
    assert result["steps"] == 1
    assert len(result["messages"]) == 1
    assert closed == [True]
