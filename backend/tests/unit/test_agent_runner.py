from types import SimpleNamespace

import pytest

from app.agent.runner import RunControl, run_resumable_agent


pytestmark = pytest.mark.unit


def _initial_state():
    return {
        "messages": [{"role": "user", "content": "需要一个可暂停的回答"}],
        "knowledge_base_id": None,
        "conversation_id": "conversation-runner",
        "active_skill": None,
        "steps": 0,
        "tools_used": [],
        "matched_chunks": 0,
        "retrieved_chunks": [],
        "final_answer": None,
        "tool_traces": [],
        "llm_calls": 0,
        "llm_duration_ms": 0.0,
        "allowed_tools": None,
    }


class FakeGraph:
    def __init__(self):
        self.nodes = []

    def stream(self, state, *, stream_mode):
        assert stream_mode == "updates"
        node = state.get("next_node", "agent")
        self.nodes.append(node)
        if node == "agent":
            tool_call = SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(name="calculator", arguments="{}"),
            )
            yield {
                "agent": {
                    "messages": [
                        SimpleNamespace(
                            role="assistant", content=None, tool_calls=[tool_call]
                        )
                    ],
                    "steps": 1,
                    "llm_calls": 1,
                    "next_node": "tools",
                }
            }
        elif node == "tools":
            yield {
                "tools": {
                    "messages": [
                        {"role": "assistant", "content": "完成"}
                    ],
                    "steps": 2,
                    "tools_used": ["calculator"],
                    "final_answer": "完成",
                    "next_node": "end",
                }
            }


def test_pause_saves_next_node_and_resume_does_not_repeat_completed_node(monkeypatch):
    graph = FakeGraph()
    monkeypatch.setattr("app.agent.runner.agent_graph", graph)
    checkpoints = []
    control = RunControl(timeout_seconds=60)

    def checkpoint_callback(state):
        checkpoints.append(state)
        if len(checkpoints) == 1:
            control.request_pause()

    paused = run_resumable_agent(
        _initial_state(),
        control=control,
        checkpoint_callback=checkpoint_callback,
        status_callback=lambda _status: None,
        on_token=None,
    )

    assert paused.status == "paused"
    assert graph.nodes == ["agent"]
    assert checkpoints[-1]["next_node"] == "tools"

    completed = run_resumable_agent(
        checkpoints[-1],
        control=RunControl(timeout_seconds=60),
        checkpoint_callback=lambda state: checkpoints.append(state),
        status_callback=lambda _status: None,
        on_token=None,
    )

    assert completed.status == "completed"
    assert completed.answer == "完成"
    assert graph.nodes == ["agent", "tools"]


def test_cancel_after_safe_boundary_never_enters_next_tool(monkeypatch):
    graph = FakeGraph()
    monkeypatch.setattr("app.agent.runner.agent_graph", graph)
    control = RunControl(timeout_seconds=60)

    def checkpoint_callback(_state):
        control.cancel()

    result = run_resumable_agent(
        _initial_state(),
        control=control,
        checkpoint_callback=checkpoint_callback,
        status_callback=lambda _status: None,
        on_token=None,
    )

    assert result.status == "stopped"
    assert graph.nodes == ["agent"]


def test_timeout_after_model_boundary_never_enters_next_tool(monkeypatch):
    graph = FakeGraph()
    monkeypatch.setattr("app.agent.runner.agent_graph", graph)
    now = [0.0]
    control = RunControl(timeout_seconds=5, clock=lambda: now[0])

    def checkpoint_callback(_state):
        now[0] = 6.0

    result = run_resumable_agent(
        _initial_state(),
        control=control,
        checkpoint_callback=checkpoint_callback,
        status_callback=lambda _status: None,
        on_token=None,
    )

    assert result.status == "timed_out"
    assert graph.nodes == ["agent"]
