from concurrent.futures import ThreadPoolExecutor

import pytest

from app.memory import repository


pytestmark = pytest.mark.unit


def _conversation_with_message(owner_id: str) -> tuple[str, int]:
    conversation_id = repository.create_conversation(owner_id=owner_id)
    message_id = repository.save_message(
        conversation_id,
        "user",
        "需要暂停的请求",
        owner_id=owner_id,
    )
    return conversation_id, message_id


def test_agent_task_is_isolated_by_owner_and_supports_conditional_transitions():
    conversation_id, message_id = _conversation_with_message("owner-a")

    task = repository.create_agent_task(
        conversation_id,
        "owner-a",
        message_id,
        checkpoint={"steps": 0},
        quota_reserved=True,
    )

    assert task["status"] == "queued"
    assert task["quota_reserved"] is True
    assert task["checkpoint"] == {"steps": 0}
    assert repository.get_agent_task(task["run_id"], "owner-b") is None
    assert repository.list_agent_tasks(conversation_id, owner_id="owner-b") == []

    assert repository.update_agent_task(
        task["run_id"],
        "owner-a",
        expected_statuses={"queued"},
        status="running",
    ) is True
    with pytest.raises(ValueError, match="状态转换"):
        repository.update_agent_task(
            task["run_id"],
            "owner-a",
            expected_statuses={"queued"},
            status="paused",
        )
    assert repository.update_agent_task(
        task["run_id"],
        "owner-a",
        expected_statuses={"running"},
        status="pause_requested",
    ) is True
    assert repository.get_agent_task(task["run_id"], "owner-a")["status"] == (
        "pause_requested"
    )


def test_only_one_concurrent_terminal_update_wins():
    conversation_id, message_id = _conversation_with_message("owner-race")
    task = repository.create_agent_task(
        conversation_id,
        "owner-race",
        message_id,
    )
    assert repository.update_agent_task(
        task["run_id"],
        "owner-race",
        expected_statuses={"queued"},
        status="running",
    ) is True

    def finish(status: str) -> bool:
        return repository.update_agent_task(
            task["run_id"],
            "owner-race",
            expected_statuses={"running"},
            status=status,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(finish, ["completed", "stopped"]))

    assert sorted(results) == [False, True]
    assert repository.get_agent_task(task["run_id"], "owner-race")["status"] in {
        "completed",
        "stopped",
    }


def test_completed_task_write_does_not_add_answer_after_stop():
    conversation_id, message_id = _conversation_with_message("owner-atomic")
    task = repository.create_agent_task(
        conversation_id,
        "owner-atomic",
        message_id,
    )
    assert repository.update_agent_task(
        task["run_id"],
        "owner-atomic",
        expected_statuses={"queued"},
        status="running",
    )
    assert repository.update_agent_task(
        task["run_id"],
        "owner-atomic",
        expected_statuses={"running"},
        status="stopped",
    )

    assert repository.complete_agent_task_with_trace(
        task["run_id"],
        "owner-atomic",
        conversation_id,
        "不应保存的回答",
        {"request_id": task["run_id"]},
    ) is False
    messages = repository.get_messages(conversation_id, owner_id="owner-atomic")
    assert [message["content"] for message in messages] == ["需要暂停的请求"]
