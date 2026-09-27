import sqlite3

import pytest

from app.memory import repository
from app.memory.database import init_database
from app.memory.service import ConversationNotFoundError, ConversationService


def test_existing_conversations_are_migrated_to_admin_without_losing_history(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "legacy.db"
    monkeypatch.setenv("APP_DATABASE_PATH", str(database_path))
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE agent_runs (
                assistant_message_id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL,
                trace_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            INSERT INTO conversations VALUES ('legacy', '2026-01-01', '2026-01-02');
            INSERT INTO messages VALUES (1, 'legacy', 'assistant', '旧回答', '2026-01-02');
            INSERT INTO agent_runs VALUES (1, 'legacy', '{"steps":1}', '2026-01-02');
            """
        )

    init_database()

    assert repository.conversation_exists("legacy", owner_id="admin")
    assert not repository.conversation_exists("legacy", owner_id="guest:visitor-a")
    assert repository.list_conversations(owner_id="admin")[0]["id"] == "legacy"
    assert repository.list_conversations(owner_id="guest:visitor-a") == []
    assert (
        repository.get_messages_with_traces("legacy", owner_id="admin")[0]["trace"]
        == {"steps": 1}
    )


def test_conversation_reads_and_mutations_are_scoped_to_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DATABASE_PATH", str(tmp_path / "owners.db"))
    init_database()
    owner_a = "guest:visitor-a"
    owner_b = "guest:visitor-b"
    conversation_a = repository.create_conversation("A", owner_id=owner_a)
    conversation_b = repository.create_conversation("B", owner_id=owner_b)
    repository.save_message(conversation_a, "user", "仅 A 可见", owner_id=owner_a)
    repository.save_message(conversation_b, "user", "仅 B 可见", owner_id=owner_b)

    assert [item["id"] for item in repository.list_conversations(owner_id=owner_a)] == [
        conversation_a
    ]
    assert repository.get_messages(conversation_a, owner_id=owner_b) == []
    assert (
        repository.get_messages(conversation_a, owner_id=owner_a)[0]["content"]
        == "仅 A 可见"
    )
    assert not repository.rename_conversation(conversation_a, "越权", owner_id=owner_b)
    assert not repository.delete_conversation(conversation_a, owner_id=owner_b)
    assert repository.conversation_exists(conversation_a, owner_id=owner_a)


def test_service_rejects_other_owners_conversation_before_agent_runs(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("APP_DATABASE_PATH", str(tmp_path / "service.db"))
    init_database()
    owner = "guest:visitor-a"
    foreign = repository.create_conversation(owner_id="guest:visitor-b")
    service = ConversationService()

    with pytest.raises(ConversationNotFoundError):
        service.chat(
            message="不应执行",
            knowledge_base_id=None,
            conversation_id=foreign,
            owner_id=owner,
        )
