from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.agent.service import AgentResult
from app.main import app
from app.memory import repository
from app.memory.database import get_database_path
from app.memory.service import ConversationNotFoundError, ConversationService


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def clean_test_database():
    path = get_database_path()
    path.unlink(missing_ok=True)
    yield
    path.unlink(missing_ok=True)


def _agent_result(answer: str = "明白。") -> AgentResult:
    return AgentResult(
        answer=answer,
        tool_called=False,
        tool_name=None,
        tools_used=[],
        active_skill=None,
        steps=1,
        matched_chunks=0,
    )


def test_conversation_and_messages_persist_in_sqlite():
    conversation_id = repository.create_conversation()
    repository.save_message(conversation_id, "user", "我的代号是 BlueFox。")

    # 每次仓库调用都会重新打开连接，因此这里也验证了磁盘持久化。
    assert get_database_path().is_file()
    assert repository.conversation_exists(conversation_id)
    assert repository.get_messages(conversation_id)[0]["content"] == (
        "我的代号是 BlueFox。"
    )


def test_conversations_are_strictly_isolated():
    conversation_a = repository.create_conversation()
    conversation_b = repository.create_conversation()
    repository.save_message(conversation_a, "user", "苹果")
    repository.save_message(conversation_b, "user", "香蕉")

    assert [m["content"] for m in repository.get_messages(conversation_a)] == [
        "苹果"
    ]
    assert [m["content"] for m in repository.get_messages(conversation_b)] == [
        "香蕉"
    ]


def test_sliding_window_returns_latest_20_in_chronological_order():
    conversation_id = repository.create_conversation()
    for index in range(25):
        repository.save_message(conversation_id, "user", f"message-{index}")

    messages = repository.get_messages(conversation_id, limit=20)
    assert len(messages) == 20
    assert messages[0]["content"] == "message-5"
    assert messages[-1]["content"] == "message-24"


def test_current_user_message_enters_agent_context_once_and_tools_are_not_saved():
    captured = {}

    def fake_run_agent(**kwargs):
        captured.update(kwargs)
        return _agent_result()

    with patch("app.memory.service.run_agent", side_effect=fake_run_agent):
        result = ConversationService().chat(
            message="我正在学习 LangGraph",
            knowledge_base_id=None,
            conversation_id=None,
        )

    current_messages = [
        item
        for item in captured["history_messages"]
        if item["content"] == "我正在学习 LangGraph"
    ]
    assert len(current_messages) == 1
    assert result.history_messages == 1
    stored = repository.get_messages(result.conversation_id)
    assert [item["role"] for item in stored] == ["user", "assistant"]


def test_unknown_conversation_is_rejected_instead_of_created():
    with pytest.raises(ConversationNotFoundError):
        ConversationService().resolve_conversation("0" * 32)

    response = TestClient(app).get(
        "/api/conversations/00000000000000000000000000000000/messages"
    )
    assert response.status_code == 404


def test_message_content_is_saved_as_data_not_executed_as_sql():
    conversation_id = repository.create_conversation()
    payload = "'); DROP TABLE conversations; --"
    repository.save_message(conversation_id, "user", payload)
    assert repository.get_messages(conversation_id)[0]["content"] == payload
    assert repository.conversation_exists(conversation_id)


def test_chat_api_reuses_conversation_and_history_api_restores_messages():
    client = TestClient(app)
    captured_histories = []

    def fake_run_agent(**kwargs):
        captured_histories.append(kwargs["history_messages"])
        return _agent_result(f"回答-{len(captured_histories)}")

    with patch("app.memory.service.run_agent", side_effect=fake_run_agent):
        first = client.post("/api/chat", json={"message": "我叫小明"})
        assert first.status_code == 200
        conversation_id = first.json()["conversation_id"]

        second = client.post(
            "/api/chat",
            json={
                "message": "我叫什么？",
                "conversation_id": conversation_id,
            },
        )

    assert second.status_code == 200
    assert second.json()["conversation_id"] == conversation_id
    assert second.json()["history_messages"] == 3
    assert [item["content"] for item in captured_histories[1]] == [
        "我叫小明",
        "回答-1",
        "我叫什么？",
    ]

    restored = client.get(
        f"/api/conversations/{conversation_id}/messages"
    )
    assert restored.status_code == 200
    assert [item["role"] for item in restored.json()["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_conversation_crud_api_persists_title_and_deletes_messages():
    """会话列表、重命名和级联删除都应由 SQLite 持久保存。"""
    client = TestClient(app)
    created = client.post("/api/conversations", json={"title": "项目讨论"})
    assert created.status_code == 201
    conversation_id = created.json()["id"]
    repository.save_message(conversation_id, "user", "测试消息")

    listed = client.get("/api/conversations")
    assert listed.status_code == 200
    assert listed.json()["conversations"][0]["title"] == "项目讨论"
    assert listed.json()["conversations"][0]["message_count"] == 1

    renamed = client.patch(
        f"/api/conversations/{conversation_id}", json={"title": "新的标题"}
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "新的标题"

    deleted = client.delete(f"/api/conversations/{conversation_id}")
    assert deleted.status_code == 204
    assert not repository.conversation_exists(conversation_id)


def test_cors_allows_vite_fallback_development_port():
    """Vite 自动切换到 5174 等端口后，浏览器预检请求仍应成功。"""
    response = TestClient(app).options(
        "/api/chat",
        headers={
            "Origin": "http://localhost:5174",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "http://localhost:5174"
    )


def test_agent_chat_endpoint_keeps_conversation_memory():
    """独立 Agent 接口应复用原有持久化链路，而不是产生临时回答。"""
    with patch("app.memory.service.run_agent", return_value=_agent_result("完成")):
        response = TestClient(app).post(
            "/api/agent/chat", json={"message": "现在几点？"}
        )

    assert response.status_code == 200
    conversation_id = response.json()["conversation_id"]
    assert [item["role"] for item in repository.get_messages(conversation_id)] == [
        "user",
        "assistant",
    ]
