import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

from app import main
from app.auth import AuthenticationError
from app.main import app
from app.auth import Principal
from app.agent.llm import ModelServiceUnavailableError
from app.main import ChatRequest
from app.memory import repository
from app.quotas import GuestQuotaExceeded
from app.security import InMemoryRateLimiter
from app.rag.catalog import register_knowledge_base


@pytest.fixture(autouse=True)
def configure_ip_hash_key_for_tests(monkeypatch):
    monkeypatch.setenv("IP_HASH_HMAC_KEY", "i" * 40)
    monkeypatch.setattr(main, "guest_upload_rate_limiter", InMemoryRateLimiter(3))


def test_public_session_is_guest_and_legacy_access_token_is_not_required(monkeypatch):
    monkeypatch.setenv("GUEST_SESSION_HMAC_KEY", "g" * 40)
    monkeypatch.setenv("APP_ACCESS_TOKEN", "legacy-token")

    with TestClient(app) as client:
        session = client.get("/api/session")
        conversations = client.get("/api/conversations")
        health = client.get("/api/health")

    assert session.status_code == 200
    assert session.json()["role"] == "guest"
    assert session.json()["quota"]["limit"] == 10
    assert "agent_guest" in session.headers["set-cookie"]
    assert conversations.status_code == 200
    assert health.status_code == 200


def test_guest_conversations_are_isolated_and_request_cannot_claim_admin(monkeypatch):
    monkeypatch.setenv("GUEST_SESSION_HMAC_KEY", "g" * 40)
    headers = {"Origin": "http://localhost:3000"}

    with TestClient(app) as guest_a, TestClient(app) as guest_b:
        created = guest_a.post(
            "/api/conversations",
            json={"title": "访客 A", "role": "admin", "owner_id": "admin"},
            headers=headers,
        )
        assert created.status_code == 201

        hidden = guest_b.get(
            f"/api/conversations/{created.json()['id']}/messages"
        )
        assert hidden.status_code == 404
        assert guest_a.get("/api/conversations").json()["conversations"][0]["title"] == "访客 A"
        assert guest_b.get("/api/conversations").json()["conversations"] == []


def test_guest_can_create_private_knowledge_base_but_cannot_access_anothers(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("APP_DATABASE_PATH", str(tmp_path / "guest-private-kb.db"))
    monkeypatch.setenv("GUEST_SESSION_HMAC_KEY", "g" * 40)
    monkeypatch.delenv("PUBLIC_KNOWLEDGE_BASE_IDS", raising=False)
    principal = [
        Principal(
            role="guest",
            owner_id="guest:alpha",
            guest_session_hash="a" * 64,
        )
    ]
    app.dependency_overrides[main.get_current_principal] = lambda: principal[0]
    try:
        with (
            patch("app.main.index_document", return_value=2),
            patch.object(main, "find_relevant_chunks", return_value=[]) as search,
            TestClient(app) as client,
        ):
            uploaded = client.post(
                "/api/documents/upload",
                files={
                    "file": (
                        "private-note.txt",
                        b"my private notes",
                        "text/plain",
                    )
                },
                headers={"Origin": "http://localhost:3000"},
            )
            knowledge_base_id = uploaded.json()["knowledge_base_id"]
            own_list = client.get("/api/knowledge-bases")

            principal[0] = Principal(
                role="guest", owner_id="guest:beta", guest_session_hash="b" * 64
            )
            other_list = client.get("/api/knowledge-bases")
            hidden_search = client.post(
                "/api/documents/search",
                json={
                    "knowledge_base_id": knowledge_base_id,
                    "query": "private",
                },
                headers={"Origin": "http://localhost:3000"},
            )
            with (
                patch.object(main, "reserve_guest_ai_request") as reserve,
                patch.object(main.conversation_service, "chat") as chat,
            ):
                hidden_chat = client.post(
                    "/api/agent/chat",
                    json={
                        "knowledge_base_id": knowledge_base_id,
                        "message": "读取别人的个人知识库",
                    },
                    headers={"Origin": "http://localhost:3000"},
                )
            hidden_delete = client.delete(
                f"/api/knowledge-bases/{knowledge_base_id}",
                headers={"Origin": "http://localhost:3000"},
            )
    finally:
        app.dependency_overrides.pop(main.get_current_principal, None)

    assert uploaded.status_code == 200
    assert [item["id"] for item in own_list.json()["knowledge_bases"]] == [
        knowledge_base_id
    ]
    assert own_list.json()["knowledge_bases"][0]["can_edit"] is True
    assert other_list.json()["knowledge_bases"] == []
    assert hidden_search.status_code == 404
    assert hidden_chat.status_code == 404
    assert hidden_delete.status_code == 404
    search.assert_not_called()
    reserve.assert_not_called()
    chat.assert_not_called()


def test_invalid_bearer_does_not_fall_back_to_guest(monkeypatch):
    monkeypatch.setenv("GUEST_SESSION_HMAC_KEY", "g" * 40)
    with patch(
        "app.auth.verify_supabase_access_token",
        AsyncMock(side_effect=AuthenticationError()),
    ):
        response = TestClient(app).get(
            "/api/session", headers={"Authorization": "Bearer expired-or-fake"}
        )

    assert response.status_code == 401
    assert "agent_guest" not in response.headers.get("set-cookie", "")


def test_only_verified_admin_identity_can_read_legacy_admin_conversations(monkeypatch):
    monkeypatch.setenv("ADMIN_USER_ID", "verified-admin-uuid")
    monkeypatch.setenv("GUEST_SESSION_HMAC_KEY", "g" * 40)
    title = "legacy-admin-regression"
    conversation_id = repository.create_conversation(title, owner_id="admin")

    with patch(
        "app.auth.verify_supabase_access_token",
        AsyncMock(return_value="verified-admin-uuid"),
    ), TestClient(app) as client:
        admin = client.get(
            "/api/conversations",
            headers={"Authorization": "Bearer verified-valid-token"},
        )
    assert admin.status_code == 200
    assert any(
        item["id"] == conversation_id
        for item in admin.json()["conversations"]
    )

    with patch(
        "app.auth.verify_supabase_access_token",
        AsyncMock(return_value="some-other-valid-user"),
    ), TestClient(app) as client:
        guest = client.get(
            "/api/conversations",
            headers={"Authorization": "Bearer verified-valid-token"},
        )
    assert guest.status_code == 200
    assert all(
        item["id"] != conversation_id
        for item in guest.json()["conversations"]
    )


def test_browser_write_requires_an_allowed_origin(monkeypatch):
    monkeypatch.setenv("GUEST_SESSION_HMAC_KEY", "g" * 40)
    with TestClient(app) as client:
        rejected = client.post("/api/conversations", json={"title": "无来源"})
        accepted = client.post(
            "/api/conversations",
            json={"title": "本地来源"},
            headers={"Origin": "http://localhost:3000"},
        )

    assert rejected.status_code == 403
    assert accepted.status_code == 201


def test_both_chat_routes_stop_before_agent_when_guest_quota_is_exhausted(monkeypatch):
    monkeypatch.setenv("GUEST_SESSION_HMAC_KEY", "g" * 40)
    retry_at = datetime.now(timezone.utc) + timedelta(minutes=3)
    for path in ("/api/chat", "/api/agent/chat"):
        with TestClient(app) as client:
            assert client.get("/api/session").status_code == 200
            with (
                patch.object(
                    main,
                    "reserve_guest_ai_request",
                    side_effect=GuestQuotaExceeded(retry_at),
                ) as reserve,
                patch.object(main.conversation_service, "chat") as chat,
            ):
                response = client.post(
                    path,
                    json={"message": "应该被额度拦截"},
                    headers={"Origin": "http://localhost:3000"},
                )

        assert response.status_code == 429
        assert int(response.headers["retry-after"]) > 0
        assert response.headers["x-quota-reset"] == retry_at.isoformat()
        reserve.assert_called_once()
        chat.assert_not_called()


def test_admin_bypasses_guest_quota(monkeypatch):
    request = main.Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/agent/chat",
            "headers": [],
            "client": ("127.0.0.1", 1234),
        }
    )
    with (
        patch.object(
            main,
            "reserve_guest_ai_request",
            side_effect=AssertionError("管理员不应消耗访客额度"),
        ) as reserve,
        patch.object(
            main.conversation_service,
            "chat",
            side_effect=ModelServiceUnavailableError("模型暂不可用"),
        ),
        pytest.raises(HTTPException) as error,
    ):
        main._chat(
            ChatRequest(message="管理员请求"),
            Principal(role="admin", owner_id="admin"),
            request,
        )

    assert error.value.status_code == 503
    reserve.assert_not_called()


def test_accepted_guest_request_consumes_quota_even_if_model_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DATABASE_PATH", str(tmp_path / "failed-chat.db"))
    monkeypatch.setenv("GUEST_SESSION_HMAC_KEY", "g" * 40)
    with TestClient(app) as client:
        session = client.get("/api/session")
        assert session.json()["quota"]["remaining"] == 10
        with patch(
            "app.memory.service.run_agent", side_effect=RuntimeError("temporary failure")
        ):
            failed = client.post(
                "/api/agent/chat",
                json={"message": "计入已接受请求"},
                headers={"Origin": "http://localhost:3000"},
            )
        refreshed = client.get("/api/session")

    assert failed.status_code == 502
    assert refreshed.json()["quota"]["remaining"] == 9


def test_guest_knowledge_listing_is_empty_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DATABASE_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setenv("GUEST_SESSION_HMAC_KEY", "g" * 40)
    monkeypatch.delenv("PUBLIC_KNOWLEDGE_BASE_IDS", raising=False)
    register_knowledge_base(
        "private-base", [{"filename": "private.txt", "chunk_count": 1}]
    )
    with TestClient(app) as client:
        response = client.get("/api/knowledge-bases")

    assert response.status_code == 200
    assert response.json()["knowledge_bases"] == []


def test_guest_can_only_search_explicitly_public_knowledge_bases(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DATABASE_PATH", str(tmp_path / "allowlist.db"))
    monkeypatch.setenv("GUEST_SESSION_HMAC_KEY", "g" * 40)
    monkeypatch.setenv("PUBLIC_KNOWLEDGE_BASE_IDS", "public-base")
    register_knowledge_base(
        "public-base", [{"filename": "public.txt", "chunk_count": 1}]
    )
    register_knowledge_base(
        "private-base", [{"filename": "private.txt", "chunk_count": 1}]
    )

    with (
        TestClient(app) as client,
        patch.object(main, "find_relevant_chunks", return_value=[]) as search,
    ):
        listed = client.get("/api/knowledge-bases")
        private = client.post(
            "/api/documents/search",
            json={"knowledge_base_id": "private-base", "query": "private"},
            headers={"Origin": "http://localhost:3000"},
        )
        assert private.status_code == 404
        search.assert_not_called()

        public = client.post(
            "/api/documents/search",
            json={"knowledge_base_id": "public-base", "query": "public"},
            headers={"Origin": "http://localhost:3000"},
        )

    assert [item["id"] for item in listed.json()["knowledge_bases"]] == [
        "public-base"
    ]
    assert public.status_code == 200
    search.assert_called_once_with("public-base", "public")


def test_guest_cannot_select_private_knowledge_base_for_chat(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DATABASE_PATH", str(tmp_path / "chat-private-kb.db"))
    monkeypatch.setenv("GUEST_SESSION_HMAC_KEY", "g" * 40)
    monkeypatch.setenv("PUBLIC_KNOWLEDGE_BASE_IDS", "public-base")
    register_knowledge_base(
        "private-base", [{"filename": "private.txt", "chunk_count": 1}]
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/agent/chat",
            json={"message": "查一下", "knowledge_base_id": "private-base"},
            headers={"Origin": "http://localhost:3000"},
        )

    assert response.status_code == 404


def test_rate_limiter_rejects_requests_over_limit():
    limiter = InMemoryRateLimiter(requests_per_minute=2)

    assert limiter.allow("client") is True
    assert limiter.allow("client") is True
    assert limiter.allow("client") is False


def test_auth_endpoint_adapter_never_exposes_upstream_error_details():
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/session",
            "headers": [],
        }
    )
    expected = AuthenticationError("登录状态无效，请重新登录")

    with patch.object(main, "resolve_principal", AsyncMock(side_effect=expected)):
        with pytest.raises(HTTPException) as error:
            asyncio.run(main.get_current_principal(request, Response()))

    assert error.value.status_code == 401
    assert error.value.detail == "登录状态无效，请重新登录"
