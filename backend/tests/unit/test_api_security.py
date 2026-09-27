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


def test_guest_cannot_upload_or_delete_knowledge_resources(monkeypatch):
    monkeypatch.setenv("GUEST_SESSION_HMAC_KEY", "g" * 40)
    with TestClient(app) as client:
        upload = client.post(
            "/api/documents/upload",
            files={"file": ("note.txt", b"hello", "text/plain")},
            headers={"Origin": "http://localhost:3000"},
        )
        delete = client.delete(
            "/api/knowledge-bases/not-public", headers={"Origin": "http://localhost:3000"}
        )

    assert upload.status_code == 403
    assert delete.status_code == 403


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
