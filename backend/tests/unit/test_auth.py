import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from starlette.requests import Request
from starlette.responses import Response

from app import auth


def _request(*, authorization: str | None = None, cookie: str | None = None) -> Request:
    headers = []
    if authorization:
        headers.append((b"authorization", authorization.encode()))
    if cookie:
        headers.append((b"cookie", cookie.encode()))
    return Request(
        {"type": "http", "method": "GET", "path": "/api/session", "headers": headers}
    )


def _auth_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "public-test-key")
    monkeypatch.setenv("ADMIN_USER_ID", "admin-user-id")
    monkeypatch.setenv("GUEST_SESSION_HMAC_KEY", "g" * 48)


def test_supabase_verifier_checks_user_endpoint_and_returns_verified_id(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co/")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "public-test-key")
    seen = {}

    def respond(request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        return httpx.Response(200, json={"id": "verified-user-id"})

    user_id = asyncio.run(
        auth.verify_supabase_access_token(
            "valid-access-token", transport=httpx.MockTransport(respond)
        )
    )

    assert user_id == "verified-user-id"
    assert seen["url"] == "https://project.supabase.co/auth/v1/user"
    assert seen["headers"]["apikey"] == "public-test-key"
    assert seen["headers"]["authorization"] == "Bearer valid-access-token"


def test_invalid_supabase_token_is_rejected_without_exposing_upstream_body(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "public-test-key")

    with pytest.raises(auth.AuthenticationError, match="登录状态无效"):
        asyncio.run(
            auth.verify_supabase_access_token(
                "expired-token",
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        401, json={"message": "secret detail"}
                    )
                ),
            )
        )


def test_supabase_timeout_fails_closed(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "public-test-key")

    def timeout(request):
        raise httpx.ReadTimeout("upstream body must not escape")

    with pytest.raises(auth.AuthenticationError, match="身份验证服务暂时不可用"):
        asyncio.run(
            auth.verify_supabase_access_token(
                "token", transport=httpx.MockTransport(timeout)
            )
        )


def test_only_configured_supabase_user_gets_admin_role(monkeypatch):
    _auth_env(monkeypatch)
    monkeypatch.setattr(
        auth,
        "verify_supabase_access_token",
        AsyncMock(return_value="admin-user-id"),
    )

    principal = asyncio.run(
        auth.resolve_principal(
            _request(authorization="Bearer verified-token"), Response()
        )
    )

    assert principal.role == "admin"
    assert principal.owner_id == auth.ADMIN_OWNER_ID
    assert principal.guest_session_hash is None


def test_valid_non_admin_supabase_user_is_guest(monkeypatch):
    _auth_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setattr(
        auth,
        "verify_supabase_access_token",
        AsyncMock(return_value="some-other-user-id"),
    )
    response = Response()

    principal = asyncio.run(
        auth.resolve_principal(
            _request(authorization="Bearer valid-but-not-admin"), response
        )
    )

    assert principal.role == "guest"
    assert principal.owner_id.startswith("guest:")
    assert "httponly" in response.headers["set-cookie"].lower()
    assert "samesite=lax" in response.headers["set-cookie"].lower()
    assert "secure" in response.headers["set-cookie"].lower()


def test_missing_admin_id_does_not_promote_a_valid_user(monkeypatch):
    _auth_env(monkeypatch)
    monkeypatch.delenv("ADMIN_USER_ID")
    monkeypatch.setattr(
        auth,
        "verify_supabase_access_token",
        AsyncMock(return_value="admin-user-id"),
    )

    principal = asyncio.run(
        auth.resolve_principal(
            _request(authorization="Bearer valid-token"), Response()
        )
    )

    assert principal.role == "guest"


def test_invalid_bearer_never_falls_back_to_guest(monkeypatch):
    _auth_env(monkeypatch)
    verifier = AsyncMock(side_effect=auth.AuthenticationError("登录状态无效，请重新登录"))
    monkeypatch.setattr(auth, "verify_supabase_access_token", verifier)

    with pytest.raises(auth.AuthenticationError):
        asyncio.run(
            auth.resolve_principal(
                _request(authorization="Bearer expired"), Response()
            )
        )
    verifier.assert_awaited_once_with("expired")


def test_guest_cookie_tampering_is_detected_and_cookie_ids_have_separate_digests(
    monkeypatch,
):
    _auth_env(monkeypatch)
    cookie_a, digest_a = auth.new_guest_cookie()
    cookie_b, digest_b = auth.new_guest_cookie()

    assert digest_a != digest_b
    assert auth.verify_guest_cookie(cookie_a) == digest_a
    with pytest.raises(auth.AuthenticationError):
        auth.verify_guest_cookie(cookie_a + "x")
