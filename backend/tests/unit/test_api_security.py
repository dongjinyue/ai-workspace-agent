import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

from app import main
from app.auth import AuthenticationError
from app.main import app
from app.security import InMemoryRateLimiter


def test_optional_access_token_protects_private_api():
    with patch.dict("os.environ", {"APP_ACCESS_TOKEN": "strong-test-token"}):
        denied = TestClient(app).get("/api/conversations")
        allowed = TestClient(app).get(
            "/api/conversations",
            headers={"Authorization": "Bearer strong-test-token"},
        )
        health = TestClient(app).get("/api/health")

    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert health.status_code == 200


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
