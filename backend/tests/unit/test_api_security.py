from unittest.mock import patch

from fastapi.testclient import TestClient

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
