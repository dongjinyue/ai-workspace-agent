from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import sqlite3
from threading import Barrier, get_ident

import pytest
from starlette.requests import Request

from app.quotas import (
    GuestQuotaExceeded,
    get_client_ip,
    get_guest_quota_status,
    hash_client_ip,
    reserve_guest_ai_request,
)


def _now(value: str = "2026-09-27T10:00:00+00:00") -> datetime:
    return datetime.fromisoformat(value)


def test_guest_has_ten_top_level_ai_requests_per_shanghai_day(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DATABASE_PATH", str(tmp_path / "quota.db"))
    session = "a" * 64
    now = _now()

    for number in range(10):
        # 不触发独立 IP 保护，以单独验证访客会话日额度。
        status = reserve_guest_ai_request(session, f"ip-{number}", now)
    assert status.remaining == 0
    assert status.limit == 10
    assert status.reset_at.isoformat() == "2026-09-28T00:00:00+08:00"
    with pytest.raises(GuestQuotaExceeded) as error:
        reserve_guest_ai_request(session, "another-ip", now)
    assert error.value.retry_at == status.reset_at
    assert get_guest_quota_status(session, now).remaining == 0

    tomorrow = now + timedelta(hours=16)
    assert reserve_guest_ai_request(session, "tomorrow-ip", tomorrow).remaining == 9


def test_ip_limit_is_five_per_rolling_minute_and_rejected_attempt_is_not_charged(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("APP_DATABASE_PATH", str(tmp_path / "minute.db"))
    now = _now()
    for number in range(5):
        reserve_guest_ai_request(f"session-{number}", "same-ip", now)
    with pytest.raises(GuestQuotaExceeded) as error:
        reserve_guest_ai_request("session-6", "same-ip", now)
    assert error.value.retry_at == now + timedelta(seconds=60)

    status = reserve_guest_ai_request(
        "session-7", "same-ip", now + timedelta(seconds=60)
    )
    assert status.remaining == 9


def test_ip_daily_limit_is_thirty_across_sessions_and_persists_after_reopen(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "daily-ip.db"
    monkeypatch.setenv("APP_DATABASE_PATH", str(database_path))
    now = _now()
    for number in range(30):
        # 每分钟间隔一次，单 IP 分钟限额不会先于每日限额触发。
        reserve_guest_ai_request(
            f"session-{number}", "same-ip", now + timedelta(minutes=number)
        )
    with pytest.raises(GuestQuotaExceeded) as error:
        reserve_guest_ai_request("new-session", "same-ip", now)
    assert error.value.retry_at.isoformat() == "2026-09-28T00:00:00+08:00"

    monkeypatch.setenv("APP_DATABASE_PATH", str(database_path))
    with pytest.raises(GuestQuotaExceeded):
        reserve_guest_ai_request("another-session", "same-ip", now)


def test_two_concurrent_requests_cannot_both_take_the_last_session_slot(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("APP_DATABASE_PATH", str(tmp_path / "concurrent.db"))
    now = _now()
    session = "same-session"
    for number in range(9):
        reserve_guest_ai_request(session, f"ip-{number}", now)
    barrier = Barrier(2)

    def reserve():
        barrier.wait()
        try:
            reserve_guest_ai_request(
                session, f"concurrent-ip-{get_ident()}", now
            )
            return "accepted"
        except GuestQuotaExceeded:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: reserve(), range(2)))
    assert sorted(results) == ["accepted", "rejected"]
    assert get_guest_quota_status(session, now).remaining == 0


def _request(peer: str | None, headers: list[tuple[bytes, bytes]]) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
        "client": (peer, 12345) if peer is not None else None,
        "server": ("127.0.0.1", 80),
        "scheme": "http",
        "query_string": b"",
    }
    return Request(scope)


def test_real_ip_header_only_trusted_from_configured_nginx_peer(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_IPS", "127.0.0.1,::1")
    request = _request(
        "127.0.0.1", [(b"x-real-ip", b"203.0.113.19"), (b"x-forwarded-for", b"198.51.100.2")]
    )
    assert get_client_ip(request) == "203.0.113.19"

    forged = _request(
        "203.0.113.8",
        [(b"x-real-ip", b"203.0.113.19"), (b"x-forwarded-for", b"198.51.100.2")],
    )
    assert get_client_ip(forged) == "203.0.113.8"

    ipv6 = _request("::1", [(b"x-real-ip", b"2001:db8::25")])
    assert get_client_ip(ipv6) == "2001:db8::25"


def test_invalid_real_ip_and_missing_peer_ip_fall_back_safely(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_IPS", "127.0.0.1")
    invalid = _request("127.0.0.1", [(b"x-real-ip", b"attacker, 203.0.113.4")])
    assert get_client_ip(invalid) == "127.0.0.1"

    missing = _request(None, [(b"x-real-ip", b"203.0.113.4")])
    assert get_client_ip(missing) == "unknown"


def test_quota_database_stores_only_ip_digest_not_raw_address(tmp_path, monkeypatch):
    database_path = tmp_path / "private-ip.db"
    monkeypatch.setenv("APP_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("GUEST_SESSION_HMAC_KEY", "g" * 40)
    monkeypatch.setenv("IP_HASH_HMAC_KEY", "i" * 40)
    monkeypatch.setenv("TRUSTED_PROXY_IPS", "127.0.0.1")
    raw_ip = "203.0.113.25"
    ip_hash = hash_client_ip(
        _request("127.0.0.1", [(b"x-real-ip", raw_ip.encode("ascii"))])
    )

    reserve_guest_ai_request("session-digest", ip_hash, _now())
    with sqlite3.connect(database_path) as connection:
        stored_ip_hash = connection.execute(
            "SELECT ip_hash FROM guest_ai_events"
        ).fetchone()[0]

    assert stored_ip_hash == ip_hash
    assert raw_ip not in stored_ip_hash


def test_ip_hash_requires_its_own_key(monkeypatch):
    monkeypatch.setenv("GUEST_SESSION_HMAC_KEY", "g" * 40)
    monkeypatch.delenv("IP_HASH_HMAC_KEY", raising=False)

    with pytest.raises(RuntimeError, match="安全密钥"):
        hash_client_ip(_request("127.0.0.1", []))
