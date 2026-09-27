"""访客 AI 请求的持久化额度与反向代理地址识别。"""

import hashlib
import hmac
import ipaddress
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import Request

from app.memory import repository


SHANGHAI_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")
GUEST_SESSION_DAILY_LIMIT = int(os.getenv("GUEST_DAILY_AI_LIMIT", "10"))
IP_ROLLING_MINUTE_LIMIT = int(os.getenv("IP_AI_LIMIT_PER_MINUTE", "5"))
IP_DAILY_LIMIT = int(os.getenv("IP_AI_LIMIT_PER_DAY", "30"))


@dataclass(frozen=True)
class QuotaStatus:
    remaining: int
    limit: int
    reset_at: datetime


class GuestQuotaExceeded(Exception):
    """访客额度已用尽，retry_at 表示最早可重试的时间。"""

    def __init__(self, retry_at: datetime):
        self.retry_at = retry_at
        super().__init__("访客今日 AI 请求次数已达上限")


def _as_utc(now: datetime) -> datetime:
    return now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)


def _day_window(now: datetime) -> tuple[str, datetime]:
    local_now = _as_utc(now).astimezone(SHANGHAI_TIMEZONE)
    next_midnight = local_now.replace(
        hour=0, minute=0, second=0, microsecond=0
    ) + timedelta(days=1)
    return local_now.date().isoformat(), next_midnight


def _ip_hash(ip_address: str) -> str:
    key = os.getenv("IP_HASH_HMAC_KEY", "").encode("utf-8")
    if len(key) < 32:
        raise RuntimeError("访客额度服务尚未配置安全密钥")
    # 只将带用途前缀的 HMAC 摘要写入数据库，避免保存可还原的原始 IP。
    return hmac.new(key, b"ip:" + ip_address.encode("ascii"), hashlib.sha256).hexdigest()


def reserve_guest_ai_request(
    session_hash: str, ip_hash: str, now: datetime
) -> QuotaStatus:
    """原子预占一次 AI 请求；被额度拒绝的请求不会扣次。"""
    utc_now = _as_utc(now)
    local_date, day_reset = _day_window(utc_now)
    accepted, session_count, retry_at = repository.reserve_guest_ai_event(
        session_hash,
        ip_hash,
        utc_now,
        local_date,
        day_reset,
        session_limit=GUEST_SESSION_DAILY_LIMIT,
        ip_minute_limit=IP_ROLLING_MINUTE_LIMIT,
        ip_day_limit=IP_DAILY_LIMIT,
    )
    if not accepted:
        raise GuestQuotaExceeded(retry_at or day_reset)
    return QuotaStatus(
        remaining=max(0, GUEST_SESSION_DAILY_LIMIT - session_count),
        limit=GUEST_SESSION_DAILY_LIMIT,
        reset_at=day_reset,
    )


def get_guest_quota_status(session_hash: str, now: datetime) -> QuotaStatus:
    """读取访客当日剩余次数，不会预占或消耗额度。"""
    local_date, day_reset = _day_window(now)
    used = repository.get_guest_ai_request_count(session_hash, local_date)
    return QuotaStatus(
        remaining=max(0, GUEST_SESSION_DAILY_LIMIT - used),
        limit=GUEST_SESSION_DAILY_LIMIT,
        reset_at=day_reset,
    )


def _parse_ip(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return None


def _is_trusted_proxy(peer: str) -> bool:
    configured = os.getenv("TRUSTED_PROXY_IPS", "127.0.0.1,::1")
    try:
        peer_ip = ipaddress.ip_address(peer)
    except ValueError:
        return False
    for entry in configured.split(","):
        try:
            if peer_ip in ipaddress.ip_network(entry.strip(), strict=False):
                return True
        except (TypeError, ValueError):
            continue
    return False


def get_client_ip(request: Request) -> str:
    """仅在 TCP 对端是明确配置的可信代理时，读取 Nginx 覆盖的 X-Real-IP。"""
    peer = request.client.host if request.client else None
    peer_ip = _parse_ip(peer)
    if peer_ip is None:
        return "unknown"
    if _is_trusted_proxy(peer_ip):
        real_ip = _parse_ip(request.headers.get("x-real-ip"))
        if real_ip is not None:
            return real_ip
    return peer_ip


def hash_client_ip(request: Request) -> str:
    """把经过代理信任验证的来源地址转换成仅供限额计数的 HMAC 摘要。"""
    return _ip_hash(get_client_ip(request))
