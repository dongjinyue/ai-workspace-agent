"""服务端管理员身份验证与访客会话签发。"""

import hashlib
import hmac
import os
import re
import secrets
from dataclasses import dataclass
from typing import Literal

import httpx
from fastapi import Request, Response


ADMIN_OWNER_ID = "admin"
GUEST_COOKIE_NAME = "agent_guest"
# 访客知识库绑定到签名 Cookie；保留 30 天，AI 额度仍按北京时间每日重置。
GUEST_COOKIE_MAX_AGE = 60 * 60 * 24 * 30
SUPABASE_AUTH_TIMEOUT_SECONDS = 5.0
_GUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43}$")


@dataclass(frozen=True)
class Principal:
    """由服务端确认的请求身份，不采信请求正文中的角色或用户 ID。"""

    role: Literal["admin", "guest"]
    owner_id: str
    guest_session_hash: str | None = None


class AuthenticationError(Exception):
    """可安全返回客户端的认证失败，不携带上游响应或令牌内容。"""

    def __init__(self, detail: str = "登录状态无效，请重新登录", status_code: int = 401):
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


def _guest_hmac_key() -> bytes:
    key = os.getenv("GUEST_SESSION_HMAC_KEY", "").encode("utf-8")
    if len(key) < 32:
        raise AuthenticationError(
            "访客会话暂时不可用，请稍后重试", status_code=503
        )
    return key


def _guest_signature(session_id: str, key: bytes) -> str:
    return hmac.new(key, b"cookie:" + session_id.encode(), hashlib.sha256).hexdigest()


def _guest_digest(session_id: str, key: bytes) -> str:
    """不同用途使用不同前缀，避免把签名直接当作数据库身份摘要。"""
    return hmac.new(key, b"owner:" + session_id.encode(), hashlib.sha256).hexdigest()


def new_guest_cookie() -> tuple[str, str]:
    """生成带签名的随机访客 Cookie 和仅供后端持久化的 HMAC 摘要。"""
    key = _guest_hmac_key()
    session_id = secrets.token_urlsafe(32)
    cookie_value = f"{session_id}.{_guest_signature(session_id, key)}"
    return cookie_value, _guest_digest(session_id, key)


def verify_guest_cookie(cookie_value: str) -> str:
    """校验不可伪造的 Cookie，返回不含原始访客 ID 的摘要。"""
    try:
        session_id, supplied_signature = cookie_value.split(".", maxsplit=1)
    except ValueError as error:
        raise AuthenticationError("访客会话无效，请刷新页面") from error

    if not _GUEST_ID_PATTERN.fullmatch(session_id):
        raise AuthenticationError("访客会话无效，请刷新页面")
    key = _guest_hmac_key()
    expected_signature = _guest_signature(session_id, key)
    if not hmac.compare_digest(supplied_signature, expected_signature):
        raise AuthenticationError("访客会话无效，请刷新页面")
    return _guest_digest(session_id, key)


def _secure_cookie() -> bool:
    configured = os.getenv("GUEST_SESSION_COOKIE_SECURE")
    if configured is not None:
        return configured.strip().lower() in {"1", "true", "yes", "on"}
    environment = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", ""))
    return environment.strip().lower() in {"production", "prod"}


def _guest_principal(request: Request, response: Response) -> Principal:
    cookie_value = request.cookies.get(GUEST_COOKIE_NAME)
    if cookie_value:
        session_hash = verify_guest_cookie(cookie_value)
    else:
        cookie_value, session_hash = new_guest_cookie()
        response.set_cookie(
            key=GUEST_COOKIE_NAME,
            value=cookie_value,
            max_age=GUEST_COOKIE_MAX_AGE,
            path="/",
            secure=_secure_cookie(),
            httponly=True,
            samesite="lax",
        )
    return Principal(
        role="guest",
        owner_id=f"guest:{session_hash}",
        guest_session_hash=session_hash,
    )


async def verify_supabase_access_token(
    token: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """通过 Supabase Auth 的远程用户端点验证访问令牌真实性与有效期。"""
    project_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    publishable_key = os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()
    if not project_url or not publishable_key:
        raise AuthenticationError("管理员登录暂时不可用", status_code=503)

    try:
        async with httpx.AsyncClient(
            timeout=SUPABASE_AUTH_TIMEOUT_SECONDS,
            transport=transport,
            follow_redirects=False,
        ) as client:
            response = await client.get(
                f"{project_url}/auth/v1/user",
                headers={
                    "apikey": publishable_key,
                    "Authorization": f"Bearer {token}",
                },
            )
    except httpx.TimeoutException as error:
        raise AuthenticationError(
            "身份验证服务暂时不可用，请稍后重试", status_code=503
        ) from error
    except httpx.RequestError as error:
        raise AuthenticationError(
            "身份验证服务暂时不可用，请稍后重试", status_code=503
        ) from error

    if response.status_code in {401, 403}:
        raise AuthenticationError()
    if response.status_code < 200 or response.status_code >= 300:
        raise AuthenticationError(
            "身份验证服务暂时不可用，请稍后重试", status_code=503
        )
    try:
        user_id = response.json().get("id")
    except (ValueError, AttributeError) as error:
        raise AuthenticationError("登录状态无效，请重新登录") from error
    if not isinstance(user_id, str) or not user_id:
        raise AuthenticationError("登录状态无效，请重新登录")
    return user_id


async def resolve_principal(request: Request, response: Response) -> Principal:
    """验证 Supabase Bearer；没有凭据时签发/读取访客身份。"""
    authorization = request.headers.get("authorization")
    if authorization is None:
        return _guest_principal(request, response)

    scheme, separator, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not token.strip():
        raise AuthenticationError()

    user_id = await verify_supabase_access_token(token.strip())
    configured_admin_id = os.getenv("ADMIN_USER_ID", "").strip()
    if configured_admin_id and hmac.compare_digest(user_id, configured_admin_id):
        return Principal(role="admin", owner_id=ADMIN_OWNER_ID)
    return _guest_principal(request, response)
