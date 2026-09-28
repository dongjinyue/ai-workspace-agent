"""启动前校验公开部署必需配置，避免服务以不安全默认值对外运行。"""

import ipaddress
import os
from urllib.parse import urlparse
from uuid import UUID


class ConfigurationError(RuntimeError):
    """配置缺失或不安全；错误信息只列配置名称，不回显配置值。"""


def get_rag_top_k() -> int:
    """读取 RAG（检索增强生成）返回数量，并限制调试和资源消耗范围。"""
    raw_value = os.getenv("RAG_TOP_K", "5").strip()
    try:
        top_k = int(raw_value)
    except ValueError as error:
        raise ConfigurationError("RAG_TOP_K 必须是 1 到 20 之间的整数") from error
    if not 1 <= top_k <= 20:
        raise ConfigurationError("RAG_TOP_K 必须是 1 到 20 之间的整数")
    return top_k


def validate_runtime_configuration() -> None:
    """生产模式要求认证、独立 HMAC 密钥、可信代理和 HTTPS 来源。"""
    environment = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "")).strip().lower()
    if environment not in {"production", "prod"}:
        return

    required = (
        "SUPABASE_URL",
        "SUPABASE_PUBLISHABLE_KEY",
        "ADMIN_USER_ID",
        "GUEST_SESSION_HMAC_KEY",
        "IP_HASH_HMAC_KEY",
        "CORS_ALLOWED_ORIGINS",
        "TRUSTED_PROXY_IPS",
    )
    missing = [name for name in required if not os.getenv(name, "").strip()]
    if missing:
        raise ConfigurationError(
            "生产配置缺少必填项：" + ", ".join(missing)
        )

    for name in ("GUEST_SESSION_HMAC_KEY", "IP_HASH_HMAC_KEY"):
        if len(os.getenv(name, "").encode("utf-8")) < 32:
            raise ConfigurationError(f"{name} 必须至少包含 32 字节")

    parsed_url = urlparse(os.getenv("SUPABASE_URL", "").strip())
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ConfigurationError("SUPABASE_URL 必须使用 HTTPS")

    try:
        UUID(os.getenv("ADMIN_USER_ID", "").strip())
    except ValueError as error:
        raise ConfigurationError("ADMIN_USER_ID 必须是 Supabase 用户 UUID") from error

    origins = [
        item.strip()
        for item in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    ]
    if not origins or any(
        urlparse(origin).scheme != "https" or not urlparse(origin).netloc
        for origin in origins
    ):
        raise ConfigurationError("生产 CORS_ALLOWED_ORIGINS 必须仅包含 HTTPS 网站来源")

    try:
        for peer in os.getenv("TRUSTED_PROXY_IPS", "").split(","):
            ipaddress.ip_network(peer.strip(), strict=False)
    except ValueError as error:
        raise ConfigurationError("TRUSTED_PROXY_IPS 包含无效 IP 或网段") from error

    secure_cookie = os.getenv("GUEST_SESSION_COOKIE_SECURE", "true").strip().lower()
    if secure_cookie not in {"1", "true", "yes", "on"}:
        raise ConfigurationError("生产环境必须启用 GUEST_SESSION_COOKIE_SECURE")

    for name, default in (
        ("GUEST_DAILY_AI_LIMIT", "10"),
        ("IP_AI_LIMIT_PER_MINUTE", "5"),
        ("IP_AI_LIMIT_PER_DAY", "30"),
    ):
        try:
            limit = int(os.getenv(name, default))
        except ValueError as error:
            raise ConfigurationError(f"{name} 必须是正整数") from error
        if limit < 1:
            raise ConfigurationError(f"{name} 必须是正整数")
