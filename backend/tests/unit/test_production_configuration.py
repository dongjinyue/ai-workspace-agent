import pytest

from app.configuration import ConfigurationError, validate_runtime_configuration


def test_production_configuration_fails_closed_with_missing_values(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    for name in (
        "SUPABASE_URL",
        "SUPABASE_PUBLISHABLE_KEY",
        "ADMIN_USER_ID",
        "GUEST_SESSION_HMAC_KEY",
        "IP_HASH_HMAC_KEY",
        "CORS_ALLOWED_ORIGINS",
        "TRUSTED_PROXY_IPS",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ConfigurationError, match="SUPABASE_URL"):
        validate_runtime_configuration()


def test_production_configuration_accepts_separate_secure_keys(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_example")
    monkeypatch.setenv("ADMIN_USER_ID", "c6e2ef00-22f3-4c54-81f8-8ab817ed61e7")
    monkeypatch.setenv("GUEST_SESSION_HMAC_KEY", "g" * 40)
    monkeypatch.setenv("IP_HASH_HMAC_KEY", "i" * 40)
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://agent.example.com")
    monkeypatch.setenv("TRUSTED_PROXY_IPS", "127.0.0.1,::1")
    monkeypatch.setenv("GUEST_SESSION_COOKIE_SECURE", "true")

    validate_runtime_configuration()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("IP_HASH_HMAC_KEY", "short", "至少包含 32 字节"),
        ("SUPABASE_URL", "http://project.supabase.co", "必须使用 HTTPS"),
        ("ADMIN_USER_ID", "not-a-uuid", "必须是 Supabase 用户 UUID"),
        ("CORS_ALLOWED_ORIGINS", "http://agent.example.com", "必须仅包含 HTTPS"),
        ("TRUSTED_PROXY_IPS", "not-an-ip", "包含无效 IP"),
        ("GUEST_SESSION_COOKIE_SECURE", "false", "必须启用"),
        ("GUEST_DAILY_AI_LIMIT", "0", "必须是正整数"),
    ],
)
def test_production_configuration_rejects_unsafe_values(monkeypatch, name, value, message):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_example")
    monkeypatch.setenv("ADMIN_USER_ID", "c6e2ef00-22f3-4c54-81f8-8ab817ed61e7")
    monkeypatch.setenv("GUEST_SESSION_HMAC_KEY", "g" * 40)
    monkeypatch.setenv("IP_HASH_HMAC_KEY", "i" * 40)
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://agent.example.com")
    monkeypatch.setenv("TRUSTED_PROXY_IPS", "127.0.0.1,::1")
    monkeypatch.setenv("GUEST_SESSION_COOKIE_SECURE", "true")
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=message):
        validate_runtime_configuration()
