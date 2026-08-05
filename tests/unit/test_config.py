"""Smoke test for core configuration loading.

This is intentionally the first test in the repo: if `Settings` can't be
constructed and validated, nothing else in the system can run. Keeping this
test green is a fast signal that the environment (Python version, deps,
.env handling) is set up correctly.
"""

from gita_engine.core.config import Settings, get_settings


def test_settings_loads_with_required_fields(monkeypatch):
    """Settings should construct successfully when required env vars are set."""
    monkeypatch.setenv("POSTGRES_PASSWORD", "test-password")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.app_env == "development"
    assert settings.postgres_db == "gita_engine"
    assert "test-password" in settings.postgres_dsn


def test_get_settings_is_cached(monkeypatch):
    """get_settings() should return the same cached instance across calls."""
    monkeypatch.setenv("POSTGRES_PASSWORD", "test-password")
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second
