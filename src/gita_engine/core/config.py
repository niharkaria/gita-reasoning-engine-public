"""Centralized, typed application configuration.

Why this file exists:
    Every other module that needs a config value (DB URL, model name, API key)
    should import `get_settings()` from here rather than calling `os.environ`
    directly. This gives us one validated source of truth, catches missing/
    malformed config at startup instead of deep inside a pipeline run, and
    makes it trivial to see the full surface area of what the app depends on.

Design notes:
    - `pydantic-settings` validates types and required fields at import time.
    - `lru_cache` on `get_settings()` ensures the .env file is parsed once per
      process, not on every call site.
    - Fields are grouped by subsystem (app, postgres, neo4j, models, eval) to
      mirror the architecture, not just alphabetized.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Comma-separated list of allowed CORS origins. Defaults to local dev
    # only. On Railway, set CORS_ALLOWED_ORIGINS to a comma-separated
    # list including the deployed Vercel URL, e.g.
    # "http://localhost:3000,https://your-app.vercel.app"
    cors_allowed_origins: str = "http://localhost:3000"

    # --- PostgreSQL ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "gita_engine"
    postgres_user: str = "gita_engine"
    postgres_password: str = Field(default=..., repr=False)

    # --- Neo4j (placeholder — usage decision deferred to Phase 3 ADR) ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = Field(default="changeme", repr=False)

    # --- Models ---
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Generation now runs via Groq's free tier (real rate limits, not a
    # tiny spending credit like Hugging Face's Inference API). Verify the
    # exact model ID in your Groq console before relying on this default —
    # model slugs on hosted providers change.
    generation_model: str = "qwen/qwen3.8-27b"
    groq_api_key: str = Field(default="", repr=False)
    hf_api_token: str = Field(default="", repr=False)

    # --- Evaluation / Observability ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = Field(default="", repr=False)
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        """Parsed, whitespace-trimmed list of allowed CORS origins."""
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]

    @property
    def postgres_dsn(self) -> str:
        """Assembled DSN for SQLAlchemy / psycopg."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return a cached, validated Settings instance for the current process."""
    return Settings()
