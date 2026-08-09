"""FastAPI application entry point.

Why this file exists:
    The actual ASGI app the Docker container runs (see
    docker/Dockerfile.api's CMD: `uvicorn gita_engine.api.main:app`).
    An app factory (create_app) rather than a bare module-level `app` so
    tests can construct fresh instances if needed later.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from gita_engine.api.rate_limit import limiter
from gita_engine.api.routes import router
from gita_engine.core.config import get_settings
from gita_engine.core.logging import configure_logging

configure_logging()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Gita Reasoning Engine API",
        description="Retrieval-grounded Q&A over a defined Pushtimarg Bhagavad Gita corpus.",
        version="0.1.0",
    )

    app.state.limiter = limiter
    # slowapi's handler signature doesn't exactly match Starlette's typed
    # exception-handler protocol, but it's the documented, correct way to
    # wire this up and works fine at runtime.
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # Allowed origins are read from CORS_ALLOWED_ORIGINS (see config.py),
    # defaulting to local dev only. Set this env var on Railway to
    # include the deployed Vercel URL for real public deployment.
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
