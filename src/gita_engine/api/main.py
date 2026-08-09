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

    # Permissive CORS for local dev — the Next.js frontend runs on a
    # different port (3000) than the API (8000). Tighten this before
    # any real public deployment.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
