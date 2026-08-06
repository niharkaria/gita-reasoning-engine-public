"""FastAPI application entry point.

Why this file exists:
    The actual ASGI app the Docker container runs (see
    docker/Dockerfile.api's CMD: `uvicorn gita_engine.api.main:app`).
    An app factory (create_app) rather than a bare module-level `app` so
    tests can construct fresh instances if needed later.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gita_engine.api.routes import router
from gita_engine.core.logging import configure_logging

configure_logging()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Gita Reasoning Engine API",
        description="Retrieval-grounded Q&A over a defined Pushtimarg Bhagavad Gita corpus.",
        version="0.1.0",
    )

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
