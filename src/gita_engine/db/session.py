"""Database engine and session management.

Why this file exists:
    Every module that touches the database (API routes, ingestion scripts,
    Kaggle notebooks) should get its session the same way — through
    `get_session()` — rather than each constructing its own engine. This
    keeps connection pooling behavior consistent and makes it trivial to
    swap the DSN (e.g. for tests) in one place.
"""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from gita_engine.core.config import get_settings

_engine = create_engine(get_settings().postgres_dsn, pool_pre_ping=True)
_SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a database session, committing on success and rolling back on error."""
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
