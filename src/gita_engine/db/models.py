"""SQLAlchemy ORM models for the Gita Engine schema.

Why this file exists:
    Single source of truth for the database shape. Alembic autogenerates
    migrations by diffing this against the live database, so this file
    should always reflect what we intend the schema to be — not what's
    currently deployed.

Design notes:
    - `source` decouples "who wrote this translation/commentary" from the
      text itself, so we can add sources later without touching existing
      rows, and can gate retrieval to `is_accepted = True` sources only
      (enforces FR7: never answer from outside the accepted corpus).
    - `translation` and `commentary` both carry their own `embedding`
      column (pgvector) sized for BGE-M3 (1024-dim). They're kept as
      separate tables rather than unified into one "content" table because
      they have different shapes: commentary is chunked (can be long-form),
      translation is not.
    - `query_log` exists from Phase 3, not Phase 7, so we don't lose
      historical query data before evaluation tooling is built.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    ForeignKey,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 1024  # BGE-M3 output dimension


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class Verse(Base):
    """A canonical Bhagavad Gita verse reference (chapter + verse number)."""

    __tablename__ = "verse"
    __table_args__ = (UniqueConstraint("chapter", "verse_number"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    chapter: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    verse_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    sanskrit_text: Mapped[str] = mapped_column(Text, nullable=False)
    transliteration: Mapped[str | None] = mapped_column(Text)
    variant_note: Mapped[str | None] = mapped_column(Text)

    translations: Mapped[list["Translation"]] = relationship(back_populates="verse")
    commentaries: Mapped[list["Commentary"]] = relationship(back_populates="verse")


class Source(Base):
    """A translator or commentator whose work is (or was) part of the accepted corpus."""

    __tablename__ = "source"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    name: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)  # 'translation' | 'commentary'
    tradition: Mapped[str] = mapped_column(String(50), nullable=False, default="pushtimarg")
    is_accepted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    citation_label: Mapped[str] = mapped_column(Text, nullable=False)


class Translation(Base):
    """A translation of a single verse, attributed to one source."""

    __tablename__ = "translation"
    __table_args__ = (UniqueConstraint("verse_id", "source_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    verse_id: Mapped[int] = mapped_column(ForeignKey("verse.id"), nullable=False)
    source_id: Mapped[int] = mapped_column(ForeignKey("source.id"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))

    verse: Mapped["Verse"] = relationship(back_populates="translations")
    source: Mapped["Source"] = relationship()


class Commentary(Base):
    """A chunk of commentary text on a verse, attributed to one source.

    Long commentaries are split across multiple rows sharing (verse_id,
    source_id) but distinct chunk_index, so each embeddable unit stays
    within the embedding model's effective context while citations still
    resolve back to "which commentary, on which verse."
    """

    __tablename__ = "commentary"
    __table_args__ = (UniqueConstraint("verse_id", "source_id", "chunk_index"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    verse_id: Mapped[int] = mapped_column(ForeignKey("verse.id"), nullable=False)
    source_id: Mapped[int] = mapped_column(ForeignKey("source.id"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    chunk_index: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)

    verse: Mapped["Verse"] = relationship(back_populates="commentaries")
    source: Mapped["Source"] = relationship()


class QueryLog(Base):
    """A record of a user question, what was retrieved, and the final answer.

    Exists from Phase 3 (not Phase 7) so historical query data isn't lost
    before evaluation tooling is built on top of it.
    """

    __tablename__ = "query_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    user_question: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_ids: Mapped[dict[str, list[int]]] = mapped_column(JSON, nullable=False)
    final_answer: Mapped[str | None] = mapped_column(Text)  
