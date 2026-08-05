"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-08-05

Creates the pgvector extension and the five core tables: verse, source,
translation, commentary, query_log. See docs/phases/phase_3_database.md
for the design rationale.
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1024


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "verse",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("chapter", sa.SmallInteger, nullable=False),
        sa.Column("verse_number", sa.SmallInteger, nullable=False),
        sa.Column("sanskrit_text", sa.Text, nullable=False),
        sa.Column("transliteration", sa.Text),
        sa.Column("variant_note", sa.Text),
        sa.UniqueConstraint("chapter", "verse_number", name="uq_verse_chapter_verse_number"),
    )

    op.create_table(
        "source",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("source_type", sa.String(20), nullable=False),
        sa.Column("tradition", sa.String(50), nullable=False, server_default="pushtimarg"),
        sa.Column("is_accepted", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("citation_label", sa.Text, nullable=False),
    )

    op.create_table(
        "translation",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("verse_id", sa.BigInteger, sa.ForeignKey("verse.id"), nullable=False),
        sa.Column("source_id", sa.BigInteger, sa.ForeignKey("source.id"), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM)),
        sa.UniqueConstraint("verse_id", "source_id", name="uq_translation_verse_source"),
    )
    op.create_index("idx_translation_verse", "translation", ["verse_id"])
    op.execute(
        "CREATE INDEX idx_translation_embedding ON translation "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "commentary",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("verse_id", sa.BigInteger, sa.ForeignKey("verse.id"), nullable=False),
        sa.Column("source_id", sa.BigInteger, sa.ForeignKey("source.id"), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM)),
        sa.Column("chunk_index", sa.SmallInteger, nullable=False, server_default="0"),
        sa.UniqueConstraint(
            "verse_id", "source_id", "chunk_index", name="uq_commentary_verse_source_chunk"
        ),
    )
    op.create_index("idx_commentary_verse", "commentary", ["verse_id"])
    op.execute(
        "CREATE INDEX idx_commentary_embedding ON commentary "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "query_log",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("user_question", sa.Text, nullable=False),
        sa.Column("retrieved_ids", sa.JSON, nullable=False),
        sa.Column("final_answer", sa.Text),
        sa.Column("langfuse_trace_id", sa.String(100)),
    )


def downgrade() -> None:
    op.drop_table("query_log")
    op.execute("DROP INDEX IF EXISTS idx_commentary_embedding")
    op.drop_index("idx_commentary_verse", table_name="commentary")
    op.drop_table("commentary")
    op.execute("DROP INDEX IF EXISTS idx_translation_embedding")
    op.drop_index("idx_translation_verse", table_name="translation")
    op.drop_table("translation")
    op.drop_table("source")
    op.drop_table("verse")
