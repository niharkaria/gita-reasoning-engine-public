"""remove unused langfuse_trace_id column from query_log

Revision ID: 78f69a2c2223
Revises: 0001
Create Date: 2026-09-20 12:19:35.497809

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '78f69a2c2223'
down_revision: Union[str, Sequence[str], None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Only change: drop the unused langfuse_trace_id column from query_log.
    # (Autogenerate also picked up unrelated created_at type diffs and
    # pgvector/other index diffs that already existed in the DB before this
    # change -- those were deliberately stripped out, not part of this
    # migration.)
    op.drop_column('query_log', 'langfuse_trace_id')


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        'query_log',
        sa.Column('langfuse_trace_id', sa.VARCHAR(length=100), autoincrement=False, nullable=True),
    )
