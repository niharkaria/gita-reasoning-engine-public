"""Import embeddings generated on Kaggle back into Postgres.

Why this file exists:
    Completes the local -> Kaggle -> local round trip: export_for_embedding.py
    produced the input, kaggle_generate_embeddings.py (run on Kaggle) produced
    the output, this script writes the resulting vectors into the
    translation.embedding / commentary.embedding pgvector columns.

Usage:
    python import_embeddings.py embedding_output.jsonl
"""

import json
import sys

from sqlalchemy import update

from gita_engine.db.models import Commentary, Translation
from gita_engine.db.session import get_session

EXPECTED_DIM = 1024  # BGE-M3 output dimension, must match the Vector(1024) column


def import_embeddings(input_path: str) -> None:
    updated = {"translation": 0, "commentary": 0}
    skipped = 0

    with get_session() as session, open(input_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            table = record.get("table")
            row_id = record.get("id")
            embedding = record.get("embedding")

            if table not in ("translation", "commentary"):
                print(f"  line {line_number}: unknown table {table!r}, skipping")
                skipped += 1
                continue
            if not isinstance(embedding, list) or len(embedding) != EXPECTED_DIM:
                actual_len = len(embedding) if isinstance(embedding, list) else "n/a"
                print(
                    f"  line {line_number}: bad embedding dimension "
                    f"(expected {EXPECTED_DIM}, got {actual_len}), skipping"
                )
                skipped += 1
                continue

            model_cls = Translation if table == "translation" else Commentary
            session.execute(
                update(model_cls).where(model_cls.id == row_id).values(embedding=embedding)
            )
            updated[table] += 1

    print("=== Import summary ===")
    print(f"Translation embeddings updated: {updated['translation']}")
    print(f"Commentary embeddings updated:  {updated['commentary']}")
    if skipped:
        print(f"Skipped (bad data): {skipped}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "embedding_output.jsonl"
    import_embeddings(path)
