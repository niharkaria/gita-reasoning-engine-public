"""Export translation/commentary text that needs embeddings, as JSONL, for
a Kaggle notebook to consume (Kaggle has GPU; your laptop doesn't).

Why this file exists:
    BGE-M3 embedding generation for ~1200 texts (translations + commentaries)
    is impractical on an 8GB/i3 laptop with no GPU. The split: this script
    runs locally against your real Postgres data and produces a JSONL file
    you upload to Kaggle as a dataset; kaggle_generate_embeddings.py runs
    ON Kaggle to produce embeddings; import_embeddings.py runs back here
    locally to load the results into Postgres.

Output format (one JSON object per line):
    {"table": "translation", "id": 123, "text": "..."}
    {"table": "commentary",  "id": 456, "text": "..."}

Usage:
    python export_for_embedding.py output.jsonl
"""

import json
import sys

from sqlalchemy import select

from gita_engine.db.models import Commentary, Translation
from gita_engine.db.session import get_session


def export(output_path: str) -> None:
    count = 0
    with get_session() as session, open(output_path, "w", encoding="utf-8") as out:
        translations = session.execute(
            select(Translation.id, Translation.text).where(Translation.embedding.is_(None))
        ).all()
        for row in translations:
            out.write(json.dumps({"table": "translation", "id": row.id, "text": row.text}) + "\n")
            count += 1

        commentaries = session.execute(
            select(Commentary.id, Commentary.text).where(Commentary.embedding.is_(None))
        ).all()
        for row in commentaries:
            out.write(json.dumps({"table": "commentary", "id": row.id, "text": row.text}) + "\n")
            count += 1

    print(f"Exported {count} texts needing embeddings to {output_path}")
    print(f"  Translations: {len(translations)}")
    print(f"  Commentaries: {len(commentaries)}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "embedding_export.jsonl"
    export(path)
