"""Throwaway script -- NOT part of the real codebase, do not commit.

Checks REAL commentary/translation text lengths currently in the DB, so
any chunking/truncation threshold for item #5 is chosen from real data
rather than a guess. Prints word-count distribution (min/max/avg/median)
and the 5 longest real rows for each table, restricted to accepted
sources only (matching retriever.py's real query behavior).
"""

import statistics

from sqlalchemy import select

from gita_engine.db.models import Commentary, Source, Translation
from gita_engine.db.session import get_session


def report(label: str, rows: list[tuple[int, str]]) -> None:
    if not rows:
        print(f"\n{label}: no rows found")
        return

    word_counts = [len(text.split()) for _id, text in rows]
    print(f"\n{label} ({len(rows)} rows, accepted sources only)")
    print(f"  min words:    {min(word_counts)}")
    print(f"  max words:    {max(word_counts)}")
    print(f"  avg words:    {statistics.mean(word_counts):.1f}")
    print(f"  median words: {statistics.median(word_counts)}")

    longest = sorted(zip(rows, word_counts), key=lambda x: -x[1])[:5]
    print(f"  5 longest real rows:")
    for (row_id, text), wc in longest:
        preview = text[:80].replace("\n", " ")
        print(f"    id={row_id} words={wc} preview={preview!r}...")


with get_session() as session:
    commentary_rows = session.execute(
        select(Commentary.id, Commentary.text)
        .join(Source, Source.id == Commentary.source_id)
        .where(Source.is_accepted.is_(True))
    ).all()

    translation_rows = session.execute(
        select(Translation.id, Translation.text)
        .join(Source, Source.id == Translation.source_id)
        .where(Source.is_accepted.is_(True))
    ).all()

    report("COMMENTARY", [(r.id, r.text) for r in commentary_rows])
    report("TRANSLATION", [(r.id, r.text) for r in translation_rows])
