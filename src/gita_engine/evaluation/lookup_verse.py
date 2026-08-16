"""Quick lookup: print translation + commentary text for a given verse.

Usage:
    uv run python src/gita_engine/evaluation/lookup_verse.py 2 20
    uv run python src/gita_engine/evaluation/lookup_verse.py 9 24
"""
import sys

from sqlalchemy import select

from gita_engine.db.models import Commentary, Source, Translation, Verse
from gita_engine.db.session import get_session


def lookup(chapter: int, verse_number: int) -> None:
    with get_session() as session:
        verse = session.execute(
            select(Verse).where(
                Verse.chapter == chapter, Verse.verse_number == verse_number
            )
        ).scalar_one_or_none()

        if verse is None:
            print(f"No verse found for {chapter}.{verse_number}")
            return

        print(f"\n=== {chapter}.{verse_number} ===")
        print(f"Sanskrit: {verse.sanskrit_text}\n")

        translations = session.execute(
            select(Translation, Source.citation_label)
            .join(Source, Source.id == Translation.source_id)
            .where(Translation.verse_id == verse.id, Source.is_accepted.is_(True))
        ).all()
        for t, citation in translations:
            print(f"--- Translation ({citation}) ---")
            print(t.text)
            print()

        commentaries = session.execute(
            select(Commentary, Source.citation_label)
            .join(Source, Source.id == Commentary.source_id)
            .where(Commentary.verse_id == verse.id, Source.is_accepted.is_(True))
        ).all()
        for c, citation in commentaries:
            print(f"--- Commentary ({citation}) ---")
            print(c.text)
            print()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: lookup_verse.py <chapter> <verse_number>")
        sys.exit(1)
    lookup(int(sys.argv[1]), int(sys.argv[2]))
