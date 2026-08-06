"""Load parsed verses from full_book.txt into the Postgres database.

Why this file exists:
    Bridges Phase 4 (OCR + parsing) to the Phase 3 schema — takes the
    ParsedVerse objects from parse_gita_text.py and writes them into the
    verse/source/translation/commentary tables.

Design decisions:
    - Idempotent / safe to re-run: every insert checks for an existing row
      first (by the relevant unique constraint) rather than assuming a
      fresh database. Running this twice does not create duplicate rows.
    - Two Source rows, not one: the schema's Source.source_type
      distinguishes 'translation' vs 'commentary'. Our single source book
      provides both (શ્લોકાર્થ as translation, વિવેચન as commentary), so
      we register it twice under the same author/citation, once per role,
      matching how the schema is meant to be queried (filter by type).
    - Duplicate (chapter, verse_number) pairs from parsing (see
      diagnose.py) are MERGED, not discarded — and merged correctly, not
      naively. We traced the actual mechanism: the FIRST entry's
      sanskrit_text is reliably the real shloka; every later entry's
      sanskrit_text is actually overflow translation/commentary text that
      got mislabeled as a shloka by a stray repeated verse-number marker.
      A naive "keep whichever field is longest" merge was tried and
      rejected — it actively overwrote the real shloka with translation
      text, since the mislabeled overflow is usually longer. Instead we
      trust entry 0's shloka as-is and re-run every other captured
      fragment through the same label-based (શ્લોકાર્થ/વિવેચન) splitter
      used during normal parsing.
    - Verses with no shlokartha or no vivechan simply don't get that
      child row; the Verse row itself is still created as long as it has
      real shloka text.

Usage:
    python load_to_db.py full_book.txt
"""

import sys
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from gita_engine.core.logging import configure_logging, get_logger
from gita_engine.db.models import Commentary, Source, Translation, Verse
from gita_engine.db.session import get_session
from gita_engine.ingestion.parse_gita_text import ParsedVerse, parse_ocr_text

# --- Source metadata ---
# Based on the front-matter seen in this PDF ("ગીતાતાત્પર્યમ્ : શ્રીવિટ્ઠલનાથ
# પ્રભુચરણ વિરચિત" — commentary in the Pushtimarg tradition). Edit these if
# you have more precise attribution.
SOURCE_NAME = "Gujarati Pushtimarg commentary (OCR'd PDF, Vitthalnathji tradition)"
SOURCE_CITATION_LABEL = "Gujarati Pushtimarg Commentary"
SOURCE_TRADITION = "pushtimarg"

configure_logging()
logger = get_logger(__name__)


def merge_duplicate_verses(
    parsed_verses: list[ParsedVerse],
) -> tuple[list[ParsedVerse], list[tuple[int, int]]]:
    """Group parsed verses by (chapter, verse_number) and merge any group
    with more than one entry.

    Confirmed mechanism (verified by tracing the actual parse output): the
    FIRST entry is reliably the real verse, with correct shloka,
    shlokartha, and vivechan — duplicates arise from a stray repeated
    verse-number marker appearing AFTER the real verse's content was
    already fully captured (e.g. a citation sentence coincidentally ending
    in marker-shaped digits). Later entries are near-empty fragments or
    stray overflow text from whatever follows the spurious marker.

    We therefore simply prefer entry 0's fields, falling back to a later
    entry's field only if entry 0 is missing it. We do NOT attempt to
    re-split or recombine text across entries — an earlier version of this
    function did that by concatenating already-split shlokartha/vivechan
    text and re-running the label-based splitter, which failed because the
    labels had already been stripped out during the original split,
    causing correct content to be silently misfiled as "no label found."
    """
    groups: dict[tuple[int, int], list[ParsedVerse]] = defaultdict(list)
    for pv in parsed_verses:
        if pv.chapter is not None:
            groups[(pv.chapter, pv.verse_number)].append(pv)

    merged: list[ParsedVerse] = []
    merged_keys: list[tuple[int, int]] = []

    for (chapter, verse_number), group in groups.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        merged_keys.append((chapter, verse_number))

        base = group[0]
        shlokartha = base.shlokartha or next(
            (g.shlokartha for g in group[1:] if g.shlokartha), None
        )
        vivechan = base.vivechan or next((g.vivechan for g in group[1:] if g.vivechan), None)

        all_warnings = sorted({w for g in group for w in g.warnings})
        all_warnings.append(f"merged_from_{len(group)}_duplicate_entries")

        merged.append(
            ParsedVerse(
                chapter=chapter,
                verse_number=verse_number,
                sanskrit_text=base.sanskrit_text,
                shlokartha=shlokartha,
                vivechan=vivechan,
                page_number=base.page_number,
                warnings=all_warnings,
            )
        )

    return merged, merged_keys


def get_or_create_source(session: Session, source_type: str) -> Source:
    """Return the existing Source row for our book (by name + type), or create it."""
    existing = session.execute(
        select(Source).where(Source.name == SOURCE_NAME, Source.source_type == source_type)
    ).scalar_one_or_none()
    if existing:
        return existing

    source = Source(
        name=SOURCE_NAME,
        source_type=source_type,
        tradition=SOURCE_TRADITION,
        is_accepted=True,
        citation_label=SOURCE_CITATION_LABEL,
    )
    session.add(source)
    session.flush()  # assigns source.id without committing yet
    return source


def get_or_create_verse(session: Session, parsed: ParsedVerse) -> Verse:
    """Return the existing Verse row for (chapter, verse_number), or create it."""
    assert parsed.chapter is not None  # caller guarantees this
    existing = session.execute(
        select(Verse).where(
            Verse.chapter == parsed.chapter, Verse.verse_number == parsed.verse_number
        )
    ).scalar_one_or_none()
    if existing:
        return existing

    verse = Verse(
        chapter=parsed.chapter,
        verse_number=parsed.verse_number,
        sanskrit_text=parsed.sanskrit_text,
    )
    session.add(verse)
    session.flush()
    return verse


def upsert_translation(session: Session, verse: Verse, source: Source, text: str) -> bool:
    """Create a Translation row if one doesn't already exist for (verse, source).
    Returns True if a new row was created."""
    existing = session.execute(
        select(Translation).where(
            Translation.verse_id == verse.id, Translation.source_id == source.id
        )
    ).scalar_one_or_none()
    if existing:
        return False
    session.add(Translation(verse_id=verse.id, source_id=source.id, text=text))
    return True


def upsert_commentary(session: Session, verse: Verse, source: Source, text: str) -> bool:
    """Create a Commentary row (chunk_index=0) if one doesn't already exist.
    Returns True if a new row was created."""
    existing = session.execute(
        select(Commentary).where(
            Commentary.verse_id == verse.id,
            Commentary.source_id == source.id,
            Commentary.chunk_index == 0,
        )
    ).scalar_one_or_none()
    if existing:
        return False
    session.add(Commentary(verse_id=verse.id, source_id=source.id, text=text, chunk_index=0))
    return True


def load(ocr_text_path: str) -> None:
    with open(ocr_text_path, encoding="utf-8") as f:
        full_text = f.read()

    parsed_verses = parse_ocr_text(full_text)
    logger.info("parsed_verses_loaded", count=len(parsed_verses))

    no_chapter_count = sum(1 for pv in parsed_verses if pv.chapter is None)
    if no_chapter_count:
        logger.warning("skipping_verses_no_chapter", count=no_chapter_count)

    merged_verses, merged_keys = merge_duplicate_verses(parsed_verses)
    if merged_keys:
        logger.info("merged_duplicate_verses", keys=merged_keys)

    to_load = [pv for pv in merged_verses if pv.sanskrit_text.strip()]
    skipped_empty = len(merged_verses) - len(to_load)
    if skipped_empty:
        logger.warning("skipping_verses_empty_shloka", count=skipped_empty)

    verses_created = 0
    translations_created = 0
    commentaries_created = 0

    with get_session() as session:
        translation_source = get_or_create_source(session, "translation")
        commentary_source = get_or_create_source(session, "commentary")

        for pv in to_load:
            verse = get_or_create_verse(session, pv)
            verses_created += 1

            if pv.shlokartha and upsert_translation(
                session, verse, translation_source, pv.shlokartha
            ):
                translations_created += 1

            if pv.vivechan and upsert_commentary(session, verse, commentary_source, pv.vivechan):
                commentaries_created += 1

    print("=== Load summary ===")
    print(f"Verses processed:      {verses_created}")
    print(f"Translations inserted: {translations_created}")
    print(f"Commentaries inserted: {commentaries_created}")
    print(f"Duplicate verse-numbers merged: {len(merged_keys)}")
    if merged_keys:
        print(f"  Merged: {merged_keys}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "full_book.txt"
    load(path)
