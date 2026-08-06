"""Locate the exact page number around each missing/duplicate verse, using
our own parser's page-tracking, so we can inspect the real OCR text there
instead of guessing.
"""

from collections import defaultdict

from gita_engine.ingestion.parse_gita_text import ParsedVerse, parse_ocr_text

CANONICAL_COUNTS = {
    1: 47, 2: 72, 3: 43, 4: 42, 5: 29, 6: 47, 7: 30, 8: 28,
    9: 34, 10: 42, 11: 55, 12: 20, 13: 34, 14: 27, 15: 20,
    16: 24, 17: 28, 18: 78,
}

with open("full_book.txt", encoding="utf-8") as f:
    text = f.read()
verses = parse_ocr_text(text)

by_chapter: dict[int, list[ParsedVerse]] = defaultdict(list)
for v in verses:
    if v.chapter is not None:
        by_chapter[v.chapter].append(v)

print("=== Pages around each missing verse (focus: verse 18 pattern) ===\n")
for ch in sorted(CANONICAL_COUNTS):
    parsed = sorted(by_chapter.get(ch, []), key=lambda v: v.verse_number)
    parsed_numbers = {v.verse_number for v in parsed}
    expected = CANONICAL_COUNTS[ch]
    missing = sorted(set(range(1, expected + 1)) - parsed_numbers)

    if not missing:
        continue

    for m in missing:
        before = next((v for v in reversed(parsed) if v.verse_number < m), None)
        after = next((v for v in parsed if v.verse_number > m), None)
        before_page = before.page_number if before else "?"
        after_page = after.page_number if after else "?"
        print(f"Ch {ch}, missing V{m}: appears between page {before_page} and page {after_page}")

print("\n=== Duplicate verse numbers (exact locations) ===\n")
for ch in sorted(CANONICAL_COUNTS):
    parsed = by_chapter.get(ch, [])
    seen: dict[int, list[ParsedVerse]] = defaultdict(list)
    for v in parsed:
        seen[v.verse_number].append(v)
    for num, group in seen.items():
        if len(group) > 1:
            pages = [v.page_number for v in group]
            print(f"Ch {ch}, V{num} appears {len(group)} times, on pages {pages}")
