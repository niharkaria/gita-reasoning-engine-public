"""Diagnose gaps and warnings in the full parsed Gita corpus.

Compares parsed verse counts per chapter against the traditional canonical
count (which totals 700 across all 18 chapters), finds specific missing
verse numbers within each chapter, flags duplicates, and groups warnings
by chapter so we can see whether they cluster or spread evenly.
"""

from collections import defaultdict

from gita_engine.ingestion.parse_gita_text import ParsedVerse, parse_ocr_text

# Traditional canonical verse count per chapter (sums to 700).
CANONICAL_COUNTS = {
    1: 47,
    2: 72,
    3: 43,
    4: 42,
    5: 29,
    6: 47,
    7: 30,
    8: 28,
    9: 34,
    10: 42,
    11: 55,
    12: 20,
    13: 34,
    14: 27,
    15: 20,
    16: 24,
    17: 28,
    18: 78,
}

with open("full_book.txt", encoding="utf-8") as f:
    text = f.read()
verses = parse_ocr_text(text)

by_chapter: dict[int, list["ParsedVerse"]] = defaultdict(list)
for v in verses:
    if v.chapter is not None:
        by_chapter[v.chapter].append(v)

print("=== Per-chapter verse counts vs canonical ===")
total_missing = 0
for ch in sorted(CANONICAL_COUNTS):
    parsed_verses = by_chapter.get(ch, [])
    parsed_numbers = sorted({v.verse_number for v in parsed_verses})
    expected = CANONICAL_COUNTS[ch]
    actual = len(parsed_numbers)
    diff = expected - actual
    total_missing += max(diff, 0)

    missing_numbers = sorted(set(range(1, expected + 1)) - set(parsed_numbers))
    dupes = len(parsed_verses) - len(parsed_numbers)

    flag = ""
    if diff != 0 or dupes:
        flag = f"  <-- missing {missing_numbers}" if missing_numbers else ""
        if dupes:
            flag += f"  <-- {dupes} duplicate verse number(s)"

    print(f"Ch {ch:>2}: expected {expected:>3}, got {actual:>3}{flag}")

print(f"\nTotal missing across all chapters: {total_missing}")

print("\n=== Warnings by chapter ===")
warning_counts: dict[int, int] = defaultdict(int)
for v in verses:
    if v.warnings and v.chapter is not None:
        warning_counts[v.chapter] += 1

for ch in sorted(CANONICAL_COUNTS):
    count = warning_counts.get(ch, 0)
    total = len(by_chapter.get(ch, []))
    if count:
        pct = 100 * count / total if total else 0
        print(f"Ch {ch:>2}: {count}/{total} verses warned ({pct:.0f}%)")
