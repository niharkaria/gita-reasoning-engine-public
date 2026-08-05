"""Parse OCR'd Gujarati text into structured verse/translation/commentary units.

Why this file exists:
    Raw OCR output is just a wall of text with page markers. This module
    finds verse boundaries, chapter numbers, and splits each verse's
    surrounding text into shloka / shlokartha ("translation") / vivechan
    ("commentary") — the shapes our Phase 3 schema (Verse, Translation,
    Commentary) expects.

Parsing strategy (heuristic, not guaranteed perfect — see Known limitations):
    1. Chapter number ("અધ્યાય N") appears in page footers; we track the
       most recently seen chapter number and apply it going forward.
    2. A verse ends wherever we see the Gujarati verse-number marker
       (e.g. "।૨૪।।" — danda, Gujarati digits, double danda).
    3. Splitting the full text on that marker gives us chunks. Each chunk
       contains: [commentary for the PREVIOUS verse][blank line][shloka
       lines for the CURRENT verse]. We find the last blank line in the
       chunk to separate the two.
    4. Within the commentary portion, "શ્લોકાર્થ" (literal meaning) and
       "વિવેચન" (elaboration) are separated by their own labels.

Known limitations (expect to refine once run against the full 320 pages):
    - Relies on a blank line separating commentary from the next shloka.
      OCR noise or unusual page layout can break this.
    - Assumes શ્લોકાર્થ/વિવેચન labels are OCR'd cleanly; minor OCR errors
      in the label text itself would cause a miss (falls back to storing
      the whole commentary chunk as "commentary" with no split).
    - Does not yet handle verses split across a page boundary.
    - The verse-end marker regex tolerates two known OCR defects seen in
      real output: (a) a missing danda (single "।" instead of "।।"), and
      (b) a stray Latin "1" inserted before the real Gujarati digits
      (e.g. "।।1૩૦।।" for what should be "।।૩૦।।"). Any non-Gujarati-digit
      character inside the number is treated as OCR noise and discarded
      when converting to an int — see gujarati_number_to_int.
    - Chapter-ending colophon lines (e.g. "નવમોડધ્યાયઃ ।।૯।।" — "thus ends
      the ninth chapter") match the verse-marker pattern but are NOT
      verses; they're excluded by checking for "ધ્યાયઃ" immediately
      before the match.
    - Inline citations to other texts (e.g. "ભાગ.૧૦।૨।૨૬") also contain
      danda-separated numbers but are excluded because real verse markers
      always sit at the end of a line, while citations continue with more
      text on the same line.
"""

import re
from dataclasses import dataclass, field

GUJARATI_DIGIT_CHARS = "૦૧૨૩૪૫૬૭૮૯"
GUJARATI_DIGITS = str.maketrans(GUJARATI_DIGIT_CHARS, "0123456789")

CHAPTER_RE = re.compile(r"અધ્યાય\s*([૦-૯]+)")
# Tolerant of missing/extra dandas (।+) and a stray Latin digit (1) that
# OCR occasionally inserts; anchored to end-of-line to avoid matching
# inline citations like "ભાગ.૧૦।૨।૨૬" which continue on the same line.
VERSE_END_RE = re.compile(r"।+\s*([૦-૯1]{1,4})\s*।+(?=[ \t]*\n)")
COLOPHON_MARKER_RE = re.compile(r"ધ્યાયઃ\s*$")
SHLOKARTH_LABEL_RE = re.compile(r"શ્લોકાર્થ\s*[:ઃ]?")
# વિવેચન ("elaboration") is the standard label, but some verses — notably
# chapter-opening ones — use વિશેષ ("special note") instead for the same
# role. This is a real convention in the source text, not OCR noise.
VIVECHAN_LABEL_RE = re.compile(r"(?:વિવેચન|વિશેષ)\s*[:ઃ]?")
PAGE_MARKER_RE = re.compile(r"===PAGE (\d+)===")


def gujarati_number_to_int(text: str) -> int:
    """Convert a Gujarati-numeral string to an int, discarding any stray
    non-Gujarati-digit characters (known OCR noise, e.g. an inserted "1").
    """
    gujarati_only = "".join(ch for ch in text if ch in GUJARATI_DIGIT_CHARS)
    return int(gujarati_only.translate(GUJARATI_DIGITS))


def _is_colophon(full_text: str, match_start: int) -> bool:
    """True if this verse-marker-shaped match is actually a chapter-ending
    colophon (e.g. "નવમોડધ્યાયઃ ।।૯।।") rather than a real verse."""
    preceding = full_text[max(0, match_start - 40) : match_start]
    return bool(COLOPHON_MARKER_RE.search(preceding.rstrip()))


@dataclass
class ParsedVerse:
    """One parsed verse unit, ready to map onto Verse/Translation/Commentary rows."""

    chapter: int | None
    verse_number: int
    sanskrit_text: str
    shlokartha: str | None = None
    vivechan: str | None = None
    page_number: int | None = None
    warnings: list[str] = field(default_factory=list)


def _split_commentary(commentary_text: str) -> tuple[str | None, str | None, list[str]]:
    """Split a commentary block into (shlokartha, vivechan, warnings)."""
    warnings: list[str] = []

    shlokarth_match = SHLOKARTH_LABEL_RE.search(commentary_text)
    vivechan_match = VIVECHAN_LABEL_RE.search(commentary_text)

    if not shlokarth_match and not vivechan_match:
        warnings.append("no_shlokartha_or_vivechan_label_found")
        return None, commentary_text.strip() or None, warnings

    shlokartha = None
    vivechan = None

    if shlokarth_match:
        start = shlokarth_match.end()
        end = vivechan_match.start() if vivechan_match else len(commentary_text)
        shlokartha = commentary_text[start:end].strip() or None
    else:
        warnings.append("shlokartha_label_missing")

    if vivechan_match:
        vivechan = commentary_text[vivechan_match.end() :].strip() or None
    else:
        warnings.append("vivechan_label_missing")

    return shlokartha, vivechan, warnings


def _is_verse_like(segment: str, *, require_danda_ending: bool = True) -> bool:
    """Heuristic: does this blank-line-separated segment look like a shloka
    pada (short poetic line) rather than commentary prose?

    Shloka padas in this text are short AND end with a single danda (।) —
    the traditional pada-separator punctuation. Prose sentences, even short
    ones, end with a Gujarati/Latin period. Checking the ending punctuation
    (not just length) avoids misclassifying short prose fragments — e.g. a
    trailing "...has been done in the Subodhini above." — as verse text.
    """
    stripped = segment.strip()
    if not stripped:
        return False
    if SHLOKARTH_LABEL_RE.search(stripped) or VIVECHAN_LABEL_RE.search(stripped):
        return False
    if len(stripped) > 120:
        return False
    return not require_danda_ending or stripped.endswith("।")


def _split_chunk_into_commentary_and_shloka(chunk: str) -> tuple[str, str]:
    """Split a chunk (text between two verse-end markers) into
    (commentary_for_previous_verse, shloka_for_this_verse).

    Shlokas are often printed as 2+ blank-line-separated padas, so we can't
    just split on the single last blank line — we walk backward through
    segments, greedily claiming verse-like ones (up to a sane cap) as part
    of the shloka, and treat everything else as commentary. The segment
    immediately adjacent to the verse-end marker is always included as
    shloka (it never carries a trailing danda itself, since the marker's
    own leading danda is what terminates it) — everything further back
    must end in a danda to qualify.
    """
    segments = [s for s in re.split(r"\n\s*\n", chunk) if s.strip()]
    if not segments:
        return "", ""

    max_padas = 4  # generous cap; typical anushtubh shlokas are 2 padas
    shloka_segments: list[str] = [segments[-1]]
    split_index = len(segments) - 1

    for seg in reversed(segments[:-1]):
        if len(shloka_segments) < max_padas and _is_verse_like(seg, require_danda_ending=True):
            shloka_segments.insert(0, seg)
            split_index -= 1
        else:
            break

    commentary_text = "\n\n".join(segments[:split_index])
    shloka_text = "\n\n".join(shloka_segments)
    return commentary_text, shloka_text


def parse_ocr_text(full_text: str) -> list[ParsedVerse]:
    """Parse OCR'd text (with ===PAGE N=== markers) into a list of ParsedVerse."""
    # Track page number per character offset so we can attribute each verse
    # to the page it was found on (useful for debugging/spot-checking later).
    page_positions: list[tuple[int, int]] = [
        (m.start(), int(m.group(1))) for m in PAGE_MARKER_RE.finditer(full_text)
    ]

    def page_for_offset(offset: int) -> int | None:
        current = None
        for pos, page_num in page_positions:
            if pos <= offset:
                current = page_num
            else:
                break
        return current

    # Track chapter number similarly, but by nearest preceding occurrence.
    chapter_positions: list[tuple[int, int]] = [
        (m.start(), gujarati_number_to_int(m.group(1))) for m in CHAPTER_RE.finditer(full_text)
    ]

    def chapter_for_offset(offset: int) -> int | None:
        current = None
        for pos, chapter_num in chapter_positions:
            if pos <= offset:
                current = chapter_num
            else:
                break
        return current

    verses: list[ParsedVerse] = []
    matches = [
        m for m in VERSE_END_RE.finditer(full_text) if not _is_colophon(full_text, m.start())
    ]

    for i, match in enumerate(matches):
        chunk_start = matches[i - 1].end() if i > 0 else 0
        chunk_end = match.start()
        chunk = full_text[chunk_start:chunk_end]

        # Correctly handles shlokas printed as multiple blank-line-separated
        # padas, not just a single trailing line.
        chunk_clean = re.sub(r"===PAGE \d+===", "", chunk)
        commentary_text, shloka_text = _split_chunk_into_commentary_and_shloka(chunk_clean)
        shloka_text = shloka_text.strip()

        warnings: list[str] = []
        if not shloka_text:
            warnings.append("empty_shloka_text")

        # Attach the commentary we just isolated to the PREVIOUS verse, if any.
        if verses and commentary_text.strip():
            shlokartha, vivechan, split_warnings = _split_commentary(commentary_text.strip())
            verses[-1].shlokartha = shlokartha
            verses[-1].vivechan = vivechan
            verses[-1].warnings.extend(split_warnings)

        verses.append(
            ParsedVerse(
                chapter=chapter_for_offset(match.start()),
                verse_number=gujarati_number_to_int(match.group(1)),
                sanskrit_text=shloka_text,
                page_number=page_for_offset(match.start()),
                warnings=warnings,
            )
        )

    # The very last verse's commentary lives AFTER the final marker — attach it.
    if matches and verses:
        tail = full_text[matches[-1].end() :]
        tail_clean = re.sub(r"===PAGE \d+===", "", tail).strip()
        if tail_clean:
            shlokartha, vivechan, split_warnings = _split_commentary(tail_clean)
            verses[-1].shlokartha = shlokartha
            verses[-1].vivechan = vivechan
            verses[-1].warnings.extend(split_warnings)

    return verses
