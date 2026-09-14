"""Evaluate whether generated answers stay grounded in retrieved passages.

Why this file exists:
    Hit Rate/MRR (evaluate_retrieval.py) only measure whether retrieval
    found the right verses. They say nothing about whether the final
    generated answer actually stays faithful to what was retrieved, or
    invents a citation for a verse that was never shown to it. This is a
    direct, deterministic test of FR6/FR7 (grounded answers, real
    citations) -- not the answer's overall "quality," just whether every
    chapter.verse citation it makes corresponds to something that was
    really in the retrieved passage block for that question.

Design notes:
    - Deliberately NOT using an LLM judge (unlike a Ragas-style
      faithfulness score). This check is narrower but fully
      deterministic: parse citations out of the generated text with a
      regex, and check each one against the real (chapter, verse_number)
      set of RetrievedPassage rows for that question. No second LLM
      call, no per-run variance, nothing to tune a judge prompt for.
    - This does NOT catch subtler faithfulness problems -- e.g. citing a
      real, retrieved verse but mischaracterizing what it actually says.
      A citation being "grounded" here only means the cited chapter.verse
      was among the passages fed to the model, not that the model
      described its content accurately. Noted as a real limitation and a
      follow-up candidate for a Ragas-style check, if this eval shows
      citation-grounding itself is already solid.
    - Calls answer_question(..., log=False) so eval runs don't pollute
      query_log with synthetic traffic.
    - Individual question failures (GenerationError, EmbeddingError --
      both real, previously-observed failure modes) are caught and
      reported per-question rather than crashing the whole run, same
      pattern as the transient timeouts hit in evaluate_retrieval.py.
    - PACING (added 2026-09-13, after real testing): a single real
      generation call uses ~6500 of the 8000 TPM budget on Groq's free
      tier (measured directly -- see llm_client.py's docstring history).
      Firing 18 questions back-to-back reliably 429s most of them, even
      with retry-with-backoff in generate() itself, because the token
      window simply can't refill fast enough between heavy calls. A
      fixed pause between questions is more reliable than reactive
      retries alone -- retries were still failing after 3 attempts
      because ~30s of cumulative backoff isn't enough for an
      almost-fully-drained ~60s-reset window. This ONLY matters for this
      batch eval script firing many requests in quick succession -- it
      does not reflect how the live app behaves for a single real user
      asking one question at a time, who would never generate this kind
      of rapid-fire load. NOTE: Groq's rate limits apply at the
      organization level, not per API key (confirmed via Groq's own
      community docs) -- a fresh API key would NOT help here, since it's
      the same account either way.
    - LOGGING FIX (2026-09-13, hardening pass): this script previously
      never called configure_logging(), so structlog ran on unconfigured
      defaults -- meaning generation_possibly_truncated /
      generation_unclosed_think_block log events from llm_client.py were
      NOT reliably visible in this script's output. Now calls
      configure_logging() explicitly at the start of evaluate(), and also
      surfaces a possibly-truncated finish_reason directly in this
      script's own per-question status line and summary counts, rather
      than relying on a log line alone.

Metrics:
    - Grounded Citation Rate: (citations that matched a retrieved
      passage) / (total citations made), across all questions.
    - Zero-Citation Count: how many answers made NO citations at all --
      a different failure mode, an unsupported answer with no
      traceability even if its content happens to be accurate.
    - Hallucinated Citation Count: citations that do NOT match any
      retrieved passage -- the sharpest failure mode, meaning the model
      referenced a verse it was never actually shown.
    - Possibly-Truncated Count: answers where finish_reason == "length"
      -- a real, non-empty answer was produced, but Groq cut it off
      before the model naturally finished. Distinct from the above --
      the answer may still be citation-grounded so far as it goes, but
      could be missing content or ending mid-sentence.

Usage:
    python evaluate_faithfulness.py [path/to/golden_set.json]
"""

import json
import re
import sys
import time
from pathlib import Path

from gita_engine.core.logging import configure_logging, get_logger
from gita_engine.generation.llm_client import GenerationError
from gita_engine.reasoning.graph import answer_question
from gita_engine.retrieval.retriever import EmbeddingError

logger = get_logger(__name__)

DEFAULT_GOLDEN_SET = Path(__file__).parent / "golden_set.json"

# Seconds to wait between questions. Sized with real margin over the
# ~60s TPM reset window observed in testing (a single heavy call can use
# ~6500/8000 tokens) -- see PACING note above.
INTER_QUESTION_PAUSE_SECONDS = 100

# Matches "Chapter 9, Verse 15", "Chapter 9, Verses 15-16",
# "Chapter 9, Verses 15, 16 and 18" -- case-insensitive. This is a
# best-effort parse of the exact format the system prompt asks for
# (SYSTEM_PROMPT rule 2 in graph.py: 'Cite the chapter and verse (e.g.
# "Chapter 9, Verse 15")'). If the model deviates from this phrasing,
# those citations simply won't be counted -- a known limitation, not a
# silent bug. Worth spot-checking real output against this regex the
# first time this eval is run, before trusting the numbers.
CITATION_RE = re.compile(
    r"Chapter\s+(\d+),\s*Verses?\s+([0-9][0-9,\-\s]*(?:and\s+\d+)?)",
    re.IGNORECASE,
)


def parse_citations(text: str) -> set[tuple[int, int]]:
    """Extract every (chapter, verse_number) pair cited in generated text."""
    citations: set[tuple[int, int]] = set()
    for match in CITATION_RE.finditer(text):
        chapter = int(match.group(1))
        verses_part = match.group(2)
        pieces = re.split(r",|\band\b", verses_part)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if "-" in piece:
                start_str, _, end_str = piece.partition("-")
                try:
                    start, end = int(start_str.strip()), int(end_str.strip())
                except ValueError:
                    continue
                for verse in range(start, end + 1):
                    citations.add((chapter, verse))
            else:
                try:
                    citations.add((chapter, int(piece)))
                except ValueError:
                    continue
    return citations


def evaluate(golden_set_path: Path) -> None:
    configure_logging()

    with open(golden_set_path, encoding="utf-8") as f:
        golden_set = json.load(f)

    cases = golden_set["cases"]

    total_citations = 0
    grounded_citations = 0
    hallucinated_citations = 0
    zero_citation_questions = 0
    failed_questions = 0
    possibly_truncated_questions = 0

    print(f"Evaluating citation grounding on {len(cases)} test cases...\n")
    print(
        f"(pacing {INTER_QUESTION_PAUSE_SECONDS}s between questions to stay under "
        "Groq's free-tier TPM limit -- full run will take a while)\n"
    )

    for i, case in enumerate(cases):
        question = case["question"]

        if i > 0:
            time.sleep(INTER_QUESTION_PAUSE_SECONDS)

        try:
            result = answer_question(question, log=False)
        except (GenerationError, EmbeddingError) as e:
            failed_questions += 1
            print(f"[FAILED] {question}")
            print(f"  Error: {e}\n")
            continue

        retrieved_keys = {(p.chapter, p.verse_number) for p in result["retrieved"]}
        cited_keys = parse_citations(result["answer"])

        finish_reason = result.get("finish_reason")
        truncated = finish_reason == "length"
        if truncated:
            possibly_truncated_questions += 1

        if not cited_keys:
            zero_citation_questions += 1
            status = "ZERO CITATIONS"
        else:
            grounded = cited_keys & retrieved_keys
            hallucinated = cited_keys - retrieved_keys
            total_citations += len(cited_keys)
            grounded_citations += len(grounded)
            hallucinated_citations += len(hallucinated)
            status = "OK" if not hallucinated else "HALLUCINATED CITATION"

        if truncated:
            status += " [POSSIBLY TRUNCATED]"

        print(f"[{status}] {question}")
        print(f"  Retrieved: {sorted(retrieved_keys)}")
        print(f"  Cited:     {sorted(cited_keys)}")
        if cited_keys and (cited_keys - retrieved_keys):
            print(f"  Hallucinated: {sorted(cited_keys - retrieved_keys)}")
        print()

    answered = len(cases) - failed_questions
    print("=== Summary ===")
    print(
        f"Questions answered: {answered}/{len(cases)} ({failed_questions} failed with API errors)"
    )
    print(f"Questions with zero citations: {zero_citation_questions}")
    print(f"Possibly-truncated answers (finish_reason=length): {possibly_truncated_questions}")
    if total_citations:
        grounded_rate = grounded_citations / total_citations
        print(f"Total citations made: {total_citations}")
        print(
            f"Grounded Citation Rate: {grounded_rate:.2%} ({grounded_citations}/{total_citations})"
        )
        print(f"Hallucinated citations: {hallucinated_citations}")
    else:
        print("No citations were made across any answered question.")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GOLDEN_SET
    evaluate(path)
