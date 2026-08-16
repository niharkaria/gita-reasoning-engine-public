# Phase 7b: Golden Set Expansion & Retrieval Evaluation Findings

## What this covers

Expanding `golden_set.json` from 3 questions (covering 3 of 18 chapters) to
18 questions (one per chapter, full coverage), then using it to get a
trustworthy Hit Rate@5 / MRR reading on retrieval quality -- and auditing
every miss against the actual corpus text before trusting the number.

This follows directly from the Phase 7 known gap: "The golden set has only
3 questions, covering 3 of 18 chapters. Worth expanding before trusting the
retrieval metrics as representative."

## How the 15 new questions were built

Each new question was grounded in a real verse + commentary read directly
from the source PDF (`Bhagvat_Geeta_Guij.pdf`, Gujarati Pushtimarg
commentary, Vitthalnathji tradition) -- not invented from general
knowledge of the Gita. The PDF uses a legacy, non-Unicode Gujarati font,
so text extraction was unusable; verses were read by rasterizing pages and
visually reading the commentary chapter by chapter.

Questions were phrased as a modern person would actually ask them --
first-person, life-problem framing (e.g. "I feel lost and confused about
which way to go in life") -- rather than academic Gita-quiz phrasing (e.g.
"What does chapter 10 say about knowledge?"), since the target use case is
someone bringing a real problem to the system, not a scripture trivia
question.

## Evaluation results, in order

| Stage | Hit Rate@5 | MRR | Cases | Notes |
|---|---|---|---|---|
| Baseline (previous session) | 33% | 0.333 | 3 | Too few data points to trust |
| Full 18-question coverage | 38.89% | 0.361 | 18 | First trustworthy-sized run |
| + dedup fix (see below) | 44.44% | 0.417 | 18 | Same corpus, cleaner scoring |
| + 3 golden-set corrections (see below) | **61.11%** | **0.556** | 18 | Final, audited number |

## Bug found and fixed: duplicate verses within one top-5

`retrieve()` in `retriever.py` queries `translation` and `commentary`
tables separately (top-k each), merges the 10 candidates, and returns the
overall top-k by similarity. When a single verse's translation *and*
commentary both ranked highly, both could land in the same merged top-5 --
two `RetrievedPassage` objects, same `(chapter, verse_number)`, different
`passage_type`. `evaluate_retrieval.py` only compares `(chapter,
verse_number)` tuples, so it silently counted the same verse twice,
wasting a scoring slot.

Decision: fixed the *evaluation script* only (dedupe `retrieved_keys` with
`dict.fromkeys()`, preserving rank order), not `retrieve()` itself.
Getting both translation and commentary for the single best-matching verse
may genuinely help generation quality (richer context for one verse); that
is a different question from retrieval *diversity*, which is what
Hit Rate@5 is meant to measure. `retrieve()` behavior in production is
unchanged.

## Golden-set corrections: 3 false negatives found via manual audit

Every one of the 10 raw misses was checked against the actual corpus text
before being accepted as a real retrieval failure, using a new helper
script, `evaluation/lookup_verse.py` (prints translation + commentary for
any chapter.verse directly from the DB -- built specifically for this
audit, keeps checks in text rather than screenshots).

Three misses turned out to be **the golden set being too narrow**, not
retrieval being wrong:

1. **"What happens to the soul when the body dies?"** -- expected only
   2.27-2.28; retrieval also (correctly) surfaced 2.20 ("the soul is never
   born nor does it ever die...") and 2.26, both part of the same
   continuous teaching block. Expanded to all four: 2.20, 2.26, 2.27, 2.28.

2. **"I want to feel God's presence in daily life, not just in a crisis.
   How do I actually stay connected?"** -- expected only 8.14; retrieval
   surfaced 12.8 ("keep your mind fixed on Me, place your intellect in
   Me"), an equally direct answer to sustained daily connection. Expanded
   to 8.14 and 12.8.

3. **"I don't feel spiritually advanced or wise myself. Is it enough to
   simply listen to and trust those who are?"** -- expected only 13.25;
   retrieval surfaced 18.71 ("whoever hears this dialogue with faith...
   shall also be liberated"), directly on-theme. Expanded to 13.25 and
   18.71.

This mirrors a pattern already seen once before (the "true enjoyer of
sacrifices" question, corrected earlier the same session: 9.24 and 5.29
both legitimately answer it). Lesson for future golden-set entries: a
single verse citation is often too strict for thematic questions:
tradition texts frequently restate the same teaching at multiple points,
and a good retrieval system finding a *different but equally correct*
verse should not be scored as wrong.

## Confirmed genuine retrieval weaknesses (7 remaining misses)

Every one of these was checked against real corpus text and confirmed to
be an actual mismatch, not a golden-set gap:

| Question | Expected | Top wrong hit | Why it's wrong |
|---|---|---|---|
| Guilt/unworthiness from past mistakes | 4.36 | 1.30 | Arjuna's physical panic (bow slipping, skin burning) -- shares distress *intensity* language, unrelated content |
| Balance in daily habits (food/work/rest) | 6.16-17 | 18.66 | The "surrender all dharmas" verse -- shares "everything/all" language, unrelated content |
| Turning to God only in crisis | 7.16 | 16.14 | The demonic/egotistical "I am the lord, I am the enjoyer" verse -- unrelated |
| Confusion, seeking life direction | 10.10-11 | 10.42 | Chapter 10's closing summary verse on Krishna's cosmic scale -- same chapter, wrong theme |
| Mood swings from external circumstance | 14.26 | 1.30 | Same false match as above -- Arjuna's panic, not the three-gunas teaching |
| Stuck in old attachments | 15.3-4 | 5.1 | Arjuna's opening question about sannyasa vs. karma yoga -- doctrinal setup, not about attachment |
| Food/lifestyle affecting mind and spirit | 17.8 | 16.14 | Same false match as above -- demonic arrogance verse |

**Pattern observed:** questions phrased in modern, abstract emotional
language ("at their mercy," "cut free," "stay connected," "unworthy") miss
more often than questions closer to the text's own vocabulary and
imagery (Krishna's universal form, Arjuna's paralysis, anger/cravings).
Plausible cause: `BAAI/bge-m3` may be weaker at bridging casual English
emotional phrasing to formal Gujarati/Sanskrit commentary content across
this specific register gap -- a hypothesis from one evaluation run, not
yet proven.

**Pattern observed, second:** two verses, 1.30 (Arjuna's panic) and 16.14
(demonic arrogance), each appeared as a false top-5 match across *multiple
unrelated* questions. This suggests these two verses' embeddings may be
unusually "central" or generic-sounding in vector space, attracting
queries that don't actually relate to their content. Worth a closer look
if retrieval work resumes -- not yet investigated further.

## What was decided NOT to do (and why)

Chose to stop here rather than immediately attempt fixes for the 7
remaining misses. Reasoning: each plausible fix (query rewriting, hybrid
keyword+vector search, model fine-tuning) is a real, separate scope of
work with its own risk of new failure modes -- not a quick follow-on.
Given the reranker experiment's earlier evidence-based reversal (Phase 7),
the standing rule is: don't attempt further retrieval changes without a
specific, testable hypothesis and a way to measure it against this now-
audited golden set. This is a decision to defer, not a decision to ignore.

## Next steps (not started)

- Cheapest next experiment, if resumed: try prefixing queries with an
  instruction like `"Represent this question for retrieving relevant
  Bhagavad Gita commentary: "` before embedding. `bge-m3`-family models
  are often sensitive to this kind of query-side prompting; this is a
  small, cheap, A/B-testable change against the now-trustworthy 18-
  question golden set.
- Investigate why 1.30 and 16.14 specifically attract unrelated queries.
- Per-stage latency logging (embed/retrieve/generate) -- still not done,
  carried over from Phase 7.
- Faithfulness/groundedness eval via Ragas -- still not wired up, carried
  over from Phase 7.
