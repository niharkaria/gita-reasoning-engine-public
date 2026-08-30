# Phase 7c: Reranker Re-Test Against Full Golden Set

## Context

On 2026-08-15, a cross-encoder reranking step (BAAI/bge-reranker-v2-m3)
was added to the retrieval pipeline, tested, and reverted the same day.
That test used a single ground-truth question — "who is the true enjoyer
and lord of all sacrifices?" (answered by verse 5.29) — checked across
3 different text-pairing strategies (Gujarati commentary text, Sanskrit
shlok text, both combined). All 3 gave identical results, consistently
ranking 5.29 below 4 less-relevant passages. The reranker code was
reverted out of the live pipeline (`reranker.py` kept, unused) and the
conclusion recorded as: "this reranker was not reliably improving
relevance for this corpus/language."

That conclusion was correct, but the evidence behind it was thin — one
question, however carefully tested, cannot establish whether a reranker
works for a corpus. By the time of Phase 7b (golden set expansion to 18
audited questions, Hit Rate@5 = 61.11%, MRR = 0.556), it was worth
re-testing the reranker properly against the full set instead of
leaving the original single-question conclusion unchallenged.

## What Was Done

1. Re-wired the reranker into the live pipeline (`graph.py`):
   `retrieve()` widened to a pool of 20 candidates, then `rerank()`
   narrowed that pool to the final top 5 before generation.
2. Added a `--rerank` flag to `evaluate_retrieval.py` so the same
   18-question golden set could be run through both the embedding-only
   pipeline and the reranked pipeline for a direct comparison, rather
   than duplicating the eval script.
3. Ran both, hit an HF hosted API timeout on the reranker call (20-pair
   cross-encoder batches take noticeably longer than a single embedding
   call — the initial 60s timeout wasn't quite enough), raised the
   timeout to 120s, and re-ran clean to completion.

## Results

| Metric | Baseline (embedding-only) | Reranked (top-20 → top-5) |
|---|---|---|
| Hit Rate @ 5 | 61.11% (11/18) | 44.44% (8/18) |
| MRR | 0.556 | 0.301 |

Reranking made retrieval measurably worse across the full golden set,
not just on the original single question:

- 7 questions that were correct hits at baseline became misses under
  reranking: staying connected to God daily (8.14/12.8), turning to God
  only in crisis (7.16), confusion about life direction (10.10-11),
  reactivity and inner steadiness (12.13-15), mood swings (14.26), old
  attachments (15.3-4), food/lifestyle affecting the mind (17.8), and
  the "simplest refuge" verse (18.66) — the last of these was a clean
  rank-1 hit at baseline and dropped out of the top 5 entirely.
- None of the 7 genuine misses identified in Phase 7b were fixed by
  reranking.
- The "attractor verse" pattern noted in Phase 7b (1.30 and 16.14
  scoring falsely high across unrelated questions) got *worse* under
  reranking, not better — both verses appeared as false top-hits in
  more questions than at baseline, and directly caused the regression
  on the original 5.29 test question again (16.14 outranked 5.29 in the
  reranked run too, same failure mode as the original 2026-08-15 test).
- Only one question improved: the soul-after-death question moved from
  rank 2 to rank 1 — but it was already a hit either way, so this
  doesn't offset the 7 regressions.

## Conclusion

The 2026-08-15 revert decision was correct. It was reached on
insufficient evidence at the time (one question is not enough to judge
a reranker's fit for a corpus), but a full re-test against the audited
18-question golden set confirms the same underlying finding: this
reranker (BAAI/bge-reranker-v2-m3) is not a good fit for this
corpus/language pairing — English queries against Gujarati/Sanskrit
commentary text — and the attractor-verse problem is a real,
reproducible weakness of this cross-encoder on this data, not an
artifact of one unlucky test case.

This should not be read as "reranking can never help this project." A
different reranker model, or a different text-pairing strategy (e.g.
translating passages to English before scoring, or pairing against
Sanskrit shlok text only), would be a genuinely new experiment with its
own hypothesis — not a retry of this one. `reranker.py`'s `rerank()`
function and `evaluate_retrieval.py`'s `--rerank` flag are left in
place specifically so that future experiment has a ready-made,
already-proven-fair A/B harness to run against, rather than needing to
be rebuilt from scratch.

The live pipeline (`graph.py`) has been reverted to embedding-only
retrieval, matching the Phase 7b baseline (61.11% / 0.556).