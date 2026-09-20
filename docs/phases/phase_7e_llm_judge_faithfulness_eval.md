# Phase 7e: LLM-Judge Claim-Level Faithfulness Evaluation (Small-Sample)

## Context

Phase 7d's `evaluate_faithfulness.py` is a deterministic, code-only check: it verifies
every cited chapter.verse number corresponds to a real retrieved passage (100.00%,
55/55, zero hallucinated citations — see `phase_7d_faithfulness_eval.md`). It explicitly
does not catch a subtler failure mode: citing a *real, retrieved* verse while
mischaracterizing what that verse actually says.

`evaluate_faithfulness_llm_judge.py` closes that gap: it decomposes a generated answer
into atomic factual claims and asks an LLM judge to verify each claim against the actual
retrieved passage text, not just the verse numbers. Built by hand rather than with
`ragas` (broken in this environment's Python 3.14.4 venv — a real `ModuleNotFoundError`
in ragas' own dependency chain, not chased further). Judge model is the same model as the
generator (`qwen/qwen3.8-27b`), a documented self-grading-bias tradeoff, not an
oversight.

## What this doc corrects

Prior informal notes (outside this `docs/phases/` series, not backed by any saved
evidence or phase doc) cited a claim support rate of ~96.15% across the full 18-question
golden set, with an adversarial control test flagging 0/5 fabricated claims as supported.
**No log, script output, or phase doc for that run could be located** (`docs/evidence/`
only contains Phase 7f truncation logs; there is no `phase_7e` predecessor). That number
should be treated as unverified and is superseded by this document's real, reproduced
findings until a full, evidence-backed run is done.

## Methodology

Each question costs two real Groq calls (generation + judge) with `PAUSE_BEFORE_JUDGE_SECONDS
= 60` and `PAUSE_BETWEEN_QUESTIONS_SECONDS = 100` built into the script, plus generate()'s
own 429 retry/backoff on top when the free-tier budget is tight. Given real, observed 429s
even at this pacing (see raw log), a full 18-question run costs 45-60+ minutes. Given that
cost, this pass deliberately ran a **3-question sample** (`max_questions=3`) rather than
the full golden set, and ran it twice to check stability before deciding whether a full
run was worth the time.

## Results

| Run | Questions | Claims judged | Supported | Partially supported | Unsupported | Failures/unparseable |
|---|---|---|---|---|---|---|
| 1 | 3 | 23 | 91.30% (21/23) | 2 | 0 | 0 |
| 2 | 3 | 25 | 100.00% (25/25) | 0 | 0 | 0 |

Same 3 questions, same script, judge called at `temperature=0.0` both times — yet a
different result each run, including a different total claim count (23 vs 25). Run 1's
full stage-by-stage log was not captured to file (only the printed summary above is
available). Run 2's full log is saved at
`docs/evidence/phase7e_llm_judge_run2_2026-09-20.log`.

## Real finding: run-to-run variance, and its cause

The judge call itself is deterministic (`temperature=0.0`), but the **answer-generation
call runs at `temperature=0.2`** (the main pipeline's normal setting — grounded-but-not-
rigid answers). That means the exact wording of the generated answer differs slightly
between runs, which changes how the judge decomposes it into claims (different claim
count) and can shift a borderline claim's verdict between "supported" and "partially
supported."

This means **a single run of this eval, at any sample size, reports one sample from a
distribution, not a fixed ground-truth score.** A single number (91.30%, 100.00%, or the
previously-circulated 96.15%) presented without this caveat overstates precision the
underlying measurement doesn't have.

## What this does and doesn't tell us

- Both runs: **zero fully unsupported claims** across both passes (23+25 = 48 claims,
  0 unsupported). This is consistent with Phase 7d's 100% citation-grounding result — the
  model isn't inventing content or fabricating support, at least not on this 3-question
  sample.
- The only disagreement between runs was on "partially supported" vs "supported" for 2
  claims out of 23-25 — a real but narrow band of judge/wording sensitivity, not a
  wholesale faithfulness problem.
- This is **3 of 18 golden-set questions**, run twice — real evidence, but not a
  full-corpus claim-support rate. Extrapolating a full-18 percentage from this sample
  would overstate confidence the sample size doesn't support.

## Honest number to cite

Across both runs combined: **48 claims judged, 0 unsupported, 2 partially supported
(both in Run 1)** — a combined "fully or partially supported" rate of 100% (48/48), and a
"fully supported" rate that varies 91.30%-100.00% depending on run. State it as a range
with the variance explained, not a single decimal.

## Follow-up (not done here)

- A full 18-question run (single pass, budgeting for the ~45-60 min real cost including
  429 retries) would give a larger-sample number, but would still be a single sample from
  the same underlying distribution — running it 2-3 times and reporting a range would be
  more honest than treating one 18-question run as definitive.
- Running the answer-generation step at `temperature=0.0` (matching the judge) for a
  faithfulness-specific eval run would remove this specific source of variance, at the
  cost of no longer testing the pipeline's actual production settings.
