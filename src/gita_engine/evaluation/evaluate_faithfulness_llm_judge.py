"""LLM-judge faithfulness check: subtler than evaluate_faithfulness.py's
deterministic citation-grounding check.

Why this file exists:
    evaluate_faithfulness.py checks whether cited chapter.verse numbers
    correspond to real retrieved passages -- but it explicitly does NOT
    catch a model citing a REAL verse while mischaracterizing what that
    verse actually says. This script closes that gap using an LLM judge:
    it decomposes the generated answer into individual factual claims,
    then asks the judge model to verify each claim against the real
    retrieved passage text (not just the verse numbers).

Design notes (decided 2026-09-13, after ragas package proved broken on
this Python 3.14.4 .venv -- a real ModuleNotFoundError inside ragas'
own dependency chain on langchain_community.chat_models.vertexai, not
fixed, deliberately not chased further):
    - Built by hand instead of using the `ragas` package. This is a
      genuine tradeoff, not a shortcut: the ragas package is a
      recognizable name, but building this by hand means every part of
      the judge methodology (the exact prompt, exactly what "faithful"
      means here, how JSON parsing failures are handled) is understood
      and owned, not a black box.
    - Claim decomposition + verification happens in ONE combined judge
      call per question (not separate extract-then-verify calls) to
      keep this at 2 real Groq calls per question (answer + judge)
      instead of 3 -- OTPM pressure is a real, already-proven-painful
      constraint in this project (see llm_client.py's docstring
      history), not a hypothetical one to ignore.
    - Judge model is the SAME model (qwen/qwen3.6-27b) as the generator.
      Real, known limitation: self-grading bias -- a model may be more
      lenient toward its own phrasing/reasoning style than an
      independent judge would be. A stronger design would use a
      different model as judge. Documented here as a deliberate,
      explained tradeoff for this project's scope, not an oversight.
    - reasoning_effort="none" and max_tokens capped, same as the main
      generation path, for the same confirmed reasons (see
      llm_client.py). The judge is asked explicitly to keep
      justifications brief to fit real answers' claim counts in one
      OTPM-safe call.
    - Judge output is requested as JSON, parsed with a real try/except
      and markdown-fence stripping (LLMs asked for "JSON only" still
      sometimes wrap it in ```json fences) -- a malformed judge response
      for one question is reported and skipped, not treated as a crash.

Metrics:
    - Total claims judged, across all answered questions.
    - Claim Support Rate: (claims judged "supported") / (total claims).
    - Per-question breakdown of unsupported/partially-supported claims,
      so a low score is immediately actionable (which specific claim,
      which specific question) rather than just a number.

Usage:
    python evaluate_faithfulness_llm_judge.py [max_questions]
    (max_questions: optional int, e.g. 3 for a quick validation run
    before committing to the full golden set)
"""

import json
import re
import sys
import time
from pathlib import Path

from gita_engine.core.logging import configure_logging, get_logger
from gita_engine.generation.llm_client import GenerationError, generate
from gita_engine.reasoning.graph import _format_passages_for_prompt, answer_question
from gita_engine.retrieval.retriever import EmbeddingError

logger = get_logger(__name__)

DEFAULT_GOLDEN_SET = Path(__file__).parent / "golden_set.json"

# Two real Groq calls per question now (answer + judge) instead of one --
# pacing needs real margin on both fronts: between the two calls WITHIN
# a question, and between questions. Sized conservatively given tonight's
# real evidence of OTPM "debt" causing 15-30 MINUTE retry-after waits
# under sustained back-to-back load, not just the ~20s isolated-call
# reset seen in isolation. generate()'s own 429 retry logic is the real
# safety net if these pauses prove insufficient -- these are a first
# line of defense, not a guarantee.
PAUSE_BEFORE_JUDGE_SECONDS = 60
PAUSE_BETWEEN_QUESTIONS_SECONDS = 100

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checking judge for a Bhagavad Gita Q&A system.

You will be given:
1. A set of retrieved source passages (Sanskrit verses + commentary).
2. A generated answer that claims to be based ONLY on those passages.

Your task: break the generated answer into individual, atomic factual claims, then verify EACH claim against the source passages.

For each claim, decide:
- "supported": the source passages directly state or clearly entail this claim.
- "unsupported": the source passages do NOT contain evidence for this claim (even if it sounds plausible).
- "partially_supported": the source passages support part of the claim but the claim adds something not present in the source.

Respond with ONLY a JSON array (no markdown fences, no other text), in this exact shape:
[
  {"claim": "<short paraphrase of the claim, one sentence>", "verdict": "supported|unsupported|partially_supported", "justification": "<one brief sentence citing which passage supports/fails to support it>"}
]

Keep each claim and justification short. Aim for 3-8 claims total covering the substantive content of the answer -- do not create claims for filler phrasing."""


def _strip_json_fences(text: str) -> str:
    """LLMs asked for 'JSON only' sometimes still wrap it in ```json
    fences. Strip those if present before parsing."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def judge_answer(question: str, answer: str, passages_block: str) -> list[dict] | None:
    """Run the judge call for one question's answer. Returns a list of
    claim-verdict dicts, or None if the judge call failed or returned
    unparseable JSON (reported by the caller, not raised -- one bad
    judge response shouldn't crash the whole eval run)."""
    judge_user_prompt = (
        f"Source passages:\n\n{passages_block}\n\n"
        f"Question that was asked: {question}\n\n"
        f"Generated answer to verify:\n{answer}\n\n"
        "Break this answer into atomic claims and verify each against the source passages above. "
        "Respond with ONLY the JSON array."
    )

    try:
        result = generate(JUDGE_SYSTEM_PROMPT, judge_user_prompt, temperature=0.0)
    except GenerationError as e:
        logger.error("judge_call_failed", error=str(e))
        return None

    cleaned = _strip_json_fences(result.answer)
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            raise ValueError("Judge response was valid JSON but not a list")
        return parsed
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("judge_response_unparseable", error=str(e), raw_response=cleaned[:500])
        return None


def evaluate(golden_set_path: Path, max_questions: int | None = None) -> None:
    configure_logging()

    with open(golden_set_path, encoding="utf-8") as f:
        golden_set = json.load(f)

    cases = golden_set["cases"]
    if max_questions is not None:
        cases = cases[:max_questions]

    total_claims = 0
    supported_claims = 0
    partially_supported_claims = 0
    unsupported_claims = 0
    failed_questions = 0
    unparseable_judge_responses = 0

    print(f"Evaluating LLM-judge faithfulness on {len(cases)} test case(s)...\n")

    for i, case in enumerate(cases):
        question = case["question"]

        if i > 0:
            time.sleep(PAUSE_BETWEEN_QUESTIONS_SECONDS)

        try:
            result = answer_question(question, log=False)
        except (GenerationError, EmbeddingError) as e:
            failed_questions += 1
            print(f"[ANSWER FAILED] {question}")
            print(f"  Error: {e}\n")
            continue

        passages_block = _format_passages_for_prompt(result["retrieved"])

        print(f"(pausing {PAUSE_BEFORE_JUDGE_SECONDS}s before judge call...)")
        time.sleep(PAUSE_BEFORE_JUDGE_SECONDS)

        claims = judge_answer(question, result["answer"], passages_block)

        if claims is None:
            unparseable_judge_responses += 1
            print(f"[JUDGE FAILED] {question}\n")
            continue

        print(f"[{len(claims)} claims] {question}")
        for c in claims:
            verdict = c.get("verdict", "unknown")
            total_claims += 1
            if verdict == "supported":
                supported_claims += 1
            elif verdict == "partially_supported":
                partially_supported_claims += 1
            elif verdict == "unsupported":
                unsupported_claims += 1
            if verdict != "supported":
                print(f"  [{verdict.upper()}] {c.get('claim', '(no claim text)')}")
                print(f"    -> {c.get('justification', '(no justification)')}")
        print()

    print("=== Summary ===")
    print(f"Questions evaluated: {len(cases)}")
    print(f"Answer generation failures: {failed_questions}")
    print(f"Unparseable judge responses: {unparseable_judge_responses}")
    if total_claims:
        support_rate = supported_claims / total_claims
        print(f"Total claims judged: {total_claims}")
        print(f"Claim Support Rate: {support_rate:.2%} ({supported_claims}/{total_claims})")
        print(f"Partially supported: {partially_supported_claims}")
        print(f"Unsupported: {unsupported_claims}")
    else:
        print("No claims were judged.")


if __name__ == "__main__":
    max_q = int(sys.argv[1]) if len(sys.argv) > 1 else None
    evaluate(DEFAULT_GOLDEN_SET, max_questions=max_q)
