"""Evaluate retrieval accuracy against a hand-curated golden set.

Why this file exists:
    Before trusting the full generation pipeline, we need to know
    retrieval itself is finding the right passages — if retrieval is
    weak, no amount of prompt engineering fixes the final answer. This
    harness measures that directly and locally: no LLM judge required,
    so it's fully testable without any hosted API access, unlike
    faithfulness-style metrics (see evaluate_faithfulness.py).

Metrics:
    - Hit Rate @ k: fraction of test questions where AT LEAST ONE expected
      verse appears among the top-k retrieved passages.
    - Mean Reciprocal Rank (MRR): rewards finding the expected verse
      EARLY in the ranked results, not just somewhere in the top-k.

Usage:
    python evaluate_retrieval.py [path/to/golden_set.json]
"""

import json
import sys
from pathlib import Path

from gita_engine.core.logging import get_logger
from gita_engine.db.session import get_session
from gita_engine.retrieval.retriever import embed_query, retrieve

logger = get_logger(__name__)

DEFAULT_GOLDEN_SET = Path(__file__).parent / "golden_set.json"


def evaluate(golden_set_path: Path, top_k: int = 5) -> None:
    with open(golden_set_path, encoding="utf-8") as f:
        golden_set = json.load(f)

    cases = golden_set["cases"]
    hits = 0
    reciprocal_ranks: list[float] = []

    print(f"Evaluating retrieval on {len(cases)} test cases (top_k={top_k})...\n")

    for case in cases:
        question = case["question"]
        expected = {(v["chapter"], v["verse_number"]) for v in case["expected_verses"]}

        query_embedding = embed_query(question)
        with get_session() as session:
            results = retrieve(session, query_embedding, top_k=top_k)

        retrieved_keys = [(p.chapter, p.verse_number) for p in results]

        rank = next(
            (i + 1 for i, key in enumerate(retrieved_keys) if key in expected),
            None,
        )

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
            status = f"HIT  (rank {rank})"
        else:
            reciprocal_ranks.append(0.0)
            status = "MISS"

        print(f"[{status}] {question}")
        print(f"  Expected: {sorted(expected)}")
        print(f"  Retrieved: {retrieved_keys}")
        print()

    hit_rate = hits / len(cases) if cases else 0.0
    mrr = sum(reciprocal_ranks) / len(cases) if cases else 0.0

    print("=== Summary ===")
    print(f"Hit Rate @ {top_k}: {hit_rate:.2%} ({hits}/{len(cases)})")
    print(f"Mean Reciprocal Rank: {mrr:.3f}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GOLDEN_SET
    evaluate(path)
