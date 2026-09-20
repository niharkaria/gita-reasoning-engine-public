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
    python evaluate_retrieval.py --rerank
    python evaluate_retrieval.py --prefix   (embed queries with a
        retrieval instruction prefix before embedding -- see
        retriever.py's embed_query() prefix parameter)
    python evaluate_retrieval.py --translate   (translate the English
        question to Gujarati via Groq before embedding -- tests whether
        same-language query/corpus matching helps, since the corpus is
        entirely Gujarati commentary/translation text)
    python evaluate_retrieval.py --rerank --prefix   (combine any flags)
"""

import json
import sys
from pathlib import Path

from gita_engine.core.logging import get_logger
from gita_engine.db.session import get_session
from gita_engine.generation.llm_client import translate_query
from gita_engine.retrieval.reranker import rerank
from gita_engine.retrieval.retriever import embed_query, retrieve

logger = get_logger(__name__)

DEFAULT_GOLDEN_SET = Path(__file__).parent / "golden_set.json"

# Pool size to retrieve before reranking narrows it down. Matches
# RERANK_POOL_SIZE in reasoning/graph.py — keep these in sync if either
# changes, since we want the eval to test the exact same pipeline shape
# that production actually runs.
RERANK_POOL_SIZE = 20

# Instruction prefix for the --prefix experiment (2026-09-13). BGE-family
# models are documented to sometimes benefit from an instruction prefix
# on the query side for asymmetric retrieval (short query vs long
# passage). Testing this specific wording first per the original idea
# noted in project handoff docs -- confirmed 2026-09-20 to HURT this
# corpus (Hit Rate@5 61.11% -> 44.44%), kept only as a documented,
# reproducible negative result, not a recommended flag.
QUERY_PREFIX = "Represent this question for retrieving relevant Bhagavad Gita commentary: "


def evaluate(
    golden_set_path: Path,
    top_k: int = 5,
    use_rerank: bool = False,
    use_prefix: bool = False,
    use_translate: bool = False,
) -> None:
    with open(golden_set_path, encoding="utf-8") as f:
        golden_set = json.load(f)

    cases = golden_set["cases"]
    hits = 0
    reciprocal_ranks: list[float] = []

    mode_parts = []
    mode_parts.append("WITH reranking" if use_rerank else "embedding-only")
    mode_parts.append("WITH query prefix" if use_prefix else "no query prefix")
    mode_parts.append("WITH translation" if use_translate else "no translation")
    mode = ", ".join(mode_parts)
    print(f"Evaluating retrieval on {len(cases)} test cases (top_k={top_k}, {mode})...\n")

    for case in cases:
        question = case["question"]
        expected = {(v["chapter"], v["verse_number"]) for v in case["expected_verses"]}

        embed_query_text = question
        if use_translate:
            embed_query_text = translate_query(question)

        embed_prefix = QUERY_PREFIX if use_prefix else None
        query_embedding = embed_query(embed_query_text, prefix=embed_prefix)
        with get_session() as session:
            retrieve_k = RERANK_POOL_SIZE if use_rerank else top_k
            results = retrieve(session, query_embedding, top_k=retrieve_k)

        if use_rerank:
            results = rerank(question, results, top_k=top_k)

        retrieved_keys = list(dict.fromkeys((p.chapter, p.verse_number) for p in results))

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
        if use_translate:
            print(f"  Translated: {embed_query_text}")
        print(f"  Expected: {sorted(expected)}")
        print(f"  Retrieved: {retrieved_keys}")
        print()

    hit_rate = hits / len(cases) if cases else 0.0
    mrr = sum(reciprocal_ranks) / len(cases) if cases else 0.0

    print("=== Summary ===")
    print(f"Hit Rate @ {top_k}: {hit_rate:.2%} ({hits}/{len(cases)})")
    print(f"Mean Reciprocal Rank: {mrr:.3f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    use_rerank = "--rerank" in args
    use_prefix = "--prefix" in args
    use_translate = "--translate" in args
    positional = [a for a in args if a not in ("--rerank", "--prefix", "--translate")]
    path = Path(positional[0]) if positional else DEFAULT_GOLDEN_SET
    evaluate(path, use_rerank=use_rerank, use_prefix=use_prefix, use_translate=use_translate)
