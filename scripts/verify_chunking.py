"""Throwaway script -- NOT part of the real codebase, do not commit.

Fires two REAL questions through the real pipeline (real DB, real
embedding call, real Groq call) to verify commentary chunking behaves
as designed: a question expected to retrieve a long commentary passage
should show a real "commentary_truncated_for_prompt" log line; an
ordinary question should show no such line at all.
"""

from gita_engine.reasoning.graph import answer_question

print("=" * 70)
print("QUESTION 1 (expected to possibly hit a long commentary passage)")
print("=" * 70)
result1 = answer_question(
    "What is the nature of the soul and how is it related to the body?",
    log=False,
)
print(f"\nReal answer (first 200 chars): {result1['answer'][:200]}...")
print(f"Real finish_reason: {result1['finish_reason']}")

print()
print("=" * 70)
print("QUESTION 2 (ordinary control question)")
print("=" * 70)
result2 = answer_question("What is dharma?", log=False)
print(f"\nReal answer (first 200 chars): {result2['answer'][:200]}...")
print(f"Real finish_reason: {result2['finish_reason']}")
