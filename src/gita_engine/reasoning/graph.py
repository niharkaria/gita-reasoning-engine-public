"""LangGraph orchestration: question -> embed -> retrieve -> grounded answer.

Why this file exists:
    Ties together retrieval (Phase 5) and generation (this phase) into one
    pipeline, and is the thing the eventual API layer (Phase 8+) will call.
    Using LangGraph (rather than a plain function) per the Phase 0 stack
    decision — gives us a typed, inspectable state machine instead of
    ad-hoc control flow, and a natural place to add branching later (e.g.
    "if retrieval confidence is too low, refuse to answer" as its own node).

Design notes:
    - The system prompt is the actual enforcement mechanism for FR6/FR7
      (grounded answers, citations, refuse if uncovered) — it explicitly
      instructs the model to only use the provided passages and to say so
      plainly if they don't address the question, rather than falling
      back on outside knowledge.
    - Every retrieved passage is labeled with its chapter.verse in the
      prompt, so the model has concrete citation anchors to reference.
    - query_log writes happen here (not in retriever.py or llm_client.py)
      since this is the one place that has the full picture: the
      question, what was retrieved, and what was generated.
"""

from typing import Any, TypedDict

from gita_engine.core.logging import get_logger
from gita_engine.db.models import QueryLog
from gita_engine.db.session import get_session
from gita_engine.generation.llm_client import generate
from gita_engine.retrieval.retriever import RetrievedPassage, embed_query, retrieve

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a reasoning engine that answers questions about the Bhagavad Gita STRICTLY according to the accepted commentary passages provided to you below.

Rules:
1. Base your answer ONLY on the passages provided. Do not use any outside knowledge of the Gita, Hindu philosophy, or Sanskrit texts.
2. Cite the chapter and verse (e.g. "Chapter 9, Verse 15") for every specific claim you make.
3. If the provided passages do not address the question, say so plainly rather than guessing or filling in from general knowledge.
4. Be faithful to the specific tradition/commentary represented in the passages — do not blend in other schools of interpretation.
5. Preserve core Sanskrit/Gujarati philosophical and religious terms in their original form rather than translating them into generic English equivalents — for example, keep "yagna" rather than substituting "sacrifice," keep "dharma" rather than substituting "duty" or "righteousness," keep "atman" rather than substituting "soul," keep "karma" as "karma." Do this consistently throughout your answer, not just on first use. Only translate or explain a term if the passage itself explains it — do not supply your own gloss.
6. Give a full, explanatory answer, not a one-line fact. The retrieved passages often contain rich detail and reasoning (in the Vivechan/commentary text) beyond the bare translation — draw on that detail to explain the "why" and "how," not just the "what," whenever the passages support it. Do not pad with repetition or outside knowledge to make the answer longer — only elaborate using what the passages actually say.
"""


class ReasoningState(TypedDict):
    query: str
    query_embedding: list[float]
    retrieved: list[RetrievedPassage]
    answer: str


def _embed_node(state: ReasoningState) -> ReasoningState:
    embedding = embed_query(state["query"])
    return {**state, "query_embedding": embedding}


def _retrieve_node(state: ReasoningState) -> ReasoningState:
    with get_session() as session:
        passages = retrieve(session, state["query_embedding"])
    return {**state, "retrieved": passages}


def _format_passages_for_prompt(passages: list[RetrievedPassage]) -> str:
    if not passages:
        return "(No relevant passages were found in the accepted corpus.)"

    blocks = []
    for p in passages:
        kind = "Translation" if p.passage_type == "translation" else "Commentary"
        blocks.append(
            f"[Chapter {p.chapter}, Verse {p.verse_number} — {kind}, source: {p.source_citation}]\n"
            f"Sanskrit: {p.sanskrit_text}\n"
            f"{kind}: {p.text}"
        )
    return "\n\n".join(blocks)


def _generate_node(state: ReasoningState) -> ReasoningState:
    passages_block = _format_passages_for_prompt(state["retrieved"])
    user_prompt = (
        f"Relevant passages:\n\n{passages_block}\n\n"
        f"Question: {state['query']}\n\n"
        "Answer the question using only the passages above, with chapter.verse citations."
    )
    answer = generate(SYSTEM_PROMPT, user_prompt)
    return {**state, "answer": answer}


def build_graph() -> Any:  # noqa: ANN401
    """Build and compile the LangGraph reasoning pipeline.

    Return type intentionally left as Any — LangGraph's compiled-graph
    type isn't easily spelled out here. Callers use answer_question()
    below rather than the graph object directly.
    """
    from langgraph.graph import END, StateGraph

    graph = StateGraph(ReasoningState)
    graph.add_node("embed", _embed_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("generate", _generate_node)

    graph.set_entry_point("embed")
    graph.add_edge("embed", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def answer_question(query: str) -> ReasoningState:
    """Run the full pipeline for a user question and log the result.

    This is the main entry point other code (API routes, eval harness)
    should call — it also writes to query_log, which is what Phase 7
    evaluation and future analytics will read from.
    """
    app = build_graph()
    result: ReasoningState = app.invoke({"query": query})

    with get_session() as session:
        session.add(
            QueryLog(
                user_question=query,
                retrieved_ids={
                    "translation": [
                        p.verse_id for p in result["retrieved"] if p.passage_type == "translation"
                    ],
                    "commentary": [
                        p.verse_id for p in result["retrieved"] if p.passage_type == "commentary"
                    ],
                },
                final_answer=result.get("answer"),
            )
        )

    logger.info("question_answered", query=query, num_retrieved=len(result["retrieved"]))
    return result
