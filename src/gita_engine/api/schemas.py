"""Request/response schemas for the API.

Why this file exists:
    Keeps the wire format (what JSON goes over HTTP) separate from our
    internal dataclasses (ParsedVerse, RetrievedPassage, ReasoningState).
    The API's shape shouldn't have to change every time we refactor
    internal pipeline code, and vice versa.
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class CitedPassage(BaseModel):
    chapter: int
    verse_number: int
    passage_type: str  # 'translation' | 'commentary'
    sanskrit_text: str
    text: str
    similarity: float


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitedPassage]


class HealthResponse(BaseModel):
    status: str
    database_connected: bool
