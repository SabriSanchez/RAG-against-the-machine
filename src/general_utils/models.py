# Core Pydantic schemas for document chunking, source tracking, and evaluation.

import uuid
from typing import List
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represents minimal location metadata for a source text segment.

    Attributes:
        file_path: Relative or absolute path to the source file.
        first_character_index: Starting character offset in the source file.
        last_character_index: Ending character offset in the source file.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class DocumentChunk(MinimalSource):
    """Represents a processed text chunk extracted from a document.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        text: Sanitized text content of the chunk.
    """

    chunk_id: str
    text: str


class UnansweredQuestion(BaseModel):
    """Represents a standalone user query awaiting retrieval or generation.

    Attributes:
        question_id: Unique UUID identifier generated automatically by
            default.
        question: The raw query text asked by the user.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represents a question paired with ground-truth sources and an answer.

    Attributes:
        sources: List of minimal source references used to ground the answer.
        answer: Ground-truth or generated response text.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represents a collection of RAG evaluation benchmarks.

    Attributes:
        rag_questions: List containing answered or unanswered questions for
            evaluation.
    """

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Encapsulates retrieval results for a single query.

    Attributes:
        question_id: Unique identifier matching the input question.
        question: Text of the queried question.
        retrieved_sources: List of sources retrieved by the search algorithm.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Encapsulates retrieval results alongside a generated answer.

    Attributes:
        answer: Generated response derived from the retrieved sources.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Encapsulates evaluation submission payloads for top-K search results.

    Attributes:
        search_results: List of search results corresponding to test queries.
        k: Top-K depth parameter used during retrieval.
    """

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Encapsulates evaluation submission payloads for
    search results and answers.

    Attributes:
        search_results: List of search results with generated answers.
        k: Top-K depth parameter used during retrieval.
    """

    search_results: List[MinimalAnswer]
    k: int
