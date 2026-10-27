"""Centralized input parsing and output serialization layer.

Validates RAG datasets and formats outputs for evaluation schemas.
"""

from typing import Any, Union, List, Tuple
from src.general_utils.models import (
    RagDataset,
    AnsweredQuestion,
    UnansweredQuestion,
    DocumentChunk,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)


def parse_answered_questions(
        data: Union[dict[str, Any], str, bytes]
) -> list[AnsweredQuestion]:
    """
    Parses a data structure (dict, str, or bytes) and returns only
    the questions that have been parsed as instances of AnsweredQuestion.
    """

    if isinstance(data, (str, bytes)):
        dataset = RagDataset.model_validate_json(data)
    else:
        dataset = RagDataset.model_validate(data)

    return [
        q for q in dataset.rag_questions if isinstance(q, AnsweredQuestion)
    ]


def parse_unanswered_questions(
        data: Union[dict[str, Any], str, bytes]
) -> list[UnansweredQuestion]:
    """
    Parses a data structure (dict, str, or bytes) and returns only
    the questions that have not yet been answered.
    """
    if isinstance(data, (str, bytes)):
        dataset = RagDataset.model_validate_json(data)
    else:
        dataset = RagDataset.model_validate(data)

    return [q for q in dataset.rag_questions if type(q) is UnansweredQuestion]


def format_search_results(
    query_results: List[
        Tuple[UnansweredQuestion, List[Tuple[DocumentChunk, float]]]
    ],
    top_k: int,
) -> StudentSearchResults:
    """
    Transforms raw retrieval hits into the required
    StudentSearchResults format.
    """
    formatted_results: List[MinimalSearchResults] = []

    for question_obj, hits in query_results:
        retrieved_sources = [
            MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index,
            )
            for chunk, _ in hits
        ]

        formatted_results.append(
            MinimalSearchResults(
                question_id=question_obj.question_id,
                question=question_obj.question,
                retrieved_sources=retrieved_sources,
            )
        )

    return StudentSearchResults(search_results=formatted_results, k=top_k)
