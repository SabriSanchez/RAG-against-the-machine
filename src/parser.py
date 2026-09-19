"""Centralized input parsing and output serialization layer.

Validates RAG datasets and formats outputs for evaluation schemas.
"""
from typing import Any, Union
from models import (
    RagDataset,
    AnsweredQuestion,
    UnansweredQuestion
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

# TODO: Implement output
