import re
from pathlib import Path
from typing import Union


def read_corpus_file(path: Union[str, Path]) -> str:
    """Read a corpus file exactly as the indexer does.
    """
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def clean_code_text(text: str) -> str:
    """
    Removes special characters and distinguishes between camelCase
    and snake_case to improve tokenization in lexical searches.
    """
    text = text.replace("_", " ").replace(".", " ")
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text


def indexing_text(file_path: str, text: str) -> str:
    """
    Prepends the words of the file path to the chunk text. Only the text
    sent to the indexes changes: the stored chunk and its offsets do not.
    """
    path = re.sub(r"^data/raw/[^/]+/", "", file_path)
    return re.sub(r"[/_.\-]", " ", path) + "\n" + text
