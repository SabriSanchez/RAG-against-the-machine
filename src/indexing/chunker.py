# Handles file ingestion, recursive text splitting,
# and SAST security inspection.

import hashlib
import logging
from pathlib import Path
from typing import List, Literal, Tuple, Union

from src.indexing.security import SASTScanner
from src.general_utils.models import DocumentChunk
from src.general_utils.text_utils import read_corpus_file

logger = logging.getLogger(__name__)


class RecursiveTextSplitter:
    """Recursively partitions text based on a hierarchy of separators."""

    def __init__(self, max_chunk_size: int = 2000, overlap: int = 200):
        """Initializes the RecursiveTextSplitter.

        Args:
            max_chunk_size (int): Maximum character length per chunk.
                Defaults to 2000.
            overlap (int): Number of overlapping characters between adjacent
                chunks. Defaults to 200.
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def _partition_text(
        self, text: str, sep: str, base_offset: int
    ) -> List[Tuple[str, int]]:
        """Splits text by separator and computes exact character offsets.

        Args:
            text (str): Raw text string to partition.
            sep (str): Separator string to split upon.
            base_offset (int): Starting character index offset in source file.

        Returns:
            List[Tuple[str, int]]: List of tuples containing text piece and
                character offset.
        """
        raw_parts = text.split(sep)
        partitioned: List[Tuple[str, int]] = []
        pos = base_offset

        for i, part in enumerate(raw_parts):
            piece = part if i == 0 else sep + part
            partitioned.append((piece, pos))
            pos += len(piece)

        return partitioned

    def _combine_parts(
        self,
        parts: List[Tuple[str, int]],
        next_seps: List[str],
        base_offset: int,
    ) -> List[Tuple[str, int, int]]:
        """Combines text parts into valid chunk sizes using dynamic overlap.

        Args:
            parts (List[Tuple[str, int]]): Text parts with start offsets.
            next_seps (List[str]): Fallback separators for oversized parts.
            base_offset (int): Base character index offset.

        """
        chunks: List[Tuple[str, int, int]] = []
        current_parts: List[Tuple[str, int]] = []
        current_len = 0

        for piece, piece_offset in parts:
            piece_len = len(piece)

            if piece_len > self.max_chunk_size:
                if current_parts:
                    chunk_text = "".join(p[0] for p in current_parts)
                    if chunk_text.strip():
                        chunk_start = current_parts[0][1]
                        chunks.append(
                            (
                                chunk_text,
                                chunk_start,
                                chunk_start + len(chunk_text),
                            )
                        )
                    current_parts = []
                    current_len = 0

                sub_splits = self.split(
                    piece, next_seps, base_offset=piece_offset
                )
                chunks.extend(sub_splits)
                continue

            if (
                current_len + piece_len > self.max_chunk_size
                and current_len > 0
            ):
                chunk_text = "".join(p[0] for p in current_parts)
                if chunk_text.strip():
                    chunk_start = current_parts[0][1]
                    chunks.append(
                        (
                            chunk_text,
                            chunk_start,
                            chunk_start + len(chunk_text),
                        )
                    )

                max_allowed_overlap = min(
                    self.overlap, self.max_chunk_size - piece_len
                )

                overlap_parts: List[Tuple[str, int]] = []
                overlap_len = 0

                for p in reversed(current_parts):
                    if overlap_len + len(p[0]) <= max_allowed_overlap:
                        overlap_parts.insert(0, p)
                        overlap_len += len(p[0])
                    else:
                        break

                current_parts = overlap_parts
                current_len = overlap_len

            current_parts.append((piece, piece_offset))
            current_len += piece_len

        if current_parts:
            chunk_text = "".join(p[0] for p in current_parts)
            if chunk_text.strip():
                chunk_start = current_parts[0][1]
                chunks.append(
                    (chunk_text, chunk_start, chunk_start + len(chunk_text))
                )

        return chunks

    def split(
        self, text: str, separators: List[str], base_offset: int = 0
    ) -> List[Tuple[str, int, int]]:
        """Recursively splits text using the provided separator hierarchy.

        Args:
            text (str): Input text string to split.
            separators (List[str]): Ordered list of string separators.
            base_offset (int): Base character index offset. Defaults to 0.

        Returns:
            List[Tuple[str, int, int]]: List of tuples containing chunk text,
                start character offset, and end character offset.
        """
        if len(text) <= self.max_chunk_size:
            if not text.strip():
                return []
            return [(text, base_offset, base_offset + len(text))]

        sep = separators[0] if separators else ""
        next_seps = separators[1:] if len(separators) > 1 else []

        if not sep:
            chunks = []
            for i in range(0, len(text), self.max_chunk_size):
                piece = text[i: i + self.max_chunk_size]
                if piece.strip():
                    chunks.append(
                        (
                            piece,
                            base_offset + i,
                            base_offset + i + len(piece),
                        )
                    )
            return chunks

        parts = self._partition_text(text, sep, base_offset)
        return self._combine_parts(parts, next_seps, base_offset)


class MarkdownChunker:
    """Dedicated splitter for Markdown documents using heading hierarchies."""

    SEPARATORS = [
        "\n# ",
        "\n## ",
        "\n### ",
        "\n#### ",
        "\n##### ",
        "\n###### ",
        "\n```",
        "\n\n\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    def __init__(self, text_splitter: RecursiveTextSplitter):
        """Initializes the MarkdownChunker instance.

        Args:
            text_splitter (RecursiveTextSplitter): Base recursive text splitter
                instance.
        """
        self.text_splitter = text_splitter

    def split(self, text: str) -> List[Tuple[str, int, int]]:
        """Splits Markdown text using predefined hierarchy separators.

        Args:
            text (str): Raw Markdown source text.

        Returns:
            List[Tuple[str, int, int]]: List of chunk tuples containing chunk
                text, start character offset, and end character offset.
        """
        return self.text_splitter.split(text, self.SEPARATORS)


class PythonChunker:
    """Dedicated splitter for Python source code using structural code tags."""

    SEPARATORS = [
        "\nclass ",
        "\ndef ",
        "\nasync def ",
        "\n\n\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    def __init__(self, text_splitter: RecursiveTextSplitter):
        """Initializes the PythonChunker instance.

        Args:
            text_splitter (RecursiveTextSplitter): Base recursive text splitter
                instance.
        """
        self.text_splitter = text_splitter

    def split(self, text: str) -> List[Tuple[str, int, int]]:
        """Splits Python source code using structural code separators.

        Args:
            text (str): Raw Python code string.

        Returns:
            List[Tuple[str, int, int]]: List of chunk tuples containing chunk
                text, start character offset, and end character offset.
        """
        return self.text_splitter.split(text, self.SEPARATORS)


class CodeAndDocChunker:
    """Facade orchestrator for file ingestion and strategy dispatching."""

    def __init__(self, max_chunk_size: int = 2000, overlap: int = 150):
        """Initializes the CodeAndDocChunker facade.

        Args:
            max_chunk_size (int): Maximum allowable character size per chunk.
                Defaults to 2000.
            overlap (int): Overlap character count. Defaults to 150.

        Raises:
            ValueError: If max_chunk_size exceeds 2000 characters.
        """
        if max_chunk_size > 2000:
            raise ValueError(
                "The maximum chunk size cannot exceed 2,000 characters."
            )

        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.text_splitter = RecursiveTextSplitter(
            max_chunk_size=max_chunk_size, overlap=overlap
        )
        self.python_splitter = PythonChunker(
            text_splitter=self.text_splitter
        )
        self.markdown_splitter = MarkdownChunker(
            text_splitter=self.text_splitter
        )

    def chunk_text(
        self,
        raw_text: str,
        file_path: str,
        strategy: Literal["python", "markdown"],
    ) -> List[DocumentChunk]:
        """Splits raw text into DocumentChunk obj without modifying content.

        Args:
            raw_text (str): Raw input text content.
            file_path (str): Relative or absolute path to the original file.
            strategy (Literal["python", "markdown"]): Chunking strategy to
                apply.

        Returns:
            List[DocumentChunk]: List of structured DocumentChunk objects.

        Raises:
            ValueError: If strategy is unknown or unsupported.
        """
        if strategy == "python":
            raw_chunks = self.python_splitter.split(raw_text)
        elif strategy == "markdown":
            raw_chunks = self.markdown_splitter.split(raw_text)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if not raw_chunks:
            raw_chunks = self.text_splitter.split(
                raw_text,
                MarkdownChunker.SEPARATORS
                if strategy == "markdown"
                else PythonChunker.SEPARATORS,
            )

        chunks: List[DocumentChunk] = []

        for i, (chunk_text, start_idx, end_idx) in enumerate(raw_chunks):
            chunk_id = (
                hashlib.md5(f"{file_path}_{i}".encode("utf-8")).hexdigest()
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    text=chunk_text,
                    first_character_index=start_idx,
                    last_character_index=end_idx,
                )
            )

        return chunks

    def chunk_file(self, file_path: Union[str, Path]) -> List[DocumentChunk]:
        """Reads a file from disk, runs SAST inspection, and chunks content.

        Args:
            file_path (Union[str, Path]): Path to the target file.

        Returns:
            List[DocumentChunk]: List of generated DocumentChunk objects.

        Raises:
            ValueError: If the file extension is not supported.
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        strategy: Literal["python", "markdown"]
        if ext == ".py":
            strategy = "python"
        elif ext in [".md", ".markdown", ".rst", ".txt"]:
            strategy = "markdown"
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        content = read_corpus_file(path)

        warnings = SASTScanner.scan(
            content, is_python=(strategy == "python")
        )
        if warnings:
            for warning in warnings:
                logger.warning(f"[SAST Alert] [{path.name}]: {warning}")

        return self.chunk_text(
            raw_text=content, file_path=str(path), strategy=strategy
        )
