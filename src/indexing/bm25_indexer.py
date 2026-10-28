# Builds and persists a BM25 index from processed document chunks.

import os
import bm25s
from typing import List
from src.general_utils.models import DocumentChunk
from src.general_utils.text_utils import clean_code_text, indexing_text


class BM25Indexer:
    """Builds and manages BM25 search indices for document chunks."""

    def __init__(self, save_dir: str = "data/processed/bm25_index"):
        """Initializes the BM25Indexer with a target directory.

        Args:
            save_dir (str): Path to the directory where the BM25 index and
                corpus metadata will be saved. Defaults to
                "data/processed/bm25_index".
        """
        self.save_dir = save_dir

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Constructs and persists a BM25 index from a list of DocumentChunks.

        Args:
            chunks (List[DocumentChunk]): List of document chunks containing
                text content and location metadata to be indexed.
        """
        corpus = []
        texts = []

        for chunk in chunks:
            corpus.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "file_path": chunk.file_path,
                    "text": chunk.text,
                    "first_character_index": chunk.first_character_index,
                    "last_character_index": chunk.last_character_index,
                }
            )
            texts.append(indexing_text(chunk.file_path, chunk.text))

        cleaned_texts = [clean_code_text(t) for t in texts]
        tokens = bm25s.tokenize(cleaned_texts)
        retriever = bm25s.BM25()
        retriever.index(tokens)

        os.makedirs(self.save_dir, exist_ok=True)
        retriever.save(self.save_dir, corpus=corpus)
