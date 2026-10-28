import torch
import pickle
from pathlib import Path
from typing import List, Union
from src.general_utils.models import DocumentChunk
from src.general_utils.text_utils import indexing_text
from sentence_transformers import SentenceTransformer


class VectorIndexer:
    """Builds and persists semantic vector embeddings for text chunks."""
    def __init__(
            self,
            save_dir: Union[str, Path] = "data/processed/vector_index",
            model_name: Union[
                str, Path
            ] = "sentence-transformers/all-MiniLM-L6-v2",
            max_seq_length: int = 128) -> None:
        """Initialize the vector indexer.

        Args:
            save_dir: Directory where the index and metadata will be saved.
            model_name: HuggingFace model name for sentence embeddings.
            max_seq_length: Maximum sequence length for tokenization.
        """
        self.save_dir = Path(save_dir)
        self.model = SentenceTransformer(str(model_name))
        self.model.max_seq_length = max_seq_length
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Encode chunks into embeddings and persist them to disk.
        Args:
            chunks: List of chunk objects containing text and metadata.
        """
        texts = [indexing_text(c.file_path, c.text) for c in chunks]

        with torch.inference_mode():
            embeddings = self.model.encode(
                texts,
                batch_size=64,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=True
            )

        torch.save(embeddings, self.save_dir / "embeddings.pt")

        with open(self.save_dir / "chunks_metadata.pkl", "wb") as f:
            pickle.dump(chunks, f)
