import os
import logging
import numpy as np
from typing import List, Union
from src.config import config

logger = logging.getLogger(__name__)


class SentenceEmbedder:
    """Embeds textual queries and documents using SentenceTransformers with fallback support."""

    def __init__(self, model_name: str = config.model.embedding_model_name, device: str = config.model.device):
        self.model_name = model_name
        self.device = device
        self._model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2

    @property
    def model(self):
        if self._model is None:
            if os.getenv("FAST_TEST", "0") == "1" or os.getenv("USE_FALLBACK_EMBEDDER", "0") == "1":
                self._model = "fallback"
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer: {self.model_name} on {self.device}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self.embedding_dim = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer ({e}). Using deterministic pseudo-embedder fallback.")
                self._model = "fallback"
        return self._model

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, normalize_embeddings: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        if self.model != "fallback":
            try:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=normalize_embeddings,
                    convert_to_numpy=True
                )
                return embeddings.astype(np.float32)
            except Exception as e:
                logger.warning(f"Error during neural encoding: {e}. Falling back to deterministic pseudo-embedder.")

        # Deterministic hashing embedding fallback
        return self._deterministic_encode(texts, normalize=normalize_embeddings)

    def _deterministic_encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = np.zeros(self.embedding_dim, dtype=np.float32)
            tokens = text.lower().split()
            if not tokens:
                vectors.append(vec)
                continue
            for token in tokens:
                h = hash(token)
                idx = abs(h) % self.embedding_dim
                sign = 1.0 if (h >> 3) % 2 == 0 else -1.0
                vec[idx] += sign
            if normalize:
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)
