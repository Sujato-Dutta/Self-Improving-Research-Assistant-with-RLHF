import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from src.config import config
from src.retrieval.embedder import SentenceEmbedder

logger = logging.getLogger(__name__)


class FaissIndexer:
    """FAISS vector store for semantic retrieval of research paper passages."""

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        index_dir: Path = config.faiss_dir
    ):
        self.embedder = embedder or SentenceEmbedder()
        self.index_dir = index_dir
        self.index_file = index_dir / "faiss.index"
        self.meta_file = index_dir / "metadata.json"
        self.documents: List[Dict[str, Any]] = []
        self.index = None
        self._load_or_init()

    def _load_or_init(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.meta_file.exists():
            try:
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load FAISS metadata: {e}")
                self.documents = []

        if self.index_file.exists():
            try:
                import faiss
                self.index = faiss.read_index(str(self.index_file))
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors.")
                return
            except Exception as e:
                logger.warning(f"Failed to load FAISS index file: {e}")

        # Initialize new index if not loaded
        self._init_empty_index()

    def _init_empty_index(self) -> None:
        try:
            import faiss
            dim = self.embedder.embedding_dim
            # Inner product on normalized vectors = Cosine similarity
            self.index = faiss.IndexFlatIP(dim)
        except Exception:
            logger.warning("FAISS native library not available, using in-memory NumPy fallback index.")
            self.index = None

    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)

        if self.index is not None:
            import faiss
            self.index.add(embeddings)
        self.documents.extend(chunks)

        self.save()
        logger.info(f"Added {len(chunks)} chunks to index. Total: {len(self.documents)}")
        return len(chunks)

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2)

        if self.index is not None:
            try:
                import faiss
                faiss.write_index(self.index, str(self.index_file))
            except Exception as e:
                logger.warning(f"Failed to write FAISS index: {e}")

    def search(
        self,
        query: str,
        top_k: int = config.retrieval.top_k,
        deduplicate_by_paper: bool = True,
        allowed_arxiv_ids: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        if not self.documents:
            return []

        top_k = min(top_k, len(self.documents))
        query_vec = self.embedder.encode([query], normalize_embeddings=True)

        # Normalize allowed_arxiv_ids to clean lowercase strings if provided
        allowed_set = {str(a).strip().lower() for a in allowed_arxiv_ids} if allowed_arxiv_ids else None

        if self.index is not None and self.index.ntotal > 0:
            fetch_k = len(self.documents) if allowed_set else min(len(self.documents), top_k * 6)
            distances, indices = self.index.search(query_vec, fetch_k)
            retrieved = []
            seen_papers = set()
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                doc = dict(self.documents[idx])
                clean_arxiv = str(doc.get("arxiv_id", "")).strip().lower().split("v")[0]
                if allowed_set is not None and clean_arxiv not in allowed_set:
                    continue

                paper_id = (doc.get("arxiv_id") or doc.get("title", "")).strip().lower()
                if deduplicate_by_paper and paper_id:
                    if paper_id in seen_papers:
                        continue
                    seen_papers.add(paper_id)
                doc["similarity_score"] = float(score)
                retrieved.append(doc)
                if len(retrieved) >= top_k:
                    break

            # If not enough distinct papers, fill with remaining highest-scoring chunks
            if len(retrieved) < top_k:
                retrieved_chunks = {d.get("chunk_id") for d in retrieved if "chunk_id" in d}
                for score, idx in zip(distances[0], indices[0]):
                    if idx < 0 or idx >= len(self.documents):
                        continue
                    doc = dict(self.documents[idx])
                    clean_arxiv = str(doc.get("arxiv_id", "")).strip().lower().split("v")[0]
                    if allowed_set is not None and clean_arxiv not in allowed_set:
                        continue

                    if doc.get("chunk_id") not in retrieved_chunks:
                        doc["similarity_score"] = float(score)
                        retrieved.append(doc)
                        retrieved_chunks.add(doc.get("chunk_id"))
                        if len(retrieved) >= top_k:
                            break

            for rank, doc in enumerate(retrieved, start=1):
                doc["citation_index"] = rank
            return retrieved

        # In-memory NumPy cosine similarity fallback
        texts = [d["text"] for d in self.documents]
        doc_embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        sims = np.dot(doc_embeddings, query_vec.T).flatten()
        top_indices = np.argsort(-sims)

        retrieved = []
        seen_papers = set()
        for idx in top_indices:
            doc = dict(self.documents[idx])
            clean_arxiv = str(doc.get("arxiv_id", "")).strip().lower().split("v")[0]
            if allowed_set is not None and clean_arxiv not in allowed_set:
                continue

            paper_id = (doc.get("arxiv_id") or doc.get("title", "")).strip().lower()
            if deduplicate_by_paper and paper_id:
                if paper_id in seen_papers:
                    continue
                seen_papers.add(paper_id)
            doc["similarity_score"] = float(sims[idx])
            retrieved.append(doc)
            if len(retrieved) >= top_k:
                break

        if len(retrieved) < top_k:
            retrieved_chunks = {d.get("chunk_id") for d in retrieved if "chunk_id" in d}
            for idx in top_indices:
                doc = dict(self.documents[idx])
                clean_arxiv = str(doc.get("arxiv_id", "")).strip().lower().split("v")[0]
                if allowed_set is not None and clean_arxiv not in allowed_set:
                    continue

                if doc.get("chunk_id") not in retrieved_chunks:
                    doc["similarity_score"] = float(sims[idx])
                    retrieved.append(doc)
                    retrieved_chunks.add(doc.get("chunk_id"))
                    if len(retrieved) >= top_k:
                        break

        for rank, doc in enumerate(retrieved, start=1):
            doc["citation_index"] = rank
        return retrieved
