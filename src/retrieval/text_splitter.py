import re
from typing import List, Dict, Any


class TextSplitter:
    """Splits academic texts into overlapping semantic passages."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 60):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        # Clean text
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split(" ")
        if len(words) <= self.chunk_size:
            return [text] if text else []

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            if end >= len(words):
                break
            start += max(1, self.chunk_size - self.chunk_overlap)
        return chunks

    def chunk_paper(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Splits a paper's title and summary/content into chunks with metadata."""
        title = paper.get("title", "").strip()
        summary = paper.get("summary", "").strip()
        full_text = f"{title}. {summary}" if title and summary else (title or summary)
        raw_chunks = self.split_text(full_text)

        chunks_with_meta = []
        for idx, chunk in enumerate(raw_chunks):
            chunks_with_meta.append({
                "chunk_id": f"{paper.get('arxiv_id', 'unknown')}_{idx}",
                "arxiv_id": paper.get("arxiv_id", ""),
                "title": paper.get("title", "Untitled"),
                "authors": paper.get("authors", []),
                "url": paper.get("url", ""),
                "published": paper.get("published", ""),
                "text": chunk,
                "chunk_index": idx
            })
        return chunks_with_meta
