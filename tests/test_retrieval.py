import pytest
from src.retrieval.arxiv_fetcher import ArxivFetcher
from src.retrieval.text_splitter import TextSplitter
from src.retrieval.embedder import SentenceEmbedder
from src.retrieval.indexer import FaissIndexer


def test_arxiv_fetcher_seeds():
    fetcher = ArxivFetcher()
    papers = fetcher.get_all_papers()
    assert len(papers) >= 5
    # Verify paper fields
    first = papers[0]
    assert "arxiv_id" in first
    assert "title" in first
    assert "summary" in first


def test_text_splitter():
    splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
    sample_paper = {
        "arxiv_id": "9999.0001",
        "title": "Test Title on Reinforcement Learning",
        "summary": "This is a sentence. " * 30,
        "authors": ["Author A", "Author B"]
    }
    chunks = splitter.chunk_paper(sample_paper)
    assert len(chunks) >= 2
    assert chunks[0]["arxiv_id"] == "9999.0001"
    assert "chunk_id" in chunks[0]


def test_sentence_embedder():
    embedder = SentenceEmbedder()
    vecs = embedder.encode(["What is self-attention?", "Transformers in NLP"], normalize_embeddings=True)
    assert vecs.shape[0] == 2
    assert vecs.shape[1] == embedder.embedding_dim


def test_faiss_indexer(tmp_path):
    embedder = SentenceEmbedder()
    indexer = FaissIndexer(embedder=embedder, index_dir=tmp_path / "faiss_test")

    chunks = [
        {"chunk_id": "c1", "title": "Paper 1", "text": "Self-attention replaces recurrence in deep transformers.", "arxiv_id": "0001"},
        {"chunk_id": "c2", "title": "Paper 2", "text": "Reinforcement learning from human feedback aligns language models.", "arxiv_id": "0002"}
    ]
    indexer.add_documents(chunks)

    results = indexer.search("Tell me about attention mechanism", top_k=1)
    assert len(results) == 1
    assert "Self-attention" in results[0]["text"]
    assert results[0]["citation_index"] == 1
