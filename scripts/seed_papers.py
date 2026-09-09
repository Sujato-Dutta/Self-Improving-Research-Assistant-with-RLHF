#!/usr/bin/env python
"""
Seeds the FAISS vector index with foundational research papers from arXiv and curated seed corpus.
"""
import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.arxiv_fetcher import ArxivFetcher
from src.retrieval.text_splitter import TextSplitter
from src.retrieval.embedder import SentenceEmbedder
from src.retrieval.indexer import FaissIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SeedPapers")


def main():
    logger.info("Initializing arXiv fetcher and FAISS indexer...")
    fetcher = ArxivFetcher()
    splitter = TextSplitter(chunk_size=350, chunk_overlap=50)
    embedder = SentenceEmbedder()
    indexer = FaissIndexer(embedder=embedder)

    # 1. Fetch online papers across core categories
    search_queries = [
        "transformer attention language model",
        "reinforcement learning human feedback rlhf",
        "direct preference optimization language model",
        "retrieval augmented generation dense retrieval",
        "low rank adaptation lora fine tuning",
        "mixture of experts distributed training gpu"
    ]

    for q in search_queries:
        logger.info(f"Querying arXiv for: '{q}'...")
        fetcher.fetch_papers_online(q, max_results=5)

    # 2. Extract all papers (online + curated seeds)
    all_papers = fetcher.get_all_papers()
    logger.info(f"Total papers to index: {len(all_papers)}")

    # 3. Chunk papers and index in FAISS
    total_chunks = 0
    all_chunks = []
    for p in all_papers:
        chunks = splitter.chunk_paper(p)
        all_chunks.extend(chunks)

    indexed_count = indexer.add_documents(all_chunks)
    logger.info(f"Successfully indexed {indexed_count} chunks into FAISS vector database at: {indexer.index_dir}")


if __name__ == "__main__":
    main()
