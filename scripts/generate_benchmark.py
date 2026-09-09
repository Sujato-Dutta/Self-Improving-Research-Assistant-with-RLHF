#!/usr/bin/env python
"""
Generates the 350-question held-out research benchmark across 6 scientific domains.
"""
import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.benchmark import BenchmarkManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GenerateBenchmark")


def main():
    logger.info("Generating held-out benchmark suite (350 technical questions)...")
    manager = BenchmarkManager()
    questions = manager.get_questions(limit=350)
    logger.info(f"Successfully generated/verified {len(questions)} benchmark questions at: {manager.file_path}")


if __name__ == "__main__":
    main()
