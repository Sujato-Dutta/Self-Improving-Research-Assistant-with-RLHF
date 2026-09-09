#!/usr/bin/env python
"""
Runs the production FastAPI server using Uvicorn.
"""
import sys
import uvicorn
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config


def main():
    print(f"Starting Self-Improving Research Assistant on http://{config.host}:{config.port}")
    uvicorn.run(
        "src.api.app:app",
        host=config.host,
        port=config.port,
        reload=False
    )


if __name__ == "__main__":
    main()
