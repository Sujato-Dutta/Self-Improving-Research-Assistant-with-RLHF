import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.config import config
from src.db.database import init_db
from src.api.routes import router

app = FastAPI(
    title="Self-Improving Research Assistant with Preference-Based RLHF",
    description="Academic research assistant with arXiv evidence retrieval, grounded citations, multi-modal feedback capture, and periodic PPO RLHF self-improvement.",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(router)

# Mount static frontend
STATIC_DIR = config.base_dir / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/", include_in_schema=False)
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Self-Improving Research Assistant API is running. Visit /docs for API schema."}


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "Self-Improving Research Assistant with RLHF",
        "version": "1.0.0"
    }
