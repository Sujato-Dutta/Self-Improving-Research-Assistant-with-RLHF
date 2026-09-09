from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Research question or query text")
    dual_response: bool = Field(False, description="Whether to generate Candidate A and Candidate B for A/B preference comparison")
    use_ollama: bool = Field(False, description="Whether to use local Ollama model instead of local PyTorch/HF model")
    session_id: Optional[str] = Field(None, description="Optional session identifier")


class CitationItem(BaseModel):
    citation_index: int
    paper_title: str
    arxiv_id: Optional[str] = None
    authors: Optional[str] = None
    excerpt: str
    similarity_score: float
    url: Optional[str] = None


class SingleResponsePayload(BaseModel):
    id: Optional[int] = None
    text: str
    citations: List[int] = []
    latency_ms: float = 0.0


class QueryResponse(BaseModel):
    query_id: int
    model_version: str
    dual_mode: bool = False
    response_id: Optional[int] = None
    response_text: Optional[str] = None
    citations: List[int] = []
    latency_ms: float = 0.0
    response_a: Optional[SingleResponsePayload] = None
    response_b: Optional[SingleResponsePayload] = None
    evidence: List[Dict[str, Any]] = []


class FeedbackRequest(BaseModel):
    response_id: int
    thumbs: int = Field(0, ge=-1, le=1, description="+1 for up, -1 for down, 0 for neutral")
    rating: Optional[int] = Field(None, ge=1, le=5, description="1 to 5 star rating")
    citation_accepted: Optional[bool] = Field(None, description="Whether user accepted cited sources")
    regenerated: bool = Field(False, description="Whether user requested regeneration")
    user_correction: Optional[str] = Field(None, description="Text correction or feedback note")
    task_success: Optional[bool] = Field(None, description="Whether task completed successfully")
    metadata: Optional[Dict[str, Any]] = None


class PreferenceRequest(BaseModel):
    query_id: int
    response_a_id: int
    response_b_id: int
    preferred: str = Field(..., pattern="^(A|B)$", description="Preferred response candidate: 'A' or 'B'")
    metadata: Optional[Dict[str, Any]] = None


class RLHFTriggerRequest(BaseModel):
    round_name: Optional[str] = None
    round_number: Optional[int] = None


class ModelInfo(BaseModel):
    name: str
    status: str
    model_path: str
    metrics: Dict[str, Any] = {}
