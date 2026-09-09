import json
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from src.db.database import get_db
from src.db.models import QueryRecord, ResponseRecord, BenchmarkMetricRecord, RLHFRoundRecord
from src.api.schemas import (
    QueryRequest, QueryResponse, FeedbackRequest, PreferenceRequest,
    RLHFTriggerRequest, ModelInfo
)
from src.pipeline import SelfImprovingAssistantPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# Singleton pipeline instance initialized on demand
_pipeline: SelfImprovingAssistantPipeline = None


def get_pipeline(db: Session = Depends(get_db)) -> SelfImprovingAssistantPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SelfImprovingAssistantPipeline(db_session=db)
    else:
        _pipeline.set_db(db)
    return _pipeline


@router.post("/query", response_model=QueryResponse)
def query_assistant(req: QueryRequest, pipeline: SelfImprovingAssistantPipeline = Depends(get_pipeline)):
    """Receives research query, retrieves evidence from FAISS, and generates grounded answer with citations."""
    try:
        res = pipeline.query(
            query_text=req.query,
            dual_response=req.dual_response,
            use_ollama=req.use_ollama,
            session_id=req.session_id
        )
        return res
    except Exception as e:
        logger.error(f"Error handling query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
def submit_feedback(req: FeedbackRequest, pipeline: SelfImprovingAssistantPipeline = Depends(get_pipeline)):
    """Records multi-modal feedback: thumbs (+1/-1), rating (1-5), citation accepted/rejected, correction, etc."""
    try:
        fb = pipeline.feedback_collector.record_feedback(
            response_id=req.response_id,
            thumbs=req.thumbs,
            rating=req.rating,
            citation_accepted=req.citation_accepted,
            regenerated=req.regenerated,
            user_correction=req.user_correction,
            task_success=req.task_success,
            metadata=req.metadata
        )
        return {"status": "success", "feedback_id": fb.id}
    except Exception as e:
        logger.error(f"Error recording feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback/preference")
def submit_ab_preference(req: PreferenceRequest, pipeline: SelfImprovingAssistantPipeline = Depends(get_pipeline)):
    """Records explicit human A/B candidate preference pair."""
    try:
        pair = pipeline.preference_converter.record_ab_preference(
            query_id=req.query_id,
            response_a_id=req.response_a_id,
            response_b_id=req.response_b_id,
            preferred=req.preferred,
            metadata=req.metadata
        )
        return {"status": "success", "preference_pair_id": pair.id}
    except Exception as e:
        logger.error(f"Error recording preference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
def get_model_registry(pipeline: SelfImprovingAssistantPipeline = Depends(get_pipeline)):
    """Returns active model checkpoint and progression registry across rounds."""
    active_version = pipeline.checkpoint_manager.get_active_version()
    rounds = pipeline.checkpoint_manager.list_all_rounds()
    return {
        "active_version": active_version,
        "active_path": pipeline.checkpoint_manager.get_active_model_path(),
        "rounds": rounds
    }


@router.post("/rlhf/trigger")
def trigger_rlhf_round(req: RLHFTriggerRequest, pipeline: SelfImprovingAssistantPipeline = Depends(get_pipeline)):
    """Manually triggers a self-improvement cycle: accumulate pairs -> train RM -> PPO RLHF -> Gate -> Promote."""
    try:
        active_rounds = pipeline.checkpoint_manager.list_all_rounds()
        next_num = len(active_rounds)
        default_names = ["Base", "RLHF Round 1", "Round 2", "Final"]
        round_name = req.round_name or (default_names[next_num] if next_num < len(default_names) else f"RLHF Round {next_num}")
        round_number = req.round_number or next_num

        result = pipeline.trigger_self_improvement_round(round_name=round_name, round_number=round_number)
        return {
            "status": "completed",
            "round_name": round_name,
            "gate_passed": result["gate_passed"],
            "metrics": result["benchmark_eval"],
            "rm_evaluation": result["rm_eval"]
        }
    except Exception as e:
        logger.error(f"Error triggering RLHF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
def get_evaluation_metrics(db: Session = Depends(get_db)):
    """Fetches benchmark progression across rounds, including bootstrap 95% CIs and failure distributions."""
    records = db.query(BenchmarkMetricRecord).order_by(BenchmarkMetricRecord.created_at.asc()).all()
    results = []
    for r in records:
        failure_dist = {}
        try:
            failure_dist = json.loads(r.failure_distribution_json)
        except Exception:
            pass

        results.append({
            "round_name": r.round_name,
            "benchmark_size": r.benchmark_size,
            "win_rate": r.win_rate,
            "win_rate_ci": [r.win_rate_ci_lower, r.win_rate_ci_upper],
            "avg_reward": r.avg_reward,
            "avg_reward_ci": [r.avg_reward_ci_lower, r.avg_reward_ci_upper],
            "citation_accuracy": r.citation_accuracy,
            "citation_accuracy_ci": [r.citation_accuracy_ci_lower, r.citation_accuracy_ci_upper],
            "groundedness": r.groundedness,
            "groundedness_ci": [r.groundedness_ci_lower, r.groundedness_ci_upper],
            "hallucination_rate": r.hallucination_rate,
            "recall_at_k": r.recall_at_k,
            "latency_ms": r.latency_ms,
            "failure_analysis": failure_dist,
            "created_at": r.created_at.isoformat()
        })
    return {"metrics_history": results}


@router.get("/papers")
def list_indexed_papers(pipeline: SelfImprovingAssistantPipeline = Depends(get_pipeline)):
    """Returns all papers currently indexed in the FAISS vector database."""
    papers = pipeline.arxiv_fetcher.get_all_papers()
    return {
        "count": len(papers),
        "papers": papers
    }
