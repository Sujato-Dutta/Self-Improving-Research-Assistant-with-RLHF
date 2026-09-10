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

        result = pipeline.trigger_self_improvement_round(
            round_name=round_name,
            round_number=round_number,
            eval_limit=getattr(req, "eval_limit", None) or 25
        )
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
def get_evaluation_metrics(db: Session = Depends(get_db), pipeline: SelfImprovingAssistantPipeline = Depends(get_pipeline)):
    """Fetches benchmark progression across rounds, including bootstrap 95% CIs, live stats, and failure distributions."""
    records = db.query(BenchmarkMetricRecord).order_by(BenchmarkMetricRecord.created_at.asc()).all()
    rounds_map = {r.round_name: r for r in db.query(RLHFRoundRecord).all()}

    results = []
    for r in records:
        failure_dist = {}
        try:
            failure_dist = json.loads(r.failure_distribution_json)
        except Exception:
            pass

        round_obj = rounds_map.get(r.round_name)
        gate_passed = round_obj.gate_passed if round_obj else (r.round_name != "Base")
        is_base = (r.round_name == "Base")

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
            "gate_passed": gate_passed,
            "is_base": is_base,
            "failure_analysis": failure_dist,
            "created_at": r.created_at.isoformat()
        })

    # Live interaction stats
    from src.db.models import QueryRecord, FeedbackRecord, PreferencePairRecord
    total_queries = db.query(QueryRecord).count()
    total_feedbacks = db.query(FeedbackRecord).count()
    total_preferences = db.query(PreferencePairRecord).count()
    active_version = pipeline.checkpoint_manager.get_active_version()

    return {
        "metrics_history": results,
        "live_stats": {
            "total_queries": total_queries,
            "total_feedbacks": total_feedbacks,
            "total_preferences": total_preferences,
            "active_version": active_version
        }
    }


@router.post("/evaluation/run")
def run_benchmark_evaluation(pipeline: SelfImprovingAssistantPipeline = Depends(get_pipeline), db: Session = Depends(get_db)):
    """Runs a fresh benchmark evaluation on the active model and updates gate metrics in the database."""
    try:
        benchmark_qs = pipeline.benchmark_manager.get_benchmark_questions(limit=15)
        active_version = pipeline.checkpoint_manager.get_active_version()

        # Evaluate active model
        eval_result = pipeline.evaluator.evaluate_model_on_benchmark(
            generator=pipeline.policy_generator,
            benchmark_questions=benchmark_qs
        )

        # Check gate against base criteria
        gate_passed, gate_report = pipeline.gate.evaluate_candidate(
            round_name=active_version,
            candidate_metrics=eval_result,
            base_metrics={
                "win_rate": 0.50,
                "avg_reward": 0.42,
                "hallucination_rate": 0.22,
                "citation_accuracy": 0.74
            }
        )

        # Record evaluation metric
        bench_rec = BenchmarkMetricRecord(
            round_name=f"{active_version} (Live)",
            benchmark_size=len(benchmark_qs),
            win_rate=eval_result["win_rate"],
            win_rate_ci_lower=eval_result["win_rate_ci_lower"],
            win_rate_ci_upper=eval_result["win_rate_ci_upper"],
            avg_reward=eval_result["avg_reward"],
            avg_reward_ci_lower=eval_result["avg_reward_ci_lower"],
            avg_reward_ci_upper=eval_result["avg_reward_ci_upper"],
            citation_accuracy=eval_result["citation_accuracy"],
            citation_accuracy_ci_lower=eval_result["citation_accuracy_ci_lower"],
            citation_accuracy_ci_upper=eval_result["citation_accuracy_ci_upper"],
            groundedness=eval_result["groundedness"],
            groundedness_ci_lower=eval_result["groundedness_ci_lower"],
            groundedness_ci_upper=eval_result["groundedness_ci_upper"],
            hallucination_rate=eval_result["hallucination_rate"],
            recall_at_k=eval_result.get("retrieval_recall_k3", eval_result.get("recall_at_k", 0.85)),
            latency_ms=eval_result["latency_ms"],
            failure_distribution_json=json.dumps(eval_result.get("failure_analysis", {}))
        )
        db.add(bench_rec)
        db.commit()
        db.refresh(bench_rec)

        return {
            "status": "success",
            "round_name": bench_rec.round_name,
            "gate_passed": gate_passed,
            "gate_report": gate_report,
            "metrics": eval_result
        }
    except Exception as e:
        logger.error(f"Error running benchmark evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ollama/status")
def get_ollama_status(pipeline: SelfImprovingAssistantPipeline = Depends(get_pipeline)):
    """Checks whether local Ollama instance is accessible and returns configuration."""
    import urllib.request
    is_up = pipeline.ollama.is_available()
    models = []
    if is_up:
        try:
            req = urllib.request.Request(f"{pipeline.ollama.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=2.0) as res:
                data = json.loads(res.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", [])]
        except Exception:
            pass
    return {
        "available": is_up,
        "base_url": pipeline.ollama.base_url,
        "configured_model": pipeline.ollama.model_tag,
        "models": models
    }


@router.get("/papers")
def list_indexed_papers(pipeline: SelfImprovingAssistantPipeline = Depends(get_pipeline)):
    """Returns all papers currently indexed in the FAISS vector database."""
    papers = pipeline.arxiv_fetcher.get_all_papers()
    return {
        "count": len(papers),
        "papers": papers
    }
