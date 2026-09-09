#!/usr/bin/env python
"""
Executes the full end-to-end self-improvement loop:
Base -> RLHF Round 1 -> Round 2 -> Final.
Generates genuine empirical benchmark evaluations with bootstrap 95% confidence intervals.
"""
import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.pipeline import SelfImprovingAssistantPipeline
from src.db.database import init_db, SessionLocal
from src.db.models import BenchmarkMetricRecord, RLHFRoundRecord
from src.evaluation.benchmark import BenchmarkManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SelfImprovementLoop")


def print_comparison_table(records):
    print("\n" + "=" * 115)
    print(f"{'Round':<16} | {'Win Rate (95% CI)':<22} | {'Reward (95% CI)':<22} | {'Citation Acc (95% CI)':<22} | {'Hallucination':<14} | {'Recall@3':<10}")
    print("=" * 115)
    for r in records:
        win_str = f"{r.win_rate:.3f} [{r.win_rate_ci_lower:.3f}, {r.win_rate_ci_upper:.3f}]"
        rew_str = f"{r.avg_reward:.3f} [{r.avg_reward_ci_lower:.3f}, {r.avg_reward_ci_upper:.3f}]"
        cit_str = f"{r.citation_accuracy:.3f} [{r.citation_accuracy_ci_lower:.3f}, {r.citation_accuracy_ci_upper:.3f}]"
        hal_str = f"{r.hallucination_rate:.3f}"
        rec_str = f"{r.recall_at_k:.3f}"
        print(f"{r.round_name:<16} | {win_str:<22} | {rew_str:<22} | {cit_str:<22} | {hal_str:<14} | {rec_str:<10}")
    print("=" * 115 + "\n")


def main():
    init_db()
    db = SessionLocal()
    pipeline = SelfImprovingAssistantPipeline(db_session=db)

    logger.info("Step 1: Ensuring papers and FAISS index are seeded...")
    if len(pipeline.indexer.documents) == 0:
        all_papers = pipeline.arxiv_fetcher.get_all_papers()
        all_chunks = []
        for p in all_papers:
            all_chunks.extend(pipeline.text_splitter.chunk_paper(p))
        pipeline.indexer.add_documents(all_chunks)
        logger.info(f"Seeded {len(all_chunks)} chunks into FAISS.")

    # Reset previous run records for clean progression
    db.query(BenchmarkMetricRecord).delete()
    db.query(RLHFRoundRecord).delete()
    db.commit()

    logger.info("Step 2: Evaluating Base policy on 350-question benchmark...")
    benchmark_qs = pipeline.benchmark_manager.get_questions(limit=350)
    base_eval = pipeline.evaluator.evaluate_model_on_benchmark(
        generator=pipeline.policy_generator,
        benchmark_questions=benchmark_qs
    )
    pipeline.base_rewards = base_eval["raw_rewards"]

    base_rec = BenchmarkMetricRecord(
        round_name="Base",
        benchmark_size=len(benchmark_qs),
        win_rate=base_eval["win_rate"],
        win_rate_ci_lower=base_eval["win_rate_ci_lower"],
        win_rate_ci_upper=base_eval["win_rate_ci_upper"],
        avg_reward=base_eval["avg_reward"],
        avg_reward_ci_lower=base_eval["avg_reward_ci_lower"],
        avg_reward_ci_upper=base_eval["avg_reward_ci_upper"],
        citation_accuracy=base_eval["citation_accuracy"],
        citation_accuracy_ci_lower=base_eval["citation_accuracy_ci_lower"],
        citation_accuracy_ci_upper=base_eval["citation_accuracy_ci_upper"],
        groundedness=base_eval["groundedness"],
        groundedness_ci_lower=base_eval["groundedness_ci_lower"],
        groundedness_ci_upper=base_eval["groundedness_ci_upper"],
        hallucination_rate=base_eval["hallucination_rate"],
        recall_at_k=base_eval["retrieval_recall_k3"],
        latency_ms=base_eval["latency_ms"],
        failure_distribution_json=json.dumps(base_eval["failure_analysis"])
    )
    db.add(base_rec)

    base_round = RLHFRoundRecord(
        round_number=0,
        round_name="Base",
        base_checkpoint=pipeline.checkpoint_manager.get_active_model_path(),
        promoted_checkpoint=pipeline.checkpoint_manager.get_active_model_path(),
        num_samples=0,
        reward_loss=0.0,
        avg_reward_before=base_eval["avg_reward"],
        avg_reward_after=base_eval["avg_reward"],
        gate_passed=True,
        metrics_json=json.dumps(base_eval)
    )
    db.add(base_round)
    db.commit()
    logger.info(f"Base benchmark complete: Avg Reward = {base_eval['avg_reward']:.3f}, Citation Acc = {base_eval['citation_accuracy']:.3f}")

    # 2. RLHF Round 1
    logger.info("\n" + "=" * 60)
    logger.info("Executing RLHF Round 1 (Post-Training & Evaluation Gate)")
    logger.info("=" * 60)
    res_r1 = pipeline.trigger_self_improvement_round("RLHF Round 1", round_number=1)
    logger.info(f"Round 1 Gate Passed: {res_r1['gate_passed']}")

    # 3. RLHF Round 2
    logger.info("\n" + "=" * 60)
    logger.info("Executing RLHF Round 2 (Post-Training & Evaluation Gate)")
    logger.info("=" * 60)
    res_r2 = pipeline.trigger_self_improvement_round("Round 2", round_number=2)
    logger.info(f"Round 2 Gate Passed: {res_r2['gate_passed']}")

    # 4. Final Checkpoint Promotion
    logger.info("\n" + "=" * 60)
    logger.info("Executing Final Round Promotion")
    logger.info("=" * 60)
    res_final = pipeline.trigger_self_improvement_round("Final", round_number=3)
    logger.info(f"Final Round Gate Passed: {res_final['gate_passed']}")

    # 5. Display All Experiment Results
    records = db.query(BenchmarkMetricRecord).order_by(BenchmarkMetricRecord.created_at.asc()).all()
    print_comparison_table(records)

    # Save summary markdown table to data/benchmark/results_summary.json
    summary_data = [
        {
            "round": r.round_name,
            "win_rate": r.win_rate,
            "win_rate_ci": [r.win_rate_ci_lower, r.win_rate_ci_upper],
            "avg_reward": r.avg_reward,
            "avg_reward_ci": [r.avg_reward_ci_lower, r.avg_reward_ci_upper],
            "citation_accuracy": r.citation_accuracy,
            "citation_accuracy_ci": [r.citation_accuracy_ci_lower, r.citation_accuracy_ci_upper],
            "groundedness": r.groundedness,
            "hallucination_rate": r.hallucination_rate,
            "recall_at_3": r.recall_at_k,
            "latency_ms": r.latency_ms
        }
        for r in records
    ]
    with open(config.benchmark_dir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Saved complete experiment summary to: {config.benchmark_dir / 'results_summary.json'}")


if __name__ == "__main__":
    main()
