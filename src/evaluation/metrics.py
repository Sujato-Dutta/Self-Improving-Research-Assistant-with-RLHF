import time
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from src.retrieval.indexer import FaissIndexer
from src.reward.model import LightweightRewardModel, MultiObjectiveFeatureExtractor
from src.retrieval.embedder import SentenceEmbedder
from src.evaluation.bootstrap import bootstrap_ci, bootstrap_paired_win_rate
from src.evaluation.failure_analysis import FailureAnalyzer

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Runs rigorous benchmark evaluation across held-out queries, computing multi-metric scores and bootstrap 95% CIs."""

    def __init__(
        self,
        indexer: FaissIndexer,
        reward_model: LightweightRewardModel,
        embedder: SentenceEmbedder,
        feature_extractor: MultiObjectiveFeatureExtractor
    ):
        self.indexer = indexer
        self.reward_model = reward_model
        self.embedder = embedder
        self.feature_extractor = feature_extractor
        self.failure_analyzer = FailureAnalyzer()

    def evaluate_model_on_benchmark(
        self,
        generator,
        benchmark_questions: List[Dict[str, Any]],
        base_rewards: List[float] = None
    ) -> Dict[str, Any]:
        """Evaluates a policy generator on the benchmark suite."""
        rewards = []
        citation_accuracies = []
        groundednesses = []
        relevances = []
        hallucination_indicators = []
        task_successes = []
        latencies = []
        diagnostics = []

        # Recall@K tracking for retrieval
        retrieval_hits_k1 = 0
        retrieval_hits_k3 = 0
        retrieval_hits_k5 = 0

        for item in benchmark_questions:
            query = item["question"]
            t0 = time.time()

            # 1. Retrieval
            retrieved = self.indexer.search(query, top_k=5)
            # Evaluate Recall@K: check if query category / keywords appear in top-K
            q_terms = set(query.lower().split())
            for rank, doc in enumerate(retrieved, start=1):
                doc_terms = set((doc.get("title", "") + " " + doc.get("text", "")).lower().split())
                overlap = len(q_terms.intersection(doc_terms))
                if overlap >= 2:
                    if rank <= 1:
                        retrieval_hits_k1 += 1
                    if rank <= 3:
                        retrieval_hits_k3 += 1
                    if rank <= 5:
                        retrieval_hits_k5 += 1
                    break

            # 2. Generation
            evidence_context = "\n".join([doc.get("text", "") for doc in retrieved])
            resp_text, citations, gen_latency = generator.generate_single(query, retrieved, temperature=0.3)
            latencies.append(gen_latency)

            # 3. Multi-factor features
            features = self.feature_extractor.compute_features(
                query, evidence_context, resp_text, evidence_citations_count=len(retrieved)
            )

            # 4. Neural + composite reward
            device = next(self.reward_model.parameters()).device
            q_vec = torch.tensor(self.embedder.encode([query], normalize_embeddings=True), dtype=torch.float32, device=device)
            r_vec = torch.tensor(self.embedder.encode([resp_text], normalize_embeddings=True), dtype=torch.float32, device=device)
            f_vec = torch.tensor([[
                features["relevance"], features["citation_correctness"], features["groundedness"],
                features["completeness"], features["hallucination_penalty"], features["verbosity_penalty"]
            ]], dtype=torch.float32, device=device)

            with torch.no_grad():
                reward_val = float(self.reward_model(q_vec, r_vec, f_vec).item())
            rewards.append(reward_val)

            # Metrics
            citation_accuracies.append(features["citation_correctness"])
            groundednesses.append(features["groundedness"])
            relevances.append(features["relevance"])

            is_hallucinating = 1.0 if features["hallucination_penalty"] > 0.4 else 0.0
            hallucination_indicators.append(is_hallucinating)

            is_success = 1.0 if (features["groundedness"] > 0.3 and features["citation_correctness"] >= 0.7 and is_hallucinating == 0) else 0.0
            task_successes.append(is_success)

            # Failure diagnosis
            diag = self.failure_analyzer.diagnose_response(query, retrieved, resp_text, citations, features)
            diagnostics.append(diag)

        n = max(1, len(benchmark_questions))

        # Bootstrap 95% Confidence Intervals
        reward_ci = bootstrap_ci(rewards)
        citation_ci = bootstrap_ci(citation_accuracies)
        groundedness_ci = bootstrap_ci(groundednesses)

        # Paired win rate vs baseline if base_rewards are provided
        if base_rewards is not None and len(base_rewards) == len(rewards):
            win_rate_ci = bootstrap_paired_win_rate(rewards, base_rewards)
        else:
            win_rate_ci = {"mean": 0.50, "ci_lower": 0.50, "ci_upper": 0.50, "std": 0.0}

        failure_summary = self.failure_analyzer.aggregate_failures(diagnostics)

        summary = {
            "num_questions": n,
            "win_rate": round(win_rate_ci["mean"], 4),
            "win_rate_ci_lower": round(win_rate_ci["ci_lower"], 4),
            "win_rate_ci_upper": round(win_rate_ci["ci_upper"], 4),
            "avg_reward": round(reward_ci["mean"], 4),
            "avg_reward_ci_lower": round(reward_ci["ci_lower"], 4),
            "avg_reward_ci_upper": round(reward_ci["ci_upper"], 4),
            "citation_accuracy": round(citation_ci["mean"], 4),
            "citation_accuracy_ci_lower": round(citation_ci["ci_lower"], 4),
            "citation_accuracy_ci_upper": round(citation_ci["ci_upper"], 4),
            "groundedness": round(groundedness_ci["mean"], 4),
            "groundedness_ci_lower": round(groundedness_ci["ci_lower"], 4),
            "groundedness_ci_upper": round(groundedness_ci["ci_upper"], 4),
            "hallucination_rate": round(float(np.mean(hallucination_indicators)), 4),
            "task_success_rate": round(float(np.mean(task_successes)), 4),
            "retrieval_recall_k1": round(retrieval_hits_k1 / n, 4),
            "retrieval_recall_k3": round(retrieval_hits_k3 / n, 4),
            "retrieval_recall_k5": round(retrieval_hits_k5 / n, 4),
            "relevance_score": round(float(np.mean(relevances)), 4),
            "latency_ms": round(float(np.mean(latencies)), 2),
            "failure_analysis": failure_summary,
            "raw_rewards": rewards
        }

        return summary
