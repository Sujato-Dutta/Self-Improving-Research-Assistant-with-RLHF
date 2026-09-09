import json
import logging
import torch
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from src.config import config
from src.db.database import SessionLocal, init_db
from src.db.models import QueryRecord, RetrievedEvidence, ResponseRecord, RLHFRoundRecord, BenchmarkMetricRecord
from src.retrieval.arxiv_fetcher import ArxivFetcher
from src.retrieval.text_splitter import TextSplitter
from src.retrieval.embedder import SentenceEmbedder
from src.retrieval.indexer import FaissIndexer
from src.generator.policy_model import PolicyGenerator
from src.generator.ollama_adapter import OllamaAdapter
from src.feedback.collector import FeedbackCollector
from src.feedback.pair_converter import PreferencePairConverter
from src.reward.model import LightweightRewardModel, MultiObjectiveFeatureExtractor
from src.reward.dataset import PreferencePairDataset
from src.reward.trainer import RewardModelTrainer
from src.reward.evaluator import RewardModelEvaluator
from src.rlhf.ppo_trainer import PPOTrainer
from src.rlhf.checkpoint_manager import CheckpointManager
from src.rlhf.evaluation_gate import EvaluationGate
from src.evaluation.benchmark import BenchmarkManager
from src.evaluation.metrics import BenchmarkEvaluator
from src.tracking.mlflow_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class SelfImprovingAssistantPipeline:
    """End-to-End Orchestrator for the Self-Improving Research Assistant with Preference-Based RLHF."""

    def __init__(self, db_session: Optional[Session] = None):
        init_db()
        self.db = db_session or SessionLocal()

        # Components
        self.embedder = SentenceEmbedder()
        self.indexer = FaissIndexer(embedder=self.embedder)
        self.arxiv_fetcher = ArxivFetcher()
        self.text_splitter = TextSplitter()
        self.feature_extractor = MultiObjectiveFeatureExtractor(embedder=self.embedder)

        # Policy & Generation
        self.checkpoint_manager = CheckpointManager()
        self.policy_generator = PolicyGenerator(self.checkpoint_manager.get_active_model_path())
        self.ollama = OllamaAdapter(fallback_generator=self.policy_generator)

        # Feedback & Preferences
        self.feedback_collector = FeedbackCollector(self.db)
        self.preference_converter = PreferencePairConverter(self.db)

        # Reward Model & RLHF
        dev = torch.device(config.model.device if torch.cuda.is_available() and config.model.device == "cuda" else "cpu")
        self.reward_model = LightweightRewardModel(embed_dim=self.embedder.embedding_dim).to(dev)
        self.reward_trainer = RewardModelTrainer(model=self.reward_model)
        self.reward_evaluator = RewardModelEvaluator(model=self.reward_model)
        self.ppo_trainer = PPOTrainer(
            reward_model=self.reward_model,
            embedder=self.embedder,
            feature_extractor=self.feature_extractor
        )
        self.gate = EvaluationGate(checkpoint_manager=self.checkpoint_manager)

        # Evaluation & Tracking
        self.benchmark_manager = BenchmarkManager()
        self.evaluator = BenchmarkEvaluator(
            indexer=self.indexer,
            reward_model=self.reward_model,
            embedder=self.embedder,
            feature_extractor=self.feature_extractor
        )
        self.tracker = ExperimentTracker()

    def query(
        self,
        query_text: str,
        dual_response: bool = False,
        use_ollama: bool = False,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes full research query flow: search FAISS evidence -> generate grounded response(s)."""
        # 1. Store query
        q_record = QueryRecord(query_text=query_text, session_id=session_id)
        self.db.add(q_record)
        self.db.commit()
        self.db.refresh(q_record)

        # 2. Retrieve evidence from FAISS
        retrieved_docs = self.indexer.search(query_text, top_k=config.retrieval.top_k)
        if not retrieved_docs:
            # Try fetching from arXiv cache/seed
            fallback_papers = self.arxiv_fetcher.search_local_papers(query_text, max_results=3)
            for p in fallback_papers:
                chunks = self.text_splitter.chunk_paper(p)
                self.indexer.add_documents(chunks)
            retrieved_docs = self.indexer.search(query_text, top_k=config.retrieval.top_k)

        # Save retrieved evidence in DB
        for doc in retrieved_docs:
            ev_record = RetrievedEvidence(
                query_id=q_record.id,
                citation_index=doc.get("citation_index", 1),
                paper_title=doc.get("title", "Untitled"),
                arxiv_id=doc.get("arxiv_id", ""),
                authors=", ".join(doc.get("authors", [])),
                excerpt=doc.get("text", ""),
                similarity_score=doc.get("similarity_score", 0.0),
                url=doc.get("url", "")
            )
            self.db.add(ev_record)
        self.db.commit()

        active_version = self.checkpoint_manager.get_active_version()

        if dual_response:
            # Generate Candidate A & Candidate B for A/B testing
            resp_a, resp_b = self.policy_generator.generate_pair(query_text, retrieved_docs)

            rec_a = ResponseRecord(
                query_id=q_record.id,
                model_version=active_version,
                variant="A",
                response_text=resp_a["text"],
                citations_json=json.dumps(resp_a["citations"]),
                latency_ms=resp_a["latency_ms"]
            )
            rec_b = ResponseRecord(
                query_id=q_record.id,
                model_version=active_version,
                variant="B",
                response_text=resp_b["text"],
                citations_json=json.dumps(resp_b["citations"]),
                latency_ms=resp_b["latency_ms"]
            )
            self.db.add(rec_a)
            self.db.add(rec_b)
            self.db.commit()
            self.db.refresh(rec_a)
            self.db.refresh(rec_b)

            return {
                "query_id": q_record.id,
                "model_version": active_version,
                "dual_mode": True,
                "response_a": {
                    "id": rec_a.id,
                    "text": rec_a.response_text,
                    "citations": resp_a["citations"],
                    "latency_ms": rec_a.latency_ms
                },
                "response_b": {
                    "id": rec_b.id,
                    "text": rec_b.response_text,
                    "citations": resp_b["citations"],
                    "latency_ms": rec_b.latency_ms
                },
                "evidence": retrieved_docs
            }

        # Single response mode
        if use_ollama:
            gen_res = self.ollama.generate(query_text, retrieved_docs)
            resp_text = gen_res["text"]
            citations = gen_res["citations"]
            latency_ms = gen_res.get("latency_ms", 50.0)
        else:
            resp_text, citations, latency_ms = self.policy_generator.generate_single(query_text, retrieved_docs)

        rec = ResponseRecord(
            query_id=q_record.id,
            model_version=active_version,
            variant="single",
            response_text=resp_text,
            citations_json=json.dumps(citations),
            latency_ms=latency_ms
        )
        self.db.add(rec)
        self.db.commit()
        self.db.refresh(rec)

        return {
            "query_id": q_record.id,
            "response_id": rec.id,
            "model_version": active_version,
            "response_text": resp_text,
            "citations": citations,
            "latency_ms": latency_ms,
            "evidence": retrieved_docs
        }

    def trigger_self_improvement_round(self, round_name: str, round_number: int) -> Dict[str, Any]:
        """Executes a full self-improvement round:
        1. Accumulates preference dataset from DB.
        2. Trains & evaluates PyTorch Reward Model.
        3. Runs PPO RLHF training step with KL regularization.
        4. Evaluates candidate model on the 350-question benchmark.
        5. Runs regression gatekeeper tests.
        6. Promotes checkpoint if passed, logs to MLflow & DB.
        """
        logger.info(f"=== Starting Self-Improvement: {round_name} (Round {round_number}) ===")
        self.tracker.start_run(run_name=f"RLHF_{round_name.replace(' ', '_')}")

        # 1. Convert implicit signals and fetch all preference pairs
        self.preference_converter.convert_implicit_feedback_to_pairs()
        pairs = self.preference_converter.export_preference_dataset()

        # If preference dataset is small in initial runs, bootstrap with seed preference comparisons
        if len(pairs) < config.rlhf.min_preference_buffer_size:
            pairs = self._generate_seed_preference_data(count=30)

        # 2. Train Reward Model
        rm_dataset = PreferencePairDataset(pairs, self.embedder, self.feature_extractor)
        rm_history = self.reward_trainer.train(
            rm_dataset,
            epochs=config.reward.epochs,
            batch_size=config.reward.batch_size,
            save_path=config.models_dir / f"reward_model_{round_number}.pt"
        )
        rm_eval = self.reward_evaluator.evaluate(rm_dataset)

        # 3. PPO RLHF Policy Update Step
        queries = [p["query"] for p in pairs[:20]]
        responses = [p["chosen"] for p in pairs[:20]]
        evidences = [p.get("evidence", "") for p in pairs[:20]]
        ppo_metrics = self.ppo_trainer.train_step(queries, responses, evidences)

        # 4. Evaluate Candidate on 350 Held-Out Benchmark Questions
        benchmark_qs = self.benchmark_manager.get_questions(limit=350)
        # Fetch base metrics for comparative win rate calculation
        base_record = self.db.query(BenchmarkMetricRecord).filter(BenchmarkMetricRecord.round_name == "Base").first()
        base_rewards = getattr(self, "base_rewards", None)
        base_metrics_dict = {
            "win_rate": 0.50,
            "avg_reward": 0.42,
            "hallucination_rate": 0.22,
            "citation_accuracy": 0.74
        }
        if base_record:
            base_metrics_dict = {
                "win_rate": base_record.win_rate,
                "avg_reward": base_record.avg_reward,
                "hallucination_rate": base_record.hallucination_rate,
                "citation_accuracy": base_record.citation_accuracy
            }

        candidate_eval = self.evaluator.evaluate_model_on_benchmark(
            generator=self.policy_generator,
            benchmark_questions=benchmark_qs,
            base_rewards=base_rewards
        )

        # 5. Gatekeeper Regression Test
        gate_passed, gate_report = self.gate.evaluate_candidate(
            round_name=round_name,
            candidate_metrics=candidate_eval,
            base_metrics=base_metrics_dict
        )

        # 6. Checkpoint Registry Update
        candidate_path = str(config.checkpoints_dir / f"checkpoint_round_{round_number}")
        self.checkpoint_manager.register_candidate(round_name, round_number, candidate_path, candidate_eval)
        if gate_passed:
            self.checkpoint_manager.promote_checkpoint(round_name, candidate_eval)

        # 7. Record in DB
        round_rec = RLHFRoundRecord(
            round_number=round_number,
            round_name=round_name,
            base_checkpoint=self.checkpoint_manager.get_active_model_path(),
            promoted_checkpoint=candidate_path if gate_passed else None,
            num_samples=len(pairs),
            reward_loss=rm_history["train_loss"][-1] if rm_history["train_loss"] else 0.0,
            avg_reward_before=base_metrics_dict["avg_reward"],
            avg_reward_after=candidate_eval["avg_reward"],
            gate_passed=gate_passed,
            metrics_json=json.dumps(candidate_eval)
        )
        self.db.add(round_rec)

        bench_rec = BenchmarkMetricRecord(
            round_name=round_name,
            benchmark_size=len(benchmark_qs),
            win_rate=candidate_eval["win_rate"],
            win_rate_ci_lower=candidate_eval["win_rate_ci_lower"],
            win_rate_ci_upper=candidate_eval["win_rate_ci_upper"],
            avg_reward=candidate_eval["avg_reward"],
            avg_reward_ci_lower=candidate_eval["avg_reward_ci_lower"],
            avg_reward_ci_upper=candidate_eval["avg_reward_ci_upper"],
            citation_accuracy=candidate_eval["citation_accuracy"],
            citation_accuracy_ci_lower=candidate_eval["citation_accuracy_ci_lower"],
            citation_accuracy_ci_upper=candidate_eval["citation_accuracy_ci_upper"],
            groundedness=candidate_eval["groundedness"],
            groundedness_ci_lower=candidate_eval["groundedness_ci_lower"],
            groundedness_ci_upper=candidate_eval["groundedness_ci_upper"],
            hallucination_rate=candidate_eval["hallucination_rate"],
            recall_at_k=candidate_eval["retrieval_recall_k3"],
            latency_ms=candidate_eval["latency_ms"],
            failure_distribution_json=json.dumps(candidate_eval["failure_analysis"])
        )
        self.db.add(bench_rec)
        self.db.commit()

        # 8. Log in MLflow
        self.tracker.log_params({
            "round_name": round_name,
            "round_number": round_number,
            "num_samples": len(pairs),
            "gate_passed": gate_passed
        })
        self.tracker.log_metrics({
            "rm_loss": round_rec.reward_loss,
            "rm_accuracy": rm_eval["pairwise_accuracy"],
            "rm_roc_auc": rm_eval["roc_auc"],
            "rm_brier_score": rm_eval["brier_score"],
            "ppo_kl_div": ppo_metrics["kl_divergence"],
            "ppo_loss": ppo_metrics["loss"],
            "win_rate": candidate_eval["win_rate"],
            "avg_reward": candidate_eval["avg_reward"],
            "citation_accuracy": candidate_eval["citation_accuracy"],
            "groundedness": candidate_eval["groundedness"],
            "hallucination_rate": candidate_eval["hallucination_rate"]
        })
        self.tracker.end_run()

        logger.info(f"=== Completed Round: {round_name} | Gate: {'PASSED' if gate_passed else 'FAILED'} ===")
        return {
            "round_name": round_name,
            "gate_passed": gate_passed,
            "rm_eval": rm_eval,
            "ppo_metrics": ppo_metrics,
            "benchmark_eval": candidate_eval,
            "gate_report": gate_report
        }

    def _generate_seed_preference_data(self, count: int = 30) -> List[Dict[str, Any]]:
        """Bootstraps a curated set of grounded vs hallucinated/uncited research answers."""
        seed_questions = [
            ("How does attention replace recurrence?", "Attention Is All You Need", "1706.03762"),
            ("How does RLHF align language models with human preferences?", "Training language models with human feedback", "2203.02155"),
            ("What is Direct Preference Optimization (DPO)?", "Direct Preference Optimization", "2305.18290"),
            ("How does LoRA reduce trainable parameter count?", "LoRA: Low-Rank Adaptation", "2106.09685"),
            ("What are the IO-aware tiling benefits of FlashAttention?", "FlashAttention", "2205.14135")
        ]

        pairs = []
        for i in range(count):
            q, title, aid = seed_questions[i % len(seed_questions)]
            # Chosen: factual, grounded, inline citations
            chosen = (
                f"Based on empirical findings in {title} [1], self-attention mechanisms compute weighted associations "
                f"directly across arbitrary sequence positions without recurrence. This enables exact global context modelling "
                f"with sub-quadratic training iteration speed [1].\n\n### References\n- [1] {title} (arXiv: {aid})"
            )
            # Rejected: unsupported hallucination, no citations, verbose drift
            rejected = (
                f"Recurrence is replaced because attention uses huge neural networks that memorize all facts. "
                f"It is known that transformers do not need any math and are basically magic classifiers with zero latency."
            )
            pairs.append({
                "query": q,
                "chosen": chosen,
                "rejected": rejected,
                "evidence": f"Title: {title} (arXiv: {aid})\nAbstract excerpt describing architecture and results.",
                "weight": 1.0,
                "source": "seed_curated",
                "pair_id": i + 1000
            })
        return pairs
