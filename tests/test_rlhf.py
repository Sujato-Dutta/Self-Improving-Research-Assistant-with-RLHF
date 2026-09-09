import pytest
from src.retrieval.embedder import SentenceEmbedder
from src.reward.model import LightweightRewardModel, MultiObjectiveFeatureExtractor
from src.rlhf.ppo_trainer import PPOTrainer
from src.rlhf.checkpoint_manager import CheckpointManager
from src.rlhf.evaluation_gate import EvaluationGate


def test_ppo_step():
    embedder = SentenceEmbedder()
    extractor = MultiObjectiveFeatureExtractor(embedder)
    rm = LightweightRewardModel(embed_dim=embedder.embedding_dim)
    trainer = PPOTrainer(rm, embedder, extractor, device="cpu")

    queries = ["What is RLHF?"]
    responses = ["RLHF aligns models with human preferences via reward modeling [1]."]
    evidences = ["InstructGPT shows RLHF aligns LLMs with human intent."]

    metrics = trainer.train_step(queries, responses, evidences)
    assert "loss" in metrics
    assert "kl_divergence" in metrics
    assert "mean_regularized_reward" in metrics


def test_checkpoint_manager(tmp_path):
    mgr = CheckpointManager(checkpoints_dir=tmp_path / "checkpoints")
    assert mgr.get_active_version() == "Base"

    mgr.register_candidate("RLHF Round 1", 1, "path/to/model")
    promoted = mgr.promote_checkpoint("RLHF Round 1", {"win_rate": 0.58})
    assert promoted is True
    assert mgr.get_active_version() == "RLHF Round 1"


def test_evaluation_gate(tmp_path):
    mgr = CheckpointManager(checkpoints_dir=tmp_path / "checkpoints")
    gate = EvaluationGate(mgr, min_win_rate=0.52, min_reward_delta=0.03)

    # Candidate that passes
    candidate_good = {
        "win_rate": 0.59,
        "avg_reward": 0.55,
        "hallucination_rate": 0.12,
        "citation_accuracy": 0.88
    }
    base_metrics = {
        "win_rate": 0.50,
        "avg_reward": 0.40,
        "hallucination_rate": 0.22,
        "citation_accuracy": 0.74
    }
    passed, report = gate.evaluate_candidate("Round 1", candidate_good, base_metrics)
    assert passed is True
    assert report["gate_passed"] is True

    # Candidate that degrades hallucination rate
    candidate_bad = dict(candidate_good)
    candidate_bad["hallucination_rate"] = 0.35
    failed, bad_report = gate.evaluate_candidate("Round Bad", candidate_bad, base_metrics)
    assert failed is False
    assert bad_report["gate_passed"] is False
