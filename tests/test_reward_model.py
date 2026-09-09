import pytest
import torch
from src.retrieval.embedder import SentenceEmbedder
from src.reward.model import LightweightRewardModel, MultiObjectiveFeatureExtractor
from src.reward.dataset import PreferencePairDataset
from src.reward.trainer import RewardModelTrainer, bradley_terry_loss
from src.reward.evaluator import RewardModelEvaluator


def test_reward_model_forward():
    model = LightweightRewardModel(embed_dim=384)
    q_emb = torch.randn(2, 384)
    r_emb = torch.randn(2, 384)
    feats = torch.randn(2, 6)

    rewards = model(q_emb, r_emb, feats)
    assert rewards.shape == (2,)


def test_bradley_terry_loss():
    r_chosen = torch.tensor([2.0, 1.5])
    r_rejected = torch.tensor([0.5, -0.5])
    loss = bradley_terry_loss(r_chosen, r_rejected)
    assert loss.item() > 0.0


def test_reward_evaluator():
    embedder = SentenceEmbedder()
    extractor = MultiObjectiveFeatureExtractor(embedder)
    model = LightweightRewardModel(embed_dim=embedder.embedding_dim)

    pairs = [
        {
            "query": "What is self-attention?",
            "chosen": "Self-attention computes global dependencies directly [1].",
            "rejected": "It is magic with zero computation.",
            "evidence": "Attention Is All You Need: self-attention operates across arbitrary positions.",
            "weight": 1.0
        },
        {
            "query": "How does LoRA work?",
            "chosen": "LoRA injects trainable rank decomposition matrices [1].",
            "rejected": "LoRA changes all billions of weights equally.",
            "evidence": "LoRA freezes pre-trained weights and injects rank decomposition matrices.",
            "weight": 1.0
        }
    ]
    dataset = PreferencePairDataset(pairs, embedder, extractor)
    evaluator = RewardModelEvaluator(model)
    res = evaluator.evaluate(dataset)

    assert "pairwise_accuracy" in res
    assert "roc_auc" in res
    assert "brier_score" in res
    assert 0.0 <= res["brier_score"] <= 1.0
