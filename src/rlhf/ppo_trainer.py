import copy
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from src.config import config
from src.reward.model import LightweightRewardModel, MultiObjectiveFeatureExtractor
from src.retrieval.embedder import SentenceEmbedder

logger = logging.getLogger(__name__)


class PolicyValueHead(nn.Module):
    """Value head estimating V(s) for Generalized Advantage Estimation (GAE)."""

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.v_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        return self.v_head(state_embedding).squeeze(-1)


class PPOTrainer:
    """PPO RLHF optimization loop maintaining a frozen reference model, trainable policy, and learned reward model."""

    def __init__(
        self,
        reward_model: LightweightRewardModel,
        embedder: SentenceEmbedder,
        feature_extractor: MultiObjectiveFeatureExtractor,
        device: str = config.model.device
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.reward_model = reward_model.to(self.device)
        self.reward_model.eval()
        self.embedder = embedder
        self.feature_extractor = feature_extractor

        # Trainable Value Function Head for PPO critic
        self.value_head = PolicyValueHead(embed_dim=embedder.embedding_dim).to(self.device)

        # Policy parameterization: linear adaptation head on top of representations
        self.policy_head = nn.Sequential(
            nn.Linear(embedder.embedding_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embedder.embedding_dim)
        ).to(self.device)

        # Frozen reference copy of policy head for KL divergence regularization
        self.ref_policy_head = copy.deepcopy(self.policy_head).to(self.device)
        for p in self.ref_policy_head.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            list(self.policy_head.parameters()) + list(self.value_head.parameters()),
            lr=config.rlhf.learning_rate
        )
        self.kl_coef = config.rlhf.init_kl_coef

    def train_step(
        self,
        batch_queries: List[str],
        batch_responses: List[str],
        batch_evidences: List[str]
    ) -> Dict[str, float]:
        """Executes a single PPO RLHF optimization step with KL divergence regularization."""
        if not batch_queries or not batch_responses:
            return {"loss": 0.0, "mean_reward": 0.0, "kl_div": 0.0}

        # Embed queries and responses
        q_vecs = torch.tensor(self.embedder.encode(batch_queries, normalize_embeddings=True), dtype=torch.float32).to(self.device)
        r_vecs = torch.tensor(self.embedder.encode(batch_responses, normalize_embeddings=True), dtype=torch.float32).to(self.device)

        # Compute multi-factor features and raw reward from learned reward model
        features_list = []
        for q, r, ev in zip(batch_queries, batch_responses, batch_evidences):
            f = self.feature_extractor.compute_features(q, ev, r)
            features_list.append([
                f["relevance"], f["citation_correctness"], f["groundedness"],
                f["completeness"], f["hallucination_penalty"], f["verbosity_penalty"]
            ])
        feats_tensor = torch.tensor(features_list, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            raw_rewards = self.reward_model(q_vecs, r_vecs, feats_tensor)

        # Compute policy and reference representations to estimate KL divergence
        state_repr = torch.cat([q_vecs, r_vecs], dim=-1)
        active_action = self.policy_head(state_repr)
        with torch.no_grad():
            ref_action = self.ref_policy_head(state_repr)

        # Approximate KL Divergence: 0.5 * ||active - ref||^2
        kl_div = 0.5 * torch.mean((active_action - ref_action) ** 2, dim=-1)
        mean_kl = kl_div.mean().item()

        # Regularized PPO Reward: R_reg = R_raw - beta * KL
        regularized_rewards = raw_rewards - self.kl_coef * kl_div

        # Critic value predictions V(s)
        values = self.value_head(q_vecs)

        # Generalized Advantage Estimation (GAE) surrogate
        advantages = (regularized_rewards - values.detach())
        # Advantage normalization
        adv_std = advantages.std() + 1e-8
        norm_advantages = (advantages - advantages.mean()) / adv_std

        # PPO Clipped Surrogate Loss
        # Ratio r_t(theta) = exp(log_p_new - log_p_old)
        # Here we model token similarity shift as cosine distance ratio
        sim_new = torch.cosine_similarity(active_action, r_vecs, dim=-1)
        sim_old = torch.cosine_similarity(ref_action, r_vecs, dim=-1)
        ratio = torch.exp(torch.clamp(sim_new - sim_old, -5.0, 5.0))

        clip_range = config.rlhf.cliprange
        surr1 = ratio * norm_advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * norm_advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value function loss (Mean Squared Error)
        value_loss = nn.functional.mse_loss(values, regularized_rewards)

        # Total PPO objective
        total_loss = policy_loss + config.rlhf.vf_coef * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.policy_head.parameters()) + list(self.value_head.parameters()), config.rlhf.max_grad_norm)
        self.optimizer.step()

        # Adaptive KL coefficient update
        if mean_kl > 1.5 * config.rlhf.target_kl:
            self.kl_coef *= 1.2
        elif mean_kl < config.rlhf.target_kl / 1.5:
            self.kl_coef = max(0.01, self.kl_coef / 1.2)

        return {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "mean_raw_reward": float(raw_rewards.mean().item()),
            "mean_regularized_reward": float(regularized_rewards.mean().item()),
            "kl_divergence": float(mean_kl),
            "kl_coef": float(self.kl_coef)
        }
