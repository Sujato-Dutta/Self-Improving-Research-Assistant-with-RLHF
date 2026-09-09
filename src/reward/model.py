import re
import math
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from src.config import config
from src.retrieval.embedder import SentenceEmbedder
from src.generator.policy_model import extract_citations

logger = logging.getLogger(__name__)


class MultiObjectiveFeatureExtractor:
    """Computes interpretable reward sub-components: relevance, citations, groundedness, completeness, hallucination, verbosity."""

    def __init__(self, embedder: SentenceEmbedder):
        self.embedder = embedder

    def compute_features(
        self,
        query: str,
        evidence: str,
        response: str,
        evidence_citations_count: int = 5
    ) -> Dict[str, float]:
        # 1. Relevance: cosine similarity between query and response
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        r_emb = self.embedder.encode([response], normalize_embeddings=True)
        relevance = float(np.dot(q_emb, r_emb.T)[0, 0])

        # 2. Citation correctness: are cited indices within [1 .. evidence_citations_count]?
        cited_indices = extract_citations(response)
        if not cited_indices:
            citation_correctness = 0.0
        else:
            valid_count = sum(1 for idx in cited_indices if 1 <= idx <= max(1, evidence_citations_count))
            citation_correctness = valid_count / len(cited_indices)

        # 3. Groundedness: lexical and entity recall from evidence in response
        ev_tokens = set(re.findall(r"\b\w{4,}\b", evidence.lower()))
        resp_tokens = set(re.findall(r"\b\w{4,}\b", response.lower()))
        if not ev_tokens or not resp_tokens:
            groundedness = 0.2
        else:
            intersection = resp_tokens.intersection(ev_tokens)
            groundedness = min(1.0, len(intersection) / max(10, len(resp_tokens) * 0.4))

        # 4. Completeness: coverage of query intent words
        q_tokens = set(re.findall(r"\b\w{3,}\b", query.lower()))
        if q_tokens:
            q_overlap = len(resp_tokens.intersection(q_tokens)) / len(q_tokens)
            completeness = min(1.0, q_overlap + (0.3 if len(response.split()) > 40 else 0.0))
        else:
            completeness = 0.5

        # 5. Hallucination penalty: high if citations are present but claim text does not overlap with evidence
        has_citations = len(cited_indices) > 0
        if has_citations and groundedness < 0.15:
            hallucination_penalty = 1.0  # High hallucination suspicion
        elif not has_citations and len(evidence.strip()) > 50:
            hallucination_penalty = 0.5  # Missed citing available evidence
        else:
            hallucination_penalty = max(0.0, 1.0 - groundedness * 1.5)

        # 6. Verbosity penalty: penalize both overly terse (<30 words) and overly verbose (>450 words)
        word_count = len(response.split())
        if word_count < 30:
            verbosity_penalty = (30 - word_count) / 30.0
        elif word_count > 450:
            verbosity_penalty = min(1.0, (word_count - 450) / 200.0)
        else:
            verbosity_penalty = 0.0

        return {
            "relevance": relevance,
            "citation_correctness": citation_correctness,
            "groundedness": groundedness,
            "completeness": completeness,
            "hallucination_penalty": hallucination_penalty,
            "verbosity_penalty": verbosity_penalty
        }


class LightweightRewardModel(nn.Module):
    """PyTorch Multi-Factor Reward Model combining deep contextual representations with multi-objective alignment features."""

    def __init__(self, embed_dim: int = 384, num_features: int = 6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_features = num_features

        # Projection layers for text embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + num_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Learnable multi-factor balancing weights
        self.objective_weights = nn.Parameter(torch.tensor([
            config.reward.relevance_weight,
            config.reward.citation_correctness_weight,
            config.reward.groundedness_weight,
            config.reward.completeness_weight,
            -config.reward.hallucination_penalty,
            -config.reward.verbosity_penalty
        ], dtype=torch.float32))

    def forward(
        self,
        query_embeddings: torch.Tensor,     # (batch, embed_dim)
        response_embeddings: torch.Tensor,  # (batch, embed_dim)
        features: torch.Tensor              # (batch, num_features)
    ) -> torch.Tensor:
        device = self.objective_weights.device
        query_embeddings = query_embeddings.to(device)
        response_embeddings = response_embeddings.to(device)
        features = features.to(device)
        # Concatenate neural features and multi-factor objective signals
        combined = torch.cat([query_embeddings, response_embeddings, features], dim=-1)
        neural_scalar = self.mlp(combined).squeeze(-1)  # (batch,)

        # Linear multi-objective contribution
        factor_scalar = torch.matmul(features, self.objective_weights)  # (batch,)

        total_reward = neural_scalar + factor_scalar
        return total_reward
