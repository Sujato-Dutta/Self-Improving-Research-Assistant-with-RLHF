import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
from src.retrieval.embedder import SentenceEmbedder
from src.reward.model import MultiObjectiveFeatureExtractor


class PreferencePairDataset(Dataset):
    """PyTorch Dataset for Bradley-Terry pairwise preference optimization."""

    def __init__(
        self,
        pairs: List[Dict[str, Any]],
        embedder: SentenceEmbedder,
        feature_extractor: MultiObjectiveFeatureExtractor
    ):
        self.pairs = pairs
        self.embedder = embedder
        self.feature_extractor = feature_extractor
        self._precompute()

    def _precompute(self):
        self.samples = []
        if not self.pairs:
            return

        queries = [p["query"] for p in self.pairs]
        chosen_texts = [p["chosen"] for p in self.pairs]
        rejected_texts = [p["rejected"] for p in self.pairs]

        q_vecs = self.embedder.encode(queries, normalize_embeddings=True)
        c_vecs = self.embedder.encode(chosen_texts, normalize_embeddings=True)
        r_vecs = self.embedder.encode(rejected_texts, normalize_embeddings=True)

        for idx, p in enumerate(self.pairs):
            evidence = p.get("evidence", "")
            f_chosen = self.feature_extractor.compute_features(p["query"], evidence, p["chosen"])
            f_rejected = self.feature_extractor.compute_features(p["query"], evidence, p["rejected"])

            feat_c_vec = [
                f_chosen["relevance"],
                f_chosen["citation_correctness"],
                f_chosen["groundedness"],
                f_chosen["completeness"],
                f_chosen["hallucination_penalty"],
                f_chosen["verbosity_penalty"]
            ]
            feat_r_vec = [
                f_rejected["relevance"],
                f_rejected["citation_correctness"],
                f_rejected["groundedness"],
                f_rejected["completeness"],
                f_rejected["hallucination_penalty"],
                f_rejected["verbosity_penalty"]
            ]

            self.samples.append({
                "query_emb": torch.tensor(q_vecs[idx], dtype=torch.float32),
                "chosen_emb": torch.tensor(c_vecs[idx], dtype=torch.float32),
                "rejected_emb": torch.tensor(r_vecs[idx], dtype=torch.float32),
                "chosen_feats": torch.tensor(feat_c_vec, dtype=torch.float32),
                "rejected_feats": torch.tensor(feat_r_vec, dtype=torch.float32),
                "weight": torch.tensor(p.get("weight", 1.0), dtype=torch.float32)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
