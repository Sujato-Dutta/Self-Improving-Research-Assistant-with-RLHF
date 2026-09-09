import math
import logging
from typing import Dict, Any, List
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, brier_score_loss
from src.reward.model import LightweightRewardModel
from src.reward.dataset import PreferencePairDataset

logger = logging.getLogger(__name__)


class RewardModelEvaluator:
    """Evaluates Reward Model on statistical accuracy, ROC-AUC, Brier score, and calibration."""

    def __init__(self, model: LightweightRewardModel, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def evaluate(self, dataset: PreferencePairDataset) -> Dict[str, float]:
        if len(dataset) == 0:
            return {
                "pairwise_accuracy": 0.0,
                "roc_auc": 0.5,
                "brier_score": 0.25,
                "ece": 0.0,
                "avg_reward_margin": 0.0
            }

        self.model.eval()
        margins = []
        probs = []
        labels = []  # 1 for chosen > rejected, 0 for reversed

        with torch.no_grad():
            for sample in dataset:
                q_emb = sample["query_emb"].unsqueeze(0).to(self.device)
                c_emb = sample["chosen_emb"].unsqueeze(0).to(self.device)
                r_emb = sample["rejected_emb"].unsqueeze(0).to(self.device)
                c_f = sample["chosen_feats"].unsqueeze(0).to(self.device)
                r_f = sample["rejected_feats"].unsqueeze(0).to(self.device)

                r_w = self.model(q_emb, c_emb, c_f).item()
                r_l = self.model(q_emb, r_emb, r_f).item()

                margin = r_w - r_l
                prob = 1.0 / (1.0 + math.exp(-margin))

                # Chosen vs rejected pair (positive)
                margins.append(margin)
                probs.append(prob)
                labels.append(1)

                # Symmetrical reverse pair (negative) for robust ROC-AUC evaluation
                rev_margin = r_l - r_w
                rev_prob = 1.0 / (1.0 + math.exp(-rev_margin))
                margins.append(rev_margin)
                probs.append(rev_prob)
                labels.append(0)

        # Pairwise accuracy on true chosen pairs
        true_chosen_margins = margins[0::2]
        pairwise_accuracy = float(np.mean([1.0 if m > 0 else 0.0 for m in true_chosen_margins]))
        avg_reward_margin = float(np.mean(true_chosen_margins))

        # ROC-AUC across symmetric evaluation pairs
        try:
            auc = float(roc_auc_score(labels, probs))
        except Exception:
            auc = 0.5

        # Brier Score (lower is better calibration: (p - y)^2)
        brier = float(brier_score_loss(labels, probs))

        # Expected Calibration Error (ECE) with 10 bins
        ece = self._compute_ece(probs, labels, n_bins=10)

        results = {
            "pairwise_accuracy": pairwise_accuracy,
            "roc_auc": auc,
            "brier_score": brier,
            "ece": ece,
            "avg_reward_margin": avg_reward_margin
        }
        logger.info(f"RM Evaluation | Acc: {pairwise_accuracy:.3f} | ROC-AUC: {auc:.3f} | Brier: {brier:.3f} | ECE: {ece:.3f}")
        return results

    def _compute_ece(self, probs: List[float], labels: List[int], n_bins: int = 10) -> float:
        probs = np.array(probs)
        labels = np.array(labels)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(labels[in_bin])
                avg_confidence_in_bin = np.mean(probs[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece)
