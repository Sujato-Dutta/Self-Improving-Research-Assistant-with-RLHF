import logging
from typing import Dict, Any, Tuple
from src.config import config
from src.rlhf.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


class EvaluationGate:
    """Automated gatekeeper executing regression tests before promoting RLHF checkpoints."""

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        min_win_rate: float = config.gate.min_win_rate_vs_base,
        min_reward_delta: float = config.gate.min_reward_delta,
        max_hallucination_rate: float = config.gate.max_hallucination_rate,
        min_citation_accuracy: float = config.gate.min_citation_accuracy
    ):
        self.checkpoint_manager = checkpoint_manager
        self.min_win_rate = min_win_rate
        self.min_reward_delta = min_reward_delta
        self.max_hallucination_rate = max_hallucination_rate
        self.min_citation_accuracy = min_citation_accuracy

    def evaluate_candidate(
        self,
        round_name: str,
        candidate_metrics: Dict[str, float],
        base_metrics: Dict[str, float]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluates whether the candidate checkpoint passes all regression gates against Base."""
        win_rate = candidate_metrics.get("win_rate", 0.5)
        candidate_reward = candidate_metrics.get("avg_reward", 0.0)
        base_reward = base_metrics.get("avg_reward", 0.0)
        reward_delta = candidate_reward - base_reward
        hallucination_rate = candidate_metrics.get("hallucination_rate", 1.0)
        citation_acc = candidate_metrics.get("citation_accuracy", 0.0)

        checks = {
            "win_rate_gate": {
                "metric": "Win Rate vs Base",
                "value": win_rate,
                "threshold": f">= {self.min_win_rate:.2f}",
                "passed": win_rate >= self.min_win_rate
            },
            "reward_improvement_gate": {
                "metric": "Reward Delta",
                "value": reward_delta,
                "threshold": f">= +{self.min_reward_delta:.2f}",
                "passed": reward_delta >= self.min_reward_delta
            },
            "hallucination_gate": {
                "metric": "Hallucination Rate",
                "value": hallucination_rate,
                "threshold": f"<= {self.max_hallucination_rate:.2f}",
                "passed": hallucination_rate <= self.max_hallucination_rate
            },
            "citation_accuracy_gate": {
                "metric": "Citation Accuracy",
                "value": citation_acc,
                "threshold": f">= {self.min_citation_accuracy:.2f}",
                "passed": citation_acc >= self.min_citation_accuracy
            }
        }

        all_passed = all(c["passed"] for c in checks.values())

        report = {
            "round_name": round_name,
            "gate_passed": all_passed,
            "checks": checks,
            "candidate_metrics": candidate_metrics,
            "base_metrics": base_metrics
        }

        if all_passed:
            logger.info(f"GATE PASSED: Checkpoint for {round_name} promoted successfully!")
            self.checkpoint_manager.promote_checkpoint(round_name, candidate_metrics)
        else:
            failed_reasons = [f"{k} failed (value={v['value']:.3f}, expected {v['threshold']})" for k, v in checks.items() if not v["passed"]]
            reason_str = "; ".join(failed_reasons)
            logger.warning(f"GATE FAILED for {round_name}: {reason_str}")
            self.checkpoint_manager.reject_checkpoint(round_name, reason_str)

        return all_passed, report
