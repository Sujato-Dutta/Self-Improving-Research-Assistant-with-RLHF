import numpy as np
from typing import List, Dict, Tuple, Union


def bootstrap_ci(
    values: Union[List[float], np.ndarray],
    num_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, float]:
    """Computes empirical percentile bootstrap confidence intervals."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

    if len(arr) == 1:
        val = float(arr[0])
        return {"mean": val, "ci_lower": val, "ci_upper": val, "std": 0.0}

    rng = np.random.default_rng(random_seed)
    n = len(arr)
    # Generate B bootstrap sample means
    resample_indices = rng.integers(0, n, size=(num_resamples, n))
    bootstrap_means = arr[resample_indices].mean(axis=1)

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(bootstrap_means, lower_pct))
    ci_upper = float(np.percentile(bootstrap_means, upper_pct))
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))

    return {
        "mean": mean_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": std_val
    }


def bootstrap_paired_win_rate(
    candidate_scores: List[float],
    baseline_scores: List[float],
    num_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, float]:
    """Computes empirical bootstrap confidence interval for paired win rate (candidate > baseline)."""
    cand = np.asarray(candidate_scores, dtype=np.float64)
    base = np.asarray(baseline_scores, dtype=np.float64)
    assert len(cand) == len(base), "Candidate and baseline score arrays must have identical length"

    # 1.0 for win, 0.5 for tie, 0.0 for loss
    wins = np.where(cand > base, 1.0, np.where(cand == base, 0.5, 0.0))
    return bootstrap_ci(wins, num_resamples=num_resamples, confidence_level=confidence_level, random_seed=random_seed)
