import pytest
from src.evaluation.benchmark import BenchmarkManager
from src.evaluation.bootstrap import bootstrap_ci, bootstrap_paired_win_rate
from src.evaluation.failure_analysis import FailureAnalyzer


def test_benchmark_generation(tmp_path):
    bm = BenchmarkManager(file_path=tmp_path / "bench.json")
    qs = bm.get_questions(limit=50)
    assert len(qs) == 50
    assert "question" in qs[0]
    assert "category" in qs[0]


def test_bootstrap_ci():
    values = [0.80, 0.85, 0.90, 0.75, 0.88, 0.82, 0.79, 0.84, 0.86, 0.81]
    res = bootstrap_ci(values, num_resamples=500)
    assert res["ci_lower"] <= res["mean"] <= res["ci_upper"]
    assert 0.70 < res["mean"] < 0.95


def test_bootstrap_paired_win_rate():
    cand = [0.9, 0.8, 0.7, 0.85, 0.95]
    base = [0.5, 0.6, 0.7, 0.40, 0.50]
    res = bootstrap_paired_win_rate(cand, base, num_resamples=500)
    assert res["mean"] > 0.5


def test_failure_analyzer():
    fa = FailureAnalyzer()
    diag = fa.diagnose_response(
        query="What is attention?",
        evidence=[],  # Empty evidence -> retrieval failure
        response_text="Short answer",  # Short -> verbosity anomaly
        citations=[],
        features={"relevance": 0.5, "hallucination_penalty": 0.1, "groundedness": 0.5}
    )
    assert "RETRIEVAL_FAILURE" in diag
    assert "VERBOSITY_ANOMALY" in diag

    agg = fa.aggregate_failures([diag, ["UNSUPPORTED_CLAIMS"]])
    assert "RETRIEVAL_FAILURE" in agg
    assert agg["RETRIEVAL_FAILURE"]["count"] == 1
