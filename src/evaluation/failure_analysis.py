import logging
from typing import List, Dict, Any
from collections import Counter

logger = logging.getLogger(__name__)

FAILURE_MODES = [
    "RETRIEVAL_FAILURE",
    "CITATION_HALLUCINATION",
    "UNSUPPORTED_CLAIMS",
    "SYNTHESIS_DRIFT",
    "VERBOSITY_ANOMALY"
]


class FailureAnalyzer:
    """Classifies model responses into a taxonomy of academic synthesis failure modes."""

    def diagnose_response(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        response_text: str,
        citations: List[int],
        features: Dict[str, float]
    ) -> List[str]:
        failures = []

        # 1. Retrieval Failure: no evidence or top similarity too weak
        if not evidence or max([e.get("similarity_score", 0.0) for e in evidence], default=0.0) < 0.25:
            failures.append("RETRIEVAL_FAILURE")

        # 2. Citation Hallucination: cited invalid index or missing citations when evidence exists
        num_evidence = len(evidence)
        if not citations and num_evidence > 0:
            failures.append("CITATION_HALLUCINATION")
        elif any(c < 1 or c > num_evidence for c in citations):
            failures.append("CITATION_HALLUCINATION")

        # 3. Unsupported Claims: low groundedness or high hallucination penalty
        if features.get("hallucination_penalty", 0.0) > 0.45 or features.get("groundedness", 1.0) < 0.22:
            failures.append("UNSUPPORTED_CLAIMS")

        # 4. Synthesis Drift: response drifted from query intent
        if features.get("relevance", 1.0) < 0.35 or features.get("completeness", 1.0) < 0.25:
            failures.append("SYNTHESIS_DRIFT")

        # 5. Verbosity Anomaly
        words = len(response_text.split())
        if words < 30 or words > 480:
            failures.append("VERBOSITY_ANOMALY")

        return failures

    def aggregate_failures(self, diagnostics: List[List[str]]) -> Dict[str, Any]:
        flat = [f for d in diagnostics for f in d]
        total_evals = max(1, len(diagnostics))
        counts = Counter(flat)

        summary = {}
        for mode in FAILURE_MODES:
            c = counts.get(mode, 0)
            summary[mode] = {
                "count": c,
                "rate": round(c / total_evals, 4),
                "percentage": round((c / total_evals) * 100.0, 2)
            }

        total_clean = sum(1 for d in diagnostics if len(d) == 0)
        summary["CLEAN_SUCCESS"] = {
            "count": total_clean,
            "rate": round(total_clean / total_evals, 4),
            "percentage": round((total_clean / total_evals) * 100.0, 2)
        }
        return summary
