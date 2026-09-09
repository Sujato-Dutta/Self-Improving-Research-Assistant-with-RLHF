import json
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from src.db.models import QueryRecord, ResponseRecord, FeedbackRecord, PreferencePairRecord

logger = logging.getLogger(__name__)


class PreferencePairConverter:
    """Converts multi-modal interaction trajectories and feedback into pairwise preference tuples."""

    def __init__(self, db_session: Session):
        self.db = db_session

    def record_ab_preference(
        self,
        query_id: int,
        response_a_id: int,
        response_b_id: int,
        preferred: str,  # 'A' or 'B'
        evidence_context: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> PreferencePairRecord:
        """Explicit A/B preference choice."""
        if preferred not in ("A", "B"):
            raise ValueError("Preferred must be 'A' or 'B'")

        pair = PreferencePairRecord(
            query_id=query_id,
            response_a_id=response_a_id,
            response_b_id=response_b_id,
            preferred=preferred,
            evidence_context=evidence_context,
            weight=1.0,
            source="user_ab_test",
            metadata_json=json.dumps(metadata or {})
        )
        self.db.add(pair)
        self.db.commit()
        self.db.refresh(pair)
        logger.info(f"Created explicit A/B preference pair ID {pair.id}")
        return pair

    def convert_implicit_feedback_to_pairs(self) -> List[PreferencePairRecord]:
        """Scans queries that have multiple responses and converts implicit feedback signals into preference pairs."""
        queries = self.db.query(QueryRecord).all()
        created_pairs = []

        for q in queries:
            responses = self.db.query(ResponseRecord).filter(ResponseRecord.query_id == q.id).all()
            if len(responses) < 2:
                continue

            # Compare pairs of responses
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    resp_a = responses[i]
                    resp_b = responses[j]

                    score_a = self._compute_implicit_score(resp_a)
                    score_b = self._compute_implicit_score(resp_b)

                    # Only create pair if there is a discernible score gap
                    score_diff = score_a - score_b
                    if abs(score_diff) < 0.5:
                        continue

                    preferred = "A" if score_diff > 0 else "B"
                    confidence_weight = min(1.0, max(0.4, abs(score_diff) / 3.0))

                    # Check if this pair already exists
                    existing = self.db.query(PreferencePairRecord).filter(
                        PreferencePairRecord.query_id == q.id,
                        PreferencePairRecord.response_a_id == resp_a.id,
                        PreferencePairRecord.response_b_id == resp_b.id
                    ).first()
                    if existing:
                        continue

                    # Evidence context
                    ev_texts = [e.excerpt for e in q.evidences]
                    evidence_context = "\n".join(ev_texts)

                    meta = {
                        "score_a": score_a,
                        "score_b": score_b,
                        "score_diff": score_diff
                    }

                    pair = PreferencePairRecord(
                        query_id=q.id,
                        response_a_id=resp_a.id,
                        response_b_id=resp_b.id,
                        preferred=preferred,
                        evidence_context=evidence_context,
                        weight=confidence_weight,
                        source="implicit_signals",
                        metadata_json=json.dumps(meta)
                    )
                    self.db.add(pair)
                    created_pairs.append(pair)

        if created_pairs:
            self.db.commit()
            logger.info(f"Converted {len(created_pairs)} implicit feedback pairs.")
        return created_pairs

    def _compute_implicit_score(self, response: ResponseRecord) -> float:
        """Calculates a composite scalar score from all feedback modalities for a response."""
        feedbacks = response.feedbacks
        if not feedbacks:
            return 0.0

        score = 0.0
        for fb in feedbacks:
            # 1. Thumbs (+1 / -1)
            score += fb.thumbs * 1.5

            # 2. 1-5 Star Rating (normalized centered around 3)
            if fb.rating is not None:
                score += (fb.rating - 3.0) * 0.8

            # 3. Citation acceptance (+1.0 / -1.0)
            if fb.citation_accepted is True:
                score += 1.0
            elif fb.citation_accepted is False:
                score -= 1.2

            # 4. Regeneration (implies user was unsatisfied with current response)
            if fb.regenerated:
                score -= 1.0

            # 5. Task success
            if fb.task_success is True:
                score += 1.5
            elif fb.task_success is False:
                score -= 1.5

            # 6. User correction provided (implies imperfection, but high-signal)
            if fb.user_correction:
                score -= 0.5

        return score

    def export_preference_dataset(self) -> List[Dict[str, Any]]:
        """Exports all preference pairs in clean Bradley-Terry format for Reward Model & RLHF training."""
        pairs = self.db.query(PreferencePairRecord).all()
        dataset = []

        for p in pairs:
            chosen = p.response_a.response_text if p.preferred == "A" else p.response_b.response_text
            rejected = p.response_b.response_text if p.preferred == "A" else p.response_a.response_text

            dataset.append({
                "query": p.query.query_text,
                "chosen": chosen,
                "rejected": rejected,
                "evidence": p.evidence_context,
                "weight": p.weight,
                "source": p.source,
                "pair_id": p.id
            })
        return dataset
