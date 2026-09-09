import json
import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from src.db.models import FeedbackRecord, ResponseRecord, QueryRecord

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Collects and stores multi-modal human feedback signals."""

    def __init__(self, db_session: Session):
        self.db = db_session

    def record_feedback(
        self,
        response_id: int,
        thumbs: int = 0,                         # +1 (up), -1 (down), 0 (neutral)
        rating: Optional[int] = None,             # 1 to 5
        citation_accepted: Optional[bool] = None, # True / False
        regenerated: bool = False,               # True if user clicked regenerate
        user_correction: Optional[str] = None,   # User text edit / correction
        task_success: Optional[bool] = None,     # True / False
        metadata: Optional[Dict[str, Any]] = None
    ) -> FeedbackRecord:
        feedback = FeedbackRecord(
            response_id=response_id,
            thumbs=thumbs,
            rating=rating,
            citation_accepted=citation_accepted,
            regenerated=regenerated,
            user_correction=user_correction,
            task_success=task_success,
            feedback_metadata_json=json.dumps(metadata or {})
        )
        self.db.add(feedback)
        self.db.commit()
        self.db.refresh(feedback)
        logger.info(f"Recorded feedback ID {feedback.id} for response {response_id}")
        return feedback

    def get_feedback_history(self, limit: int = 100):
        return self.db.query(FeedbackRecord).order_by(FeedbackRecord.created_at.desc()).limit(limit).all()
