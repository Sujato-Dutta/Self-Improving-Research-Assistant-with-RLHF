from datetime import datetime
import json
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey
)
from sqlalchemy.orm import relationship
from src.db.database import Base


class QueryRecord(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    session_id = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    evidences = relationship("RetrievedEvidence", back_populates="query", cascade="all, delete-orphan")
    responses = relationship("ResponseRecord", back_populates="query", cascade="all, delete-orphan")
    preference_pairs = relationship("PreferencePairRecord", back_populates="query", cascade="all, delete-orphan")


class RetrievedEvidence(Base):
    __tablename__ = "retrieved_evidence"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("queries.id"), nullable=False, index=True)
    citation_index = Column(Integer, nullable=False)  # 1, 2, 3...
    paper_title = Column(String(512), nullable=False)
    arxiv_id = Column(String(64), nullable=True)
    authors = Column(String(512), nullable=True)
    excerpt = Column(Text, nullable=False)
    similarity_score = Column(Float, nullable=False)
    url = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    query = relationship("QueryRecord", back_populates="evidences")


class ResponseRecord(Base):
    __tablename__ = "responses"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("queries.id"), nullable=False, index=True)
    model_version = Column(String(64), default="base", nullable=False, index=True)
    variant = Column(String(16), default="single")  # 'A', 'B', or 'single'
    response_text = Column(Text, nullable=False)
    citations_json = Column(Text, default="[]")  # JSON list of cited indices
    latency_ms = Column(Float, default=0.0)
    reward_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    query = relationship("QueryRecord", back_populates="responses")
    feedbacks = relationship("FeedbackRecord", back_populates="response", cascade="all, delete-orphan")

    def get_citations(self):
        try:
            return json.loads(self.citations_json)
        except Exception:
            return []


class FeedbackRecord(Base):
    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    response_id = Column(Integer, ForeignKey("responses.id"), nullable=False, index=True)
    thumbs = Column(Integer, default=0)  # +1, -1, 0
    rating = Column(Integer, nullable=True)  # 1 to 5
    citation_accepted = Column(Boolean, nullable=True)
    regenerated = Column(Boolean, default=False)
    user_correction = Column(Text, nullable=True)
    task_success = Column(Boolean, nullable=True)
    feedback_metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    response = relationship("ResponseRecord", back_populates="feedbacks")


class PreferencePairRecord(Base):
    __tablename__ = "preference_pairs"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("queries.id"), nullable=False, index=True)
    response_a_id = Column(Integer, ForeignKey("responses.id"), nullable=False)
    response_b_id = Column(Integer, ForeignKey("responses.id"), nullable=False)
    preferred = Column(String(8), nullable=False)  # 'A' or 'B'
    evidence_context = Column(Text, nullable=False)
    weight = Column(Float, default=1.0)
    source = Column(String(64), default="user_ab_test")  # 'user_ab_test', 'heuristic_implicit', 'benchmark'
    metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    query = relationship("QueryRecord", back_populates="preference_pairs")
    response_a = relationship("ResponseRecord", foreign_keys=[response_a_id])
    response_b = relationship("ResponseRecord", foreign_keys=[response_b_id])


class RLHFRoundRecord(Base):
    __tablename__ = "rlhf_rounds"

    id = Column(Integer, primary_key=True, index=True)
    round_number = Column(Integer, nullable=False)
    round_name = Column(String(64), nullable=False)  # 'Base', 'RLHF Round 1', 'Round 2', 'Final'
    base_checkpoint = Column(String(256), nullable=False)
    promoted_checkpoint = Column(String(256), nullable=True)
    num_samples = Column(Integer, default=0)
    reward_loss = Column(Float, nullable=True)
    avg_reward_before = Column(Float, nullable=True)
    avg_reward_after = Column(Float, nullable=True)
    gate_passed = Column(Boolean, default=False)
    metrics_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class BenchmarkMetricRecord(Base):
    __tablename__ = "benchmark_metrics"

    id = Column(Integer, primary_key=True, index=True)
    round_name = Column(String(64), nullable=False, index=True)
    benchmark_size = Column(Integer, nullable=False)
    win_rate = Column(Float, nullable=False)
    win_rate_ci_lower = Column(Float, nullable=False)
    win_rate_ci_upper = Column(Float, nullable=False)
    avg_reward = Column(Float, nullable=False)
    avg_reward_ci_lower = Column(Float, nullable=False)
    avg_reward_ci_upper = Column(Float, nullable=False)
    citation_accuracy = Column(Float, nullable=False)
    citation_accuracy_ci_lower = Column(Float, nullable=False)
    citation_accuracy_ci_upper = Column(Float, nullable=False)
    groundedness = Column(Float, nullable=False)
    groundedness_ci_lower = Column(Float, nullable=False)
    groundedness_ci_upper = Column(Float, nullable=False)
    hallucination_rate = Column(Float, nullable=False)
    recall_at_k = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False)
    failure_distribution_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
