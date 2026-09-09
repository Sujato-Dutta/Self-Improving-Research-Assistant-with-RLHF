import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.db.database import Base
from src.db.models import QueryRecord, ResponseRecord
from src.feedback.collector import FeedbackCollector
from src.feedback.pair_converter import PreferencePairConverter


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_feedback_collector(test_db):
    q = QueryRecord(query_text="What is DPO?")
    test_db.add(q)
    test_db.commit()

    resp = ResponseRecord(query_id=q.id, model_version="Base", response_text="DPO optimizes policy directly.")
    test_db.add(resp)
    test_db.commit()

    collector = FeedbackCollector(test_db)
    fb = collector.record_feedback(
        response_id=resp.id,
        thumbs=1,
        rating=5,
        citation_accepted=True,
        task_success=True,
        user_correction="Good summary."
    )
    assert fb.id is not None
    assert fb.rating == 5
    assert fb.citation_accepted is True


def test_pair_converter(test_db):
    q = QueryRecord(query_text="Explain LoRA")
    test_db.add(q)
    test_db.commit()

    resp_a = ResponseRecord(query_id=q.id, model_version="Base", variant="A", response_text="LoRA freezes pre-trained weights.")
    resp_b = ResponseRecord(query_id=q.id, model_version="Base", variant="B", response_text="LoRA changes all weights.")
    test_db.add(resp_a)
    test_db.add(resp_b)
    test_db.commit()

    converter = PreferencePairConverter(test_db)
    pair = converter.record_ab_preference(
        query_id=q.id,
        response_a_id=resp_a.id,
        response_b_id=resp_b.id,
        preferred="A",
        evidence_context="LoRA paper excerpt"
    )
    assert pair.id is not None
    assert pair.preferred == "A"

    dataset = converter.export_preference_dataset()
    assert len(dataset) == 1
    assert dataset[0]["chosen"] == resp_a.response_text
    assert dataset[0]["rejected"] == resp_b.response_text
