import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_api_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "active_version" in data
    assert "rounds" in data


def test_api_papers():
    response = client.get("/api/papers")
    assert response.status_code == 200
    data = response.json()
    assert "papers" in data


def test_api_query():
    response = client.post("/api/query", json={
        "query": "How does attention replace recurrence in Transformers?",
        "dual_response": False
    })
    assert response.status_code == 200
    data = response.json()
    assert "response_text" in data
    assert "query_id" in data
    assert "citations" in data


def test_api_feedback():
    # First query to obtain a response_id
    q_res = client.post("/api/query", json={"query": "Explain LoRA"})
    resp_id = q_res.json()["response_id"]

    fb_res = client.post("/api/feedback", json={
        "response_id": resp_id,
        "thumbs": 1,
        "rating": 5,
        "citation_accepted": True,
        "task_success": True
    })
    assert fb_res.status_code == 200
    assert fb_res.json()["status"] == "success"
