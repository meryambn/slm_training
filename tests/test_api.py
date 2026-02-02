import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["model"] == "qwen2.5:7b-instruct"

def test_traces_endpoint():
    response = client.get("/traces/latest")
    assert response.status_code == 200
    data = response.json()
    assert "traces" in data
