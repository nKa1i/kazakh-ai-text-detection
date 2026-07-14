import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_verify_endpoint():
    payload = {
        "text": "Каспи маған өте ұнайды",
        "telemetry": {
            "is_pasted": False,
            "typing_speed_wpm": 45.0,
            "submission_latency_sec": 12.0,
            "backspace_ratio": 0.05
        }
    }
    response = client.post("/v1/verify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "overall_risk_score" in data
    assert "action_recommended" in data
    assert "breakdown" in data
    assert "is_authentic" in data
