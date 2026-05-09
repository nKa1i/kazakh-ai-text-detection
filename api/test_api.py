import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# We need to mock the model import in main.py to prevent it from trying to load a real model
with patch.dict('sys.modules', {
    'model': type('MockModel', (), {
        'predict': lambda text: {'label': 'ai', 'confidence': 0.95}
    })
}):
    from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_success():
    response = client.post("/predict", json={"text": "This is a test text."})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["human", "ai"]
    assert 0 <= data["confidence"] <= 1
    assert data == {"label": "ai", "confidence": 0.95}

def test_predict_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

def test_predict_whitespace_text():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422
