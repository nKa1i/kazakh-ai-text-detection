import pytest
from llama_engine import LlamaEngine

def test_generate_mock(mocker):
    engine = LlamaEngine(api_url="http://localhost:8080/v1/chat/completions")
    mock_response = {
        "choices": [{
            "message": {
                "content": "Hello world"
            }
        }]
    }
    mocker.patch("requests.post", return_value=mocker.Mock(status_code=200, json=lambda: mock_response))
    
    res = engine.generate("Say hello")
    assert res == "Hello world"

def test_extract_json_mock(mocker):
    engine = LlamaEngine(api_url="http://localhost:8080/v1/chat/completions")
    mock_response = {
        "choices": [{
            "message": {
                "content": '{"entity_name": "Kaspi Shop", "claimed_features": ["delivery"]}'
            }
        }]
    }
    mocker.patch("requests.post", return_value=mocker.Mock(status_code=200, json=lambda: mock_response))
    
    res = engine.extract_json("Extract features from text")
    assert isinstance(res, dict)
    assert res["entity_name"] == "Kaspi Shop"
    assert "delivery" in res["claimed_features"]

def test_extract_json_markdown_block(mocker):
    engine = LlamaEngine(api_url="http://localhost:8080/v1/chat/completions")
    mock_response = {
        "choices": [{
            "message": {
                "content": '```json\n{"entity_name": "Kaspi Shop", "claimed_features": ["delivery"]}\n```'
            }
        }]
    }
    mocker.patch("requests.post", return_value=mocker.Mock(status_code=200, json=lambda: mock_response))
    
    res = engine.extract_json("Extract features from text")
    assert isinstance(res, dict)
    assert res["entity_name"] == "Kaspi Shop"
    assert "delivery" in res["claimed_features"]
