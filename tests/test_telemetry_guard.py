import pytest
import unittest

try:
    from telemetry_guard import TelemetryGuard
    from telemetry_synthesizer import generate_synthetic_telemetry
except ImportError:
    raise unittest.SkipTest("telemetry_guard not installed")

def test_telemetry_evaluation():
    guard = TelemetryGuard()
    
    # Test human behavior (normal speed, no paste, edits made)
    human_telemetry = {
        "is_pasted": False,
        "typing_speed_wpm": 45.0,
        "submission_latency_sec": 15.0,
        "backspace_ratio": 0.08
    }
    res_human = guard.evaluate(human_telemetry)
    assert res_human["telemetry_risk_score"] < 0.35
    assert len(res_human["flags"]) == 0

    # Test bot behavior (instant paste, zero latency)
    bot_telemetry = {
        "is_pasted": True,
        "typing_speed_wpm": 450.0,
        "submission_latency_sec": 0.5,
        "backspace_ratio": 0.0
    }
    res_bot = guard.evaluate(bot_telemetry)
    assert res_bot["telemetry_risk_score"] > 0.70
    assert "INSTANT_PASTE_DETECTED" in res_bot["flags"]

def test_telemetry_synthesizer():
    dataset = generate_synthetic_telemetry(num_samples=10, bot_ratio=0.5)
    assert len(dataset) == 10
    human_samples = [d for d in dataset if not d["is_bot"]]
    bot_samples = [d for d in dataset if d["is_bot"]]
    assert len(human_samples) > 0
    assert len(bot_samples) > 0
    for s in dataset:
        assert "is_pasted" in s
        assert "typing_speed_wpm" in s
        assert "submission_latency_sec" in s
        assert "backspace_ratio" in s
