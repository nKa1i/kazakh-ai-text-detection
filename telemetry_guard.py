from typing import Dict, Any, List

class TelemetryGuard:
    """
    Layer 2 Telemetry Guard evaluating client submission dynamics
    (pasting, typing speed, submission latency, editing behavior).
    """
    def __init__(self):
        pass

    def evaluate(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        is_pasted = telemetry.get("is_pasted", False)
        wpm = telemetry.get("typing_speed_wpm", 40.0)
        latency = telemetry.get("submission_latency_sec", 10.0)
        backspace_ratio = telemetry.get("backspace_ratio", 0.05)

        flags: List[str] = []
        score = 0.10  # Baseline human probability

        if is_pasted:
            flags.append("INSTANT_PASTE_DETECTED")
            score += 0.45

        if wpm > 150.0:
            flags.append("BOT_TYPING_VELOCITY")
            score += 0.35

        if latency < 2.0:
            flags.append("ANOMALOUS_LOW_LATENCY")
            score += 0.20

        if backspace_ratio == 0.0 and len(flags) > 0:
            score += 0.10

        final_score = min(1.0, score)
        return {
            "telemetry_risk_score": round(final_score, 2),
            "flags": flags
        }
