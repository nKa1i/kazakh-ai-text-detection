import random
from typing import Dict, Any, List

def generate_synthetic_telemetry(
    num_samples: int = 100,
    bot_ratio: float = 0.5,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generates a synthetic dataset containing human and bot behavior telemetry records.
    """
    random.seed(seed)
    dataset = []
    
    num_bots = int(num_samples * bot_ratio)
    num_humans = num_samples - num_bots

    for _ in range(num_humans):
        is_pasted = random.random() < 0.05  # Humans occasionally paste text
        typing_speed = random.uniform(30.0, 85.0)
        submission_latency = random.uniform(8.0, 180.0)
        backspace_ratio = random.uniform(0.04, 0.15)

        dataset.append({
            "is_bot": False,
            "is_pasted": is_pasted,
            "typing_speed_wpm": round(typing_speed, 1),
            "submission_latency_sec": round(submission_latency, 1),
            "backspace_ratio": round(backspace_ratio, 3)
        })

    for _ in range(num_bots):
        is_pasted = random.random() < 0.90  # Bots almost always instant paste
        typing_speed = random.uniform(250.0, 900.0)
        submission_latency = random.uniform(0.1, 1.8)
        backspace_ratio = random.uniform(0.0, 0.01)

        dataset.append({
            "is_bot": True,
            "is_pasted": is_pasted,
            "typing_speed_wpm": round(typing_speed, 1),
            "submission_latency_sec": round(submission_latency, 1),
            "backspace_ratio": round(backspace_ratio, 3)
        })

    random.shuffle(dataset)
    return dataset
