import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = os.environ.get("MODEL_PATH", "/model")

try:
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model from {MODEL_PATH}: {e}")
    sys.exit(1)

def predict(text: str) -> dict:
    """
    Tokenizes text, runs inference, and returns {"label": "human"|"ai", "confidence": float}
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)[0]

    # Assuming label 0 is human, label 1 is ai (or vice versa - need to dynamically check id2label or assume standard binary)
    # Often standard id2label: {0: "human", 1: "ai"} or similar. Let's use config.
    predicted_class_id = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class_id].item()

    # We will use the model's id2label if available, otherwise assume 0=human, 1=ai
    label = model.config.id2label.get(predicted_class_id, "human" if predicted_class_id == 0 else "ai")

    return {
        "label": label.lower(),
        "confidence": confidence
    }