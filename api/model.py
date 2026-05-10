import os
import sys
import unicodedata
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer

MODEL_PATH = os.environ.get("MODEL_PATH", "/model")
LABEL_MAP = {0: "human", 1: "ai"}

try:
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.config.id2label = {0: "human", 1: "ai"}
    model.config.label2id = {"human": 0, "ai": 1}
    model.eval()
    explainer = SequenceClassificationExplainer(model, tokenizer)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model from {MODEL_PATH}: {e}")
    sys.exit(1)


def predict(text: str) -> dict:
    """
    Tokenizes text, runs inference, returns label + confidence.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    predicted_class_id = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class_id].item()

    return {
        "label": LABEL_MAP.get(predicted_class_id, "unknown"),
        "confidence": confidence
    }


def explain(text: str) -> str:
    """
    Returns an HTML string highlighting tokens green (human signal)
    or red (AI signal) based on integrated gradient attributions.
    """
    word_attributions = explainer(text)

    if not word_attributions:
        return "<p>No attribution data.</p>"

    # Normalize scores to [0, 1] for opacity
    scores = [abs(score) for _, score in word_attributions]
    max_score = max(scores) if max(scores) > 0 else 1.0

    # Only show tokens that meaningfully contributed (top 40% of max score)
    threshold = 0.4 * max_score

    spans = []
    for word, score in word_attributions:
        # Skip special tokens
        if word in ("[CLS]", "[SEP]", "<s>", "</s>", "[PAD]"):
            continue
        # Decode byte-level BPE tokens back to proper Unicode (Kazakh/Cyrillic)
        word = tokenizer.convert_tokens_to_string([word]).strip()
        if not word:
            continue
        # Skip replacement characters, variation selectors, zero-width chars
        if all(ord(c) in (0xFFFD, 0xFE0F, 0x200D, 0x200B, 0x200C) or
               unicodedata.category(c) in ('Cc', 'Cf') for c in word):
            continue
        intensity = min(abs(score) / max_score, 1.0)

        if abs(score) < threshold:
            # Low attribution — show as plain text, no highlight
            spans.append(f'<span style="padding: 1px 3px; margin: 1px; display: inline-block; font-size: 14px;">{word}</span>')
        else:
            # Flip score if prediction is AI so green always = human signal, red always = AI signal
            adjusted = score if explainer.predicted_class_index == 0 else -score
            if adjusted > 0:
                # Green = pushed toward HUMAN
                r, g, b = int(60 - 60 * intensity), int(180 * intensity + 60), int(60 - 60 * intensity)
            else:
                # Red = pushed toward AI
                r, g, b = int(180 * intensity + 60), int(60 - 60 * intensity), int(60 - 60 * intensity)

            alpha = 0.2 + 0.7 * intensity
            style = (
                f"background-color: rgba({r},{g},{b},{alpha:.2f}); "
                "border-radius: 3px; padding: 1px 3px; margin: 1px; "
                "display: inline-block; font-size: 14px;"
            )
            spans.append(f'<span style="{style}">{word}</span>')

    # Build key evidence list — only highlighted tokens, sorted by absolute influence
    evidence = []
    for word, score in word_attributions:
        decoded = tokenizer.convert_tokens_to_string([word]).strip()
        if not decoded or abs(score) < threshold:
            continue
        if all(ord(c) in (0xFFFD, 0xFE0F, 0x200D, 0x200B, 0x200C) or
               unicodedata.category(c) in ('Cc', 'Cf') for c in decoded):
            continue
        adjusted = score if explainer.predicted_class_index == 0 else -score
        evidence.append((decoded, adjusted, abs(score) / max_score))

    # Sort by absolute weight descending
    evidence.sort(key=lambda x: x[2], reverse=True)

    evidence_rows = ""
    for word, adjusted, weight in evidence:
        direction = "HUMAN" if adjusted > 0 else "AI"
        bar_color = "rgba(60,200,60,0.7)" if adjusted > 0 else "rgba(200,60,60,0.7)"
        bar_width = int(weight * 120)
        evidence_rows += (
            f'<tr>'
            f'<td style="padding: 3px 8px; font-size:13px;">{word}</td>'
            f'<td style="padding: 3px 8px; font-size:13px; color: {"#6f6" if adjusted > 0 else "#f66"};">{direction}</td>'
            f'<td style="padding: 3px 8px;">'
            f'<div style="width:{bar_width}px; height:10px; background:{bar_color}; border-radius:3px;"></div>'
            f'</td>'
            f'</tr>'
        )

    evidence_table = (
        '<table style="margin-top:10px; border-collapse:collapse; width:100%;">'
        '<tr><th style="text-align:left; padding:3px 8px; font-size:12px; opacity:0.6;">Token</th>'
        '<th style="text-align:left; padding:3px 8px; font-size:12px; opacity:0.6;">Signal</th>'
        '<th style="text-align:left; padding:3px 8px; font-size:12px; opacity:0.6;">Weight</th></tr>'
        + evidence_rows
        + '</table>'
    )

    html = (
        '<div style="line-height: 2.2; padding: 8px;">'
        + " ".join(spans)
        + '<hr style="opacity:0.2; margin:12px 0;">'
        + '<small style="opacity:0.6;">Key evidence (sorted by influence)</small>'
        + evidence_table
        + '</div>'
    )
    return html
