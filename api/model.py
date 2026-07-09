import os
import sys
import re
import unicodedata
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer
import fst_analyzer

# Determine model paths
DATA_DIR = "D:/roberta/data"
DEFAULT_PURE_PATH = os.path.join(DATA_DIR, "pure_native_KazRoBERTa")
DEFAULT_FST_PATH = os.path.join(DATA_DIR, "fst_native_KazRoBERTa")

# Check environment variables, then local D: drive paths, then fallback paths
MODEL_PATH_PURE = os.environ.get("MODEL_PATH_PURE", os.environ.get("MODEL_PATH", "/model_pure" if os.path.exists("/model_pure") else "/model"))
if not os.path.exists(MODEL_PATH_PURE) and os.path.exists(DEFAULT_PURE_PATH):
    MODEL_PATH_PURE = DEFAULT_PURE_PATH

MODEL_PATH_FST = os.environ.get("MODEL_PATH_FST", "/model_fst" if os.path.exists("/model_fst") else "/model")
if not os.path.exists(MODEL_PATH_FST) and os.path.exists(DEFAULT_FST_PATH):
    MODEL_PATH_FST = DEFAULT_FST_PATH

LABEL_MAP = {0: "human", 1: "ai"}

# Initialize models and explainers
try:
    print(f"Loading Pure model from {MODEL_PATH_PURE}...")
    pure_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_PURE)
    pure_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH_PURE)
    pure_model.config.id2label = {0: "human", 1: "ai"}
    pure_model.config.label2id = {"human": 0, "ai": 1}
    pure_model.eval()
    pure_explainer = SequenceClassificationExplainer(pure_model, pure_tokenizer)
    print("Pure model loaded successfully.")
except Exception as e:
    print(f"Failed to load Pure model: {e}")
    sys.exit(1)

try:
    print(f"Loading FST model from {MODEL_PATH_FST}...")
    fst_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_FST)
    fst_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH_FST)
    fst_model.config.id2label = {0: "human", 1: "ai"}
    fst_model.config.label2id = {"human": 0, "ai": 1}
    fst_model.eval()
    fst_explainer = SequenceClassificationExplainer(fst_model, fst_tokenizer)
    print("FST model loaded successfully.")
except Exception as e:
    print(f"Failed to load FST model: {e}")
    # Don't exit here if FST fails, fallback to Pure
    print("Continuing with Pure model only.")
    fst_model = None
    fst_tokenizer = None
    fst_explainer = None


def predict(text: str, mode: str = "pure") -> dict:
    """
    Tokenizes text, runs inference, returns label + confidence.
    """
    # Pick active model and tokenizer
    active_model = pure_model
    active_tokenizer = pure_tokenizer
    
    if mode == "fst" and fst_model is not None:
        active_model = fst_model
        active_tokenizer = fst_tokenizer
        # Preprocess text with FST morphological segmentation
        text = fst_analyzer.analyze_and_segment(text)

    inputs = active_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = active_model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    predicted_class_id = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class_id].item()

    return {
        "label": LABEL_MAP.get(predicted_class_id, "unknown"),
        "confidence": confidence
    }


def explain(text: str, mode: str = "pure") -> str:
    """
    Returns an HTML string highlighting tokens green (human signal)
    or red (AI signal) based on integrated gradient attributions.
    """
    active_explainer = pure_explainer
    active_tokenizer = pure_tokenizer
    
    if mode == "fst" and fst_explainer is not None:
        active_explainer = fst_explainer
        active_tokenizer = fst_tokenizer
        # Preprocess text with FST morphological segmentation
        text = fst_analyzer.analyze_and_segment(text)

    word_attributions = active_explainer(text)

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
        word = active_tokenizer.convert_tokens_to_string([word]).strip()
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
            adjusted = score if active_explainer.predicted_class_index == 0 else -score
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
        decoded = active_tokenizer.convert_tokens_to_string([word]).strip()
        if not decoded or abs(score) < threshold:
            continue
        if all(ord(c) in (0xFFFD, 0xFE0F, 0x200D, 0x200B, 0x200C) or
               unicodedata.category(c) in ('Cc', 'Cf') for c in decoded):
            continue
        adjusted = score if active_explainer.predicted_class_index == 0 else -score
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
