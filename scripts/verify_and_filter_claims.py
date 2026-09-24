# scripts/verify_and_filter_claims.py
# -*- coding: utf-8 -*-
"""
Kazakh-FEVER Claim Verification and Annotation Quality Tooling.
Implements quality filtering, schema enforcement, Cohen's Kappa calculation,
and stratified sampling for human annotation packages.
"""

import os
import json
import random
import collections
from typing import List, Dict, Tuple, Any

VALID_LABELS = {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}


def filter_and_clean_claims(
    claims: List[Dict[str, Any]],
    min_tokens: int = 8,
    max_tokens: int = 35
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Filter claims by token length, label legality, and evidence completeness."""
    valid_claims = []
    stats = {
        "total_input": len(claims),
        "valid_count": 0,
        "rejected_token_length": 0,
        "rejected_label": 0,
        "rejected_empty_evidence": 0
    }

    for c in claims:
        claim_text = str(c.get("claim", "")).strip()
        label = str(c.get("label", "")).strip().upper()
        ev = c.get("evidence_sentences") or []

        tokens = claim_text.split()
        if len(tokens) < min_tokens or len(tokens) > max_tokens:
            stats["rejected_token_length"] += 1
            continue

        if label not in VALID_LABELS:
            stats["rejected_label"] += 1
            continue

        # In FEVER, SUPPORTS and REFUTES must have evidence sentences
        if label in {"SUPPORTS", "REFUTES"} and not ev:
            stats["rejected_empty_evidence"] += 1
            continue

        cleaned = dict(c)
        cleaned["label"] = label
        if label == "NOT_ENOUGH_INFO":
            cleaned["evidence_sentences"] = []
        else:
            cleaned["evidence_sentences"] = list(ev)

        valid_claims.append(cleaned)
        stats["valid_count"] += 1

    return valid_claims, stats


def compute_cohens_kappa(rater_a: List[str], rater_b: List[str]) -> Dict[str, float]:
    """Calculate Cohen's Kappa inter-annotator agreement between two raters."""
    if len(rater_a) != len(rater_b) or len(rater_a) == 0:
        raise ValueError("Rater lists must be non-empty and of identical length.")

    n = len(rater_a)
    unique_labels = sorted(list(set(
        [a.strip().upper() for a in rater_a] +
        [b.strip().upper() for b in rater_b]
    )))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    # Build confusion matrix
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    agree_count = 0
    for a, b in zip(rater_a, rater_b):
        a_clean = a.strip().upper()
        b_clean = b.strip().upper()
        if a_clean == b_clean:
            agree_count += 1
        matrix[label_to_idx[a_clean]][label_to_idx[b_clean]] += 1

    p_o = agree_count / n

    # Marginal probabilities
    p_e = 0.0
    for i in range(num_classes):
        row_sum = sum(matrix[i][j] for j in range(num_classes))
        col_sum = sum(matrix[j][i] for j in range(num_classes))
        p_e += (row_sum / n) * (col_sum / n)

    if abs(1.0 - p_e) < 1e-12:
        kappa = 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return {
        "sample_size": float(n),
        "observed_agreement": round(p_o, 4),
        "expected_chance_agreement": round(p_e, 4),
        "kappa": round(kappa, 4)
    }


def export_human_annotation_package(
    claims: List[Dict[str, Any]],
    sample_size: int = 300,
    output_path: str = "data/human_annotation_sample.jsonl"
) -> None:
    """Export balanced subset for human evaluation."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if len(claims) <= sample_size:
        selected = list(claims)
    else:
        # Group by label
        by_label: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for c in claims:
            label = str(c.get("label", "")).strip().upper()
            by_label[label].append(c)

        # Distribute sample_size as evenly as possible across labels
        labels = sorted(list(by_label.keys()))
        target_per_label = sample_size // len(labels)
        selected = []
        remaining_pool = []

        # First pass: take up to target_per_label from each
        for label in labels:
            items = list(by_label[label])
            random.shuffle(items)
            take_count = min(len(items), target_per_label)
            selected.extend(items[:take_count])
            remaining_pool.extend(items[take_count:])

        # Second pass: fill remaining slots from remaining pool if needed
        needed = sample_size - len(selected)
        if needed > 0 and remaining_pool:
            random.shuffle(remaining_pool)
            selected.extend(remaining_pool[:needed])

    random.shuffle(selected)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/kazakh_fever_3k_generated.jsonl"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/kazakh_fever_3k_cleaned.jsonl"

    if os.path.exists(input_file):
        claims_data = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    claims_data.append(json.loads(line))

        valid_claims, stats = filter_and_clean_claims(claims_data)
        print("Filtering Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        with open(output_file, "w", encoding="utf-8") as f:
            for c in valid_claims:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"Wrote {len(valid_claims)} cleaned claims to {output_file}")
    else:
        print(f"Input file {input_file} not found; skipping batch execution.")
