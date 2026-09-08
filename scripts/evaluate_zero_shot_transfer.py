import argparse
import json
import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def compute_transfer_metrics(records: list) -> dict:
    """
    Computes comprehensive zero-shot transfer and robustness metrics
    stratified by generator and length bracket.
    """
    total = len(records)
    if total == 0:
        return {}

    correct = sum(1 for r in records if r["label"] == r["prediction"])
    overall_accuracy = round(correct / total, 4)

    # By generator
    generators = {}
    for r in records:
        gen = r.get("generator", "unknown")
        if gen not in generators:
            generators[gen] = {"total": 0, "correct": 0, "false_positives": 0, "false_negatives": 0}
        generators[gen]["total"] += 1
        if r["label"] == r["prediction"]:
            generators[gen]["correct"] += 1
        elif r["label"] == 0 and r["prediction"] == 1:
            generators[gen]["false_positives"] += 1
        elif r["label"] == 1 and r["prediction"] == 0:
            generators[gen]["false_negatives"] += 1

    for gen, stats in generators.items():
        stats["accuracy"] = round(stats["correct"] / max(1, stats["total"]), 4)

    # By length bracket
    length_brackets = {"short": [], "medium": [], "long": []}
    for r in records:
        bracket = r.get("length_bracket", "medium")
        if bracket in length_brackets:
            length_brackets[bracket].append(r)

    stratified_length = {}
    for bracket, items in length_brackets.items():
        if items:
            b_total = len(items)
            b_correct = sum(1 for x in items if x["label"] == x["prediction"])
            b_fp = sum(1 for x in items if x["label"] == 0 and x["prediction"] == 1)
            b_fn = sum(1 for x in items if x["label"] == 1 and x["prediction"] == 0)
            stratified_length[bracket] = {
                "total": b_total,
                "accuracy": round(b_correct / b_total, 4),
                "false_positives": b_fp,
                "false_negatives": b_fn
            }
        else:
            stratified_length[bracket] = {"total": 0, "accuracy": 0.0, "false_positives": 0, "false_negatives": 0}

    return {
        "total_samples": total,
        "overall_accuracy": overall_accuracy,
        "generators": generators,
        "length_stratification": stratified_length
    }

def format_markdown_table(metrics: dict) -> str:
    """Formats the metrics into a publication-ready Markdown table."""
    lines = [
        "### Zero-Shot Cross-Generator Transfer & Robustness Summary",
        "",
        "| Generator | Category | Total Samples | Accuracy (%) | False Positives | False Negatives |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    for gen, stats in metrics.get("generators", {}).items():
        cat = "Human (Real)" if gen == "human" else "AI (Synthetic)"
        acc_pct = f"{stats['accuracy'] * 100:.2f}%"
        lines.append(f"| `{gen}` | {cat} | {stats['total']} | {acc_pct} | {stats['false_positives']} | {stats['false_negatives']} |")

    lines.extend([
        "",
        "#### Length Stratification (RAID Protocol)",
        "| Length Bracket | Character Boundary | Total Samples | Accuracy (%) | False Positives |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ])
    boundaries = {"short": "≤ 60 chars", "medium": "61–85 chars", "long": "> 85 chars"}
    for b, stats in metrics.get("length_stratification", {}).items():
        acc_pct = f"{stats['accuracy'] * 100:.2f}%"
        lines.append(f"| `{b.capitalize()}` | {boundaries.get(b, '-')} | {stats['total']} | {acc_pct} | {stats['false_positives']} |")

    return "\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", type=str, required=True, help="JSON file with predictions")
    parser.add_argument("--output_file", type=str, default="data/zero_shot_transfer_results.json")
    args = parser.parse_args()

    with open(args.predictions_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = compute_transfer_metrics(data)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(format_markdown_table(results))
    print(f"\nSaved metrics to {args.output_file}")
