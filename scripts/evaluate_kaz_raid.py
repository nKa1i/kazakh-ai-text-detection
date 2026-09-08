"""
Kaz-RAID: Adversarial Robustness Benchmark Evaluation & Reporting Engine.

Computes:
1. Multi-tier degradation curves (ROC-AUC, Accuracy, EER) across 28 conditions.
2. Attack Success Rate (ASR) measuring adversarial evasion on AI-generated text.
3. 95% Bootstrap Confidence Intervals (B = 1,000 resamples).
4. RAID-compliant length stratification (Short <= 60, Medium 61-85, Long > 85).
5. Publication-ready Markdown & LaTeX evaluation tables for ACL/EMNLP.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.metrics_evaluator import compute_comprehensive_metrics


def compute_bootstrap_ci(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute 95% Bootstrap Confidence Intervals for ROC-AUC and Accuracy.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Continuous prediction probabilities in [0.0, 1.0].
        n_bootstraps: Number of bootstrap iterations (default: 1000).
        alpha: Significance level (default: 0.05 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with auc_mean, auc_ci_lower, auc_ci_upper, eer_mean, etc.
    """
    n = len(y_true)
    if n == 0:
        return {
            "auc_mean": 0.5, "auc_ci_lower": 0.5, "auc_ci_upper": 0.5,
            "eer_mean": 0.5, "eer_ci_lower": 0.5, "eer_ci_upper": 0.5
        }

    rng = random.Random(seed)
    auc_scores: List[float] = []
    eer_scores: List[float] = []

    for _ in range(n_bootstraps):
        sample_indices = [rng.randint(0, n - 1) for _ in range(n)]
        b_true = [y_true[i] for i in sample_indices]
        b_prob = [y_prob[i] for i in sample_indices]

        # Only evaluate if both classes are present in the bootstrap sample
        if len(set(b_true)) < 2:
            continue

        metrics = compute_comprehensive_metrics(b_true, b_prob)
        auc_scores.append(metrics["roc_auc"])
        eer_scores.append(metrics["eer"])

    if not auc_scores:
        base_metrics = compute_comprehensive_metrics(y_true, y_prob)
        auc_scores = [base_metrics["roc_auc"]]
        eer_scores = [base_metrics["eer"]]

    auc_scores.sort()
    eer_scores.sort()

    lower_idx = int(math.floor(len(auc_scores) * (alpha / 2.0)))
    upper_idx = int(math.ceil(len(auc_scores) * (1.0 - alpha / 2.0))) - 1
    upper_idx = min(upper_idx, len(auc_scores) - 1)

    return {
        "auc_mean": round(float(sum(auc_scores) / len(auc_scores)), 4),
        "auc_ci_lower": round(float(auc_scores[lower_idx]), 4),
        "auc_ci_upper": round(float(auc_scores[upper_idx]), 4),
        "eer_mean": round(float(sum(eer_scores) / len(eer_scores)), 4),
        "eer_ci_lower": round(float(eer_scores[lower_idx]), 4),
        "eer_ci_upper": round(float(eer_scores[upper_idx]), 4),
    }


def compute_attack_success_rate(
    y_true: Sequence[int],
    y_pred_clean: Sequence[int],
    y_pred_adv: Sequence[int]
) -> float:
    """
    Compute Attack Success Rate (ASR) for AI-evasion attacks:
        ASR = Fraction of AI texts detected as AI under clean conditions
              that are misclassified as human under adversarial perturbation.
    """
    ai_clean_detected = [
        i for i, (yt, yc) in enumerate(zip(y_true, y_pred_clean))
        if yt == 1 and yc == 1
    ]

    if not ai_clean_detected:
        # Fallback: simple False Negative Rate on all AI texts
        all_ai = [i for i, yt in enumerate(y_true) if yt == 1]
        if not all_ai:
            return 0.0
        evaded = sum(1 for i in all_ai if y_pred_adv[i] == 0)
        return round(float(evaded / len(all_ai)), 4)

    successful_evasions = sum(1 for i in ai_clean_detected if y_pred_adv[i] == 0)
    return round(float(successful_evasions / len(ai_clean_detected)), 4)


def evaluate_benchmark_results(
    y_true: Sequence[int],
    condition_predictions: Dict[str, Sequence[float]],
    lengths: Optional[Sequence[int]] = None,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluate detector performance across all Kaz-RAID conditions.

    Args:
        y_true: Ground truth binary labels (0=human, 1=ai).
        condition_predictions: Dict mapping condition name (e.g. 'clean', 'homoglyph_swap_rate_0.10')
                                to predicted probabilities for class 1.
        lengths: Optional character lengths for RAID stratification.
        n_bootstraps: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        Structured evaluation dict with condition metrics, degradation, and tier summaries.
    """
    if "clean" not in condition_predictions:
        raise ValueError("condition_predictions must contain a 'clean' condition baseline.")

    clean_probs = condition_predictions["clean"]
    clean_metrics = compute_comprehensive_metrics(y_true, clean_probs, lengths=lengths)
    calibrated_threshold = clean_metrics["optimal_threshold"]
    clean_preds = [1 if p >= calibrated_threshold else 0 for p in clean_probs]

    clean_ci = compute_bootstrap_ci(y_true, clean_probs, n_bootstraps=n_bootstraps, seed=seed)

    conditions_eval: Dict[str, Dict[str, Any]] = {}
    conditions_eval["clean"] = {
        **clean_metrics,
        **clean_ci,
        "asr": 0.0,
        "delta_auc": 0.0,
        "delta_acc": 0.0,
    }

    # Evaluate all perturbed conditions at the calibrated threshold
    for cond_name, adv_probs in condition_predictions.items():
        if cond_name == "clean":
            continue

        adv_metrics = compute_comprehensive_metrics(
            y_true,
            adv_probs,
            lengths=lengths,
            threshold=calibrated_threshold
        )
        adv_preds = [1 if p >= calibrated_threshold else 0 for p in adv_probs]
        asr = compute_attack_success_rate(y_true, clean_preds, adv_preds)
        adv_ci = compute_bootstrap_ci(y_true, adv_probs, n_bootstraps=n_bootstraps, seed=seed)

        delta_auc = round(clean_metrics["roc_auc"] - adv_metrics["roc_auc"], 4)
        delta_acc = round(clean_metrics["accuracy"] - adv_metrics["accuracy"], 4)

        conditions_eval[cond_name] = {
            **adv_metrics,
            **adv_ci,
            "asr": asr,
            "delta_auc": delta_auc,
            "delta_acc": delta_acc,
        }

    # Group and aggregate degradation by Tier
    tier_mapping = {
        "tier1_orthographic": ["homoglyph_swap", "keyboard_typo", "zero_width_injection"],
        "tier2_morphological": ["suffix_tamperer", "colloquial_contractor"],
        "tier3_lexical": ["loanword_swap", "discourse_particle"],
        "tier4_semantic": ["back_translation", "llm_paraphrase"],
    }

    tier_summary: Dict[str, Dict[str, float]] = {}
    for tier_name, ops in tier_mapping.items():
        tier_aucs = []
        tier_asrs = []
        tier_delta_aucs = []
        for cond_name, res in conditions_eval.items():
            if any(cond_name.startswith(op) for op in ops):
                tier_aucs.append(res["roc_auc"])
                tier_asrs.append(res["asr"])
                tier_delta_aucs.append(res["delta_auc"])

        if tier_aucs:
            tier_summary[tier_name] = {
                "mean_auc": round(float(sum(tier_aucs) / len(tier_aucs)), 4),
                "mean_asr": round(float(sum(tier_asrs) / len(tier_asrs)), 4),
                "mean_delta_auc": round(float(sum(tier_delta_aucs) / len(tier_delta_aucs)), 4),
            }

    overall_delta_auc = [
        v["delta_auc"] for k, v in conditions_eval.items() if k != "clean"
    ]
    mean_overall_degradation = (
        round(float(sum(overall_delta_auc) / len(overall_delta_auc)), 4)
        if overall_delta_auc else 0.0
    )

    return {
        "calibrated_threshold": calibrated_threshold,
        "conditions": conditions_eval,
        "degradation": {
            "mean_overall_delta_auc": mean_overall_degradation,
            "tier_summary": tier_summary
        }
    }


def generate_kaz_raid_markdown_table(benchmark_results: Dict[str, Any]) -> str:
    """
    Generate publication-ready Markdown table summarizing Kaz-RAID performance across tiers.
    """
    conditions = benchmark_results.get("conditions", {})
    clean = conditions.get("clean", {})

    lines = [
        "### Kaz-RAID Adversarial Robustness Benchmark Results",
        "",
        f"**Calibrated Clean Threshold ($\\tau^*$):** `{benchmark_results.get('calibrated_threshold', 0.5):.4f}`",
        "",
        "| Condition | Rate ($\\epsilon$) | ROC-AUC (95% CI) | EER (%) | Accuracy (%) | ASR (%) | $\\Delta$ AUC |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Clean Baseline** | 0.00 | **{clean.get('roc_auc', 0.0):.4f}** [{clean.get('auc_ci_lower', 0.0):.4f}, {clean.get('auc_ci_upper', 0.0):.4f}] | {clean.get('eer', 0.0)*100:.2f}% | {clean.get('accuracy', 0.0)*100:.2f}% | 0.00% | 0.0000 |"
    ]

    sorted_conds = sorted([k for k in conditions.keys() if k != "clean"])
    for cond in sorted_conds:
        m = conditions[cond]
        # Extract rate from condition name if possible
        rate_str = cond.split("_rate_")[-1] if "_rate_" in cond else "-"
        op_name = cond.split("_rate_")[0] if "_rate_" in cond else cond
        lines.append(
            f"| `{op_name}` | {rate_str} | {m.get('roc_auc', 0.0):.4f} [{m.get('auc_ci_lower', 0.0):.4f}, {m.get('auc_ci_upper', 0.0):.4f}] | "
            f"{m.get('eer', 0.0)*100:.2f}% | {m.get('accuracy', 0.0)*100:.2f}% | {m.get('asr', 0.0)*100:.2f}% | {m.get('delta_auc', 0.0):.4f} |"
        )

    lines.append("")
    # Add Tier Summary
    tier_summary = benchmark_results.get("degradation", {}).get("tier_summary", {})
    if tier_summary:
        lines.append("#### Tier-Level Summary")
        lines.append("| Tier | Mean ROC-AUC | Mean ASR (%) | Mean $\\Delta$ AUC |")
        lines.append("| :--- | :---: | :---: | :---: |")
        for tier, ts in tier_summary.items():
            lines.append(
                f"| `{tier}` | {ts.get('mean_auc', 0.0):.4f} | {ts.get('mean_asr', 0.0)*100:.2f}% | {ts.get('mean_delta_auc', 0.0):.4f} |"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Kaz-RAID Benchmark predictions.")
    parser.add_argument("--predictions_file", type=str, required=True, help="Path to predictions JSON")
    parser.add_argument("--labels_file", type=str, default="data/kazakh_aigc_paired_2k.json", help="Path to test records with labels")
    parser.add_argument("--output_file", type=str, default="data/kaz_raid_benchmark_results.json", help="Path to save evaluation results")
    parser.add_argument("--report_file", type=str, default="data/kaz_raid_paper_report.md", help="Path to save markdown report")
    parser.add_argument("--n_bootstraps", type=int, default=1000, help="Number of bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    with open(args.labels_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    y_true = [r["label"] for r in records]
    lengths = [r.get("char_length", len(r["text"])) for r in records]

    with open(args.predictions_file, "r", encoding="utf-8") as f:
        condition_predictions = json.load(f)

    print(f"Evaluating Kaz-RAID benchmark across {len(condition_predictions)} conditions...")
    results = evaluate_benchmark_results(
        y_true=y_true,
        condition_predictions=condition_predictions,
        lengths=lengths,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed
    )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    table_md = generate_kaz_raid_markdown_table(results)
    rep_path = Path(args.report_file)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(table_md)

    print(f"[Done] Evaluation completed. Results saved to {out_path} and report to {rep_path}")


if __name__ == "__main__":
    main()
