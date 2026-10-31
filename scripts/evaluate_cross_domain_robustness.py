# -*- coding: utf-8 -*-
"""
scripts/evaluate_cross_domain_robustness.py: Phase 4 Cross-Domain Multi-Genre
Robustness Benchmark Evaluation Suite.

Evaluates Stage 1 (Binary AI Detection at calibrated threshold tau = 0.9980) and
Stage 2 (5-Family Closed-Set Attribution & Unseen Calque OOD Detection) across
5 Kazakh genres:
  - news
  - academic
  - official_legal
  - fiction_literature
  - social_casual

Outputs:
  - JSON artifact: data/cross_domain_evaluation_results.json
  - Markdown/LaTeX report: docs/reports/2026-10-31-cross-domain-benchmark-report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.domain_calibrator import (
    DOMAIN_DISPLAY_LABELS,
    SUPPORTED_DOMAINS,
    DomainCalibrator,
)
from models.generator_profiler import KNOWN_GENERATOR_FAMILIES
from models.heuristic_detector import OfflineHeuristicDetector
from scripts.build_cross_domain_benchmark import (
    UNSEEN_GENERATOR,
    build_cross_domain_benchmark_dataset,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cross-domain multi-genre robustness across 5 Kazakh functional styles."
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "cross_domain_benchmark.json"),
        help="Path to the 5-genre cross-domain benchmark JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "cross_domain_evaluation_results.json"),
        help="Destination path for the evaluation results JSON.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "docs", "reports", "2026-10-31-cross-domain-benchmark-report.md"
        ),
        help="Destination path for the Markdown/LaTeX benchmark report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate 2 records per domain (1 human + 1 AI) for fast CI/unit verification.",
    )
    return parser.parse_args(argv)


def _select_records_for_run(
    records: List[Dict[str, Any]],
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    if not dry_run:
        return list(records)

    subset: List[Dict[str, Any]] = []
    for dom in SUPPORTED_DOMAINS:
        dom_recs = [r for r in records if r.get("domain") == dom]
        first_human = next((r for r in dom_recs if int(r.get("label", 0)) == 0), None)
        first_ai = next((r for r in dom_recs if int(r.get("label", 0)) == 1), None)
        if first_human is not None:
            subset.append(first_human)
        if first_ai is not None:
            subset.append(first_ai)
    return subset


def evaluate_cross_domain_benchmark(
    records: List[Dict[str, Any]],
    detector: Optional[OfflineHeuristicDetector] = None,
    calibrator: Optional[DomainCalibrator] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Runs batched document inference and computes per-domain and aggregate metrics.
    """
    active_detector = detector or OfflineHeuristicDetector(calibrated_threshold=0.9980)
    active_calibrator = calibrator or DomainCalibrator(default_threshold=0.9980, max_safe_fpr=0.05)
    eval_records = _select_records_for_run(records, dry_run=dry_run)

    per_domain_raw: Dict[str, List[Dict[str, Any]]] = {d: [] for d in SUPPORTED_DOMAINS}
    all_y_true: List[int] = []
    all_y_scores: List[float] = []
    all_known_total = 0
    all_known_correct = 0
    all_unseen_total = 0
    all_unseen_hits = 0

    for rec in eval_records:
        dom = rec["domain"]
        label = int(rec["label"])
        gen = str(rec["generator"])
        doc_res = active_detector.predict_document(rec["text"])
        prob = float(doc_res.document_ai_probability)
        prof = doc_res.generator_profile or {}

        per_domain_raw[dom].append(
            {
                "label": label,
                "score": prob,
                "generator": gen,
                "predicted_family": prof.get("predicted_family", ""),
                "is_unseen_generator": bool(prof.get("is_unseen_generator", False)),
            }
        )
        all_y_true.append(label)
        all_y_scores.append(prob)

    per_domain_results: Dict[str, Dict[str, Any]] = {}
    for dom in SUPPORTED_DOMAINS:
        items = per_domain_raw[dom]
        y_true = [it["label"] for it in items]
        y_scores = [it["score"] for it in items]

        dom_metrics = active_calibrator.evaluate_domain_predictions(dom, y_true, y_scores)
        curve = active_calibrator.compute_threshold_stability_curve(y_true, y_scores)

        known_items = [it for it in items if it["generator"] in KNOWN_GENERATOR_FAMILIES]
        known_correct = sum(
            1
            for it in known_items
            if it["predicted_family"] == it["generator"] and not it["is_unseen_generator"]
        )
        attr_acc = round(known_correct / float(max(1, len(known_items))), 4) if known_items else 1.0

        unseen_items = [it for it in items if it["generator"] == UNSEEN_GENERATOR]
        unseen_hits = sum(1 for it in unseen_items if it["is_unseen_generator"])
        unseen_recall = round(unseen_hits / float(max(1, len(unseen_items))), 4) if unseen_items else 1.0

        all_known_total += len(known_items)
        all_known_correct += known_correct
        all_unseen_total += len(unseen_items)
        all_unseen_hits += unseen_hits

        dom_dict = dom_metrics.to_dict()
        dom_dict["genre_label"] = DOMAIN_DISPLAY_LABELS[dom]["en"]
        dom_dict["num_records"] = len(items)
        dom_dict["attribution_accuracy"] = attr_acc
        dom_dict["unseen_ood_recall"] = unseen_recall
        dom_dict["transfer_gap_auc"] = 0.0
        dom_dict["threshold_stability_curve"] = [pt.to_dict() for pt in curve]
        per_domain_results[dom] = dom_dict

    news_auc = per_domain_results["news"]["roc_auc"]
    for dom in SUPPORTED_DOMAINS:
        per_domain_results[dom]["transfer_gap_auc"] = round(
            float(news_auc - per_domain_results[dom]["roc_auc"]), 4
        )

    agg_metrics = active_calibrator.evaluate_domain_predictions("aggregate", all_y_true, all_y_scores)
    agg_curve = active_calibrator.compute_threshold_stability_curve(all_y_true, all_y_scores)
    agg_dict = agg_metrics.to_dict()
    agg_dict["attribution_accuracy"] = round(
        all_known_correct / float(max(1, all_known_total)), 4
    )
    agg_dict["unseen_ood_recall"] = round(
        all_unseen_hits / float(max(1, all_unseen_total)), 4
    )
    agg_dict["max_transfer_gap_auc"] = round(
        max(abs(per_domain_results[d]["transfer_gap_auc"]) for d in SUPPORTED_DOMAINS), 4
    )
    agg_dict["threshold_stability_curve"] = [pt.to_dict() for pt in agg_curve]

    return {
        "benchmark_version": "phase4_v1",
        "calibrated_threshold": round(float(active_detector.calibrated_threshold), 4),
        "dry_run": bool(dry_run),
        "total_evaluated": len(eval_records),
        "per_domain": per_domain_results,
        "aggregate": agg_dict,
    }


def render_cross_domain_markdown_report(results: Dict[str, Any]) -> str:
    """
    Renders a publication-grade Markdown and LaTeX benchmark report (zero emojis).
    """
    tau = results.get("calibrated_threshold", 0.9980)
    total_n = results.get("total_evaluated", 500)
    per_dom = results.get("per_domain", {})
    agg = results.get("aggregate", {})

    lines = [
        "# Phase 4: Cross-Domain Multi-Genre Robustness Benchmark Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Evaluated Documents:** {total_n} across 5 functional Kazakh genres (`news`, `academic`, `official_legal`, `fiction_literature`, `social_casual`)",
        f"- **Global Calibrated Threshold:** `tau = {tau:.4f}`",
        f"- **Aggregate ROC-AUC:** `{agg.get('roc_auc', 0.0):.4f}`",
        f"- **Aggregate FPR@95%TPR:** `{agg.get('fpr_at_95_tpr', 0.0):.4f}`",
        f"- **Aggregate Macro-F1 (at tau = {tau:.4f}):** `{agg.get('f1_score', 0.0):.4f}`",
        f"- **Closed-Set 5-Family Attribution Accuracy:** `{agg.get('attribution_accuracy', 0.0) * 100.0:.2f}%`",
        f"- **Unseen Calque Generator OOD Recall:** `{agg.get('unseen_ood_recall', 0.0) * 100.0:.2f}%`",
        "",
        "---",
        "",
        "## 2. Per-Domain Robustness & Transfer Gap Table",
        "",
        "| Domain Key | Genre Name | N | ROC-AUC | FPR@95%TPR | Macro-F1 | Accuracy | Transfer Gap vs News | Attribution Acc | Unseen OOD Recall |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for dom in SUPPORTED_DOMAINS:
        m = per_dom.get(dom, {})
        lines.append(
            f"| `{dom}` | {m.get('genre_label', dom)} | {m.get('num_records', 0)} | "
            f"{m.get('roc_auc', 0.0):.4f} | {m.get('fpr_at_95_tpr', 0.0):.4f} | "
            f"{m.get('f1_score', 0.0):.4f} | {m.get('accuracy', 0.0):.4f} | "
            f"{m.get('transfer_gap_auc', 0.0):+.4f} | {m.get('attribution_accuracy', 0.0) * 100.0:.1f}% | "
            f"{m.get('unseen_ood_recall', 0.0) * 100.0:.1f}% |"
        )

    lines.extend(
        [
            f"| **Aggregate** | **All 5 Genres** | **{total_n}** | "
            f"**{agg.get('roc_auc', 0.0):.4f}** | **{agg.get('fpr_at_95_tpr', 0.0):.4f}** | "
            f"**{agg.get('f1_score', 0.0):.4f}** | **{agg.get('accuracy', 0.0):.4f}** | "
            f"**{agg.get('max_transfer_gap_auc', 0.0):.4f} (max)** | "
            f"**{agg.get('attribution_accuracy', 0.0) * 100.0:.1f}%** | "
            f"**{agg.get('unseen_ood_recall', 0.0) * 100.0:.1f}%** |",
            "",
            "---",
            "",
            "## 3. Threshold Calibration Stability Curve (`tau in [0.900, 0.999]`)",
            "",
            "| Threshold (`tau`) | Aggregate FPR | Aggregate TPR | Aggregate Macro-F1 |",
            "| :---: | :---: | :---: | :---: |",
        ]
    )

    for pt in agg.get("threshold_stability_curve", []):
        lines.append(
            f"| `{pt['threshold']:.4f}` | {pt['fpr']:.4f} | {pt['tpr']:.4f} | {pt['f1_score']:.4f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. LaTeX Table Artifact for Manuscript Inclusion",
            "",
            "```latex",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Cross-Domain Multi-Genre Robustness Benchmark Across Five Kazakh Functional Styles ($\\tau = 0.9980$).}",
            "\\label{tab:phase4_cross_domain_robustness}",
            "\\begin{tabular}{lcccccc}",
            "\\hline",
            "\\textbf{Genre / Domain} & \\textbf{ROC-AUC} & \\textbf{FPR@95\\%TPR} & \\textbf{Macro-F1} & \\textbf{Acc.} & \\textbf{$\\Delta$ AUC vs. News} & \\textbf{Unseen OOD Recall} \\\\",
            "\\hline",
        ]
    )

    for dom in SUPPORTED_DOMAINS:
        m = per_dom.get(dom, {})
        lines.append(
            f"{m.get('genre_label', dom)} & {m.get('roc_auc', 0.0):.4f} & "
            f"{m.get('fpr_at_95_tpr', 0.0):.4f} & {m.get('f1_score', 0.0):.4f} & "
            f"{m.get('accuracy', 0.0):.4f} & {m.get('transfer_gap_auc', 0.0):+.4f} & "
            f"{m.get('unseen_ood_recall', 0.0) * 100.0:.1f}\\% \\\\"
        )

    lines.extend(
        [
            "\\hline",
            f"\\textbf{{Aggregate / Mean}} & \\textbf{{{agg.get('roc_auc', 0.0):.4f}}} & "
            f"\\textbf{{{agg.get('fpr_at_95_tpr', 0.0):.4f}}} & \\textbf{{{agg.get('f1_score', 0.0):.4f}}} & "
            f"\\textbf{{{agg.get('accuracy', 0.0):.4f}}} & \\textbf{{{agg.get('max_transfer_gap_auc', 0.0):.4f}}} & "
            f"\\textbf{{{agg.get('unseen_ood_recall', 0.0) * 100.0:.1f}\\%}} \\\\",
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def run_cross_domain_evaluation(
    benchmark_path: str = os.path.join(PROJECT_ROOT, "data", "cross_domain_benchmark.json"),
    output_json: str = os.path.join(PROJECT_ROOT, "data", "cross_domain_evaluation_results.json"),
    output_report: str = os.path.join(
        PROJECT_ROOT, "docs", "reports", "2026-10-31-cross-domain-benchmark-report.md"
    ),
    dry_run: bool = False,
) -> Dict[str, Any]:
    if not os.path.isfile(benchmark_path):
        build_cross_domain_benchmark_dataset(output_path=benchmark_path)

    with open(benchmark_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    results = evaluate_cross_domain_benchmark(records, dry_run=dry_run)
    report_md = render_cross_domain_markdown_report(results)

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(output_report)), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        f.write(report_md)

    return results


if __name__ == "__main__":
    cli_args = parse_args()
    summary = run_cross_domain_evaluation(
        benchmark_path=cli_args.benchmark_path,
        output_json=cli_args.output_json,
        output_report=cli_args.output_report,
        dry_run=cli_args.dry_run,
    )
    print(
        json.dumps(
            {
                "total_evaluated": summary["total_evaluated"],
                "dry_run": summary["dry_run"],
                "aggregate_roc_auc": summary["aggregate"]["roc_auc"],
                "aggregate_fpr_at_95_tpr": summary["aggregate"]["fpr_at_95_tpr"],
                "aggregate_f1": summary["aggregate"]["f1_score"],
            },
            indent=2,
        )
    )
