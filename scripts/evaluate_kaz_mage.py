"""
Kaz-MAGE: 2x2 Cross-Domain & Multi-Generator Benchmark Evaluation Suite.

Evaluates AI text detectors according to the ACL 2024 MAGE protocol:
- Q1: Seen Domain, Seen Generator (Kaspi Reviews x Sherkala-7B)
- Q2: Seen Domain, Unseen Generator (Kaspi Reviews x Qwen-2.5-7B)
- Q3: Unseen Domain, Seen Generator (News/Wiki x Sherkala-7B)
- Q4: Unseen Domain, Unseen Generator (News/Wiki x Qwen-2.5-7B)

Computes:
1. Quadrant-specific metrics: ROC-AUC, Accuracy, Youden's J optimal threshold, F1, Precision, Recall.
2. 95% Bootstrap Confidence Intervals (B = 1,000 resamples).
3. Degradation metrics:
   - Delta AUC_generator = AUC(Q1) - AUC(Q2)
   - Delta AUC_domain = AUC(Q1) - AUC(Q3)
   - Delta AUC_wild = AUC(Q1) - AUC(Q4)
4. Dynamic Gate Routing Shifts:
   - Mean gate value per domain (consumer_reviews, news, wikipedia)
   - Delta g_news = mean_g(news) - mean_g(reviews)
   - Delta g_wiki = mean_g(wiki) - mean_g(reviews)

Defensively designed with pure Python fallbacks for zero-dependency execution
in environments where NumPy, SciPy, or Scikit-Learn are not installed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Optional imports for hardware-accelerated / optimized environments
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Pure Python Metric Calculations
# ---------------------------------------------------------------------------

def _to_list(arr: Any) -> Optional[List[Any]]:
    """Convert sequence or array-like object to a Python list."""
    if arr is None:
        return None
    if hasattr(arr, "tolist"):
        return arr.tolist()
    if isinstance(arr, (list, tuple)):
        return list(arr)
    return list(arr)


def _compute_roc_and_youden_pure(
    y_true: Sequence[int],
    y_prob: Sequence[float]
) -> Tuple[float, float, float]:
    """
    Pure-Python calculation of ROC curve, ROC-AUC, Youden's J optimal threshold, and EER.
    Handles tied prediction probabilities according to standard ROC definition.
    """
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.5, 0.5

    # Sort descending by predicted probability
    pairs = sorted(zip(y_prob, y_true), key=lambda x: x[0], reverse=True)

    # Group ties
    distinct_groups = []
    for p, group in groupby(pairs, key=lambda x: x[0]):
        g = list(group)
        pos_in_g = sum(1 for _, y in g if y == 1)
        neg_in_g = len(g) - pos_in_g
        distinct_groups.append((p, pos_in_g, neg_in_g))

    thresholds = [distinct_groups[0][0] + 1.0]
    fprs = [0.0]
    tprs = [0.0]

    tp = 0
    fp = 0
    for p, pos_g, neg_g in distinct_groups:
        tp += pos_g
        fp += neg_g
        thresholds.append(p)
        fprs.append(fp / n_neg)
        tprs.append(tp / n_pos)

    # Trapezoidal rule for ROC-AUC
    auc = 0.0
    for i in range(1, len(fprs)):
        auc += (fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1]) / 2.0

    # Youden's J statistic: J = TPR - FPR
    j_scores = [t - f for t, f in zip(tprs, fprs)]
    best_idx = max(range(len(j_scores)), key=lambda i: j_scores[i])
    optimal_threshold = thresholds[best_idx]
    optimal_threshold = min(1.0, optimal_threshold)

    # Equal Error Rate (EER)
    eer_idx = min(range(len(fprs)), key=lambda i: abs(fprs[i] - (1.0 - tprs[i])))
    eer = fprs[eer_idx]

    return auc, optimal_threshold, eer


def _calc_classification_stats(
    y_true: List[int],
    y_prob: List[float],
    thresh: float
) -> Dict[str, float]:
    """Calculate accuracy, macro F1, binary F1, precision, recall at a given threshold."""
    y_pred = [1 if p >= thresh else 0 for p in y_prob]
    total = len(y_true)

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    acc = (tp + tn) / total if total > 0 else 0.0

    # Class 1 (AI)
    prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = (2 * prec_1 * rec_1) / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0

    # Class 0 (Human)
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = (2 * prec_0 * rec_0) / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0

    macro_f1 = (f1_0 + f1_1) / 2.0

    return {
        "accuracy": round(float(acc), 4),
        "f1": round(float(macro_f1), 4),
        "binary_f1": round(float(f1_1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "precision": round(float(prec_1), 4),
        "recall": round(float(rec_1), 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def compute_bootstrap_auc_ci(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Compute 95% Bootstrap Confidence Intervals for ROC-AUC.
    """
    y_true_list = [int(y) for y in y_true]
    y_prob_list = [float(p) for p in y_prob]
    n = len(y_true_list)

    if n < 2 or len(set(y_true_list)) < 2:
        return 0.5, 0.5

    rng = random.Random(seed)
    auc_scores: List[float] = []

    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        b_true = [y_true_list[i] for i in indices]
        if len(set(b_true)) < 2:
            continue
        b_prob = [y_prob_list[i] for i in indices]
        auc, _, _ = _compute_roc_and_youden_pure(b_true, b_prob)
        auc_scores.append(auc)

    if not auc_scores:
        base_auc, _, _ = _compute_roc_and_youden_pure(y_true_list, y_prob_list)
        return round(float(base_auc), 4), round(float(base_auc), 4)

    auc_scores.sort()
    lower_idx = int(math.floor(len(auc_scores) * (alpha / 2.0)))
    upper_idx = int(math.ceil(len(auc_scores) * (1.0 - alpha / 2.0))) - 1
    upper_idx = max(0, min(upper_idx, len(auc_scores) - 1))

    return round(float(auc_scores[lower_idx]), 4), round(float(auc_scores[upper_idx]), 4)


# ---------------------------------------------------------------------------
# Core Metric Evaluators
# ---------------------------------------------------------------------------

def compute_mage_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: Optional[float] = None,
    bootstrap_ci: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes MAGE evaluation metrics:
    - ROC-AUC with pure Python / sklearn fallback
    - Youden's J optimal threshold tau* = argmax(TPR - FPR)
    - Accuracy, Macro F1, Precision, Recall at tau* and at 0.5
    - 95% Bootstrap CI on ROC-AUC (ci_lower, ci_upper)

    Args:
        y_true: Ground truth binary labels (0 for Human, 1 for AI).
        y_prob: Prediction probabilities for class 1 in [0.0, 1.0].
        threshold: Optional fixed threshold. If None, optimal_threshold (tau*) is used.
        bootstrap_ci: Whether to compute bootstrap confidence intervals.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed for bootstrap reproducibility.

    Returns:
        Dictionary of computed metrics.
    """
    y_true_list = _to_list(y_true) or []
    y_prob_list = _to_list(y_prob) or []

    y_true_list = [int(y) for y in y_true_list]
    y_prob_list = [float(p) for p in y_prob_list]

    n_samples = len(y_true_list)
    if n_samples == 0:
        return {
            "roc_auc": 0.5,
            "optimal_threshold": 0.5,
            "threshold": 0.5,
            "accuracy": 0.0,
            "f1": 0.0,
            "macro_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "ci_lower": 0.5,
            "ci_upper": 0.5,
            "auc_ci_lower": 0.5,
            "auc_ci_upper": 0.5,
            "n_samples": 0
        }

    # 1. ROC curve, AUC, and Youden's J optimal threshold
    if HAS_SKLEARN and HAS_NUMPY and len(set(y_true_list)) > 1:
        try:
            yt_arr = np.asarray(y_true_list)
            yp_arr = np.asarray(y_prob_list)
            auc_val = float(roc_auc_score(yt_arr, yp_arr))
            fpr_curve, tpr_curve, thresh_curve = roc_curve(yt_arr, yp_arr)
            j_scores = tpr_curve - fpr_curve
            best_idx = int(np.argmax(j_scores))
            opt_thresh = float(thresh_curve[best_idx])
            if math.isinf(opt_thresh) or opt_thresh > 1.0:
                opt_thresh = 1.0
        except Exception:
            auc_val, opt_thresh, _ = _compute_roc_and_youden_pure(y_true_list, y_prob_list)
    else:
        auc_val, opt_thresh, _ = _compute_roc_and_youden_pure(y_true_list, y_prob_list)

    # 2. Determine effective threshold for primary accuracy and F1
    eff_thresh = float(opt_thresh) if threshold is None else float(threshold)

    # 3. Compute classification stats at effective threshold, optimal threshold, and 0.5
    stats_eff = _calc_classification_stats(y_true_list, y_prob_list, eff_thresh)
    stats_opt = _calc_classification_stats(y_true_list, y_prob_list, opt_thresh)
    stats_05 = _calc_classification_stats(y_true_list, y_prob_list, 0.5)

    # 4. Bootstrap 95% Confidence Intervals
    if bootstrap_ci and n_samples > 1 and len(set(y_true_list)) > 1:
        ci_lower, ci_upper = compute_bootstrap_auc_ci(
            y_true_list, y_prob_list, n_bootstrap=n_bootstrap, seed=seed
        )
    else:
        ci_lower, ci_upper = round(float(auc_val), 4), round(float(auc_val), 4)

    return {
        "roc_auc": round(float(auc_val), 4),
        "optimal_threshold": round(float(opt_thresh), 4),
        "threshold": round(float(eff_thresh), 4),
        "accuracy": stats_eff["accuracy"],
        "f1": stats_eff["f1"],
        "macro_f1": stats_eff["macro_f1"],
        "precision": stats_eff["precision"],
        "recall": stats_eff["recall"],
        "accuracy_optimal": stats_opt["accuracy"],
        "f1_optimal": stats_opt["f1"],
        "macro_f1_optimal": stats_opt["macro_f1"],
        "precision_optimal": stats_opt["precision"],
        "recall_optimal": stats_opt["recall"],
        "accuracy_05": stats_05["accuracy"],
        "f1_05": stats_05["f1"],
        "macro_f1_05": stats_05["macro_f1"],
        "precision_05": stats_05["precision"],
        "recall_05": stats_05["recall"],
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "auc_ci_lower": ci_lower,
        "auc_ci_upper": ci_upper,
        "tp": stats_eff["tp"],
        "tn": stats_eff["tn"],
        "fp": stats_eff["fp"],
        "fn": stats_eff["fn"],
        "n_samples": n_samples
    }


def calculate_domain_degradation(quadrant_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes ACL 2024 MAGE degradation metrics:
    - Delta AUC_generator = AUC(Q1) - AUC(Q2)
    - Delta AUC_domain = AUC(Q1) - AUC(Q3)
    - Delta AUC_wild = AUC(Q1) - AUC(Q4)

    Args:
        quadrant_results: Dict mapping quadrant names ("Q1", "Q2", "Q3", "Q4") to either
                          a metric dict containing "roc_auc" or a direct float AUC value.

    Returns:
        Dict with degradation deltas and quadrant AUCs.
    """
    def _extract_auc(val: Any) -> float:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict):
            return float(val.get("roc_auc", val.get("auc", 0.0)))
        return 0.0

    auc_q1 = _extract_auc(quadrant_results.get("Q1", 0.0))
    auc_q2 = _extract_auc(quadrant_results.get("Q2", 0.0))
    auc_q3 = _extract_auc(quadrant_results.get("Q3", 0.0))
    auc_q4 = _extract_auc(quadrant_results.get("Q4", 0.0))

    delta_generator = auc_q1 - auc_q2
    delta_domain = auc_q1 - auc_q3
    delta_wild = auc_q1 - auc_q4

    return {
        "delta_auc_generator": round(float(delta_generator), 4),
        "delta_auc_domain": round(float(delta_domain), 4),
        "delta_auc_wild": round(float(delta_wild), 4),
        "delta_auc_unseen_generator": round(float(delta_generator), 4),
        "delta_auc_unseen_domain": round(float(delta_domain), 4),
        "delta_auc_unseen_wild": round(float(delta_wild), 4),
        "q1_auc": round(float(auc_q1), 4),
        "q2_auc": round(float(auc_q2), 4),
        "q3_auc": round(float(auc_q3), 4),
        "q4_auc": round(float(auc_q4), 4),
    }


def _get_field(record: Any, field_name: str, default: Any = None) -> Any:
    """Extract a field from either a dict or an object instance."""
    if isinstance(record, dict):
        return record.get(field_name, default)
    return getattr(record, field_name, default)


def _get_label(record: Any) -> int:
    """Extract ground truth binary label from record."""
    val = _get_field(record, "y_true", _get_field(record, "label", 0))
    return int(val)


def _get_prob(record: Any) -> float:
    """Extract predicted AI probability from record."""
    val = _get_field(record, "y_prob", _get_field(record, "prob", _get_field(record, "score", _get_field(record, "prediction", 0.5))))
    return float(val)


def _get_domain(record: Any) -> str:
    """Extract domain from record."""
    return str(_get_field(record, "domain", ""))


def evaluate_quadrant_matrix(
    records: List[Union[Dict[str, Any], Any]],
    bootstrap_ci: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates the full 2x2 MAGE quadrant matrix, computes domain/generator/wild degradations,
    and analyzes dynamic gate routing shifts across domains.

    Args:
        records: List of evaluated records or sample dicts containing:
                 "quadrant", "domain", "y_true" (or "label"), "y_prob" (or "prob"),
                 and optionally "gate" (or "gate_value").
        bootstrap_ci: Whether to compute 95% bootstrap confidence intervals.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for bootstrap reproducibility.

    Returns:
        Structured evaluation dict with "quadrants", "degradation", "gate_dynamics",
        "domains", and summary stats.
    """
    quadrant_buckets: Dict[str, List[Any]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    domain_buckets: Dict[str, List[Any]] = {}
    domain_gates: Dict[str, List[float]] = {}
    quadrant_gates: Dict[str, List[float]] = {}
    human_controls: List[Any] = []

    for r in records:
        q = _get_field(r, "quadrant", "")
        dom = _get_domain(r)
        gate = _get_field(r, "gate", _get_field(r, "gate_value", None))
        label = _get_label(r)

        if label == 0:
            human_controls.append(r)

        if q in quadrant_buckets:
            quadrant_buckets[q].append(r)

        if dom:
            if dom not in domain_buckets:
                domain_buckets[dom] = []
            domain_buckets[dom].append(r)

            if gate is not None:
                if dom not in domain_gates:
                    domain_gates[dom] = []
                domain_gates[dom].append(float(gate))

        if q and gate is not None:
            if q not in quadrant_gates:
                quadrant_gates[q] = []
            quadrant_gates[q].append(float(gate))

    # If any quadrant lacks negative (human) samples, pair with human controls from matching domain
    # Q1 & Q2: consumer_reviews
    # Q3 & Q4: news & wikipedia
    for q_name in ["Q1", "Q2"]:
        q_recs = quadrant_buckets[q_name]
        has_negative = any(_get_label(r) == 0 for r in q_recs)
        if q_recs and not has_negative:
            domain_humans = [h for h in human_controls if _get_domain(h) in ("consumer_reviews", "reviews")]
            quadrant_buckets[q_name].extend(domain_humans)

    for q_name in ["Q3", "Q4"]:
        q_recs = quadrant_buckets[q_name]
        has_negative = any(_get_label(r) == 0 for r in q_recs)
        if q_recs and not has_negative:
            domain_humans = [h for h in human_controls if _get_domain(h) in ("news", "wikipedia", "wiki")]
            quadrant_buckets[q_name].extend(domain_humans)

    # 1. Evaluate each quadrant
    quadrant_results: Dict[str, Any] = {}
    for q_name in ["Q1", "Q2", "Q3", "Q4"]:
        q_recs = quadrant_buckets[q_name]
        if q_recs:
            yt = [_get_label(r) for r in q_recs]
            yp = [_get_prob(r) for r in q_recs]
            quadrant_results[q_name] = compute_mage_metrics(
                yt, yp, bootstrap_ci=bootstrap_ci, n_bootstrap=n_bootstrap, seed=seed
            )
        else:
            quadrant_results[q_name] = {
                "roc_auc": 0.0,
                "accuracy": 0.0,
                "f1": 0.0,
                "macro_f1": 0.0,
                "optimal_threshold": 0.5,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "n_samples": 0
            }

    # 2. Compute degradation metrics
    degradation = calculate_domain_degradation(quadrant_results)

    # 3. Dynamic Gate Routing Analysis
    mean_gate_by_domain: Dict[str, float] = {}
    for dom, g_vals in domain_gates.items():
        if g_vals:
            mean_gate_by_domain[dom] = round(float(sum(g_vals) / len(g_vals)), 4)

    mean_gate_by_quadrant: Dict[str, float] = {}
    for q_name, g_vals in quadrant_gates.items():
        if g_vals:
            mean_gate_by_quadrant[q_name] = round(float(sum(g_vals) / len(g_vals)), 4)

    # Reference baseline is consumer_reviews
    ref_dom = "consumer_reviews" if "consumer_reviews" in mean_gate_by_domain else (
        "reviews" if "reviews" in mean_gate_by_domain else None
    )
    ref_gate = mean_gate_by_domain.get(ref_dom) if ref_dom else None

    delta_gate_news = (
        round(float(mean_gate_by_domain["news"] - ref_gate), 4)
        if (ref_gate is not None and "news" in mean_gate_by_domain)
        else None
    )

    wiki_key = "wikipedia" if "wikipedia" in mean_gate_by_domain else (
        "wiki" if "wiki" in mean_gate_by_domain else None
    )
    delta_gate_wiki = (
        round(float(mean_gate_by_domain[wiki_key] - ref_gate), 4)
        if (ref_gate is not None and wiki_key)
        else None
    )

    gate_shifts: Dict[str, float] = {}
    if delta_gate_news is not None:
        gate_shifts["news"] = delta_gate_news
    if delta_gate_wiki is not None:
        gate_shifts["wikipedia"] = delta_gate_wiki

    gate_dynamics = {
        "mean_gate_by_domain": mean_gate_by_domain,
        "mean_gate_by_quadrant": mean_gate_by_quadrant,
        "delta_gate_news": delta_gate_news,
        "delta_gate_wiki": delta_gate_wiki,
        "delta_gate_wikipedia": delta_gate_wiki,
        "gate_shifts": gate_shifts,
        "sample_counts_by_domain": {dom: len(recs) for dom, recs in domain_buckets.items()}
    }

    # 4. Domain-level evaluations
    domain_results: Dict[str, Any] = {}
    for dom, dom_recs in domain_buckets.items():
        if dom_recs:
            yt = [_get_label(r) for r in dom_recs]
            yp = [_get_prob(r) for r in dom_recs]
            domain_results[dom] = compute_mage_metrics(
                yt, yp, bootstrap_ci=bootstrap_ci, n_bootstrap=n_bootstrap, seed=seed
            )

    return {
        "quadrants": quadrant_results,
        "degradation": degradation,
        "gate_dynamics": gate_dynamics,
        "domains": domain_results,
        "total_samples": len(records),
        "records": records
    }


def evaluate_kaz_mage_matrix(
    model: Any,
    tokenizer: Any = None,
    morpheme_tok: Any = None,
    dataset: Sequence[Any] = (),
    device: str = "cpu",
    bootstrap_ci: bool = True,
    n_bootstrap: int = 1000
) -> Dict[str, Any]:
    """
    Evaluates a PyTorch detector or hybrid model on the Kaz-MAGE benchmark dataset.
    Extracts predictions and dynamic gate values, then evaluates the full 2x2 matrix.

    Args:
        model: Trained detector model or callable.
        tokenizer: HuggingFace tokenizer.
        morpheme_tok: MorphemeTokenizer.
        dataset: List of KazMageSample objects or dicts.
        device: Torch device ("cuda" or "cpu").
        bootstrap_ci: Whether to compute bootstrap CI.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Full evaluation dictionary.
    """
    records: List[Dict[str, Any]] = []

    # If dataset is already a list of evaluated prediction dicts
    if dataset and isinstance(dataset[0], dict) and ("y_prob" in dataset[0] or "prob" in dataset[0]):
        return evaluate_quadrant_matrix(list(dataset), bootstrap_ci=bootstrap_ci, n_bootstrap=n_bootstrap)

    if model is None:
        raise ValueError("A model or mock model instance must be provided to evaluate_kaz_mage_matrix.")

    if hasattr(model, "eval") and callable(model.eval):
        model.eval()
    if hasattr(model, "to") and callable(model.to):
        model.to(device)

    _no_grad_ctx = (
        torch.no_grad
        if (HAS_TORCH and torch is not None and hasattr(torch, "no_grad"))
        else contextlib.nullcontext
    )

    for sample in dataset:
        text = str(_get_field(sample, "text", ""))
        label = _get_label(sample)
        quad = str(_get_field(sample, "quadrant", ""))
        dom = _get_domain(sample)

        prob = 0.5
        gate_val = None

        with _no_grad_ctx():
            predict_fn = None
            if hasattr(model, "predict_text") and callable(model.predict_text):
                predict_fn = model.predict_text
            elif hasattr(model, "predict") and callable(model.predict) and (
                (tokenizer is None and morpheme_tok is None) or not hasattr(model, "forward")
            ):
                predict_fn = model.predict

            if predict_fn is not None:
                res = predict_fn(text)
                if isinstance(res, dict):
                    raw_p = res.get("ai_probability", res.get("probability", 0.5))
                    prob = float(raw_p.item()) if hasattr(raw_p, "item") else float(raw_p)
                    raw_g = res.get("gate_value", res.get("gate_semantic_weight", None))
                    if raw_g is not None:
                        if hasattr(raw_g, "mean"):
                            mean_g = raw_g.mean()
                            if hasattr(mean_g, "cpu"):
                                mean_g = mean_g.cpu()
                            gate_val = float(mean_g.item()) if hasattr(mean_g, "item") else float(mean_g)
                        else:
                            gate_val = float(raw_g)
                else:
                    prob = float(res.item()) if hasattr(res, "item") else float(res)
            else:
                # Direct PyTorch forward execution
                inputs = {}
                if tokenizer is not None:
                    enc = tokenizer(text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
                    inputs["input_ids"] = enc["input_ids"].to(device)
                    inputs["attention_mask"] = enc["attention_mask"].to(device)
                if morpheme_tok is not None:
                    morph_ids = morpheme_tok.batch_encode([text], max_length=256)
                    if hasattr(morph_ids, "to"):
                        morph_ids = morph_ids.to(device)
                    inputs["morpheme_ids"] = morph_ids

                out = model(**inputs) if inputs else model(text)
                if isinstance(out, dict):
                    logits = out["logits"]
                    if "gate" in out and out["gate"] is not None:
                        g = out["gate"]
                        gate_val = float(g.mean().cpu().item()) if hasattr(g, "mean") else float(g)
                    elif "gate_value" in out and out["gate_value"] is not None:
                        g = out["gate_value"]
                        gate_val = float(g.mean().cpu().item()) if hasattr(g, "mean") else float(g)
                    elif "gate_semantic_weight" in out and out["gate_semantic_weight"] is not None:
                        g = out["gate_semantic_weight"]
                        gate_val = float(g.mean().cpu().item()) if hasattr(g, "mean") else float(g)
                elif hasattr(out, "logits"):
                    logits = out.logits
                    if hasattr(out, "gate") and out.gate is not None:
                        g = out.gate
                        gate_val = float(g.mean().cpu().item()) if hasattr(g, "mean") else float(g)
                elif isinstance(out, (tuple, list)):
                    logits = out[0]
                    if len(out) > 1 and out[1] is not None:
                        gate_val = float(out[1].mean().cpu().item()) if hasattr(out[1], "mean") else float(out[1])
                else:
                    logits = out

                if HAS_TORCH and torch is not None:
                    if hasattr(logits, "shape") and logits.shape[-1] == 2:
                        probs_t = torch.softmax(logits, dim=-1)
                        prob = float((probs_t[..., 1] if probs_t.ndim > 1 else probs_t[1]).cpu().item())
                    else:
                        prob = float(torch.sigmoid(logits).squeeze().cpu().item())
                else:
                    # Pure Python fallback for mock objects / environments without PyTorch
                    if hasattr(logits, "tolist"):
                        l_vals = logits.tolist()
                    elif isinstance(logits, (list, tuple)):
                        l_vals = list(logits)
                    else:
                        l_vals = getattr(logits, "data", [0.0, 1.0])

                    while isinstance(l_vals, list) and l_vals and isinstance(l_vals[0], list):
                        l_vals = l_vals[0]

                    if len(l_vals) == 2:
                        m = max(float(l_vals[0]), float(l_vals[1]))
                        e0 = math.exp(float(l_vals[0]) - m)
                        e1 = math.exp(float(l_vals[1]) - m)
                        prob = float(e1 / (e0 + e1))
                    else:
                        v = float(l_vals[0]) if l_vals else 0.0
                        prob = float(1.0 / (1.0 + math.exp(-v)))

        records.append({
            "quadrant": quad,
            "domain": dom,
            "y_true": label,
            "y_prob": prob,
            "gate": gate_val
        })

    return evaluate_quadrant_matrix(records, bootstrap_ci=bootstrap_ci, n_bootstrap=n_bootstrap)


# ---------------------------------------------------------------------------
# Publication Report Formatter
# ---------------------------------------------------------------------------

def generate_mage_markdown_report(
    results: Dict[str, Any],
    model_name: str = "Kazakh AI Detector"
) -> str:
    """
    Format evaluation results into publication-ready Markdown for paper inclusion.
    """
    quads = results.get("quadrants", {})
    deg = results.get("degradation", {})
    gate_dyn = results.get("gate_dynamics", {})
    mean_gate = gate_dyn.get("mean_gate_by_domain", {})

    lines: List[str] = [
        f"# Kaz-MAGE Benchmark Evaluation Report: {model_name}",
        "",
        "## 1. Quadrant Performance Matrix (ACL 2024 Protocol)",
        "",
        "| Quadrant | Domain Type | Generator Type | Domain | Generator | ROC-AUC (95% CI) | Accuracy (%) | Macro F1 (%) | Optimal $\\tau^*$ |",
        "|---|---|---|---|---|---|---|---|---|"
    ]

    quad_meta = [
        ("Q1", "Seen", "Seen", "Reviews", "Sherkala-7B"),
        ("Q2", "Seen", "Unseen", "Reviews", "Qwen-2.5-7B"),
        ("Q3", "Unseen", "Seen", "News/Wiki", "Sherkala-7B"),
        ("Q4", "Unseen", "Unseen", "News/Wiki", "Qwen-2.5-7B"),
    ]

    for q_id, dom_type, gen_type, dom, gen in quad_meta:
        q_data = quads.get(q_id, {})
        auc = q_data.get("roc_auc", 0.0)
        ci_l = q_data.get("ci_lower", auc)
        ci_u = q_data.get("ci_upper", auc)
        acc = q_data.get("accuracy", 0.0) * 100.0 if q_data.get("accuracy", 0.0) <= 1.0 else q_data.get("accuracy", 0.0)
        f1 = q_data.get("f1", 0.0) * 100.0 if q_data.get("f1", 0.0) <= 1.0 else q_data.get("f1", 0.0)
        opt_t = q_data.get("optimal_threshold", 0.5)

        lines.append(
            f"| **{q_id}** | {dom_type} | {gen_type} | {dom} | {gen} | "
            f"{auc:.4f} [{ci_l:.4f}, {ci_u:.4f}] | {acc:.1f}% | {f1:.1f}% | {opt_t:.4f} |"
        )

    lines.extend([
        "",
        "## 2. Generalization Degradation Analysis",
        "",
        "| Degradation Dimension | Formula | Delta ROC-AUC |",
        "|---|---|---|",
        f"| **Generator Degradation** | $\\text{{AUC}}(Q1) - \\text{{AUC}}(Q2)$ | **{deg.get('delta_auc_generator', 0.0):+.4f}** |",
        f"| **Domain Degradation** | $\\text{{AUC}}(Q1) - \\text{{AUC}}(Q3)$ | **{deg.get('delta_auc_domain', 0.0):+.4f}** |",
        f"| **Wild Degradation** | $\\text{{AUC}}(Q1) - \\text{{AUC}}(Q4)$ | **{deg.get('delta_auc_wild', 0.0):+.4f}** |",
        "",
        "## 3. Dynamic Gate Routing Dynamics across Domains",
        "",
        "| Domain | Domain Status | Mean Gate Routing $\\bar{g}$ | Shift $\\Delta g$ vs Reviews |",
        "|---|---|---|---|"
    ])

    rev_gate = mean_gate.get("consumer_reviews", mean_gate.get("reviews", None))
    news_gate = mean_gate.get("news", None)
    wiki_gate = mean_gate.get("wikipedia", mean_gate.get("wiki", None))

    if rev_gate is not None:
        lines.append(f"| **Consumer Reviews** | Seen (In-Domain) | `{rev_gate:.3f}` | `0.000` (baseline) |")
    if news_gate is not None:
        shift_news = gate_dyn.get("delta_gate_news", news_gate - rev_gate if rev_gate is not None else 0.0)
        lines.append(f"| **Formal News** | Unseen Domain | `{news_gate:.3f}` | `+{shift_news:.3f}` |")
    if wiki_gate is not None:
        shift_wiki = gate_dyn.get("delta_gate_wiki", wiki_gate - rev_gate if rev_gate is not None else 0.0)
        lines.append(f"| **Kazakh Wikipedia** | Unseen Domain | `{wiki_gate:.3f}` | `+{shift_wiki:.3f}` |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Kaz-MAGE 2x2 Matrix & Dynamic Gate Shifts.")
    parser.add_argument("--input", type=str, required=True, help="Path to input predictions JSON file.")
    parser.add_argument("--output", type=str, default=None, help="Path to write evaluation output.")
    parser.add_argument("--format", type=str, choices=["json", "markdown", "md"], default="json", help="Output format.")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap resamples (default: 1000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} evaluation records from '{args.input}'.")
    results = evaluate_quadrant_matrix(records, bootstrap_ci=True, n_bootstrap=args.n_bootstrap, seed=args.seed)

    if args.format in ("markdown", "md") or (args.output and args.output.endswith(".md")):
        report = generate_mage_markdown_report(results)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Markdown report written to '{args.output}'.")
        else:
            print(report)
    else:
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"JSON evaluation results written to '{args.output}'.")
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
