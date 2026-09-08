"""
Evaluation & Threshold Calibration Suite for Kazakh AI Text Detection.

Provides:
1. `compute_comprehensive_metrics`: Threshold-independent (ROC-AUC, Youden's J, EER)
   and fixed-threshold metrics (accuracy, f1, precision, recall, confusion matrix,
   FPR/FNR) along with RAID-compliant length-stratified diagnostics.
2. `compute_mcnemar_test`: Statistical significance testing (McNemar with Edwards'
   continuity correction) comparing two models on full or short-text subsets.
3. `measure_inference_efficiency`: Latency (ms/sample) and throughput (samples/sec)
   benchmarking across repeat executions.

Defensively designed with pure Python fallbacks for zero-dependency execution
in environments where NumPy, SciPy, or Scikit-Learn are not installed.
"""

import math
import time
from itertools import groupby
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

# Optional imports for hardware-accelerated / optimized environments
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    stats = None
    HAS_SCIPY = False

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def _to_list(arr: Any) -> Optional[List[Any]]:
    """Converts a sequence, array-like, or tensor to a standard Python list."""
    if arr is None:
        return None
    if hasattr(arr, "tolist"):
        return arr.tolist()
    if isinstance(arr, (list, tuple)):
        return list(arr)
    return list(arr)


def _compute_roc_curve_and_auc_pure(y_true: List[int], y_prob: List[float]):
    """
    Pure-Python calculation of the ROC curve, AUC, Youden's J optimal threshold, and EER.
    Handles tied prediction probabilities according to standard ROC definition.
    """
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.5, 0.5

    # Sort pairs descending by predicted probability
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

    # Equal Error Rate (EER): where |FPR - FNR| is minimized, FNR = 1 - TPR
    eer_idx = min(range(len(fprs)), key=lambda i: abs(fprs[i] - (1.0 - tprs[i])))
    eer = fprs[eer_idx]

    return auc, optimal_threshold, eer


def compute_comprehensive_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    lengths: Optional[Sequence[int]] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes threshold-independent (ROC-AUC, optimal threshold via Youden's J, EER)
    and fixed-threshold metrics (accuracy, F1, precision, recall, confusion matrix,
    FPR, FNR), plus length-stratified metrics across short/medium/long brackets.

    Args:
        y_true: Ground-truth binary labels (0 for Human, 1 for AI).
        y_prob: Predicted probabilities for the AI class (class 1).
        lengths: Optional sequence of character lengths for length stratification.
        threshold: Decision threshold for discrete classification (default: 0.5).

    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    y_true_list = _to_list(y_true)
    y_prob_list = _to_list(y_prob)
    lengths_list = _to_list(lengths)

    if y_true_list is None or y_prob_list is None:
        raise ValueError("y_true and y_prob must not be None")
    if len(y_true_list) != len(y_prob_list):
        raise ValueError("y_true and y_prob must have identical lengths")
    if lengths_list is not None and len(lengths_list) != len(y_true_list):
        raise ValueError("lengths and y_true must have identical lengths")

    n_samples = len(y_true_list)

    # 1. Threshold-independent metrics (ROC-AUC, Youden's J, EER)
    auc_val, opt_thresh, eer_val = 0.5, threshold, 0.5
    if HAS_SKLEARN and HAS_NUMPY:
        try:
            yt_arr = np.asarray(y_true_list)
            yp_arr = np.asarray(y_prob_list)
            auc_val = float(roc_auc_score(yt_arr, yp_arr))
            fpr_curve, tpr_curve, thresh_curve = roc_curve(yt_arr, yp_arr)

            j_scores = tpr_curve - fpr_curve
            best_idx = int(np.argmax(j_scores))
            opt_thresh = float(thresh_curve[best_idx])

            fnr_curve = 1.0 - tpr_curve
            eer_idx = int(np.nanargmin(np.abs(fpr_curve - fnr_curve)))
            eer_val = float(fpr_curve[eer_idx])
        except Exception:
            auc_val, opt_thresh, eer_val = _compute_roc_curve_and_auc_pure(y_true_list, y_prob_list)
    else:
        auc_val, opt_thresh, eer_val = _compute_roc_curve_and_auc_pure(y_true_list, y_prob_list)

    # 2. Fixed-threshold metrics at specified threshold
    y_pred = [1 if p >= threshold else 0 for p in y_prob_list]

    tn = sum(1 for yt, yp in zip(y_true_list, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true_list, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true_list, y_pred) if yt == 1 and yp == 0)
    tp = sum(1 for yt, yp in zip(y_true_list, y_pred) if yt == 1 and yp == 1)

    acc = (tp + tn) / max(1, n_samples)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)

    summary: Dict[str, Any] = {
        "accuracy": round(float(acc) * 100.0, 2),
        "f1": round(float(f1) * 100.0, 2),
        "precision": round(float(prec) * 100.0, 2),
        "recall": round(float(rec) * 100.0, 2),
        "roc_auc": round(float(auc_val), 4),
        "optimal_threshold": round(float(opt_thresh), 4),
        "eer": round(float(eer_val), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": round(float(fpr) * 100.0, 2),
        "fnr": round(float(fnr) * 100.0, 2),
    }

    # 3. Length stratification
    # Brackets: short (<= 60), medium (61..85), long (> 85)
    if lengths_list is not None:
        strat: Dict[str, Dict[str, Any]] = {}
        bracket_filters = {
            "short": lambda length: length <= 60,
            "medium": lambda length: 60 < length <= 85,
            "long": lambda length: length > 85,
        }

        for b_name, b_fn in bracket_filters.items():
            b_indices = [i for i, length in enumerate(lengths_list) if b_fn(length)]
            b_total = len(b_indices)
            if b_total > 0:
                b_yt = [y_true_list[i] for i in b_indices]
                b_yp = [y_pred[i] for i in b_indices]

                b_tn = sum(1 for yt, yp in zip(b_yt, b_yp) if yt == 0 and yp == 0)
                b_fp = sum(1 for yt, yp in zip(b_yt, b_yp) if yt == 0 and yp == 1)
                b_fn = sum(1 for yt, yp in zip(b_yt, b_yp) if yt == 1 and yp == 0)
                b_tp = sum(1 for yt, yp in zip(b_yt, b_yp) if yt == 1 and yp == 1)

                b_acc = (b_tp + b_tn) / b_total
                b_fpr = b_fp / max(1, b_fp + b_tn)
                b_fnr = b_fn / max(1, b_fn + b_tp)

                strat[b_name] = {
                    "total": int(b_total),
                    "accuracy": round(float(b_acc) * 100.0, 2),
                    "fp": int(b_fp),
                    "fn": int(b_fn),
                    "fpr": round(float(b_fpr) * 100.0, 2),
                    "fnr": round(float(b_fnr) * 100.0, 2),
                }
        summary["length_stratification"] = strat

    return summary


def compute_mcnemar_test(
    y_true: Sequence[int],
    y_pred_baseline: Sequence[int],
    y_pred_proposed: Sequence[int],
    short_mask: Optional[Sequence[bool]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Computes McNemar's hypothesis test with Edwards' continuity correction.

    Args:
        y_true: Ground-truth binary labels.
        y_pred_baseline: Predictions from baseline model.
        y_pred_proposed: Predictions from proposed model.
        short_mask: Optional boolean sequence to filter subset (e.g. short-text only).
        alpha: Significance threshold (default: 0.05).

    Returns:
        Dictionary with contingency_matrix, chi2_statistic, p_value, and is_significant.
    """
    y_true_list = _to_list(y_true)
    y_base_list = _to_list(y_pred_baseline)
    y_prop_list = _to_list(y_pred_proposed)
    mask_list = _to_list(short_mask)

    if y_true_list is None or y_base_list is None or y_prop_list is None:
        raise ValueError("Inputs y_true, y_pred_baseline, and y_pred_proposed must not be None")
    if not (len(y_true_list) == len(y_base_list) == len(y_prop_list)):
        raise ValueError("All input arrays must have the same length")

    # Apply mask if provided
    if mask_list is not None:
        if len(mask_list) != len(y_true_list):
            raise ValueError("short_mask must have the same length as y_true")
        indices = [i for i, m in enumerate(mask_list) if bool(m)]
        y_true_list = [y_true_list[i] for i in indices]
        y_base_list = [y_base_list[i] for i in indices]
        y_prop_list = [y_prop_list[i] for i in indices]

    correct_base = [int(p == t) for p, t in zip(y_base_list, y_true_list)]
    correct_prop = [int(p == t) for p, t in zip(y_prop_list, y_true_list)]

    n_00 = sum(1 for b, p in zip(correct_base, correct_prop) if b == 1 and p == 1)
    n_01 = sum(1 for b, p in zip(correct_base, correct_prop) if b == 1 and p == 0)
    n_10 = sum(1 for b, p in zip(correct_base, correct_prop) if b == 0 and p == 1)
    n_11 = sum(1 for b, p in zip(correct_base, correct_prop) if b == 0 and p == 0)

    contingency_matrix = [[int(n_00), int(n_01)], [int(n_10), int(n_11)]]

    # Edwards' continuity correction
    b, c = n_01, n_10
    diff = abs(b - c)
    total_discordant = b + c

    if total_discordant == 0 or diff < 1.0:
        chi2 = 0.0
        p_val = 1.0
    else:
        chi2 = ((diff - 1.0) ** 2) / float(total_discordant)
        if HAS_SCIPY:
            p_val = float(stats.chi2.sf(chi2, df=1))
        else:
            p_val = float(math.erfc(math.sqrt(chi2 / 2.0)))

    is_significant = bool(p_val < alpha)

    return {
        "contingency_matrix": contingency_matrix,
        "chi2_statistic": round(float(chi2), 4),
        "p_value": float(round(p_val, 6)),
        "is_significant": is_significant,
        "is_statistically_significant": is_significant,
        "n_baseline_only_correct": int(n_01),
        "n_proposed_only_correct": int(n_10),
    }


def _invoke_model_inference(model_fn: Callable[[Any], Any], texts: Sequence[str]) -> Any:
    """Invokes model inference supporting both batch and single-sample functions."""
    if isinstance(texts, str):
        texts = [texts]
    if len(texts) == 0:
        return []

    try:
        res = model_fn(texts)
        if hasattr(res, "__len__") and not isinstance(res, (dict, set)) and len(res) == len(texts):
            return res
        if isinstance(res, dict) and len(texts) > 1:
            return [model_fn(t) for t in texts]
        return res
    except (TypeError, AttributeError, ValueError):
        return [model_fn(t) for t in texts]


def measure_inference_efficiency(
    model_fn: Callable[[Any], Any],
    texts: Sequence[str],
    num_runs: int = 20,
) -> Dict[str, Any]:
    """
    Measures latency (ms/sample) and throughput (samples/sec) across repeat runs.

    Args:
        model_fn: Callable performing prediction (batch or single-text function).
        texts: Sequence of input texts to benchmark.
        num_runs: Number of benchmark evaluation runs (default: 20).

    Returns:
        Dictionary with latency_ms_per_sample, throughput_samples_per_sec,
        total_time_sec, num_samples, and num_runs.
    """
    if isinstance(texts, str):
        texts = [texts]
    texts_list = list(texts)
    n_samples = len(texts_list)

    if n_samples == 0 or num_runs <= 0:
        return {
            "latency_ms_per_sample": 0.0,
            "latency_ms": 0.0,
            "throughput_samples_per_sec": 0.0,
            "throughput": 0.0,
            "total_time_sec": 0.0,
            "num_samples": n_samples,
            "num_runs": int(num_runs),
        }

    # 1. Warm-up run (not timed)
    _invoke_model_inference(model_fn, texts_list)

    # 2. Timed benchmarking runs
    start_time = time.perf_counter()
    for _ in range(num_runs):
        _invoke_model_inference(model_fn, texts_list)
    total_time_sec = time.perf_counter() - start_time

    total_samples = n_samples * num_runs
    latency_ms = (total_time_sec / total_samples) * 1000.0
    throughput = total_samples / total_time_sec if total_time_sec > 0.0 else 0.0

    return {
        "latency_ms_per_sample": round(float(latency_ms), 3),
        "latency_ms": round(float(latency_ms), 3),
        "throughput_samples_per_sec": round(float(throughput), 2),
        "throughput": round(float(throughput), 2),
        "total_time_sec": round(float(total_time_sec), 4),
        "num_samples": n_samples,
        "num_runs": int(num_runs),
    }
