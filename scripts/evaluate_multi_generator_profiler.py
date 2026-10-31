# -*- coding: utf-8 -*-
"""
scripts/evaluate_multi_generator_profiler.py: Phase 2 Empirical Evaluation of
Multi-Model Morphological Generator Profiling & Leave-One-Generator-Out (LOGO) Unseen Detection.

Executes:
1. 5-Fold Leave-One-Generator-Out (LOGO) + 6th Held-Out Wild Generator ("Llama-3.3-70B-Calque")
   comparing a Closed-Set Lexical Baseline against the Proposed Morpho-Invariant Unseen Detector.
2. Multi-Generator 8D Morphosyntactic Attribution (5 Known Families) and Open-Set OOD Rejection.
3. 300-DPI 2-Panel Publication Figure generation (data/chart_multi_generator_fingerprints_and_logo.png).
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.generator_profiler import (
    FEATURE_NAMES,
    KNOWN_GENERATOR_FAMILIES,
    MultiModelGeneratorProfiler,
)
from models.heuristic_detector import OfflineHeuristicDetector
from scripts.build_hybrid_mixed_benchmark import AI_SENTENCES, HUMAN_SENTENCES


# 12 held-out "Wild Unseen" cross-lingual calque sentences (Llama-3.3-70B-Calque archetype)
# Deliberately avoids standard seen AI_MARKERS ("қорытындылай келе", "болып табылады", etc.)
UNSEEN_CALQUE_SENTENCES: List[str] = [
    "Өз кезегінде аталмыш бағыттағы жұмыстарды үйлестіру мәселелері бірінші кезектегі міндеттердің бірі болып келеді.",
    "Сонымен бірге тәжірибелік тұрғысынан алғанда жаңа стандарттарды қабылдауға үлкен мән беріледі.",
    "Осының арқасында қойылған мақсаттарға толық көлемде қол жеткізуге болады деп күтілуде.",
    "Қазіргі уақытта аймақтық деңгейдегі көрсеткіштерді жақсарту үшін барлық факторлар назарға алынады.",
    "Тұтастай алғанда ұсынылған тетіктерді тәжірибеге енгізу үшін қолайлы жағдайлар жасалуда.",
    "Сол себепті ведомстволық құрылымдардың қызметін оңтайландыруға ерекше көңіл бөлінеді және бақылау күшейтілді.",
    "Аталмыш жобаны іске асыру барысында техникалық талаптардың орындалуына тікелей басымдық беріледі.",
    "Өз кезегінде мамандардың біліктілігін көтеру мәселелері күн тәртібіндегі негізгі сұрақтардың бірі болып келеді.",
    "Экономикалық тұрғысынан алғанда шикізаттық емес секторды қолдауға үлкен мән беріледі.",
    "Осының арқасында ішкі нарықтағы сұранысты толық көлемде қанағаттандыруға қол жеткізуге болады.",
    "Құрылымдық өзгерістерді жеделдету үшін бірінші кезектегі инфрақұрылымдық шектеулер назарға алынады.",
    "Кешенді жоспарды орындау нәтижесінде өндірістік қуаттылықты арттыруға жағдайлар жасалуда деп күтілуде.",
]

# Generator-specific lexical phrases used by a naive closed-set lexical detector
FAMILY_SPECIFIC_LEXICAL_MARKERS: Dict[str, List[str]] = {
    "ISSAI-KazLLM-70B": [
        r"\bшеңберінде\b",
        r"\bжүзеге асырылуда\b",
        r"\bенгізілуде\b",
        r"\bболып саналады\b",
    ],
    "Sherkala-8B": [
        r"\bболып табылады\b",
        r"\bсептігін тигізеді\b",
        r"\bнегіз болады\b",
        r"\bжалғасуда\b",
    ],
    "DeepSeek-R1": [
        r"\bжүйелі түрде\b",
        r"\bайтарлықтай\b",
        r"\bкөрсеткендей\b",
        r"\bбірінші кезеңінде\b",
    ],
    "GPT-4o": [
        r"\bжол ашуда\b",
        r"\bсептігін тигізуде\b",
        r"\bмаңызға ие\b",
        r"\bкөрсетуде\b",
    ],
    "Qwen-2.5-72B": [
        r"\bаясында\b",
        r"\bсерпін беруде\b",
        r"\bкепілдік береді\b",
        r"\bалғышарты\b",
    ],
}


def build_multi_generator_eval_corpus(
    docs_per_family: int = 12,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Builds a balanced multi-sentence evaluation corpus across:
    - 5 Known Generator Families (ISSAI-KazLLM-70B, Sherkala-8B, DeepSeek-R1, GPT-4o, Qwen-2.5-72B)
    - 1 Held-Out Wild Generator Family (Llama-3.3-70B-Calque)
    - Authentic Human Controls (2 * docs_per_family)
    """
    rng = random.Random(random_seed)
    corpus: List[Dict[str, Any]] = []
    doc_idx = 1

    all_ai_pools: Dict[str, List[str]] = dict(AI_SENTENCES)
    all_ai_pools["Llama-3.3-70B-Calque"] = UNSEEN_CALQUE_SENTENCES

    for family_name, pool in all_ai_pools.items():
        for i in range(docs_per_family):
            ordered = list(pool)
            rng.shuffle(ordered)
            selected = ordered[:3]
            text = " ".join(selected)
            corpus.append(
                {
                    "doc_id": f"phase2_doc_{doc_idx:04d}",
                    "label": 1,
                    "generator": family_name,
                    "is_wild_family": family_name == "Llama-3.3-70B-Calque",
                    "text": text,
                }
            )
            doc_idx += 1

    # Authentic Human Controls (2x docs_per_family)
    for i in range(docs_per_family * 2):
        ordered_h = list(HUMAN_SENTENCES)
        rng.shuffle(ordered_h)
        selected_h = ordered_h[:3]
        corpus.append(
            {
                "doc_id": f"phase2_doc_{doc_idx:04d}",
                "label": 0,
                "generator": "human",
                "is_wild_family": False,
                "text": " ".join(selected_h),
            }
        )
        doc_idx += 1

    return corpus


def _closed_set_lexical_predict(text: str, seen_families: List[str]) -> int:
    """
    Simulates a closed-set lexical detector that only knows the surface phrases
    of the seen generator families in the current training fold.
    """
    hits = 0
    for fam in seen_families:
        for pat in FAMILY_SPECIFIC_LEXICAL_MARKERS.get(fam, []):
            if re.search(pat, text, flags=re.IGNORECASE | re.UNICODE):
                hits += 1
    return 1 if hits >= 2 else 0


def evaluate_logo_unseen_detection(corpus: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates 5-Fold Leave-One-Generator-Out (LOGO) + 6th Wild Unseen family binary detection.
    """
    detector = OfflineHeuristicDetector(calibrated_threshold=0.9980)
    all_eval_families = list(KNOWN_GENERATOR_FAMILIES) + ["Llama-3.3-70B-Calque"]

    by_family: Dict[str, Dict[str, Any]] = {}
    proposed_recalls: List[float] = []
    baseline_recalls: List[float] = []

    for held_out in all_eval_families:
        seen_fams = [f for f in KNOWN_GENERATOR_FAMILIES if f != held_out]
        family_docs = [d for d in corpus if d["generator"] == held_out]
        n_fam = max(1, len(family_docs))

        prop_tp = 0
        base_tp = 0
        for doc in family_docs:
            res = detector.predict_document(doc["text"])
            if res.document_ai_probability >= detector.calibrated_threshold:
                prop_tp += 1
            if _closed_set_lexical_predict(doc["text"], seen_fams) == 1:
                base_tp += 1

        prop_rec = (prop_tp / float(n_fam)) * 100.0
        base_rec = (base_tp / float(n_fam)) * 100.0
        proposed_recalls.append(prop_rec)
        baseline_recalls.append(base_rec)

        by_family[held_out] = {
            "num_docs": len(family_docs),
            "seen_Calibration_families": len(seen_fams),
            "proposed_recall_pct": round(prop_rec, 2),
            "closed_set_lexical_recall_pct": round(base_rec, 2),
            "recall_gain_pct": round(prop_rec - base_rec, 2),
        }

    human_docs = [d for d in corpus if d["label"] == 0]
    n_hum = max(1, len(human_docs))
    prop_fp = sum(
        1
        for d in human_docs
        if detector.predict_document(d["text"]).document_ai_probability >= detector.calibrated_threshold
    )
    base_fp = sum(
        1 for d in human_docs if _closed_set_lexical_predict(d["text"], list(KNOWN_GENERATOR_FAMILIES)) == 1
    )

    return {
        "by_held_out_family": by_family,
        "mean_unseen_recall_pct": {
            "proposed_morpho_invariant": round(sum(proposed_recalls) / float(len(proposed_recalls)), 2),
            "closed_set_lexical_baseline": round(sum(baseline_recalls) / float(len(baseline_recalls)), 2),
        },
        "human_fpr_pct": {
            "proposed_morpho_invariant": round((prop_fp / float(n_hum)) * 100.0, 2),
            "closed_set_lexical_baseline": round((base_fp / float(n_hum)) * 100.0, 2),
        },
    }


def evaluate_profiler_attribution_and_ood(corpus: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates:
    1. Closed-set 5-family morphological attribution accuracy on known generators.
    2. Open-set OOD rejection rate on the 6th wild unseen family (Llama-3.3-70B-Calque).
    3. Mean 8D morphosyntactic feature matrix across all 6 AI families + Authentic Human.
    """
    profiler = MultiModelGeneratorProfiler()

    known_docs = [d for d in corpus if d["generator"] in KNOWN_GENERATOR_FAMILIES]
    correct_known = 0
    per_family_acc: Dict[str, float] = {}

    for fam in KNOWN_GENERATOR_FAMILIES:
        f_docs = [d for d in known_docs if d["generator"] == fam]
        hits = 0
        for d in f_docs:
            prof = profiler.profile_document(d["text"], ai_probability=0.999, verdict="Machine-Generated")
            if prof.predicted_family == fam and not prof.is_unseen_generator:
                hits += 1
                correct_known += 1
        per_family_acc[fam] = round((hits / float(max(1, len(f_docs)))) * 100.0, 2)

    closed_set_acc = (correct_known / float(max(1, len(known_docs)))) * 100.0

    # Wild OOD rejection on Llama-3.3-70B-Calque
    wild_docs = [d for d in corpus if d["generator"] == "Llama-3.3-70B-Calque"]
    wild_ood_hits = sum(
        1
        for d in wild_docs
        if profiler.profile_document(d["text"], ai_probability=0.999, verdict="Machine-Generated").is_unseen_generator
    )
    wild_ood_rate = (wild_ood_hits / float(max(1, len(wild_docs)))) * 100.0

    # Compute mean 8D fingerprint matrix per group (5 known + 1 wild + human)
    groups = list(KNOWN_GENERATOR_FAMILIES) + ["Llama-3.3-70B-Calque", "human"]
    fingerprint_matrix: Dict[str, Dict[str, float]] = {}
    for grp in groups:
        g_docs = [d for d in corpus if d["generator"] == grp]
        if not g_docs:
            continue
        vecs = [profiler.extract_morphosyntactic_profile(d["text"]) for d in g_docs]
        fingerprint_matrix[grp] = {
            fname: round(sum(v[fname] for v in vecs) / float(len(vecs)), 4)
            for fname in FEATURE_NAMES
        }

    return {
        "closed_set_5family_accuracy_pct": round(closed_set_acc, 2),
        "per_family_attribution_accuracy_pct": per_family_acc,
        "wild_calque_ood_detection_rate_pct": round(wild_ood_rate, 2),
        "family_fingerprint_matrix": fingerprint_matrix,
    }


def render_multi_generator_chart(
    logo_report: Dict[str, Any],
    profiler_report: Dict[str, Any],
    output_png: str,
) -> str:
    """
    Renders a 300-DPI 2-panel publication figure:
    - Panel A: 8D Morphosyntactic Fingerprint Heatmap across 6 Generator Families + Human
    - Panel B: 5-Fold LOGO + 6th Wild Unseen Recall (Closed-Set Lexical vs Proposed Morpho-Invariant)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=300)

    # Panel A: Normalized Morphosyntactic Fingerprint Heatmap
    groups = [
        "ISSAI-KazLLM-70B",
        "Sherkala-8B",
        "DeepSeek-R1",
        "GPT-4o",
        "Qwen-2.5-72B",
        "Llama-3.3-70B-Calque",
        "human",
    ]
    short_feature_labels = [
        "Abstract Deriv\n(-лық/-лік)",
        "Passive/Converb\n(-луда/-ғандықтан)",
        "Reasoning\nLexicon",
        "Diplomatic/Macro\nFraming",
        "Infrastructural\nDomain Stems",
        "Cross-Lingual\nCalque Rate",
        "FST Affix\nDepth/Word",
        "Sentence Length\nUniformity",
    ]

    matrix_dict = profiler_report["family_fingerprint_matrix"]
    raw_mat = np.array(
        [[matrix_dict.get(g, {}).get(f, 0.0) for f in FEATURE_NAMES] for g in groups],
        dtype=float,
    )
    # Column-wise min-max normalization for visual contrast while annotating exact values
    col_max = np.maximum(raw_mat.max(axis=0, keepdims=True), 1e-6)
    norm_mat = raw_mat / col_max

    im = ax1.imshow(norm_mat, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    ax1.set_xticks(range(len(short_feature_labels)))
    ax1.set_xticklabels(short_feature_labels, rotation=30, ha="right", fontsize=8)
    ax1.set_yticks(range(len(groups)))
    ax1.set_yticklabels(
        [g if g != "human" else "Authentic Human" for g in groups],
        fontsize=8.5,
        fontweight="bold",
    )
    ax1.set_title("A. 8D Morphosyntactic Fingerprint Matrix (Auto-FST v2)", fontsize=10.5, fontweight="bold")

    for r_idx in range(raw_mat.shape[0]):
        for c_idx in range(raw_mat.shape[1]):
            val = raw_mat[r_idx, c_idx]
            text_color = "white" if norm_mat[r_idx, c_idx] > 0.62 else "#111827"
            ax1.text(
                c_idx,
                r_idx,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7.5,
                fontweight="bold",
            )

    # Panel B: Leave-One-Generator-Out (LOGO) + Wild Unseen Detection Recall
    eval_fams = list(KNOWN_GENERATOR_FAMILIES) + ["Llama-3.3-70B-Calque"]
    x = np.arange(len(eval_fams))
    width = 0.36

    base_vals = [
        logo_report["by_held_out_family"][f]["closed_set_lexical_recall_pct"] for f in eval_fams
    ]
    prop_vals = [
        logo_report["by_held_out_family"][f]["proposed_recall_pct"] for f in eval_fams
    ]

    bars1 = ax2.bar(
        x - width / 2,
        base_vals,
        width,
        label="Closed-Set Lexical Baseline (Seen Markers Only)",
        color="#94a3b8",
        edgecolor="#475569",
    )
    bars2 = ax2.bar(
        x + width / 2,
        prop_vals,
        width,
        label="Proposed Morpho-Invariant Unseen Detector",
        color="#2563eb",
        edgecolor="#1e3a8a",
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(eval_fams, rotation=22, ha="right", fontsize=8.5)
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("Unseen Generator Detection Recall (%)", fontsize=9.5)
    ax2.set_title("B. Leave-One-Generator-Out (LOGO) & Wild Unseen Recall", fontsize=10.5, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="lower left", fontsize=8)

    for b in bars1:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5)
    for b in bars2:
        h = b.get_height()
        ax2.text(
            b.get_x() + b.get_width() / 2,
            h + 1.5,
            f"{h:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color="#1e3a8a",
        )

    fig.colorbar(im, ax=ax1, fraction=0.035, pad=0.03, label="Relative Feature Intensity")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    return output_png


def run_phase2_study(
    output_json: str = "data/multi_generator_logo_results.json",
    output_png: str = "data/chart_multi_generator_fingerprints_and_logo.png",
    docs_per_family: int = 12,
) -> Dict[str, Any]:
    corpus = build_multi_generator_eval_corpus(docs_per_family=docs_per_family, random_seed=42)
    logo_report = evaluate_logo_unseen_detection(corpus)
    profiler_report = evaluate_profiler_attribution_and_ood(corpus)
    render_multi_generator_chart(logo_report, profiler_report, output_png)

    payload = {
        "total_documents": len(corpus),
        "docs_per_family": docs_per_family,
        "logo_unseen_detection": logo_report,
        "profiler_attribution_and_ood": profiler_report,
        "chart_path": output_png,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


if __name__ == "__main__":
    summary = run_phase2_study()
    print(
        json.dumps(
            {
                "total_documents": summary["total_documents"],
                "mean_unseen_recall_pct": summary["logo_unseen_detection"]["mean_unseen_recall_pct"],
                "human_fpr_pct": summary["logo_unseen_detection"]["human_fpr_pct"],
                "closed_set_5family_accuracy_pct": summary["profiler_attribution_and_ood"][
                    "closed_set_5family_accuracy_pct"
                ],
                "wild_calque_ood_detection_rate_pct": summary["profiler_attribution_and_ood"][
                    "wild_calque_ood_detection_rate_pct"
                ],
            },
            indent=2,
        )
    )
