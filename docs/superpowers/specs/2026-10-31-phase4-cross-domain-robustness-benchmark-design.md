# Phase 4: Cross-Domain Multi-Genre Robustness Benchmark Design Specification

## Executive Summary
This specification defines the architectural design, genre taxonomy, cross-domain evaluation framework, domain sensitivity analysis, and Gradio UI integration for **Phase 4: Cross-Domain Multi-Genre Robustness Benchmark**.

Following the successful completion of Phase 1 (Unified ONNX INT8 Quantized Engine), Phase 2 (Multi-Model Generator Profiler & LOGO Unseen Detector), and Phase 3 (Deep Pattern Analysis & Comparative Generator Study), Phase 4 addresses the critical scientific challenge of **domain generalization and cross-genre robustness** in agglutinative Kazakh text detection.

In practical deployment, AI text detectors often overfit to formal journalistic or encyclopedic training distributions and experience catastrophic false-positive spikes when confronted with bureaucratic legal boilerplate, creative literary metaphors, or informal social media orthography. Phase 4 stress-tests both Stage 1 (Binary Detection) and Stage 2 (Open-Set Attribution & Unseen Detection) across 5 distinct Kazakh genres to measure cross-genre transfer gaps, evaluate the safety metric **FPR@95%TPR**, and ensure threshold stability ($\tau = 0.9980$).

---

## 1. Domain Taxonomy & Linguistic Characteristics

The benchmark standardizes 5 distinct Kazakh functional styles:

| Domain Key | Genre Name | Linguistic Properties | Primary Failure Mode Addressed |
| :--- | :--- | :--- | :--- |
| `news` | **News / Journalism** | Standard journalistic Kazakh, balanced vocabulary, factual declarative syntax, Tengrinews / Kazinform register. | Serves as the canonical in-domain baseline reference. |
| `academic` | **Academic / Technical** | High lexical density, passive/causative suffix stacking, abstract nominalizations, thesis abstracts, formal definitions. | False-positive risk due to high syntactic formality mimicking LLM prose. |
| `official_legal` | **Government / Legal** | Bureaucratic register, stereotyped administrative formulas (`қаулы етеді`, `бұйырамын`), legislative decrees, long hypotactic clauses. | High analytic copula density triggering false AI classification. |
| `fiction_literature` | **Literary / Fiction** | Expressive narratives, rich figurative imagery, dialogue with conversational particles (`-ау`, `ғой`), non-standard word order. | Over-flagging creative structural burstiness as anomalous. |
| `social_casual` | **Casual / Social Media** | Informal consumer reviews (Kaspi style), forum comments, Russian/English code-switching, keyboard layout shifts, reduced punctuation. | Vulnerability to noisy orthography and unassimilated colloquial forms. |

---

## 2. System Architecture

```
                          Cross-Domain Evaluation Suite
                     (scripts/evaluate_cross_domain_robustness.py)
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
5-Genre Stratified Benchmark                                Evaluation Harness
(data/cross_domain_benchmark.json)                     (OfflineHeuristic / ONNX / Profiler)
 - 50% Authentic Human (N=50/domain)                                 │
 - 50% AI Generated (N=50/domain across 5 LLMs)                      ▼
 - Balanced Seed Shuffling                                   Batched Document Inference
 - Cryptographic SHA-256 Isolation                           - Sliding-window chunking
           │                                                 - Top-K worst-chunk pooling
           └────────────────────────────┬────────────────────┘
                                        ▼
                   ┌───────────────────────────────────────────┐
                   │        Multi-Domain Metric Engine         │
                   │                                           │
                   │ 1. Binary Detection Metrics per Domain:   │
                   │    - ROC-AUC, Macro-F1, Accuracy          │
                   │    - FPR@95%TPR (Safety Metric)           │
                   │ 2. Attribution & Unseen Metrics:          │
                   │    - Closed-set 5-family accuracy         │
                   │    - Unseen OOD recall                    │
                   │ 3. Domain Transfer Gap (Delta vs. News)   │
                   │ 4. Threshold Calibration Stability Curve  │
                   └────────────────────┬──────────────────────┘
                                        │
              ┌─────────────────────────┴─────────────────────────┐
              ▼                                                   ▼
┌───────────────────────────────┐           ┌───────────────────────────────────┐
│ JSON & LaTeX Benchmark Artifacts│         │ Gradio UI Multi-Genre Presets     │
│ - data/cross_domain_results.json│         │ (ui/presets.py & ui/app.py)       │
│ - docs/reports/               │           │ - 5 Verified Domain Quick-Presets │
│   2026-10-31-cross-domain-    │           │ - Genre-tagged verification audit │
│   benchmark-report.md         │           │   in bilingual UI panel           │
└───────────────────────────────┘           └───────────────────────────────────┘
```

---

## 3. Module & Component Specifications

### 3.1. Benchmark Dataset Schema (`data/cross_domain_benchmark.json`)
Each record adheres to the following strict data contract:
```json
{
  "id": "cross_domain_academic_001",
  "domain": "academic",
  "text": "...",
  "label": 0,
  "generator": "human",
  "word_count": 142,
  "sentence_count": 6,
  "sha256": "3a8f...",
  "metadata": {
    "source_register": "thesis_abstract",
    "has_layout_shift": false
  }
}
```
- Total benchmark size: 500 records (100 records per domain: 50 authentic human, 50 AI continuations across 5 known families + 1 unseen calque family).
- Cryptographic isolation: Strict zero-overlap audit verifying no text in `cross_domain_benchmark.json` matches training or prompt seed sets.

### 3.2. Evaluation Runner (`scripts/evaluate_cross_domain_robustness.py`)
- **CLI Flags**:
  - `--benchmark-path`: Path to benchmark JSON (default: `data/cross_domain_benchmark.json`).
  - `--output-json`: Destination for results JSON (default: `data/cross_domain_evaluation_results.json`).
  - `--output-report`: Destination for Markdown/LaTeX report (default: `docs/reports/2026-10-31-cross-domain-benchmark-report.md`).
  - `--dry-run`: Runs 2 records per domain for ultra-fast unit testing and CI verification.
- **Computed Metrics**:
  - For each of the 5 domains and the overall aggregate:
    - `roc_auc`: Area under the Receiver Operating Characteristic curve.
    - `fpr_at_95_tpr`: False Positive Rate when True Positive Rate is fixed at $\ge 0.95$.
    - `f1_score`: Macro-averaged F1 score at calibrated threshold $\tau = 0.9980$.
    - `accuracy`: Overall classification accuracy.
    - `transfer_gap_auc`: $\text{AUC}_{\text{news}} - \text{AUC}_{\text{domain}}$.
    - `attribution_accuracy`: Accuracy of 5-family closed-set attribution on synthetic samples.
    - `unseen_ood_recall`: Recall on held-out unseen calque generator samples.

### 3.3. Domain Sensitivity & Threshold Calibration (`models/domain_calibrator.py`)
- **Objective**: Evaluates whether global threshold $\tau = 0.9980$ satisfies safety constraints across all 5 genres:
  - Verifies that $\text{FPR} \le 0.05$ across all individual domains at $\tau = 0.9980$.
  - Generates domain-level sensitivity curves (FPR vs. threshold $\tau \in [0.90, 0.999]$).
  - Provides `DomainCalibrator` utility allowing optional domain-aware threshold recommendations without breaking backward compatibility.

### 3.4. Gradio UI Integration (`ui/presets.py`, `ui/app.py`, `hf_space/ui/app.py`)
- Adds a **"Domain Presets"** dropdown or tab to the UI with 5 representative samples:
  1. *News*: Kazinform formal report.
  2. *Academic*: Computational linguistics abstract.
  3. *Official*: Ministerial executive decree.
  4. *Fiction*: Contemporary Kazakh short story excerpt.
  5. *Casual*: Kaspi marketplace consumer review.
- Displays detected domain badge and domain-specific robustness indicators in the UI results header.

---

## 4. Test Strategy & Acceptance Criteria

1. **Unit Tests**:
   - `tests/test_cross_domain_benchmark.py`: Asserts dataset schema integrity, domain balance, label balance, and cryptographic isolation.
   - `tests/test_domain_calibrator.py`: Tests threshold evaluation, ROC curve generation, and FPR@95%TPR computation.
   - `tests/test_evaluate_cross_domain.py`: Runs evaluation script with `--dry-run` and verifies output JSON and Markdown artifacts.
   - `tests/test_ui_domain_presets.py`: Verifies all 5 domain preset samples load cleanly and produce expected verdicts in the UI.
2. **Acceptance Criteria**:
   - The detector achieves $\text{ROC-AUC} \ge 0.92$ across all 5 domains.
   - `FPR@95%TPR` $\le 0.08$ on Academic and Legal domains, verifying resistance to formal writing false alarms.
   - Global threshold $\tau = 0.9980$ remains safe with zero false positives on standard human benchmark samples across all 5 genres.
   - All created/modified files pass zero-emoji audit.
   - Full test suite passes cleanly with zero regressions.
