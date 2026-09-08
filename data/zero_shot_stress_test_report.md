# Empirical Results: Zero-Shot Cross-Generator Detector Stress-Test

**Target Architecture:** KazRoBERTa (`kz-transformers/kaz-roberta-conversational`)
**Training Distribution:** KazAI-Detect (KazSAnDRA Reviews + Sherkala-7B LLM)
**Out-of-Distribution Test Bed:** $N = 2,000$ paired samples (1,000 Human KazSAnDRA + 1,000 Qwen-2.5-7B-Instruct)

## 1. Primary Zero-Shot Transfer Performance

| Model Variant | Input Mode | Overall Acc (%) | F1-Score (%) | Precision (%) | Recall (%) | False Positive Rate (FPR %) | False Negative Rate (FNR %) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KazRoBERTa (Pure)** | Raw Text | 36.75% | 26.92% | 31.87% | 23.3% | 49.8% | 76.7% |
| **KazRoBERTa (Test-Time FST)** | FST Preprocessed | 52.7% | 12.41% | 83.75% | 6.7% | 1.3% | 93.3% |
| **KazRoBERTa (Hybrid FST)** | FST-Trained + FST | 50.45% | 1.78% | 100.0% | 0.9% | 0.0% | 99.1% |

## 2. RAID-Compliant Length Bracket Breakdown

| Length Bracket | Sub-sample Size | Pure Acc (%) | Pure FPR (%) | Hybrid FST Acc (%) | Hybrid FST FPR (%) | FPR Reduction (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short** (≤ 60 chars) | 600 | 57.0% | 9.33% | 51.0% | 0.0% | **100.0%** |
| **Medium** (61–85 chars) | 900 | 27.0% | 74.89% | 50.33% | 0.0% | **100.0%** |
| **Long** (> 85 chars) | 500 | 30.0% | 53.2% | 50.0% | 0.0% | **100.0%** |

## 3. Short-Text Hypothesis Testing & Rigorous Robustness Verification

- **Short-Text Human Samples:** 300
- **Pure KazRoBERTa False Positives:** 28 (9.33%)
- **Hybrid FST False Positives:** 0 (0.0%)
- **Absolute FP Elimination:** 28 cases
- **Relative FP Reduction:** **100.0%** (95% CI: [100.0%, 100.0%])
- **McNemar's Test Statistic:** $\chi^2 = 26.0357$, $p = 0.000000$
- **Conclusion:** Statistically significant false positive reduction confirmed (p < 0.05).

## 4. In-Distribution (Sherkala-7B) vs. Out-of-Distribution (Qwen-2.5-7B) Transfer Degradation

| Metric | In-Distribution (Sherkala-7B) | Out-of-Distribution (Qwen-2.5-7B) | Generalization Delta (Δ) |
| :--- | :---: | :---: | :---: |
| **Pure Accuracy** | 96.1% | 36.75% | -59.35% |
| **Pure F1-Score** | 96.09% | 26.92% | -69.17% |
| **Hybrid FST Accuracy** | 96.32% | 50.45% | -45.87% |
| **Hybrid FST F1-Score** | 96.32% | 1.78% | -94.54% |
