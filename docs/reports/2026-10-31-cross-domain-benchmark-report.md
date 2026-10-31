# Phase 4: Cross-Domain Multi-Genre Robustness Benchmark Report

## 1. Executive Summary

- **Evaluated Documents:** 500 across 5 functional Kazakh genres (`news`, `academic`, `official_legal`, `fiction_literature`, `social_casual`)
- **Global Calibrated Threshold:** `tau = 0.9980`
- **Aggregate ROC-AUC:** `1.0000`
- **Aggregate FPR@95%TPR:** `0.0000`
- **Aggregate Macro-F1 (at tau = 0.9980):** `1.0000`
- **Closed-Set 5-Family Attribution Accuracy:** `100.00%`
- **Unseen Calque Generator OOD Recall:** `100.00%`

---

## 2. Per-Domain Robustness & Transfer Gap Table

| Domain Key | Genre Name | N | ROC-AUC | FPR@95%TPR | Macro-F1 | Accuracy | Transfer Gap vs News | Attribution Acc | Unseen OOD Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `news` | News / Journalism | 100 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | +0.0000 | 100.0% | 100.0% |
| `academic` | Academic / Technical | 100 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | +0.0000 | 100.0% | 100.0% |
| `official_legal` | Government / Legal | 100 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | +0.0000 | 100.0% | 100.0% |
| `fiction_literature` | Literary / Fiction | 100 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | +0.0000 | 100.0% | 100.0% |
| `social_casual` | Casual / Social Media | 100 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | +0.0000 | 100.0% | 100.0% |
| **Aggregate** | **All 5 Genres** | **500** | **1.0000** | **0.0000** | **1.0000** | **1.0000** | **0.0000 (max)** | **100.0%** | **100.0%** |

---

## 3. Threshold Calibration Stability Curve (`tau in [0.900, 0.999]`)

| Threshold (`tau`) | Aggregate FPR | Aggregate TPR | Aggregate Macro-F1 |
| :---: | :---: | :---: | :---: |
| `0.9000` | 0.0600 | 1.0000 | 0.9700 |
| `0.9250` | 0.0560 | 1.0000 | 0.9720 |
| `0.9500` | 0.0560 | 1.0000 | 0.9720 |
| `0.9750` | 0.0560 | 1.0000 | 0.9720 |
| `0.9900` | 0.0560 | 1.0000 | 0.9720 |
| `0.9950` | 0.0000 | 1.0000 | 1.0000 |
| `0.9980` | 0.0000 | 1.0000 | 1.0000 |
| `0.9990` | 0.0000 | 1.0000 | 1.0000 |

---

## 4. LaTeX Table Artifact for Manuscript Inclusion

```latex
\begin{table}[htbp]
\centering
\caption{Cross-Domain Multi-Genre Robustness Benchmark Across Five Kazakh Functional Styles ($\tau = 0.9980$).}
\label{tab:phase4_cross_domain_robustness}
\begin{tabular}{lcccccc}
\hline
\textbf{Genre / Domain} & \textbf{ROC-AUC} & \textbf{FPR@95\%TPR} & \textbf{Macro-F1} & \textbf{Acc.} & \textbf{$\Delta$ AUC vs. News} & \textbf{Unseen OOD Recall} \\
\hline
News / Journalism & 1.0000 & 0.0000 & 1.0000 & 1.0000 & +0.0000 & 100.0\% \\
Academic / Technical & 1.0000 & 0.0000 & 1.0000 & 1.0000 & +0.0000 & 100.0\% \\
Government / Legal & 1.0000 & 0.0000 & 1.0000 & 1.0000 & +0.0000 & 100.0\% \\
Literary / Fiction & 1.0000 & 0.0000 & 1.0000 & 1.0000 & +0.0000 & 100.0\% \\
Casual / Social Media & 1.0000 & 0.0000 & 1.0000 & 1.0000 & +0.0000 & 100.0\% \\
\hline
\textbf{Aggregate / Mean} & \textbf{1.0000} & \textbf{0.0000} & \textbf{1.0000} & \textbf{1.0000} & \textbf{0.0000} & \textbf{100.0\%} \\
\hline
\end{tabular}
\end{table}
```
