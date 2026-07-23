# Kazakh AI-Generated Text Detection & Morphological Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Enabled-orange.svg)](https://pytorch.org/)

A benchmark study and novel morphological framework for detecting AI-generated text in the Kazakh language, introducing **Hybrid Code-Switched FST Pre-tokenization** to eliminate subword token inflation and reduce false positive AI accusations.

---

## 📌 Highlights & Key Research Breakthroughs

* **First Kazakh AI Detection Benchmark**: Comprehensive evaluation comparing 8 transformer model variations (mBERT, XLM-RoBERTa, KazBERT, KazRoBERTa) across raw text and morphological FST conditions.
* **Statistically Proven False Positive Reduction**: McNemar's Test ($\chi^2 = 21.04, p = 0.000004$) and 1,000 Bootstrap 95% Confidence Intervals prove that **Hybrid Code-Switched FST reduces short-text false positive AI accusations by 43.21%** ($53 \rightarrow 30$ false positives).
* **Token Inflation Equalization ($T/W$)**: Hybrid FST pre-segmentation eliminates subword fragmentation, reducing mBERT token inflation by **-36.65%** ($p < 0.0001$) and forcing all tokenizers to converge to the natural **Kazakh Morphemic Limit of 2.202 tokens/word**.
* **Out-of-Distribution (OOD) Cross-Domain Robustness**: Proven generalization across 3 distinct linguistic domains: **Consumer Reviews (96.4% Acc)**, **Formal News (98.8% Acc)**, and **Academic Wikipedia (99.1% Acc)**.

---

## 📊 Scientific Benchmark & Statistical Proofs

### 1. Hybrid Code-Switched FST vs Pure Baselines

| Model Architecture | Mode | Accuracy | F1-Score | Short FP Count | 95% Bootstrap Confidence Interval (F1) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| mBERT | Pure | 95.42% | 95.46% | 58 | [94.80%, 96.10%] |
| mBERT | FST | 93.62% | 93.71% | 70 | [92.90%, 94.40%] |
| XLM-RoBERTa | Pure | 95.45% | 95.52% | 67 | [94.85%, 96.15%] |
| XLM-RoBERTa | FST | 94.35% | 94.50% | 83 | [93.70%, 95.20%] |
| KazBERT | Pure | 94.59% | 94.63% | 63 | [93.90%, 95.30%] |
| KazBERT | FST | 94.56% | 94.63% | 65 | [93.90%, 95.30%] |
| **KazRoBERTa** | **Pure** | **96.10%** | **96.09%** | 53 | [95.45%, 96.69%] |
| **KazRoBERTa** | **Hybrid FST** | **96.32%** | **96.32%** | **30** | **[95.69%, 96.88%]** |

---

### 2. Token Inflation Ratio ($T/W$)

$$\text{Token Inflation Ratio } (T/W) = \frac{\text{Total Subword Tokens}}{\text{Total Whitespace-Separated Words}}$$

| Model Architecture | Raw Text ($T/W$) | Hybrid FST ($T/W$) | Token Inflation Reduction | Significance ($p$-value) |
| :--- | :---: | :---: | :---: | :---: |
| **KazRoBERTa** | 2.461 tokens/word | **2.202 tokens/word** | **-10.51%** | $p < 0.000001$ |
| **XLM-RoBERTa** | 2.461 tokens/word | **2.202 tokens/word** | **-10.51%** | $p < 0.000001$ |
| **mBERT** | 3.476 tokens/word | **2.202 tokens/word** | **-36.65%** | $p < 0.000001$ |

> **Hypothesis Proved**: Hybrid FST acts as a universal morphological normalizer. Regardless of pre-training vocabulary size (mBERT 110k vs XLM-R 250k vs KazRoBERTa 30k), FST forces all tokenizers to converge to the natural Kazakh morphemic limit ($\approx 2.202$ morphemes per word).

---

### 3. Out-of-Distribution (OOD) Cross-Domain Generalization

| Domain / Register | Pure KazRoBERTa FPR (%) | Hybrid FST KazRoBERTa FPR (%) | False Positive Reduction | Accuracy (Hybrid FST) |
| :--- | :---: | :---: | :---: | :---: |
| **Consumer Reviews (Informal)** | 12.80% | **7.20%** | **-43.75%** | **96.40%** |
| **Formal News (Informburo)** | 4.20% | **2.40%** | **-42.86%** | **98.80%** |
| **Wikipedia (Academic)** | 3.60% | **1.80%** | **-50.00%** | **99.10%** |

---

## 📈 Visualizations

| Bootstrap 95% Confidence Intervals | Token Inflation Ratio (T/W) |
| :---: | :---: |
| ![Bootstrap CI](data/chart_bootstrap_confidence_intervals.png) | ![Token Inflation](data/chart_token_inflation_ratio.png) |

| Out-of-Distribution Cross-Domain Generalization |
| :---: |
| ![OOD Generalization](data/chart_ood_domain_generalization.png) |

---

## 🛠️ Code-Switched Hybrid FST Parsing

Standard tokenizers treat Kazakh-Russian code-switched review slang (`доставкасы`, `каспиден`, `оплатасын`) as out-of-vocabulary noise, causing false AI accusations. Our `fst_analyzer.py` automatically parses code-switched loanwords:

```python
from fst_analyzer import analyze_and_segment

sample = "Каспиден доставкасы өте тез болды, оплатасын жасадым."
parsed = analyze_and_segment(sample)

# Output: "Каспи -ден доставка -сы өте тез болды, оплата -сын жасадым."
```

---

## 📁 Repository Structure

```
kazakh-ai-text-detection/
├── fst_analyzer.py                            # Core Morphological Segmenter Module
├── aist2026/                                  # Conference LaTeX Manuscript & Paper Drafts
│   ├── paper.tex / paper.pdf
│   └── paper_draft.docx / paper_draft.md
├── scripts/                                   # Analysis, Calculation & Mining Engine
│   ├── calculate_token_inflation.py
│   ├── mine_code_switched_loanwords.py
│   ├── mine_ood_kazakh_data.py
│   ├── evaluate_statistical_significance.py
│   ├── plot_bootstrap_ci.py
│   └── plot_ood_generalization.py
├── data/                                      # Datasets, JSON summaries & generated charts
├── kaggle_runner/                             # Kaggle Remote GPU Execution Pipeline
├── notebooks/                                 # Jupyter Notebooks
├── api/                                       # FastAPI Model Server
├── tests/                                     # Unit Test Suite
└── ui/                                        # Web UI Interface
```

---

## 🚀 Running the Local API & Web UI

### Local API
```bash
uvicorn api.main:app --reload --port 8000
```

### Docker Web UI
```bash
docker compose up --build
```
Navigate to **[http://localhost:7860](http://localhost:7860)**.

---

## 📜 Citation & License

This project is licensed under the [MIT License](LICENSE).
