# Kazakh AI Text Detection & Morphologically-Grounded Fact Verification Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Evaluation: 520+ Tests Passing](https://img.shields.io/badge/Tests-520%2B%20Passing-green.svg)](tests/)
[![Peer Review: Anonymous Double-Blind](https://img.shields.io/badge/Review-Double--Blind%20ARR%2FCOLING%202027-purple.svg)](docs/coling_2027_arr_submission_brief.md)

An end-to-end computational framework and empirical benchmark for dual-axis document auditing in low-resource agglutinative languages. The system unifies **FST-augmented multi-generator AI text detection** with **evidence-grounded factual verification (Kazakh-FEVER 3K)**, mapping document evaluations into an operational **2D Cartesian Trust Matrix** $(T_{\text{fact}}, T_{\text{gen}})$.

---

## Key Research Highlights

* **Kazakh-FEVER 3K Benchmark:** The first human-validated fact verification corpus for Kazakh, comprising 3,000 claim-evidence pairs grounded over 250,000 Wikipedia passages with strict Cohen's $\kappa = 0.840$ inter-annotator consensus.
* **Hard NEI Adversarial Protocol:** Mitigates superficial lexical overlap heuristics through systematic morphological mutations targeting negation infixes (`-ба/-бе`), reportative evidentials (`-ыпты/-іпті` vs. witnessed `-ды/-ді`), and epistemic modality particles (`болуы мүмкін`).
* **Hybrid Evidence Retrieval:** Combines morpheme-stemmed BM25 with dense multilingual representations (mContriever) via Reciprocal Rank Fusion (RRF), achieving **99.8% Recall@5** and **0.994 MRR** across 250k knowledge passages.
* **16-Dimensional Morphological NLI Cross-Encoder:** Augments deep pretrained encoders with an explicit 16-D morphotactic diagnostic vector ($\mathbf{m}_{\text{affix}}$), attaining **82.6% Accuracy**, **82.1% Macro-F1**, and **71.4% Strict Joint FEVER Score** (+11.5 pp over XLM-RoBERTa on Hard NEI).
* **Dual-Stream Generator-Invariant Detector:** Dynamic gating between transformer representations and an 83-rule FST analyzer reduces short-text false positive accusations by **43.2%** and achieves **+42.18 pp OOD gain**.
* **Memory-Bounded Document Chunker:** `SentencePreservingChunker` incorporates 10 Kazakh abbreviation guards and dynamic worst-case Top-K pooling, evaluating documents up to 25,000 words within a **1.4 GB VRAM budget**.
* **Continuous 2D Cartesian Trust Matrix:** Decouples generative style from factual veracity, routing predictions across four operational risk quadrants with real-time HUD telemetry.

---

## Scientific Benchmarks

### 1. Kazakh-FEVER Evidence Retrieval ($N = 250,000$ Passages)

| Retrieval Pipeline | Recall@1 (%) | Recall@3 (%) | Recall@5 (%) | MRR |
| :--- | :---: | :---: | :---: | :---: |
| Standard BM25 (Word Surface) | 78.4 | 89.1 | 93.2 | 0.842 |
| Morpheme-Stemmed BM25 (83-Rule FST) | 88.6 | 96.2 | 98.4 | 0.927 |
| Dense Retriever (mContriever) | 91.2 | 97.4 | 99.1 | 0.945 |
| **Hybrid Pipeline (Morpho-BM25 + mContriever RRF)** | **98.7** | **99.6** | **99.8** | **0.994** |

---

### 2. Fact Verification & Joint FEVER Performance

| Model Architecture | Input Mode | Accuracy (%) | Macro-F1 (%) | Strict Joint FEVER (%) | Hard NEI F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| mBERT-base | Surface Only | 68.4 | 67.2 | 51.3 | 48.2 |
| XLM-RoBERTa-base | Surface Only | 74.2 | 73.8 | 61.5 | 58.6 |
| XLM-RoBERTa-large | Surface Only | 76.5 | 76.1 | 64.2 | 61.3 |
| LLaMA-3-8B-Instruct | Zero-Shot Prompt | 62.1 | 60.4 | 46.8 | 42.1 |
| **Ours (Hybrid Retrieval + 16-D Morpho NLI)** | **Hybrid + Morpho** | **82.6** | **82.1** | **71.4** | **72.8** |

---

### 3. Morphological Feature Ablation Breakdown

| Configuration | Feature Set | Accuracy (%) | Macro-F1 (%) | Hard NEI F1 (%) | Delta Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Full Model** | **All 16 Features** | **82.6** | **82.1** | **72.8** | **--** |
| Config 1: w/o Negation Affix Mismatch ($\Delta_{\text{neg}}$) | 15 Features | 78.2 | 77.5 | 65.4 | -4.6 pp |
| Config 2: w/o Calendar/Numeral Distance ($\Delta_{\text{year}}, \Delta_{\text{num}}$) | 14 Features | 79.1 | 78.4 | 68.2 | -3.7 pp |
| Config 3: w/o Evidential/Modality Markers ($m_5, m_6$) | 14 Features | 80.3 | 79.8 | 69.1 | -2.3 pp |
| Config 4: w/o FST Root vs Surface Jaccard ($J_{\text{root}}, J_{\text{surf}}$) | 14 Features | 79.5 | 78.9 | 67.0 | -3.2 pp |
| Config 5: w/o Case-Agreement Alignment ($m_{13}$) | 15 Features | 81.0 | 80.4 | 70.5 | -1.7 pp |
| Config 6: Surface Only (Ablate All 16 Features) | 0 Features | 72.1 | 71.5 | 59.8 | -10.6 pp |

---

### 4. Continuous 2D Cartesian Trust Matrix

Document evaluations are mapped into continuous coordinates $(T_{\text{fact}}, T_{\text{gen}}) \in [-1, 1] \times [0, 1]$:
* **Quadrant I ($T_{\text{fact}} \ge 0, T_{\text{gen}} \ge 0.5$): Verified Human Fact** (Safe, authentic human journalism).
* **Quadrant II ($T_{\text{fact}} < 0, T_{\text{gen}} \ge 0.5$): Human Misinformation** (Debunking alert; no false AI accusation).
* **Quadrant III ($T_{\text{fact}} < 0, T_{\text{gen}} < 0.5$): AI Hallucination** (Critical risk automated disinformation).
* **Quadrant IV ($T_{\text{fact}} \ge 0, T_{\text{gen}} < 0.5$): Accurate AI Synthesis** (Benign machine generation with disclosure).

Composite scalar trust metric:
$$\mathcal{T}(x) = \sqrt{\frac{1}{2}\left(\max(0, T_{\text{fact}})^2 + T_{\text{gen}}^2\right)}$$

---

## Quickstart & Reproducibility

### 1. Environment Setup

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/kazakh-ai-text-detection-57F8.git
cd kazakh-ai-text-detection-57F8

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Morphological FST Segmentation

```python
from fst_analyzer import analyze_and_segment

sample_text = "Қаныш Сәтбаев Жезқазған мыс кен орындарын зерттемеген."
analysis = analyze_and_segment(sample_text)
print("FST Morphological Segmentation:", analysis)
```

### 3. Reproduce Benchmark Results

A standalone, dependency-light reproduction script is provided in `papers/kazakh_fever_conference/supplementary/evaluate_fever.py`:

```bash
# Reproduce Information Retrieval metrics (Table 2)
python papers/kazakh_fever_conference/supplementary/evaluate_fever.py --task retrieval

# Reproduce Fact Verification and Ablation metrics (Table 3 & Table 4)
python papers/kazakh_fever_conference/supplementary/evaluate_fever.py --task verification
```

### 4. Launch Interactive Gradio Dashboard

```bash
# Launch the 4-tab forensic inspection platform
python ui/app.py
```
Open `http://localhost:7860` to access the Document Forensic Analyzer, Morphological FST Lab, Kazakh-FEVER Browser, and 2D Trust Matrix HUD.

### 5. Run Regression Test Suite

```bash
# Execute the full automated test suite (520+ tests)
python -m unittest discover -s tests -p "test_*.py"
```

---

## Repository Structure

```
kazakh-ai-text-detection/
├── fst_analyzer.py                            # 83-Rule Kazakh Morphological FST Analyzer
├── verification/                              # Evidence Retrieval & 16-D Morphological NLI Modules
│   ├── hybrid_retriever.py                    # BM25 + Dense mContriever RRF Pipeline
│   ├── morpho_nli_verifier.py                 # 16-D Affix Feature Extraction & Classification
│   └── trust_matrix.py                        # 2D Cartesian Trust Space & Telemetry Calculations
├── ui/                                        # Gradio 4-Tab Interactive Explainability Dashboard
│   ├── app.py                                 # Main Application Entry Point
│   └── components/                            # SVG Trust Matrix Canvas, Heatmaps & Drawers
├── papers/
│   └── kazakh_fever_conference/               # COLING 2027 / ARR Double-Blind Manuscript Package
│       ├── main.tex                           # Two-Column Article using Official acl.sty
│       ├── sections/                          # Modular Section TeX Sources
│       ├── figures/                           # Standalone Vector TikZ Figures
│       └── supplementary/                     # Anonymous Benchmark Split & evaluate_fever.py
├── data/                                      # Benchmark Splits, Wikipedia Chunks & Cache
├── tests/                                     # Automated Regression Suite (520+ Passing Unit Tests)
├── LICENSE                                    # MIT Open-Source License
└── README.md                                  # Repository Documentation
```

---

## Citation & Double-Blind Review Notice

This codebase is provided as an anonymized open-science repository under double-blind peer review for the **COLING 2027 / ARR October 2026 Cycle**. Author names, affiliations, and grant acknowledgments will be reinstated upon publication.

```bibtex
@inproceedings{anonymous2027kazakhfever,
  title={Kazakh-FEVER 3K: A Morphologically-Grounded Fact Verification Benchmark and 2D Trust Matrix for Low-Resource Agglutinative Languages},
  author={Anonymous},
  booktitle={Proceedings of the 31st International Conference on Computational Linguistics (COLING 2027)},
  year={2027},
  note={Under double-blind peer review}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
