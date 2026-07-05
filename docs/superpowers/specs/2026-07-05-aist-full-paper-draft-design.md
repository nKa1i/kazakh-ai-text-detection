# AIST 2026 Full Paper Draft Design Spec

* **Date:** 2026-07-05
* **Authors:** Daulet Anekesh, Irina Ualiyeva
* **Conference:** 13th International Conference on Analysis of Images, Social Networks, and Texts (AIST 2026)
* **Status:** Draft for Review

---

## 1. Goal Description
The objective is to draft a comprehensive 15-page LNCS-formatted research paper based on the abstract submitted to AIST 2026. The paper benchmarks multilingual and monolingual BERT models on Kazakh AI-generated text detection and explores how rule-based Finite-State Transducer (FST) morphological segmentation reduces false positive rates on short texts. 

To facilitate sharing and collaboration with the supervisor, the paper will be drafted locally in Markdown (`paper_draft.md`) and compiled into a styled Word Document (`paper_draft.docx`) using a Python script.

---

## 2. Proposed Document Structure (`paper_draft.md`)

The draft will follow the standard structure of a Springer LNCS computer science paper:

1. **Title & Abstract**
   * Title: *Detecting AI-Generated User Reviews in Kazakh: A Study on BERT Model Performance and False Positive Reduction*
   * Authors: Daulet Anekesh, Irina Ualiyeva
   * Keywords: AI-generated text detection, Kazakh language, BERT models, Finite-State Transducer (FST), morphological analysis, KazSAnDRA, low-resource NLP

2. **Introduction**
   * Background on Kazakh NLP, the rise of LLMs (leading to Kazakh text generation), and the necessity of AI detectors.
   * Highlight the severe impact of false positives (falsely accusing students, writers, or users of AI usage) in precision-critical applications.
   * State the core contribution: benchmarking monolingual vs. multilingual BERT, and introducing FST morphological analyzer to reduce false positives on short texts.

3. **Related Work**
   * Brief review of general AI-generated text detection methods.
   * State of Kazakh NLP and low-resource text classification.
   * Traditional morphological tools in Turkic languages.

4. **Dataset Construction**
   * **Human Corpus:** KazSAnDRA (Kazakh Sentiment Analysis Dataset of Reviews and Advertisements).
   * **AI Corpus:** Generated using a Kazakh-specific LLM, aligned to match lengths, styles, and vocabulary distributions.
   * Data split: Train set (8,848 samples) and Eval/Test set (984 samples).

5. **Methodology: FST Morphological Analyzer**
   * Description of the Kazakh agglutinative morphology challenge.
   * The design of `AdvancedKazakhFSTAnalyzer` that splits suffixes from stems for:
     * Plurals (`-лар`, `-лер`, etc.)
     * Possessives (`-мыз`, `-міз`, `-ңыз`, `-ңіз`, etc.)
     * Cases (`-ның`, `-нің`, `-ға`, `-ге`, etc.)
   * Examples of the segmentation process.

6. **Model Configurations & Evaluation Setup**
   * Description of the 4 models: `mBERT`, `XLM-R`, `KazBERT`, `KazRoBERTa`.
   * Explain the two modes of training:
     * **Pure Baseline:** Trained on raw text.
     * **FST Segmented:** Trained on FST-preprocessed texts.

7. **Experimental Results & Analysis**
   * Incorporate the exact evaluation results:
     
     | Model | Mode | Overall Acc | F1-Score | Acc (Short $\le 60$) | Acc (Long $> 60$) | FPs (Short) |
     | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
     | mBERT | Pure | 95.42% | 95.46% | 94.43% | 96.26% | 58 |
     | mBERT | FST | 93.62% | 93.71% | 92.83% | 94.29% | 70 |
     | XLM-R | Pure | 95.45% | 95.52% | 94.08% | 96.59% | 67 |
     | XLM-R | FST | 94.35% | 94.50% | 93.23% | 95.30% | 83 |
     | KazBERT | Pure | 94.59% | 94.63% | 93.52% | 95.49% | 63 |
     | KazBERT | FST | 94.56% | 94.63% | 93.34% | 95.59% | 65 |
     | **KazRoBERTa** | **Pure** | **96.10%** | **96.15%** | **95.05%** | **96.98%** | **53** |
     | **KazRoBERTa** | **FST** | **96.07%** | **96.07%** | **94.99%** | **96.98%** | **36** |
   
   * Key findings:
     1. Monolingual pretraining (KazRoBERTa) outperforms multilingual models.
     2. FST segmentation maintains overall accuracy but dramatically reduces false positives for KazRoBERTa on short texts by **32%** (from 53 to 36).
     3. Explanations of this phenomenon: morphological segmentation exposes grammatical suffixes, preventing the model from misidentifying rare inflections as AI-generated patterns.

8. **Conclusion & Practical Recommendations**
   * Deploy "Pure" models for overall accuracy.
   * Deploy "FST-augmented" models for precision-critical systems where false positives carry high cost.

---

## 3. Compilation Script Specs (`compile_doc.py`)
A script that parses `paper_draft.md` and generates `paper_draft.docx` using `python-docx`.
* **Features:**
  * Title page formatting with author details.
  * Correct heading hierarchies (Heading 1, Heading 2, Heading 3).
  * Table rendering with border grids.
  * Bulleted and numbered lists support.

---

## 4. Verification Plan
* Validate that `paper_draft.docx` is generated without error.
* Open `paper_draft.docx` (manually) to check font sizes, layout, and table alignments.
