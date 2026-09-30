# Design Specification: 2024–2026 SOTA Literature Modernization & Related Work Enhancement

**Date:** 2026-09-30  
**Target Venue:** COLING 2027 / ARR October 2026 Cycle (Core A / CCF-B)  
**Document:** `papers/kazakh_fever_conference/main.pdf`  
**Status:** Approved by User  

---

## 1. Problem Formulation & Objective

Supervisor Dr. Irina M. Ualiyeva identified that premier NLP conferences (ACL, EMNLP, COLING) require bibliographies heavily weighted toward contemporary state-of-the-art literature. The previous bibliography contained older citations from 2009–2021 (BM25 2009, Kazakh corpus 2013, FEVER 2018, BERT 2019, Schuster et al. 2019, TyDi QA 2020, DPR 2020, XLM-R 2020, FEVEROUS 2021, Bekmanova et al. 2021).

The user explicitly requested reducing pre-2024 citations to near zero and focusing strictly on **2024–2026** publications:
- **0 papers from 2023**.
- **91.7%+ of all sources published in 2024–2026**.
- **Only 2 foundational anchors preserved**: Thorne et al. (2018) for the FEVER task definition and Conneau et al. (2020) for the XLM-RoBERTa backbone encoder.

---

## 2. Target Bibliography Architecture (24 Sources)

### Category A: Fact Verification & LLM Grounding (2024–2025)
1. `tang-etal-2024-minicheck` — Tang, Laban, Durrett (EMNLP 2024): *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents*.
2. `zheng-etal-2024-evidence` — Zheng et al. (Findings of ACL 2024): *Evidence Retrieval is almost All You Need for Fact Verification*.
3. `guan-etal-2024-language` — Guan, Dodge, Wadden, Huang, Peng (NAACL 2024): *Language Models Hallucinate, but May Excel at Fact Verification*.
4. `yue-etal-2024-retrieval` — Yue et al. (ACL 2024): *Retrieval Augmented Fact Verification by Synthesizing Contrastive Arguments*.
5. `si-etal-2024-checkwhy` — Si et al. (ACL 2024): *CHECKWHY: Causal Fact Verification via Argument Structure*.
6. `wang-etal-2024-factcheck-bench` — Wang et al. (Findings of EMNLP 2024): *Factcheck-Bench: Fine-Grained Evaluation Benchmark for Automatic Fact-checkers*.
7. `mitra-etal-2025-factlens` — Mitra, Zhang, Rahman, Hruschka (Findings of ACL 2025): *FactLens: Benchmarking Fine-Grained Fact Verification*.

### Category B: Turkic & Kazakh NLP Breakthroughs (2024–2026)
8. `togmanov-etal-2025-kazmmlu` — Togmanov et al. (ACL 2025): *KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan*.
9. `laiyk-etal-2025-instruction` — Laiyk et al. (ACL 2025): *Instruction Tuning on Public Government and Cultural Data for Low-Resource Language: A Case Study in Kazakh*.
10. `yeshpanov2024kazsandra` — Yeshpanov & Varol (LREC-COLING 2024): *KazSAnDRA: Kazakh Sentiment Analysis Dataset of Reviews and Attitudes*.
11. `sagyndyk2025kazroberta` — Sagyndyk, Murzakhmetov, Yakunin (TechRxiv 2025): *Kaz-RoBERTa Conversational Technical Report*.
12. `tukenov2026sozkz` — Tukenov (arXiv 2026): *SozKZ: Training Efficient Small Language Models for Kazakh from Scratch*.

### Category C: AI Text Detection & Adversarial Robustness (2024)
13. `wang2024raid` — Wang, Mansoor, He, Zou (ACL 2024): *RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors*.
14. `hans2024binoculars` — Hans et al. (NAACL 2024): *Spotting LLMs with Binoculars: Zero-Shot Detection of Machine-Generated Text*.
15. `bao2024fastdetectgpt` — Bao, Zhao, Teng, Yang, Zhang (ICLR 2024): *Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature*.

### Category D: Multilingual Representation, LLM Baselines & Hybrid Retrieval (2024–2026)
16. `dubey2024llama3` — Dubey et al. (arXiv 2024): *The Llama 3 Herd of Models* (Zero-shot baseline citation in Section 5).
17. `chen-etal-2024-m3` — Chen et al. (Findings of ACL 2024): *M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation* (BGE-M3 baseline in Section 5).
18. `ustun-etal-2024-aya` — Üstün et al. (ACL 2024): *Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model* (Multilingual foundation model context in Section 2).
19. `bandarkar-etal-2024-belebele` — Bandarkar et al. (ACL 2024): *The Belebele Benchmark: A Parallel Reading Comprehension Dataset in 122 Language Variants* (Multilingual reading comprehension covering Kazakh in Section 2).
20. `mudunuri-etal-2026-bounded` — Mudunuri et al. (SIGDIAL / ACL 2026): *Bounded Conversational Memory with Hybrid Retrieval and Evidence Highlighting for Multi-Session Dialogue Systems* (Sparse BM25 + dense hybrid retrieval with RRF in Section 4).
21. `izacard2022mcontriever` — Izacard et al. (TMLR 2022): *Unsupervised Dense Information Retrieval with Contrastive Learning* (Our dense retrieval backbone).
22. `guo2026kazakh` — Anonymous (TACL 2026): *Kazakh-FEVER and Multi-dimensional Trust Modeling in Central Asian Natural Language Processing*.

### Category E: Foundational Anchors (< 2024)
23. `thorne2018fever` — Thorne et al. (NAACL 2018): *FEVER: A Large-scale Dataset for Fact Extraction and VERification*.
24. `conneau2020xlmr` — Conneau et al. (ACL 2020): *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-RoBERTa backbone).

---

## 3. Sections to Modify

1. **`papers/kazakh_fever_conference/custom.bib`**:
   - Remove obsolete pre-2024 entries: `robertson2009bm25`, `makhambetov2013kazakh`, `schuster2019fever`, `devlin2019bert`, `clark2020tydi`, `karpukhin2020dpr`, `aly2021feverous`, `bekmanova2021kazakh`, `mitchell2023detectgpt`, `touvron2023llama`.
   - Add new 2024–2026 BibTeX entries: `dubey2024llama3`, `chen-etal-2024-m3`, `ustun-etal-2024-aya`, `bandarkar-etal-2024-belebele`, `wang-etal-2024-factcheck-bench`, `mitra-etal-2025-factlens`, `togmanov-etal-2025-kazmmlu`, `tukenov2026sozkz`, `mudunuri-etal-2026-bounded`.

2. **`papers/kazakh_fever_conference/sections/02_related_work.tex`**:
   - Update Section 2.1 to cite `mitra-etal-2025-factlens` and `wang-etal-2024-factcheck-bench` alongside `bandarkar-etal-2024-belebele`.
   - Update Section 2.2 to cite `togmanov-etal-2025-kazmmlu`, `ustun-etal-2024-aya`, and `tukenov2026sozkz`.
   - Update Section 2.3 to lead directly with `bao2024fastdetectgpt`, `hans2024binoculars`, and `wang2024raid`.

3. **`papers/kazakh_fever_conference/sections/04_methodology.tex`**:
   - Update Section 4.2 retrieval citation to cite modern hybrid BM25 + dense fusion `\citep{mudunuri-etal-2026-bounded}` and `\citep{izacard2022mcontriever}`.

4. **`papers/kazakh_fever_conference/sections/05_experiments.tex`**:
   - Update baseline descriptions:
     - Sparse BM25 / hybrid retrieval: `\citep{mudunuri-etal-2026-bounded}`.
     - Dense retrieval baseline: BGE-M3 `\citep{chen-etal-2024-m3}` and mContriever `\citep{izacard2022mcontriever}`.
     - LLM zero-shot baseline: LLaMA-3-8B-Instruct `\citep{dubey2024llama3}`.
     - Surface overlap shortcut pathology: `\citep{wang-etal-2024-factcheck-bench}`.

5. **`papers/kazakh_fever_conference/sections/08_limitations.tex`**:
   - Ensure clean 1-column fit on Page 9 to preserve the strict 10-page document budget.

---

## 4. Layout & Submission Constraints

- **Strict Page Budget**: Exactly 10 pages. Sections 1–7 terminate on Page 8. Section 8 begins on Page 9. References occupy Pages 9–10.
- **Compilation**: Clean compilation under XeLaTeX + BibTeX with 0 undefined citations and 0 missing font characters.
- **Double-Blind Anonymity**: Strictly 0 occurrences of author names, affiliations, or identifying grant/URL strings in compiled text and metadata.
- **Regression Suite**: All 8 tests in `tests/test_coling_submission_integrity.py` and all 521 repository regression tests must pass.
- **AIST Invariance**: Strictly 0 diff lines on `aist2026/paper.tex`.
