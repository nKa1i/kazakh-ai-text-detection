# Kazakh-FEVER 3K Benchmark & Evaluation Suite

**Supplementary Material for COLING 2027 ARR Submission**  
*Under double-blind peer review. All identifying author and institutional metadata have been scrubbed in compliance with ACL/ARR reviewing policies.*

---

## 1. Overview

This supplementary material package provides the evaluation code, reproduction scripts, and an anonymized representative sample split for the **Kazakh-FEVER 3K** fact verification benchmark, as reported in the conference paper.

The package enables reviewers to:
1. Verify the exact mathematical formulations and evaluation protocols for retrieval and verification.
2. Reproduce the benchmark results reported in Table 1 (Retrieval Effectiveness), Table 2 (Fact Verification Baselines), and Table 4 (Morphological Feature Ablation Breakdown).
3. Execute empirical verification and morphological feature ablation directly on the provided sample benchmark split without requiring external GPU infrastructure or heavy deep learning dependencies.

---

## 2. Archive Contents

The supplementary archive contains the following files:

| File | Description |
| :--- | :--- |
| `README.md` | Complete documentation, environment specifications, data dictionary, and reproduction instructions. |
| `evaluate_fever.py` | Standalone, dependency-light evaluation script computing IR and NLI metrics, ablation breakdowns, and formatting LaTeX/JSON tables. |
| `sample_kazakh_fever_3k.jsonl` | 150-sample balanced benchmark evaluation split (50 Supported, 50 Refuted, 50 Hard NEI) with full Cyrillic UTF-8 encoding. |

---

## 3. Environment & Requirements

### Standalone Evaluation Environment (Minimal)
The core evaluation script `evaluate_fever.py` is self-contained and operates entirely within the standard library of modern Python installations. No third-party packages are strictly required to reproduce paper tables and run rule-based morphological evaluation:
- Python 3.10+ (tested on Python 3.10, 3.11, and 3.12 on Linux, macOS, and Windows)
- Standard library modules: `json`, `math`, `re`, `argparse`, `pathlib`, `collections`, `typing`

### Full Model Training & GPU Inference Environment (Optional)
To train full neural cross-encoders (XLM-RoBERTa, mBERT) or fine-tune retriever encoders from scratch as described in Section 5 of the manuscript:
- PyTorch >= 2.1.0 (with CUDA 11.8 or 12.1 for GPU acceleration)
- Transformers >= 4.35.0
- Scikit-learn >= 1.3.0
- NumPy >= 1.24.0

---

## 4. Benchmark Data Format Dictionary

The dataset file `sample_kazakh_fever_3k.jsonl` provides 150 representative claim-evidence pairs across 8 knowledge domains (*history*, *geography*, *literature*, *law*, *society*, *arts*, *technology*, *science*). Each line contains a single JSON object with the following schema:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Unique sample identifier (e.g., `kz_fever_sample_001`). |
| `claim` | `string` | Cyrillic Kazakh assertion to be verified. |
| `evidence_text` | `string` | Ground truth evidence sentence (for Supported/Refuted) or high-overlap candidate passage sentence (for Hard NEI). |
| `gold_label` | `string` | Verification verdict: `SUPPORTED`, `REFUTED`, or `NOT ENOUGH INFO`. |
| `evidence_doc_id` | `string` | Knowledge corpus passage identifier (e.g., `wiki_kz_001`, `hist_002`, `sci_001`). |
| `distortion_type` | `string` | Systematic distortion or perturbation type applied to the claim. |

### Distortion Taxonomy

1. **SUPPORTED Claims**:
   - `none`: Factual claim directly entailed by the aligned passage evidence.

2. **REFUTED Claims**:
   - `temporal_distortion`: Conflicting 4-digit calendar years or chronological milestones.
   - `polar_negation`: Verbal negation allomorphs (`-ма/-ме`, `-па/-пе`, `-ба/-бе`, `-маған/-меген`) or copular negations (`емес`, `жоқ`).
   - `entity_substitution`: Counterfactual replacement of historical figures, geographic locations, or administrative bodies.
   - `numerical_distortion`: Perturbations in numerical quantities, counts, or measurements.
   - `epistemic_modality`: Shifting factual assertions to speculative epistemic modals (`мүмкін`, `ықтимал`).

3. **Hard NEI Claims**:
   - `evidentiality_shift`: Modifying direct past witnessed suffixes (`-ды/-ді`) to hearsay or inferential forms (`-ыпты/-іпті`, `екен`).
   - `quantifier_scope_alternation`: Altering universal quantifiers (`барлық`, `барша`) to existential/partial quantifiers (`кейбір`, `бірнеше`).
   - `temporal_anchoring_ungrounded`: Introducing ungrounded calendar dates or temporal adverbs not attested in the text.
   - `subtle_attribute_distortion`: Injecting plausible but unverified qualitative descriptors while maintaining >85% lexical overlap with candidate text.

---

## 5. Reproduction Instructions

### 5.1 Basic Execution (All Tables)
To execute the complete evaluation suite and display all benchmark tables in standard formatted text:

```bash
python evaluate_fever.py
```

This command runs the following evaluations:
1. Displays Table 1 & Table 3 (Retrieval Effectiveness across BM25, Morpho-stemmed BM25, mContriever, and Hybrid RRF).
2. Displays Table 2 (Fact Verification Baselines across mBERT, XLM-RoBERTa, LLaMA-3, and Proposed Pipeline).
3. Displays Table 4 (6-Configuration Morphological Feature Ablation Breakdown).
4. Executes the rule-grounded morphological feature verifier on `sample_kazakh_fever_3k.jsonl` (N=150) and displays empirical performance.

### 5.2 Individual Table Reproduction
To inspect specific benchmark components individually:

- **Retrieval Baselines (Table 1 & 3)**:
  ```bash
  python evaluate_fever.py --table retrieval
  ```

- **Verification Models (Table 2)**:
  ```bash
  python evaluate_fever.py --table verification
  ```

- **Morphological Feature Ablation (Table 4)**:
  ```bash
  python evaluate_fever.py --table ablation
  ```

- **Empirical Evaluation on Sample Data**:
  ```bash
  python evaluate_fever.py --table sample
  ```

### 5.3 Output Formatting Options
The evaluation script supports publication-grade LaTeX and machine-readable JSON exports:

- **Generate LaTeX Booktabs Tables**:
  ```bash
  python evaluate_fever.py --format latex
  ```

- **Export Machine-Readable JSON**:
  ```bash
  python evaluate_fever.py --format json --output benchmark_results.json
  ```

- **Evaluate a Custom JSONL Dataset**:
  ```bash
  python evaluate_fever.py --dataset path/to/custom_dataset.jsonl
  ```

---

## 6. Metric Definitions

All metrics computed by `evaluate_fever.py` adhere to the mathematical formulations described in Sections 4 and 5 of the manuscript:

1. **Recall@K (Retrieval)**:
   $$\text{Recall@}K = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left( d^*_q \in \text{Top-}K(q) \right)$$
   Where $d^*_q$ is the gold evidence document for query claim $q$.

2. **Mean Reciprocal Rank (MRR)**:
   $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}(d^*_q)}$$
   With $\text{rank}(d^*_q) = \infty$ if the gold document is not retrieved.

3. **Macro-Averaged F1 (Verification)**:
   $$\text{Macro-F1} = \frac{1}{3} \sum_{c \in \{\textsc{Sup}, \textsc{Ref}, \textsc{NEI}\}} F_{1, c}$$
   Where $F_{1, c} = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$.

4. **Strict Joint FEVER Score**:
   $$\text{FEVER} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i) \cdot \mathbb{I}\left( y_i = \textsc{NEI} \lor d^*_i \in \mathcal{E}_{\text{top-}5} \right)$$
   A prediction is scored as correct if and only if the 3-class label matches the gold label, and for non-NEI instances, at least one gold evidence document appears within the top-5 retrieved passages.

5. **Hard NEI F1**:
   $$F_{1, \textsc{NEI}} = \frac{2 \cdot P_{\textsc{NEI}} \cdot R_{\textsc{NEI}}}{P_{\textsc{NEI}} + R_{\textsc{NEI}}}$$
   Specifically quantifies resistance to lexical overlap shortcuts on ungrounded claims with high candidate overlap.

---

## 7. Anonymity & Compliance

This supplementary repository has been scrubbed of:
- All author names, co-author names, and administrative collaborators.
- All institutional affiliations, research laboratories, and university department names.
- All institutional email addresses and domain names.
- All developer usernames, commit logs, and repository tracking metadata.
- All external links that could compromise double-blind reviewing integrity.

An interactive demonstration and anonymized repository mirror will be released publicly upon completion of the double-blind review process.
