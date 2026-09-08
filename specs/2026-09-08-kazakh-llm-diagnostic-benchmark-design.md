# Kazakh LLM Linguistic Diagnostic & Robust Benchmark Engine: Design Specification

**Date:** 2026-09-08  
**Author:** Daulet Anekesh / AI Research Assistant  
**Status:** Approved for Implementation  
**Target Venues:** ACL / EMNLP / COLING & MSc Thesis (Chapters 3 & 4)

---

## 1. Executive Summary & Scientific Motivation

Existing AI text detection benchmarks predominantly focus on high-resource languages (English, Chinese) and long, well-formed texts (essays, Wikipedia). In low-resource, morphologically rich agglutinative languages such as Kazakh, conventional detectors exhibit severe performance degradation—most notably elevated false positive rates on short texts ($\le 60$ characters).

Following our Master's Thesis blueprint approved by Prof. Guo (*"Robust AI-Generated Text Detection and Trustworthy Content Verification for Low-Resource Kazakh"*), this project builds Phase 1: a **Diagnostic Linguistic Profiling Suite** and a **Multi-LLM Paired Benchmark Engine (`Kazakh-AIGC-Robust-4K`)**.

The goal is to quantitatively profile how modern Large Language Models (LLMs) generate Kazakh text compared to authentic human writing across morphological, lexical, and semantic dimensions before training detectors, establishing an empirical foundation for morphology-aware detection.

---

## 2. System Architecture

```
[ Human Seed Data: KazSAnDRA ] (1,000 stratified samples)
            │
            ├─► Metadata Extraction: (Domain, Category, Rating 1-5, Length Bracket)
            │
            ▼
[ Multi-LLM Generation Engine (Kaggle T4/P100) ]
      ├── Generator 1: Sherkala-7B (Kazakh-adapted LLaMA-2)
      ├── Generator 2: Qwen-2.5-7B-Instruct (SOTA Central Asian Multilingual)
      └── Generator 3: LLaMA-3.1-8B-Instruct (Global Open Foundation)
            │
            ▼
[ Unified Paired Dataset (4,000 samples) ]
            │
            ├───────────────────────────────┬───────────────────────────────┐
            ▼                               ▼                               ▼
[ 6-Factor Diagnostic Suite ]   [ RAID-Style Perturbations ]   [ Baseline Zero-Shot Drop ]
• Token Inflation Ratio (T/W)   • Cyrillic Typos               • KazRoBERTa Pure vs. FST
• Suffix Fragmentation Rate     • Paraphrasing                 • OOD-Model Evaluation
• Lexical Richness (TTR)        • Code-Switching Loanwords     • Length Breakdown (<60, etc.)
• Loanword & Register Shift     
• Length Stratification         
• Semantic Distance Matrix      
```

---

## 3. Component Details

### 3.1 Data Engine & Paired Sampling
1. **Human Seed Sampling (`KazSAnDRA`):**
   * Sample $N=1,000$ authentic reviews from the KazSAnDRA dataset.
   * Stratified sampling strategy:
     * **Length Brackets:** Short ($\le 60$ chars: 350 samples), Medium ($61–200$ chars: 450 samples), Long ($> 200$ chars: 200 samples).
     * **Domains:** Market/E-commerce, AppStore, Services/Mapping, Bookstore.
     * **Sentiment:** 1 to 5 stars (balanced distribution).
2. **Prompt Conditioning Pipeline:**
   * Each human review's metadata (domain, rating, target length) forms a structured system/user prompt template.
   * Prompts instruct the LLM to write authentic, colloquial Kazakh customer reviews matching the exact domain and rating without revealing AI-like disclaimer boilerplate.
3. **Multi-LLM Execution (Kaggle GPU):**
   * Load each model in 4-bit/8-bit quantization (`bitsandbytes`) or BF16/FP16 across dual T4 GPUs (or single P100).
   * Inference parameters: Temperature = 0.7, Top-P = 0.9, Repetition Penalty = 1.1.
   * Outputs structured as `kazakh_aigc_diagnostic_4k.parquet` containing:
     * `text`, `label` (0: Human, 1: AI), `generator` (`human`, `sherkala_7b`, `qwen_2.5_7b`, `llama_3.1_8b`), `domain`, `rating`, `length_bracket`, `char_length`, `word_count`.

---

### 3.2 Six-Factor Diagnostic Profiling Suite

#### Metric 1: Subword Token Inflation Ratio ($T/W$)
* **Definition:** Ratio of generated subword tokens to whitespace-delimited words:
  $$\text{T/W} = \frac{\text{Total Subword Tokens}}{\text{Total Whitespace Words}}$$
* **Evaluated Tokenizers:**
  * KazRoBERTa (WordPiece, 52k vocab)
  * Qwen-2.5 (Byte-level BPE, 152k vocab)
  * LLaMA-3.1 (Tiktoken BPE, 128k vocab)
  * XLM-RoBERTa / Sherkala (SentencePiece, 250k vocab)

#### Metric 2: Morphological Suffix Fragmentation & Morphemic Depth
* **Tool:** Apertium-Kaz / Rule-based Kazakh FST.
* **Metrics:**
  * Suffix Preservation Rate: Proportion of Kazakh inflectional suffixes (*-лар/лер, -ның/нің, -ға/ге, -да/де, -сы/сі*) preserved as whole morphological units vs. fragmented into arbitrary sub-tokens.
  * Average morphemic depth (number of identified suffixes per stem).

#### Metric 3: Lexical Diversity & Richness
* **Metrics:**
  * Type-Token Ratio (TTR): $\frac{|V|}{N}$ normalized across fixed-length windows.
  * Distinct-1 and Distinct-2: Proportions of unique unigrams and bigrams.
  * Measures whether LLMs suffer from repetitive vocabulary patterns compared to diverse human colloquial reviews.

#### Metric 4: Loanword & Register Shift Analysis (MultiSocial ACL 2025)
* **Rationale:** Authentic Kazakh reviews heavily incorporate Russian commercial loanwords (*доставка, заказ, каспи, возврат, скидка*) combined with Kazakh agglutinative suffixes (*доставка-сы*, *заказ-ды*).
* **Measurement:** Frequency and accuracy of code-switched commercial terms across Human vs. LLM texts.

#### Metric 5: Length-Stratified Granularity (RAID ACL 2024)
* All diagnostic metrics and classification accuracies are strictly partitioned into:
  * Short: $\le 60$ characters.
  * Medium: $61–200$ characters.
  * Long: $> 200$ characters.
* Directly exposes the short-text vulnerability identified in prior literature.

#### Metric 6: Cross-LLM Semantic Distance Matrix (MAGE ACL 2024)
* Compute mean sentence embeddings using multilingual sentence encoders (or KazRoBERTa mean-pooling).
* Measure pairwise Cosine Similarity and Centered Kernel Alignment (CKA) between Human distributions and each LLM generator to map generator proximity to authentic human Kazakh.

---

### 3.3 Zero-Shot Classifier Stress-Testing
* Run inference using existing fine-tuned **KazRoBERTa (Pure)** and **KazRoBERTa (FST)** models on the newly generated Qwen-2.5 and LLaMA-3.1 subsets.
* Report:
  * In-Distribution (Sherkala) vs. Out-of-Distribution (Qwen-2.5, LLaMA-3.1) accuracy and F1 drop.
  * False Positive Rate (FPR) and False Negative Rate (FNR) stratified by length.
  * Demonstrates baseline fragility on unseen generators prior to contrastive training.

---

## 4. Kaggle Execution Architecture

* **Kaggle Kernel Bundle:** `kazakh-llm-diagnostic-engine`
* **Hardware Target:** 2x Nvidia Tesla T4 (or 1x Tesla P100), 16GB VRAM, 30GB RAM.
* **Environment:** Python 3.10+, PyTorch 2.x, Transformers 4.40+, Accelerate, BitsAndBytes, Apertium-Kaz tools.
* **Automation Workflow:**
  1. `generate_multi_llm.py`: Batch generation with seed conditioning.
  2. `compute_linguistic_diagnostics.py`: Calculates the 6 diagnostic factors.
  3. `evaluate_zero_shot_transfer.py`: Evaluates KazRoBERTa Pure vs. FST.
  4. `generate_report_and_plots.py`: Produces final LaTeX/Markdown summary tables and publication-ready PNG/EPS graphs.

---

## 5. Deliverables & Output Specifications

1. **Benchmark Artifact:** `kazakh_aigc_diagnostic_4k.parquet` (Clean, balanced 4,000-sample paired dataset).
2. **Diagnostic Report:** `diagnostic_results_summary.md` with structured tables:
   * Table 1: Token Inflation ($T/W$) across Tokenizers & Generators.
   * Table 2: Lexical Diversity & Suffix Retention Metrics.
   * Table 3: Length-Stratified Zero-Shot Transfer Accuracy (Pure vs. FST).
   * Table 4: Cross-LLM Embedding Cosine Similarity Matrix.
3. **Visual Figures:**
   * Figure 1: Subword Inflation Density Distribution Curves.
   * Figure 2: Length vs. False Positive Rate Curve (RAID-style).
   * Figure 3: Generator Semantic Proximity Heatmap (MAGE-style).
