# Kaz-MultiDomain: Tri-Domain Contrastive Training for Robust Kazakh AI Text Detection
**Design Specification**  
**Date**: September 9, 2026  
**Status**: Approved (Brainstorming & Design Phase)  
**Target Branch**: `feat/kaz-multi-domain-training`  

---

## 1. Motivation & Problem Statement

In the previous Kaz-MAGE benchmark evaluation ($N=6{,}000$ test samples), we identified a critical domain blindspot:
- While our detector achieved **$1.0000$ ROC-AUC** on Consumer Reviews (Q1 & Q2) and **$0.9114 \text{--} 0.9731$** on zero-shot wild texts from Qwen-2.5-7B (Q4), it dropped to **$0.6449$ ROC-AUC** on Q3 (Seen Generator $\times$ Unseen Domain: Sherkala on News & Wikipedia).
- **Root Cause**: The detector was trained strictly on $100\%$ consumer reviews from Kaspi. It had never observed authentic, human-written formal Kazakh prose during training. When presented with a formal journalistic article (*Informburo*, *Tengrinews*) or encyclopedic text (*kk-wiki*), the detector misattributed formal passive syntax and high lexical density to AI generation, causing high false accusation rates on human formal text.

To build a truly well-rounded, production-ready AI detector for the Kazakh language, this specification establishes **Tri-Domain Multi-Task Contrastive Training**, mixing Consumer Reviews, Formal News, and Encyclopedic Wikipedia text with cross-domain contrastive representation alignment.

---

## 2. Training Corpus Construction & Data Isolation

### 2.1. Corpus Composition ($N=6{,}000$ Training Instances)
The training corpus maintains an exact **1:1:1 Tri-Domain Balance** ($2{,}000$ samples per domain, $50\%$ Human, $50\%$ AI):

| Domain | Human Samples ($y=0$) | AI Samples ($y=1$) | Total | Data Sources |
| :--- | :---: | :---: | :---: | :--- |
| **Consumer Reviews** | $1{,}000$ | $1{,}000$ | $2{,}000$ | Human Kaspi reviews + Sherkala review generations |
| **Formal News** | $1{,}000$ | $1{,}000$ | $2{,}000$ | Human articles (*Informburo*, *Tengrinews*, *Kazinform*) + Paired prefix continuations ($500$ Sherkala, $500$ Qwen) |
| **Wikipedia** | $1{,}000$ | $1{,}000$ | $2{,}000$ | Human Kazakh Wikipedia (*kk-wiki*) articles + Paired prefix continuations ($500$ Sherkala, $500$ Qwen) |
| **Total** | **$3{,}000$** | **$3{,}000$** | **$6{,}000$** | Fully balanced tri-domain training dataset |

### 2.2. Zero Data Leakage Guarantee
* The training corpus is stored in `data/kaz_multi_domain_train_6k.json`.
* **Verification Protocol**: A cryptographic SHA-256 hash collision audit ensures $0\%$ overlap with the held-out $6{,}000$-sample evaluation benchmark (`data/kaz_mage_eval_6k.json`). The test benchmark remains completely unseen during training.

---

## 3. Training Architecture & Loss Formulation

### 3.1. Model Architecture
We train our top-performing dual-stream architecture:
* **Semantic Stream**: KazRoBERTa backbone (`kz-transformers/kaz-roberta-conversational`, $d=768$).
* **Morphological Stream**: Explicit FST `MorphemeEncoder` ($250$-token affix vocabulary, $d=768$, `pos_embedding=512`).
* **Gated Cross-Attention Fusion**: Dynamic routing gate $\mathbf{g} = \sigma(\mathbf{W}_g [\mathbf{h}_{\text{sem}}; \mathbf{h}_{\text{morph}}])$.
* **Dual Heads**: Linear classification head ($768 \to 2$) and L2-normalized projection head ($768 \to 256 \to 128$).

### 3.2. Cross-Domain Stratified Batching
Each training mini-batch ($B=32$, effective $64$ with gradient accumulation) samples:
* $11$ Consumer Reviews
* $11$ Formal News articles
* $10$ Wikipedia paragraphs
* Balanced $50\%$ Human and $50\%$ AI.

### 3.3. Cross-Domain Supervised Contrastive Loss ($\mathcal{L}_{\text{SupCon}}$)
In the projection space $\mathbf{z} \in \mathbb{R}^{128}$, Supervised Contrastive Loss pulls all instances with the same label together:
$$\mathcal{L}_{\text{SupCon}} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$

* **Domain Invariance Mechanism**: Positive pairs $P(i)$ cross domain boundaries. A human news article is explicitly pulled toward human reviews and human Wikipedia texts. Any attempt by the encoder to cluster by genre (e.g., grouping all news articles together) incurs high contrastive penalty.
* **Combined Loss**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\hat{y}, y) + \lambda_{\text{supcon}} \mathcal{L}_{\text{SupCon}}(\mathbf{z}, y)$$
  with $\lambda_{\text{supcon}} = 0.5$ and $\tau = 0.07$.

---

## 4. Evaluation & Quantitative Success Criteria

### 4.1. Benchmark Protocol
The trained model (`Model 5-MultiDomain`) is evaluated on the held-out $6{,}000$-sample test set (`data/kaz_mage_eval_6k.json`) across all four quadrants:
- **Q1**: Seen LLM $\times$ Seen Domain (Sherkala on Reviews)
- **Q2**: Unseen LLM $\times$ Seen Domain (Qwen on Reviews)
- **Q3**: Seen LLM $\times$ Unseen Domain (Sherkala on News/Wiki)
- **Q4**: Unseen LLM $\times$ Unseen Domain (Qwen on News/Wiki)

### 4.2. Target Metrics
1. **Q3 Domain Resolution**: Q3 ROC-AUC must improve from **$0.6449 \to \ge 0.9500$** ($+0.30$ absolute gain).
2. **Tri-Domain Uniformity**: $\Delta \text{AUC}_{\text{domain}} = \text{AUC}_{\text{Q1}} - \text{AUC}_{\text{Q3}} < 0.0500$.
3. **Conversational Retention**: Kaspi review detection accuracy (Q1 & Q2) must remain $\ge 98\%$ (zero catastrophic forgetting).
4. **Overall Wild Performance**: Q4 ROC-AUC must maintain $\ge 0.9200$.

---

## 5. Global Constraints & Paper Integrity
- Under no circumstances shall `aist2026/paper.tex` be touched or modified.
- Full unit test regression must continue to pass ($78/78$ passing).
- Deterministic seeding via CRC32 and UTF-8 encoding across all data files.
