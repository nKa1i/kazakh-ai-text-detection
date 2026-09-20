# Design Specification: Multi-Domain Generalization, Kazakh-FEVER 3K Benchmark & Two-Paper Publication Roadmap

**Date**: 2026-09-20  
**Author**: Daulet Anekesh (大雷) / Antigravity Pair Programming  
**Supervisor**: Prof. Bin Guo (郭斌)  
**Co-Authors & Collaborators**: Wangna (王娜), Daulet  
**Institutions**: School of Computer Science, Northwestern Polytechnical University (NPU) & Al-Farabi Kazakh National University (KazNU)  
**Branch**: `feat/kaz-presentation-cloning-and-figures`  

---

## 1. Executive Summary & Strategic Context

Following the Master's thesis progress review with Professor Guo on September 20, 2026, two key research directives and a strategic publication mandate were established:

1. **Cross-Domain Generalization (Topics 1 & 2):**  
   *Advisor Feedback:* The detector must prove its effectiveness across diverse domains and topics beyond formal news and e-commerce, ensuring robustness when applied to unseen real-world scenarios.
2. **Factual Evidence & Dataset Scaling (Topic 3):**  
   *Advisor Feedback:* The current 120-claim pilot benchmark must be substantially supplemented with real-world factual evidence and expanded into a full-scale benchmark dataset for empirical validation.
3. **Two-Paper Publication Roadmap:**  
   *Advisor Directive:* Produce two distinct publications over the coming months:
   - **Paper 1 (Conference Paper, 8 pages):** Dedicated to Topic 3 (The Kazakh-FEVER 3K benchmark, hybrid retrieval, and the Two-Dimensional Trust Matrix).
   - **Paper 2 (Journal Paper, 12–16 pages):** Dedicated to the comprehensive unified architecture (Topic 1 Morphology Gated Fusion + Topic 2 Multi-Domain Generalization + Topic 3 Fact-Checking + 4-Tab Interactive System).

---

## 2. Publication Strategy & Target Venues

### 2.1 Paper 1: Conference Paper (8 Pages)
- **Title:** *Kazakh-FEVER: An Evidence-Grounded Benchmark and Two-Dimensional Trust Matrix for Low-Resource Fact Verification*
- **Scope & Contribution:**
  - First public, standardized evidence-grounded fact-checking benchmark for Kazakh (~3,000 verified claims across ~250 articles/topics).
  - Hybrid Evidence Retrieval Architecture combining 83-rule FST stemmed BM25 with multilingual dense retrieval (BGE-M3 / mE5) via Reciprocal Rank Fusion (RRF).
  - Two-Dimensional Trust Matrix $(\hat{y}_{AI}, R_{fact}) \in [0, 1]^2$ and epistemic uncertainty calibration for NEI claims.
- **Target Venues:**
  - **Plan A (Top-Tier CCF-B / CCF-A):** **EMNLP / COLING** (via ACL Rolling Review - ARR, or direct submission).
  - **Plan B (Fast / High-Acceptance Fallback):** **LREC-COLING / EACL / ACL Findings** (or dedicated low-resource tracks e.g., SIGTYP / AmericasNLP / TurkicNLP workshops).

### 2.2 Paper 2: Journal Paper (12–16 Pages)
- **Title:** *A Unified Morphologically-Grounded and Evidence-Verified Architecture for Generative AI Text Detection in Low-Resource Agglutinative Languages*
- **Scope & Contribution:**
  - Full end-to-end framework integrating all 3 thesis topics:
    1. Topic 1: Dual-Stream 83-Rule FST Morphology-Aware Gated Fusion.
    2. Topic 2: Robust Generalization under generator shift, long-document chunking ($L=256$, Top-$K$ worst-chunk pooling), and cross-domain social media shift.
    3. Topic 3: Kazakh-FEVER factual verification and 2D Trust Matrix.
  - Comprehensive empirical evaluation across 4 distinct domains: News, Wikipedia, Kaspi e-commerce reviews, and colloquial Social Media (Telegram/Twitter).
  - Production-ready 4-tab Gradio prototype, containerized Hugging Face Spaces deployment, and 300+ regression tests.
- **Target Venues:**
  - **Primary Target (CCF-B / SCI Q2):** **ACM TALLIP** (*ACM Transactions on Asian and Low-Resource Language Information Processing*). Perfect topical match, rolling submissions 365 days/year, recognized in Chinese academia.
  - **Secondary Target (CCF-B / SCI Q1):** **Information Processing & Management (IP&M)** or **Knowledge-Based Systems (KBS)**.

### 2.3 Strict Anti-Dual-Submission Policy
- **Absolute Rule:** Simultaneous submission of the same paper to multiple venues is strictly prohibited by academic ethics. Submissions are strictly sequential.
- **Conference-to-Journal Path:** Publishing the 8-page conference paper (Paper 1) first, followed by extending it with 30–40% new technical material, full unified system integration, and multi-domain social media experiments for the journal submission (Paper 2), is 100% compliant with IEEE/ACM policies.

---

## 3. Topic 3: Kazakh-FEVER 3K Benchmark Scaling Architecture

### 3.1 Dataset Scale & Distribution
The current 120-claim pilot benchmark is scaled by $25\times$ to **3,000 balanced claims**:
- **Reference Articles:** ~250 documents:
  - ~180 Kazakh Wikipedia articles across 5 core domains: History, Geography, Science & Technology, Law & Government, Culture & Literature.
  - ~70 Real-world debunked cases and viral social media rumors from `Factcheck.kz`.
- **Claim Distribution (3-Way Balanced):**
  - **SUPPORTS (1,000 claims):** Entailed statements directly supported by gold evidence sentences.
  - **REFUTES (1,000 claims):** Negated statements with entity substitutions, inverted dates, or contradictory causal relations.
  - **NOT ENOUGH INFO (1,000 claims):** Plausible, topical claims lacking sufficient grounding evidence in the reference corpus (epistemic uncertainty).
- **Partitioning:**
  - **Training Split:** 2,000 claims (for fine-tuning Kazakh NLI cross-encoders).
  - **Development Split:** 500 claims.
  - **Test Split:** 500 claims (250 encyclopedic + 250 out-of-domain social media rumor claims).

### 3.2 Semi-Automated Data Generation & Quality Protocol
1. **Automated Article Harvesting:** Python crawler using `wikipedia-api` and BeautifulSoup to fetch structured Kazakh encyclopedic articles and Factcheck.kz case studies with paragraph and sentence boundaries.
2. **LLM-Assisted Claim Mutation:** Batch prompting of Qwen-2.5-72B / GPT-4o with few-shot Kazakh examples to produce candidate (Claim, Gold Sentences, Label) triples.
3. **Automated Quality Filtering:**
   - Word length filtering ($8 \le \text{tokens} \le 35$).
   - FST morphological legality check (ensuring clean Kazakh orthography).
   - Deduplication via embedding cosine similarity ($\text{sim} < 0.85$).
4. **Human Verification Pass:** Daulet and Wangna split the filtered candidates (1,500 each), performing a rapid binary validation pass (confirming label correctness and evidence sentence indices).

### 3.3 Hybrid Evidence Retrieval Architecture
Upgrades the retrieval engine from sparse BM25 to a state-of-the-art hybrid architecture:
- **Stream 1 (Morphological BM25):** 83-rule FST stemmer strips inflectional suffixes, matching core root morphemes.
- **Stream 2 (Multilingual Dense Retrieval):** `BGE-M3` or `multilingual-e5-large` dense vector representations capturing semantic equivalence.
- **Reciprocal Rank Fusion (RRF):**
  $$RRF(d) = \sum_{m \in \{BM25, Dense\}} \frac{1}{60 + \text{rank}_m(d)}$$
  Expected result: Increases Evidence Recall@3 from 91.67% to **$\ge 96.5\%$**, directly resolving the strict Joint FEVER bottleneck (targeting $\ge 82.0\%$ Joint FEVER).

---

## 4. Topic 2: Social Media Cross-Domain Generalization Benchmark

### 4.1 Dataset Construction (~2,000 Samples)
- **Human Corpus (1,000 samples):** Collected from public Kazakh Telegram channels (news discussions, community chats), Twitter/X posts, and forum comments.
- **AI Corpus (1,000 samples):** Generated by prompting modern LLMs (Qwen-2.5, Llama-3.1, GPT-4o) with colloquial personas to generate comments, replies, and posts in informal Kazakh.
- **Linguistic Stressors:**
  - Agglutinative colloquial suffixes (`-сың ғой`, `-ма екен`, `-шы`).
  - Code-switching (Russian/English loanwords with Kazakh inflection, e.g., `донаттау`, `хайптану`).
  - Missing diacritics and informal transliterations (e.g., `к` for `қ`, `г` for `ғ`).

### 4.2 Cross-Domain Evaluation Matrix
1. **In-Domain:** Train on News $\rightarrow$ Test on News.
2. **Cross-Domain:** Train on News $\rightarrow$ Test on **Social Media** (testing morphological resilience).
3. **Cross-Domain + Cross-Generator:** Train on News (Sherkala-7B) $\rightarrow$ Test on **Social Media (Qwen-2.5 & Llama-3.1)**.

### 4.3 Key Hypothesis
Standard subword BPE models (KazRoBERTa, mBERT) will experience severe domain collapse due to token fragmentation on slang. The Dual-Stream 83-Rule FST Gated Fusion model will maintain high ROC-AUC ($\ge 98.5\%$) because the FST strips conversational suffixes cleanly and the dynamic gate ($g$) automatically shifts weight toward the invariant morphological stream.

---

## 5. Implementation Roadmap & Timeline

| Phase | Duration | Tasks & Deliverables |
| :--- | :--- | :--- |
| **Phase 1: Tooling & Data Pipeline** | **Weeks 1–2** | 1. Implement `scripts/build_kazakh_fever_3k.py` (crawling + LLM claim generation).<br>2. Implement `scripts/build_social_media_benchmark.py`.<br>3. Complete human verification with Wangna (1,500 claims each). |
| **Phase 2: Hybrid Retrieval & Experiments** | **Weeks 3–4** | 1. Implement `src/retrieval/hybrid_retriever.py` (BM25 + BGE-M3 + RRF).<br>2. Run 5-fold cross-validation on Kazakh-FEVER 3K.<br>3. Run Social Media domain-shift experiments. |
| **Phase 3: Paper 1 Writing & Submission** | **Weeks 5–7** | 1. Write 8-page conference draft (LaTeX template).<br>2. Review with Wangna and Prof. Guo.<br>3. Submit to ACL Rolling Review (ARR) or COLING. |
| **Phase 4: Journal Expansion & System Packaging** | **Weeks 8–12** | 1. Integrate Social Media experiments into unified framework.<br>2. Expand manuscript to 14 pages for ACM TALLIP.<br>3. Update 4-tab Gradio prototype and documentation. |

---

## 6. Self-Review & Integrity Constraints
- **Placeholders:** Zero `TODO`, `TBD`, or undefined variables.
- **Constraints:** `aist2026/paper.tex` remains completely untouched (0 diff lines).
- **Anti-Dual-Submission:** Fully compliant with ACL/IEEE/ACM policies.
- **Encoding:** Explicit UTF-8 on all file operations.
