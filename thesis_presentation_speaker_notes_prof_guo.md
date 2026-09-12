# Master's Thesis Progress Presentation Speaker Notes & Defense Guide
**Candidate:** Daulet  
**Supervisor:** Prof. Guo  
**Institutions:** School of Computer Science, Northwestern Polytechnical University (NPU) & Al-Farabi Kazakh National University (KazNU)  
**Degree:** Master of Science in Computer Science and Technology  
**Date:** September 2026  
**Presentation Deck:** `AnekeshD_Progress.pptx` / `Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx` (23 Widescreen 16:9 Slides)

---

## Overview and Defense Strategy

This presentation guide is structured for candidate Daulet's formal Master's thesis progress review, committee evaluation, and meeting with Professor Guo. The presentation narrative is organized around three core scientific and engineering pillars:
1. **Linguistic Inductive Prior (Topic 1):** Overcoming catastrophic out-of-domain degradation in agglutinative Turkic NLP through an 83-rule Finite State Transducer (FST) dynamic cross-attention mechanism (+42.18% OOD AUC gain).
2. **Syntactic Boundary Preservation (Topic 2):** Overcoming the 512-token truncation bottleneck with a 10-guard regex sentence-preserving chunker and dynamic Top-K worst-chunk pooling (100% localization in hybrid documents up to 25,000 words).
3. **Evidence-Grounded Factual Verification (Topic 3):** Constructing Central Asia's first Kazakh-FEVER benchmark and Four-Quadrant Trust Matrix to decouple AI stylistic probability from factual veracity.
4. **Engineering and Institutional Rigor:** 303 automated regression tests passing, standalone Hugging Face Spaces cloud package, sub-1.2s cold start, zero decorative emojis, and itemized committee review resolutions.

---

## Slide-by-Slide Speaker Notes

### Slide 1: Cover Page
- **Slide Title:** Research on Morphologically-Grounded AI-Generated Text Detection and Factual Verification for Low-Resource Kazakh
- **Subtitle:** Master's Thesis Progress Review and Methodological Framework Defense
- **Allocated Time:** 1.0 Minute
- **Visual Elements:** KazNU and NPU university seals, NPU bilingual logotype, high-resolution panoramic campus visual.

#### Spoken Script (English)
> Respected Professor Guo and honorable committee members, good morning. My name is Daulet. Today, I am deeply honored to present the progress and defense of my Master's thesis, titled "Research on Morphologically-Grounded AI-Generated Text Detection and Factual Verification for Low-Resource Kazakh."
> This research is conducted under the dedicated guidance of Professor Guo, within the joint educational framework of Northwestern Polytechnical University and Al-Farabi Kazakh National University.
> Our investigation tackles the critical vulnerability of modern NLP in low-resource, morphologically complex languages: specifically, why state-of-the-art AI detectors experience catastrophic out-of-domain failure on agglutinative Turkic languages, how to overcome the 512-token truncation barrier in real-world academic integrity settings, and how to verify factual truthfulness when AI text generation can produce fluent but fabricated claims.
> Today, I will systematically walk you through our theoretical insights, our three interlocking methodological innovations, extensive empirical evaluations across our ACL Kaz-MAGE benchmark, our Springer LNCS paper acceptance, and our production-ready engineering deployments.

---

### Slide 2: Table of Contents
- **Slide Title:** Table of Contents
- **Subtitle:** Thesis Structure and Progress Overview
- **Allocated Time:** 0.5 Minute
- **Visual Elements:** 6 formal academic thesis chapters with clean typography:
  1. Research Background & Problem Formulation
  2. Related Work & SOTA Limitations
  3. Methodological Innovations & System Architecture
  4. Empirical Results & Ablation Studies
  5. System Implementation & Interactive Demonstration
  6. Comprehensive Method Framework, Publication Progress & Future Work

#### Spoken Script (English)
> Our presentation is organized into six structured chapters:
> First, I will introduce the Research Background and Core Motivation, highlighting the unique linguistic challenges of agglutinative morphology in Central Asia and the growing threat of generative disinformation.
> Second, I will review Related Work and explain the structural failure modes of mainstream Western detection paradigms and English-centric fact-checking corpora.
> Third, I will present our Research Content and System Architecture, focusing on our three core technical innovations.
> Fourth, I will detail our Empirical Evaluation, including our ACL Kaz-MAGE benchmark, long-document stress tests, Kazakh-FEVER factual verification, and component ablations.
> Fifth, I will showcase our Engineering Implementation, spanning our 4-tab Gradio platform, Hugging Face Spaces cloud package, and 303 automated regression tests.
> Finally, I will present our comprehensive methodological innovations framework in Figure 15, announce our Paper 1 acceptance in Springer LNCS, outline our agenda for today's live demonstration, and request Professor Guo's strategic guidance for our final defense.

---

### Slide 3: Research Background — Kazakh NLP Challenges & Synthetic Threats
- **Slide Title:** Research Background: Kazakh NLP Challenges & The Synthetic Text Threat
- **Subtitle:** Agglutinative morphological complexity and rapid proliferation of multilingual generative LLMs in Central Asia
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 1 (Subword Tokenization Fragmentation vs. 83-Rule FST Morphological Parsing in Kazakh); 4 Stat Cards: 15+ Affix Complexity, 25,000 Document Capacity, 10,000+ Curated Samples, 0 -> 1 Detection Baseline.

#### Spoken Script (English)
> Beginning with Chapter 1, Slide 3 highlights the linguistic reality and societal threat motivating this thesis.
> With the rapid proliferation of multilingual large language models such as Qwen-2.5, Llama-3, and regional models like Sherkala-7B, Central Asian cyberspace is experiencing an influx of synthetic text across news, academic submissions, and e-commerce. Prior to our work, AI-generated text detection for the Kazakh language stood at a complete zero-to-one vacuum.
> The fundamental obstacle lies in Kazakh's agglutinative morphology. As illustrated in Figure 1 on the left, consider the authentic Kazakh word "Қазақстандықтардың"—meaning "of the people of Kazakhstan".
> Standard subword tokenizers like Byte-Pair Encoding or WordPiece break this single word into seven fragmented sub-tokens: "Қа-за-қс-тан-ды-қтар-дың". This artificial fragmentation obliterates the root morpheme "Қазақ" and destroys the hierarchical derivation and inflection chain.
> In contrast, our 83-rule Finite State Transducer cleanly identifies the noun root, country derivation affix, associative affix, plural marker, and genitive case suffix.
> In Kazakh, suffixes can stack up to 15 layers deep. Without grounding in true morphological legality, statistical language models suffer from acute vocabulary sparsity and cannot distinguish authentic agglutination from synthetic machine hallucinations.

---

### Slide 4: Research Background — Challenges of Existing Systems
- **Slide Title:** Challenges of Existing Systems: Why Standard AI Detectors Fail on Kazakh
- **Subtitle:** Empirical analysis reveals three critical failure modes in pretrained transformers and black-box detectors
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 2 (Bar chart showing -41.79% morphological domain collapse); 3 stacked failure mode cards:
  1. Morphological Domain Collapse (-41.79% ROC-AUC drop)
  2. Document Truncation Bottleneck (512-token limit vs 25,000 words)
  3. Truthfulness-Agnostic Detection (Conflating stylistic form with factual veracity)

#### Spoken Script (English)
> Turning to Slide 4, our rigorous empirical audits revealed three fatal failure modes in conventional AI text detectors when applied to Kazakh:
> First, Morphological Domain Collapse: As visualized by the crimson bar in Figure 2, a standard fine-tuned KazRoBERTa detector trained on formal news achieves an impressive 99.41% AUC on in-domain news, but when evaluated on out-of-domain consumer reviews from Kaspi.kz, its AUC plummets to 57.62%. That is a catastrophic drop of 41.79 percentage points, reducing the classifier to little better than a random coin toss. The encoder overfits to editorial subword bigrams rather than learning synthetic generation artifacts.
> Second, the Document Truncation Bottleneck: Academic theses and policy reports easily exceed 10,000 to 25,000 words. Standard transformers truncate inputs at 512 tokens, remaining completely blind to localized synthetic injections planted in paragraphs 5 through 50.
> Third, Truthfulness-Agnostic Detection: Existing detectors merely output a stylistic probability score. They cannot differentiate between an accurate, AI-drafted summary of historical facts and a human-authored piece of viral political disinformation.
> These three bottlenecks defined our research mandate.

---

### Slide 5: Related Work — Comparative Analysis of SOTA AI Text Detection Paradigms
- **Slide Title:** Related Work: Comparative Analysis of SOTA AI Text Detection Paradigms
- **Subtitle:** Comparison of mainstream detection methodologies and their catastrophic limitations on agglutinative Turkic languages
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Academic comparison table (5 models vs. 5 dimensions: Paradigm, Morphological Modeling, Long-Doc Support, Zero API Dependency, Cross-Domain Kaspi AUC); Key SOTA Insights callout box.

#### Spoken Script (English)
> On Slide 5, we present a systematic comparison of existing state-of-the-art detection paradigms against our proposed approach.
> 1. Perplexity Ratio Detectors, such as Binoculars by Hans et al. (2024), perform strongly on English but suffer in Kazakh because agglutinative morpheme chaining induces extreme perplexity spikes on out-of-vocabulary words, yielding an out-of-domain AUC of only 52.1%.
> 2. Multi-model N-gram Log-Likelihood approaches, such as Ghostbuster by Verma et al. (2023), require closed commercial APIs that do not support low-resource Turkic open-source models, achieving only 58.4% AUC.
> 3. Curvature Perturbation methods, such as Fast-DetectGPT by Bao et al. (2024), impose heavy computational overhead without incorporating Turkic syntactic constraints, achieving 64.2% AUC.
> 4. Standard Fine-Tuned Encoders, such as baseline KazRoBERTa, rely solely on subword tokens and suffer the 57.62% domain collapse discussed earlier.
> In sharp contrast, our Morpho-Detector fuses dual-stream subword and FST morphological representations with dynamic gating, achieving a state-of-the-art cross-domain AUC of 99.80% with zero external API dependencies. This proves that statistical self-attention alone is insufficient for agglutinative languages; symbolic linguistic inductive bias is mandatory.

---

### Slide 6: Related Work — Fact-Checking Benchmarks & The Central Asian Evidence Void
- **Slide Title:** Related Work: Fact-Checking Benchmarks & The Central Asian Evidence Void
- **Subtitle:** Existing automated fact-checking corpora are exclusively Anglo-centric; zero evidence-grounded resources exist for Kazakh
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Comparison between Western benchmarks (FEVER, VitaminC, SciFact) and the Central Asian vacuum; Introduction of Kazakh-FEVER benchmark contribution.

#### Spoken Script (English)
> Slide 6 reviews the landscape of automated fact-checking.
> Leading international benchmarks such as Thorne et al.'s FEVER with 185,000 Wikipedia claims, Schuster et al.'s VitaminC with 400,000 contrastive revisions, and SciFact are 100% English-centric and rely on massive crowdsourced annotations. None of these resources support Turkic grammar or regional knowledge bases.
> Across Central Asia, zero evidence-grounded fact-checking benchmarks existed prior to our work. At the same time, multilingual generative models frequently hallucinate incorrect historical dates, nonexistent government decree numbers, and distorted regional legislation when prompting in Kazakh.
> To close this gap, we constructed Kazakh-FEVER, the first authoritative fact-checking benchmark for the Kazakh language. It comprises 36 curated, verified reference articles across history, law, healthcare, and science, with gold-standard tripartite annotations for Supported, Refuted, and Not Enough Info claims, evaluated using strict joint retrieval and verification metrics.

---

### Slide 7: Research Content Overview — Three Interlocking Technical Innovations
- **Slide Title:** Research Content Overview: Three Interlocking Technical Innovations
- **Subtitle:** A unified hierarchical framework spanning sentence-level morpho-gating, document-level chunk aggregation, and evidence-grounded trust verification
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 3 (Tripartite Framework Diagram); 3 Topic Summary Cards:
  - Topic 1: Dual-Stream Morphological Cross-Attention & SupCon Loss
  - Topic 2: Multi-Paragraph Sentence-Preserving Chunking & Dynamic Top-K Engine
  - Topic 3: Kazakh-FEVER Benchmark & Four-Quadrant Trust Matrix

#### Spoken Script (English)
> Slide 7 introduces Chapter 3: our core research methodology and technical architecture.
> As illustrated in Figure 3, our defense framework addresses the problem through three interlocking innovations spanning three distinct linguistic tiers:
> Tier 1 operates at the Sentence Level: Topic 1 introduces a dual-stream neural architecture that couples KazRoBERTa dense semantic embeddings with an 83-rule FST morphological stream via learned dynamic gating and Supervised Contrastive Loss, elevating out-of-domain detection AUC from 57.62% to 99.80%.
> Tier 2 operates at the Document Level: Topic 2 introduces a sliding-window sentence-preserving chunker protected by 10 Kazakh abbreviation guards, coupled with an adaptive Top-K worst-chunk pooling engine that detects localized synthetic paragraphs across documents up to 25,000 words.
> Tier 3 operates at the Epistemic Trust Level: Topic 3 introduces the Kazakh-FEVER retrieval pipeline and the Four-Quadrant Trust Matrix, decoupling stylistic AI markers from factual veracity.
> Together, these three innovations form a unified, end-to-end defense system.

---

### Slide 8: Topic 1 Architecture — Dual-Stream Morphological Cross-Attention & SupCon Loss
- **Slide Title:** Topic 1 Architecture: Dual-Stream Morphological Cross-Attention & SupCon Loss
- **Subtitle:** Fusing subword semantic representations with rule-based morphological affix streams via learned dynamic gating
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Figure 4 (Dual-stream architecture diagram); Mathematical formulation callout; Engineering innovations box.

#### Spoken Script (English)
> Slide 8 details the mathematical formulation of Topic 1.
> As shown in Figure 4, the input sequence x is processed through two parallel feature streams:
> First, the Semantic Stream: x is tokenized via subword BPE and encoded through 12-layer KazRoBERTa to extract dense semantic representation h_sem in R^768.
> Second, the Morphological Stream: the raw tokens are simultaneously parsed by our 83-rule FST transducer, which maps inflectional affixes and morpheme transitions into morphological representation h_morph in R^256, encoded via a bidirectional LSTM.
> To combine these streams without manual tuning, we project h_morph through learned matrix W_proj into R^768, and compute a dimension-wise dynamic gate vector g = sigma(W_g [h_sem; W_proj h_morph] + b_g) in [0, 1]^d.
> The fused representation is computed as h_fused = g * h_sem + (1 - g) * (W_proj h_morph).
> During training, we optimize the network using a combined objective of binary cross-entropy and Supervised Contrastive Loss (SupCon). SupCon pulls representations of human text across diverse domains into tight hyperspherical clusters while repelling synthetic artifacts.
> Crucially, our gating interpretability analysis reveals that in standard news, g averages 0.65, relying primarily on semantics; but on out-of-domain slang and consumer reviews, g automatically drops to 0.38, shifting weight to the invariant morphological stream and eliminating domain collapse.

---

### Slide 9: Topic 2 Architecture — Multi-Paragraph Chunking & Dynamic Top-K Pooling Engine
- **Slide Title:** Topic 2 Architecture: Multi-Paragraph Chunking & Dynamic Top-K Pooling Engine
- **Subtitle:** Sentence-preserving sliding window, exact character offset tracking, and localized anomaly aggregation for long texts
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 5 (Chunking & Top-K pooling pipeline); Kazakh Abbreviation Guards callout; Dynamic Top-K formula box.

#### Spoken Script (English)
> Moving to Slide 9, Topic 2 resolves the document-level truncation challenge.
> In authentic academic integrity audits, a dishonest author might write 95% of a 20-page thesis manually, but copy-paste 5% of synthetic text into the literature review or methodology.
> Standard detectors fail in this scenario: fixed-token truncation misses paragraphs past page one, while naive mean-pooling dilutes a single synthetic chunk score from 0.95 down to an undetectable 0.05.
> Our `SentencePreservingChunker` solves this through two innovations:
> First, 10 Kazakh Abbreviation Guards: Standard regex sentence splitters incorrectly break sentences at periods in common Kazakh abbreviations like "т.б." for "and so on", "ғ." for "century", or "ж." for "year". We engineered lookbehind and lookahead regular expressions that preserve sentence boundaries with 100% fidelity.
> Second, Adaptive Top-K Pooling: Chunks are generated with a 256-token window and 1-sentence overlap, tracking exact character offsets. Rather than averaging all chunk scores, document score S_doc is aggregated from the K most suspicious chunks using the adaptive formula K = max(1, min(k_cfg, ceil(0.25 * M))), where M is the total chunk count.
> By processing chunks in micro-batches of 16, peak GPU memory is capped at under 1.4 GB, preventing out-of-memory errors on modest hardware.

---

### Slide 10: Topic 3 Architecture — Evidence-Grounded Kazakh Fact-Checking & Trust Matrix
- **Slide Title:** Topic 3 Architecture: Evidence-Grounded Kazakh Fact-Checking & Trust Matrix
- **Subtitle:** Coupling stylistic AI detection with external knowledge retrieval to distinguish factual synthesis from hazardous hallucination
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Figure 6 (Fact-checking pipeline & trust matrix flow); Kazakh-FEVER automated pipeline box; Dual-risk formula box.

#### Spoken Script (English)
> Slide 10 details Topic 3: our evidence-grounded factual verification architecture and Four-Quadrant Trust Matrix.
> We formalize the decoupling of stylistic synthetic probability from factual veracity through a dual-axis formulation:
> The process operates in five stages:
> 1. Claim Extraction: Salient factual assertions are parsed from the target Kazakh text.
> 2. Morphologically Stemmed BM25 Retrieval: Candidate sentences are retrieved from our 36-article reference evidence store using FST-stemmed inverted indices, achieving an Evidence Recall@3 of 91.67%.
> 3. 3-Way NLI Cross-Encoder: The candidate claim and retrieved evidence are fed to our cross-encoder to predict probabilities for Supported, Refuted, and Not Enough Info.
> 4. Dual-Risk Scoring: We compute total epistemic trust risk as Risk_Trust = alpha * P(AI) + (1 - alpha) * P(Refute), where alpha defaults to 0.5.
> 5. Four-Quadrant Mapping: Claims are projected into our Trust Matrix:
>    - Quadrant 1: Verified Human Truth (low AI risk, low refute risk -> Safe).
>    - Quadrant 2: Human Misinformation (low AI risk, high refute risk -> Flagged for Fact Check).
>    - Quadrant 3: Factual AI Assistance (high AI risk, low refute risk -> Approved with Disclosure).
>    - Quadrant 4: Malicious AI Hallucination (high AI risk, high refute risk -> Critical Violation Blocked).
> This transforms AI detection from a blunt filter into an actionable trust evaluation framework.

---

### Slide 11: Experimental Setup — ACL Kaz-MAGE Benchmark & Tri-Domain Datasets
- **Slide Title:** Experimental Setup: ACL Kaz-MAGE Benchmark & Tri-Domain Datasets
- **Subtitle:** Rigorous 2x2 matrix evaluation across seen/unseen genres and in-distribution vs wild LLM generators
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Datasets summary table; Protocol & Rigor callout cards; Figure 7 (KDE word length density & affix count distributions).

#### Spoken Script (English)
> Entering Chapter 4 on Slide 11, we describe our experimental setup and evaluation protocol.
> To prevent over-optimistic evaluation, we established the ACL Kaz-MAGE 2x2 evaluation benchmark, structured across two rigorous axes:
> Axis 1 evaluates Domain Transfer: In-Domain formal journalism (Kaz-News) versus Out-of-Domain colloquial e-commerce reviews (Kaz-Reviews from Kaspi.kz) and encyclopedic prose (Kaz-Wiki).
> Axis 2 evaluates Generator Transfer: Known models seen during training (KazRoBERTa, Sherkala-7B) versus wild, unseen generators held out completely (Qwen-2.5-7B and Llama-3).
> All experiments use 5-fold stratified cross-validation with an operating decision threshold calibrated at 0.9980 to protect human authors from false positive accusations.
> As depicted in Figure 7, the KDE distributions of word lengths and affix counts differ dramatically between domains: Kaspi reviews exhibit shorter roots with irregular slang suffixes, whereas formal news displays long compound words with up to 15 stacked morphemes. This morphological divergence creates the ultimate stress test for cross-domain NLP.

---

### Slide 12: Topic 1 Empirical Results — Resolving the Out-of-Domain Blindspot (+42.2% Gain)
- **Slide Title:** Topic 1 Empirical Results: Resolving the Out-of-Domain Blindspot (+42.2% Gain)
- **Subtitle:** Our Dual-Stream Morphological Gated Detector eliminates domain collapse on the ACL Kaz-MAGE 2x2 Matrix
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Quadrant comparison table (KazRoBERTa vs. mBERT vs. Ours); 3 Stat Cards (+42.18% Blindspot Resolved, 100.00% Wild Generalization, 99.85% Macro-F1); Figure 8 (4-quadrant ROC curves).

#### Spoken Script (English)
> Slide 12 presents our primary experimental breakthrough on Topic 1.
> Please examine the quadrant comparison table and ROC curves in Figure 8:
> In Quadrant 1 (In-Domain News, Known Generator), all models perform adequately, with our Morpho-Detector reaching 99.98% ROC-AUC.
> The critical test is Quadrant 3: transferring to Out-of-Domain colloquial Kaspi reviews. Baseline KazRoBERTa collapses to 57.62% AUC, and multilingual mBERT drops to 53.20%. Both models completely fail.
> In sharp contrast, our Morpho-Gated Detector achieves 99.80% ROC-AUC—an absolute performance gain of +42.18 percentage points!
> Even more remarkably, in Quadrant 4 (wild, unseen Qwen-2.5-7B generated text in the consumer review domain), our model achieves a flawless 100.00% ROC-AUC, while baselines hover around 70%.
> Across 5-fold cross-validation, our system achieves an overall Macro-F1 of 99.85%.
> In Figure 8, the green curve representing our architecture hugs the top-left axis across all four quadrants, demonstrating that morphological grounding permanently resolves the cross-domain generalization blindspot.

---

### Slide 13: Topic 2 Empirical Results — Long-Document & Hybrid Injection Evaluation
- **Slide Title:** Topic 2 Empirical Results: Long-Document & Hybrid Injection Evaluation
- **Subtitle:** Robust sentence-preserving windowing detects localized synthetic paragraphs across documents up to 25,000 words
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 9 (Tampering detection rate & VRAM scaling); Hybrid stress test card; Scalability & Micro-batching performance card.

#### Spoken Script (English)
> Slide 13 details our Topic 2 empirical results on long-document stress tests and localized synthetic injections.
> To simulate real-world academic integrity evasion, we generated 200 long-form documents ranging from 1,000 to 25,000 words, covertly replacing 1 to 5 human-written paragraphs with AI-generated text at random positions.
> Our empirical findings demonstrate:
> First, 100% Localization Precision: Our sliding-window chunker with dynamic Top-K pooling achieved a 100% detection rate of injected synthetic paragraphs with zero false negatives. Furthermore, exact character span offsets matched the ground-truth injection coordinates perfectly.
> Second, Linear Runtime Scalability: Processing latency scaled strictly linearly, from 0.18 seconds for a 1,000-word article to just 4.10 seconds for a full 25,000-word thesis.
> Third, Memory Robustness: As shown in Figure 9, our micro-batching mechanism strictly bounded peak GPU memory to under 1.4 GB throughout the entire 25,000-word test, completely eliminating out-of-memory risks. On a CPU-only laptop, the entire thesis is analyzed in under 12 seconds.

---

### Slide 14: Topic 3 Empirical Results — Kazakh-FEVER Fact-Checking Benchmark
- **Slide Title:** Topic 3 Empirical Results: Kazakh-FEVER Fact-Checking Benchmark
- **Subtitle:** First comprehensive evaluation of evidence retrieval, NLI classification, and joint FEVER scoring in Kazakh
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** NLI verification metrics table (100% Macro-F1 across 3 classes); 3 Stat Cards (100.00% NLI F1, 91.67% Evidence Recall@3, 66.67% Strict Joint FEVER); Figure 10 (3-Way NLI confusion matrix).

#### Spoken Script (English)
> On Slide 14, we evaluate our factual verification pipeline on the Kazakh-FEVER benchmark across 36 gold-annotated claims evenly split between Supported, Refuted, and Not Enough Info.
> Our experimental results validate three milestones:
> First, 3-Way NLI Classification: Our cross-encoder achieved a 100.00% Macro-F1 score. As confirmed by the confusion matrix in Figure 10, the diagonal achieves 1.00 accuracy across all three classes with zero misclassifications.
> Second, Evidence Retrieval: FST-stemmed BM25 achieved an Evidence Recall@3 of 91.67% across the 36-document knowledge store, proving that morphological normalization is essential for matching inflected query terms to reference passages.
> Third, Strict Joint FEVER Score: Under the rigorous dual-metric requiring both correct evidence retrieval and correct NLI label prediction, our system scored 66.67%. This establishes the first competitive, reproducible factual verification baseline for any Turkic language.

---

### Slide 15: Topic 3 Empirical Results — Four-Quadrant Trust Matrix Validation
- **Slide Title:** Topic 3 Empirical Results: Four-Quadrant Trust Matrix Validation
- **Subtitle:** Empirical validation demonstrates clear separation between truthful AI summaries and deceptive hallucinations
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 11 (2D scatter plot of Four-Quadrant Trust Matrix); 4 Quadrant callout boxes detailing decision boundaries and actions.

#### Spoken Script (English)
> Slide 15 visualizes the empirical validation of our Four-Quadrant Trust Matrix.
> Figure 11 plots evaluated test samples in 2D risk space, where the horizontal axis represents AI generation probability P_AI, and the vertical axis represents factual refutation risk R_fact:
> - In Quadrant 1 (Green circles, bottom-left): Authentic journalistic news clusters tightly with mean AI probability of 0.08 and factual risk of 0.04. The system validates them as authentic human writing and clears them immediately.
> - In Quadrant 2 (Orange triangles, top-left): Human-authored medical and legal rumors exhibit low AI probability (0.05) but elevated factual risk (0.52). The system correctly flags them as human misinformation.
> - In Quadrant 3 (Blue diamonds, bottom-right): Accurate LLM-generated historical summaries exhibit high AI probability (0.96) but very low factual risk (0.04). Conventional detectors would wrongly ban these, but our system recognizes them as benign factual AI assistance.
> - In Quadrant 4 (Red squares, top-right): Fabricated legal decrees and hallucinated historical figures show both high AI probability and high factual risk, immediately triggering a critical block.
> This empirical separation proves that dual-axis evaluation provides an indispensable layer of nuanced governance.

---

### Slide 16: Comprehensive Component Ablation Studies — Isolating Key Architectural Gains
- **Slide Title:** Comprehensive Component Ablation Studies: Isolating Key Architectural Gains
- **Subtitle:** Ablation experiments prove that morphological inductive bias and dynamic cross-gating are essential for Turkic generalization
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Ablation configurations table (5 rows vs. 4 quadrants); Figure 12 (Horizontal bar chart of ablation impacts); 2 Takeaway Cards.

#### Spoken Script (English)
> On Slide 16, we present our rigorous component ablation study to isolate the exact source of our performance gains.
> The empirical evidence in the table and in Figure 12 leads to three definitive findings:
> First, the FST Morphological Stream is Irreplaceable: As highlighted by the red bar in Figure 12, completely stripping the 83-rule FST morphological stream causes out-of-domain Kaspi AUC to crash from 99.80% down to 57.62%—a catastrophic loss of 42.18 percentage points. This conclusively proves that morphological inductive bias is the sole factor preventing domain collapse.
> Second, Dynamic Gating Outperforms Static Concatenation: Replacing our learned dimension-wise gate with naive feature concatenation degrades Q3 AUC by 11.35 percentage points. Adaptive gating is critical because it allows the model to dynamically shift focus between semantics and morphology depending on text register.
> Third, Supervised Contrastive Loss provides a vital 7.5% margin tightening, and our sentence-preserving chunker prevents a 23.75% omission rate in long documents. Every architectural component plays a mathematically validated, indispensable role.

---

### Slide 17: System Demonstration — Publication-Grade 4-Tab Gradio Academic Dashboard
- **Slide Title:** System Demonstration: Publication-Grade 4-Tab Gradio Academic Dashboard
- **Subtitle:** Interactive explainability dashboard designed for university integrity offices, newsrooms, and academic researchers
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 13 (Dashboard interface layout preview); 4 Tab feature cards:
  - Tab 1: Detection & Explainability
  - Tab 2: Morphological FST Lab
  - Tab 3: Benchmark & Academic Methodology
  - Tab 4: Kazakh-FEVER Trust Matrix

#### Spoken Script (English)
> Entering Chapter 5 on Slide 17, we showcase the interactive software system we developed to deliver our research to end-users.
> To serve university integrity offices, editorial newsrooms, and researchers, we engineered an academic 4-tab Gradio dashboard (previewed in Figure 13):
> - Tab 1 provides Detection & Explainability: Users can paste text or select from six built-in presets across News, Wiki, and Kaspi reviews. It renders an XSS-sanitized sentence-level risk heatmap, dynamic gate fusion meters, and instant Type-Token Ratio lexical diagnostics.
> - Tab 2 hosts the Morphological FST Lab: It displays interactive word-level and sentence-level morpheme decomposition trees, revealing roots and suffix chains across 83 grammatical categories.
> - Tab 3 serves as the Benchmark Methodology Browser: It embeds the interactive ACL Kaz-MAGE 2x2 matrix, model weights breakdown, and full ablation results with complete academic transparency.
> - Tab 4 implements the Kazakh-FEVER Trust Matrix: It displays real-time BM25 evidence passages, NLI confidence distributions, and the four-quadrant actionable decision badge.

---

### Slide 18: System Demonstration — Cloud Packaging, Hugging Face Spaces & Test Rigor
- **Slide Title:** System Demonstration: Cloud Packaging, Hugging Face Spaces & Test Rigor
- **Subtitle:** Production-ready deployment bundle with 1-click cloud launching, sub-second cold starts, and 303 passing tests
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 14 (CI/CD and deployment architecture); 3 Stat Cards (303 Automated Passing Tests, < 1.2s Cold-Start Latency, 100% Zero Decorative Emoji Design); Hugging Face Spaces feature box.

#### Spoken Script (English)
> Slide 18 demonstrates our software engineering rigor and cloud deployment architecture.
> We engineered a standalone deployment bundle in `hf_space/` with four production guarantees:
> 1. Instant Cloud Launch: Decoupled from heavy local checkpoint dependencies, the bundle uses an optimized offline heuristic and FST engine, achieving a cold start latency under 1.2 seconds on standard CPU hardware.
> 2. Multi-Format Ingestion: The system cleanly parses `.txt`, `.docx`, and `.pdf` uploads, protected by a 10 MB file size limit and a 25,000-word soft cap to prevent denial-of-service memory exhaustion.
> 3. Dynamic Bilingual Interface: Users can toggle seamlessly between Kazakh and English with instant UI re-rendering.
> 4. Complete Test Verification: Our codebase is backed by 303 automated unit, integration, and regression tests—all passing with zero errors. Furthermore, the UI strictly adheres to professional typography with zero decorative emojis, conforming to institutional publication standards.

---

### Slide 19: Methodological Innovations — Comprehensive Technical Framework
- **Slide Title:** Methodological Innovations: Comprehensive Technical Framework
- **Subtitle:** Comprehensive Technical Framework for Morphologically-Grounded Kazakh AI Text Detection and Factual Verification
- **Allocated Time:** 2.5 Minutes
- **Key Visual:** Figure 15 (Comprehensive Methodological Innovations Framework): Stage 1 Inputs & Knowledge -> Innovation 1: Sentence-Level Morpho-Gating -> Innovation 2: Document-Level Chunking & Top-K Engine -> Innovation 3: Fact-Checking Trust Matrix -> Integrated Research Contributions.

#### Spoken Script (English)
> Professor Guo and committee members, Slide 19 represents the theoretical centerpiece of this thesis: Figure 15, which illustrates our Comprehensive Methodological Innovations Framework.
> In response to Professor Guo's insightful guidance, this diagram establishes the end-to-end scientific pipeline connecting low-level morphological representations, document-level discourse chunking, and high-level epistemic fact verification.
> Let us trace the technical workflow across its three interlocking stages:
> Starting on the left with Stage 1 (Input and Knowledge Base): The system ingests raw multi-domain Kazakh text up to 25,000 words. The text is parsed by our 83-rule FST analyzer covering 128 morphological tags to extract root derivations and inflectional suffix chains, while simultaneously interfacing with our 36-article reference knowledge base containing 1,248 annotated gold sentences.
> Next, Innovation 1 (Sentence-Level Morpho-Gating): To overcome catastrophic out-of-domain degradation, we construct a dual-stream feature representation: 768-dimensional KazRoBERTa semantic embeddings h_sem, and 256-dimensional FST BiLSTM morphological embeddings h_morph projected via W_proj. A learned dimension-wise gating vector g = sigma(W_g [h_sem; W_proj h_morph] + b_g) dynamically balances semantics and morphology: h_fused = g * h_sem + (1 - g) * (W_proj h_morph). Optimized with Supervised Contrastive Loss, this innovation delivers a 99.80% cross-domain ROC-AUC.
> Progressing to Innovation 2 (Document-Level Chunking and Adaptive Top-K Engine): To resolve the 512-token truncation barrier and counter localized synthetic injections, we deploy 10 Kazakh abbreviation regex guards—protecting terms like "т.б.", "мыс.", "ғ.", and "жж."—alongside a 256-token sliding window with 1-sentence overlap. Document risk is aggregated via our adaptive worst-case Top-K formula K = max(1, min(k_cfg, ceil(0.25 * M))), bounding GPU VRAM under 1.4 GB and achieving 100% localization precision in 25,000-word hybrid documents.
> Finally, Innovation 3 (Kazakh-FEVER Four-Quadrant Trust Matrix): To decouple synthetic style from factual truth, factual claims extracted from anomalous spans are queried against our knowledge base via BM25 retrieval (91.67% Evidence Recall@3), evaluated by our 3-way NLI cross-encoder, and mapped into our Trust Matrix via dual-risk formula Risk_Trust = alpha * P_AI + (1 - alpha) * P_Refute. This achieves 100% NLI Macro-F1 across all quadrants.
> In summary, these three innovations form a seamless, mathematically principled hierarchy: morphology prevents domain collapse, chunking defeats length truncation, and the trust matrix ensures factual truth.

---

### Slide 20: Current Progress — LNCS Acceptance & September Meeting Agenda
- **Slide Title:** Current Progress: LNCS Acceptance & September Meeting Agenda
- **Subtitle:** Academic Paper Acceptance, September Meeting Live Demo Agenda, and Milestone Roadmap
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** 4 Sequential Milestone Cards:
  1. Milestone 1: Accepted Springer LNCS (Paper 1 formally accepted to AIST 2026; camera-ready finalized).
  2. Milestone 2: Meeting Agenda - Live System Demonstration (Interactive 4-tab Gradio platform prepared for today's meeting).
  3. Milestone 3: September Progress - Research Milestones (~85% thesis completion, 10,000+ benchmark dataset, Kazakh-FEVER 36 articles / 120 verified claims, 303 automated passing tests).
  4. Milestone 4: Next Steps - Guidance Requested from Prof. Guo (Paper 2 positioning, thesis draft review timeline).

#### Spoken Script (English)
> Slide 20 summarizes our current academic progress, our agenda for today's live demonstration, and our immediate roadmap toward final graduation:
> First, Milestone 1: I am thrilled to report that Paper 1, titled "Morphologically-Grounded AI Detection in Low-Resource Kazakh", has been officially accepted for publication in the prestigious Springer Lecture Notes in Computer Science series for AIST 2026! All camera-ready revisions have been finalized, and code repositories have been archived. This acceptance provides international peer-reviewed validation for our dual-stream FST cross-attention architecture and ACL Kaz-MAGE benchmark.
> Second, Milestone 2: As part of our agenda for today's meeting, I have prepared a live demonstration of our 4-tab Gradio platform to showcase our models in action:
> - On Tab 1, we will test real-world Kazakh articles and observe real-time sentence-level heatmaps, dynamic gate weighting, and linguistic diagnostic bullets.
> - On Tab 2, we will inspect the FST Morphological Lab, demonstrating root and suffix chain decomposition on complex agglutinative words.
> - On Tab 3, we will review the interactive Kaz-MAGE 2x2 matrix and empirical ablation results.
> - On Tab 4, we will run the Kazakh-FEVER fact verification module, demonstrating millisecond BM25 retrieval, 3-way NLI prediction, and Four-Quadrant Trust badge mapping.
> Third, Milestone 3: As of September, our overall thesis progress stands at approximately 85% completion. Our 10,000-sample multi-domain corpus is curated, Kazakh-FEVER benchmark annotations are finalized, and our engineering codebase is protected by 303 passing automated tests with sub-1.4 GB memory limits.
> Fourth, Milestone 4: We are now positioned to finalize our second paper and complete the final chapters of the dissertation, for which I look forward to Professor Guo's strategic guidance.

---

### Slide 21: Discussion — Guidance Requests & Strategic Questions for Prof. Guo
- **Slide Title:** Discussion: Guidance Requests & Strategic Questions for Prof. Guo
- **Subtitle:** Key strategic questions regarding Paper 2 framing, dataset scaling, and defense preparation
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Comprehensive consultation points box covering 4 strategic dimensions:
  1. Paper 2 Venue & Positioning (LREC-COLING vs. EMNLP Findings)
  2. Kazakh-FEVER Benchmark Scaling (Expanding beyond 36 articles)
  3. User Study Design for Chapter 5 (Pilot evaluation with KazNU researchers)
  4. Internal Review & Defense Timeline (Chapters 5 & 6 completion schedule)

#### Spoken Script (English)
> Turning to Slide 21, I would like to solicit Professor Guo's valuable advice and strategic direction on four key questions:
> Question 1 concerns the positioning and target venue for Paper 2: We have two viable submission paths for our Kazakh-FEVER and Four-Quadrant Trust Matrix contributions. Option A is targeting LREC-COLING as a Resource and Benchmark paper, highlighting Central Asia's first fact-checking benchmark. Option B is targeting EMNLP 2026 Findings as a Methodological paper, emphasizing the joint modeling of synthetic text detection and evidence-grounded verification. I welcome Professor Guo's recommendation on narrative emphasis.
> Question 2 concerns benchmark scaling: Our gold test set currently comprises 36 comprehensive reference articles and 120 verified claims. Would Professor Guo recommend expanding the corpus to 100+ documents via controlled LLM synthesis prior to the October submission window?
> Question 3 concerns user evaluation: For Chapter 5 of the thesis, would you advise conducting a formal user study with bilingual students and faculty at KazNU to quantify the practical utility of our Gradio explainability dashboard?
> Finally, Question 4 concerns our timeline: Chapters 1 through 4 of the thesis are fully drafted, and Chapters 5 and 6 are being finalized. I seek Professor Guo's guidance on scheduling our internal pre-defense review and subsequent blind review submissions.

---

### Slide 22: Committee Review Comments & Responses — Addressing Expert Feedback
- **Slide Title:** Committee Review Comments & Responses: Addressing Expert Feedback
- **Subtitle:** Reviewer 1 (Internal) & Reviewer 2 (International) itemized revisions: 100.00% AUC wild generalization verified [RESOLVED]
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Side-by-side comparison tables: Reviewer 1 (Internal Academic Committee) vs. Reviewer 2 (International Committee); all 6 items marked `[RESOLVED]`.

#### Spoken Script (English)
> Slide 22 details our itemized responses and technical resolutions to the feedback provided by the examination committee:
> Addressing Reviewer 1 from the Internal Academic Committee:
> 1. On Wild Generator Generalization: We evaluated held-out Qwen-2.5-7B generated reviews in Quadrant 4, achieving 100.00% ROC-AUC. This confirms that our FST morphological features capture invariant linguistic legality regardless of the generative model. [RESOLVED]
> 2. On Long-Document Truncation: We developed `SentencePreservingChunker` with dynamic Top-K pooling, achieving 100% localization of stealth AI injections across 25,000-word documents under 1.4 GB VRAM. [RESOLVED]
> 3. On Conflating Veracity with Style: We developed the Kazakh-FEVER benchmark and Four-Quadrant Trust Matrix, decoupling AI drafting probability from factual correctness. [RESOLVED]
> Addressing Reviewer 2 from the International Committee:
> 1. On Reproducibility and Deployment: We built a standalone Hugging Face Spaces cloud package with one-command deployment, backed by 303 passing tests. [RESOLVED]
> 2. On Linguistic Justification: We grounded the architecture in an 83-rule FST engine, with ablation studies proving a +42.18% cross-domain AUC gain over pure neural baselines. [RESOLVED]
> 3. On Societal Impact and Fair Thresholding: We calibrated our operating decision threshold to 0.9980 to eliminate false positives against human writers, backed by plain-language linguistic explanations in the UI. [RESOLVED]

---

### Slide 23: Closing Slide — Thank You for Your Attention
- **Slide Title:** Thank You for Your Attention!
- **Subtitle:** Research on Morphologically-Grounded AI-Generated Text Detection and Factual Verification for Low-Resource Kazakh | Open for Discussion
- **Allocated Time:** 0.5 Minute
- **Visual Elements:** Panoramic campus background, bold typography, formal academic closing.

#### Spoken Script (English)
> That concludes my presentation.
> I would like to express my heartfelt gratitude to my advisor, Professor Guo, for his continuous guidance, rigorous mentorship, and support throughout this journey. I also extend my sincere appreciation to the examination committee for your valuable time and attention.
> Preserving information integrity and linguistic vitality for low-resource languages in the generative AI era is a vital research mission. I look forward to your questions, feedback, and constructive recommendations. Thank you very much!

---

## Defense Q&A Preparation: Anticipated Challenging Questions

### Q1: Why did KazRoBERTa baseline collapse to 57.62% AUC in Q3 while your Morpho-Detector reached 99.80%?
**Model Answer:**
> "Pretrained transformers like KazRoBERTa utilize subword Byte-Pair Encoding (BPE). In in-domain data (News), the detector easily overfits to specific editorial vocabulary and high-frequency subword bigrams characteristic of journalistic style. When transferred to Q3 (Kaspi.kz consumer reviews), the text is informal, featuring colloquial slang, non-standard spelling, and product terminology. The subword statistics shift dramatically.
> However, Kazakh inflectional morphology remains invariant: whether an author writes a formal news piece or a short informal review, grammatical case suffixes (-ның/-нің, -ға/-ге) and verbal participle suffixes follow identical agglutinative transition legality. Our 83-rule FST explicitly extracts this invariant structural scaffold. Through our dynamic gating mechanism, the network automatically down-weights semantic features (g ≈ 0.38) and relies on morphological transition regularity, completely insulating the classifier from topic and style shifts."

### Q2: Is the 100.00% AUC on Q4 wild data realistic, or is there data leakage?
**Model Answer:**
> "We rigorously verified that there is zero data leakage. Qwen-2.5-7B generated samples were generated with zero-shot web prompts and held out completely during training. 
> The reason Q4 achieves 100.00% AUC is twofold: First, modern high-parameter LLMs like Qwen-2.5 exhibit pronounced, highly regular morphological fluency in Kazakh that differs systematically from the noisy, irregular suffix distributions of human-authored colloquial reviews. Second, because our classifier uses a calibrated threshold of 0.9980 and Supervised Contrastive Loss (SupCon), human colloquial text is mapped into an extremely compact hyperspherical cluster, yielding perfect separation at the decision boundary. Furthermore, we verified this across 5-fold stratified cross-validation."

### Q3: Why is the Strict Joint FEVER score 66.67% when NLI accuracy is 100.00%?
**Model Answer:**
> "In accordance with standard FEVER benchmarking conventions, the Strict Joint FEVER metric requires a dual-success criterion: the model must correctly predict the 3-way NLI label (Supports/Refutes/NEI) AND the BM25 retrieval module must retrieve the exact gold sentence containing the factual proof span. 
> In our pipeline, the NLI cross-encoder performs with 100% accuracy once provided with candidate evidence. However, the BM25 evidence retriever achieves an Evidence Recall@3 of 91.67%. In cases where multiple candidate sentences share high lexical overlap or when the claim involves multi-sentence reasoning, BM25 occasionally ranks a secondary corroborating sentence higher than the annotated gold span. This reflects the realistic challenge of retrieval in agglutinative corpora and provides an honest, rigorous baseline for future research."

### Q4: How does the 25,000-word document chunker guarantee no OOM on modest hardware?
**Model Answer:**
> "Standard chunking approaches attempt to batch all document chunks simultaneously, which leads to quadratic memory growth in attention mechanisms. In `SentencePreservingChunker`, we implement defensive micro-batching: chunks (with a maximum length of 256 tokens and a 1-sentence sliding overlap) are processed in fixed batches of 16 chunks. Intermediate tensor activations are immediately freed after computing chunk logits. As demonstrated in Figure 9, peak VRAM remains strictly capped at under 1.4 GB regardless of whether the document has 1,000 words or 25,000 words. On CPU systems, processing completes in under 12 seconds."

### Q5: How do the three methodological innovations in Figure 15 interlock, and why is a monolithic end-to-end model insufficient?
**Model Answer:**
> "A monolithic neural detector fails on Kazakh because synthetic text artifacts operate across three distinct linguistic scales that cannot be conflated into a single scalar prediction:
> 1. Sub-word and Morpheme Scale: In agglutinative languages, out-of-domain collapse stems from subword vocabulary shifts. Innovation 1 isolates invariant morphological legality via the 83-rule FST cross-attention stream.
> 2. Discourse and Document Scale: Real-world academic integrity violations involve stealth local injections where 95% of a paper is authentic human writing and 5% is machine-generated. A whole-document embedding dilutes this local signal. Innovation 2 uses 10 abbreviation guards, sliding-window chunking, and adaptive worst-case Top-K pooling to pinpoint exact tampering spans without memory explosion.
> 3. Semantic and World Knowledge Scale: Style and veracity are orthogonal; a text can be stylistically synthetic yet factually accurate, or human-authored yet malicious disinformation. Innovation 3 decouples origin from veracity via Kazakh-FEVER retrieval and the Four-Quadrant Trust Matrix.
> These three innovations interlock serially: Stage 1 produces calibrated sentence logits and FST affixes; Stage 2 aggregates them into document scores while identifying anomalous spans; Stage 3 extracts claims from anomalous spans to verify factual veracity against authoritative sources."

### Q6: What is the core scientific distinction and contribution boundary between the accepted Springer LNCS Paper 1 and Paper 2?
**Model Answer:**
> "The accepted Springer LNCS Paper 1 focuses on the fundamental morphological detection problem: establishing the dual-stream FST cross-attention architecture, proving the +42.18% cross-domain AUC gain on our ACL Kaz-MAGE benchmark, and formalizing the sentence-preserving chunking mechanism. It solves the question: 'How do we reliably detect synthetic Kazakh text across domains without catastrophic degradation?'
> Paper 2 addresses the higher-order factual verification and trust problem: introducing Central Asia's first Kazakh-FEVER benchmark, the 36-article reference corpus, 120 verified claims, and the Four-Quadrant Trust Matrix (Risk_Trust = alpha * P_AI + (1 - alpha) * P_Refute). It answers: 'Given detected text, how do we distinguish harmless AI drafting from malicious disinformation?' This modular separation allows Paper 1 to stand as a rigorous computational linguistics foundation, while Paper 2 delivers a groundbreaking resource and factual integrity framework for high-impact venues such as EMNLP Findings or LREC-COLING."

### Q7: In the live Gradio system demonstration for today's meeting, how is explainability conveyed to non-technical users?
**Model Answer:**
> "Non-technical stakeholders, such as academic integrity officers or journal editors, cannot act on opaque confidence percentages alone. In our 4-tab Gradio platform, we provide three transparent levels of interpretability:
> 1. Visual Heatmap: Tab 1 renders color-coded sentence-level risk highlighting, where green denotes verified human writing and amber or red flags synthetic or tampered sections, allowing instant visual inspection of localized insertions.
> 2. Morphological Inspection: Tab 2 provides an interactive FST decomposition tree, displaying exactly which suffixes and inflectional transitions were parsed, proving why the model arrived at its decision based on linguistic legality.
> 3. Evidence Grounding: Tab 4 presents the retrieved gold reference sentences alongside the NLI inference badge and four-quadrant actionable recommendations (e.g., 'Publishable with Citation' versus 'Manual Fact Verification Required')."
