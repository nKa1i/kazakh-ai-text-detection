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
4. **Engineering and Institutional Rigor:** 311 automated regression tests passing, standalone Hugging Face Spaces cloud package, sub-1.2s cold start, zero decorative emojis, and itemized committee review resolutions.

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
  6. Conclusion, Research Contributions & Future Work

#### Spoken Script (English)
> Our presentation is organized into six structured chapters:
> First, I will introduce the Research Background and Core Motivation, highlighting the unique linguistic challenges of agglutinative morphology in Central Asia and the growing threat of generative disinformation.
> Second, I will review Related Work and explain the structural failure modes of mainstream Western detection paradigms and English-centric fact-checking corpora.
> Third, I will present our Research Content and System Architecture, introducing our comprehensive methodological framework in Figure 3 and detailing our three core technical innovations.
> Fourth, I will detail our Empirical Evaluation, including our ACL Kaz-MAGE benchmark, long-document stress tests, Kazakh-FEVER factual verification, and component ablations.
> Fifth, I will showcase our Engineering Implementation, spanning our 4-tab Gradio platform, Hugging Face Spaces cloud package, and 311 automated regression tests.
> Finally, I will summarize our overall thesis contributions, report our manuscript writing progress (~85% complete), announce our Paper 1 acceptance in Springer LNCS, outline our agenda for today's live demonstration, and request Professor Guo's strategic guidance for our final defense.

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

### Slide 7: Research Content Overview — Comprehensive Methodological Framework
- **Slide Title:** Research Content Overview: Comprehensive Methodological Framework
- **Subtitle:** A unified 3-column architecture spanning edge ingestion, dual-stream feature fusion with ablated baselines, and 3-tier portal verification
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Figure 3 (Overall Methodological Innovation Framework) organized into 3 publication-grade architectural columns:
  - Column 1 (Edge & Ingestion Subsystem): Input Text & Segmentation node (.txt, .docx, .pdf multi-format ingestion, 10 abbreviation regex guards, L=256 token windowing with 1-sentence overlap); 83-Rule FST parsing node producing morpheme sequences; Client ingestion endpoints (Web and API clients) with in-memory guards bounded under 1.4 GB VRAM; Edge serving device with numbered badge 1.
  - Column 2 (Core Detection & Verification Engine):
    - Left Sub-Column (Processing Modules): Contextual Semantic stream (KazRoBERTa 12-layer backbone yielding h_sem in R^768); Morphological Inductive stream (83-rule FST transducer with BiLSTM morpheme embeddings projected via W_proj into R^768); Ablated Baselines node (perplexity and standard subword BPE exhibiting severe root fragmentation and -42.18% OOD collapse, explicitly excluded).
    - Right Sub-Column (Core Processing & Classifier): Feature Fusion and Dynamic Gating Unit computing gate g = sigma(W_g [h_sem; W_proj h_morph] + b_g) and fused representation h_fused; Transformer Encoder & Top-K aggregation engine (4 layers, 4 heads, d_model=64) with dynamic worst-case Top-K pooling K = max(1, floor(0.25 * N)), dual-loss optimization (BCE + SupCon in R^128 hypersphere), global pooling yielding y_hat in [0, 1], empirical metrics (ROC-AUC = 0.9980, Kaspi OOD gain = +42.18%, Tamper precision = 100%, latency < 85ms), and numbered badge 2.
  - Column 3 (Portals & Deployment): Response Delivery via Gradio 4-tab dashboard and Telegram bot; 3-tier quantitative Threshold Alert Levels:
    - CRITICAL (y_hat >= 0.70): Synthetic / Malicious text, flag for immediate inspection.
    - WARNING (0.40 <= y_hat < 0.70): Ambiguous / Hybrid text, manual editorial review required.
    - NORMAL (y_hat < 0.40): Authentic human text, certified publication safe.
  - Caption: Figure 3. Overall Methodological Innovation Framework for Kazakh AI-Generated Text Detection and Factual Verification.

#### Spoken Script (English)
> Entering Chapter 3 on Slide 7, I present our comprehensive methodological framework, illustrated in Figure 3 as a unified three-column systems architecture.
> As recommended by Professor Guo, this end-to-end framework maps our technical innovations from client ingestion to core transformer reasoning and actionable deployment portals.
> On the left, in Column 1, the Edge and Ingestion Subsystem receives multi-genre Kazakh text and documents in .txt, .docx, or .pdf format. Our SentencePreservingChunker protects Kazakh punctuation with 10 abbreviation regex guards for terms like "т.б." and "ж.б.", slicing text into 256-token windows with 1-sentence overlap. In parallel, our 83-rule FST transducer parses each word into structured morpheme chains across 128 morphological tags. Web and REST API clients operate under strict in-memory guards, bounding peak memory to under 1.4 GB. From Badge 1, preprocessed tokens and morpheme sequences flow directly into our detection engine.
> In the center, Column 2 houses our Core Detection and Verification Engine.
> In the Processing Modules sub-column, we maintain two parallel representations: the Contextual Semantic stream via 12-layer KazRoBERTa producing h_sem in R^768, and the Morphological Inductive stream via an 83-rule FST BiLSTM producing projected embeddings in R^768. Below them, we explicitly identify the Ablated Baselines—showing how standard subword BPE and perplexity-only detectors slice Kazakh roots, leading to catastrophic domain collapse with a 42.18% drop, which we rigorously exclude.
> In the Core Processing sub-column, our Dynamic Gating Unit computes a learned dimension-wise gate g, softly combining semantic and morphological representations into h_fused. This feeds our Transformer Encoder and Document Top-K Engine at Badge 2, which applies worst-case Top-K pooling K = max(1, floor(0.25 * N)) and joint Supervised Contrastive Loss to output calibrated probability y_hat. Our architecture achieves a 0.9980 ROC-AUC, a +42.18% gain on colloquial text, 100% tamper precision, and sub-85ms latency.
> Finally, on the right in Column 3, our Portals layer delivers results via an interactive 4-tab Gradio web portal and Telegram bot. The scalar probability y_hat is mapped directly into our Three-Tier Quantitative Threshold Alert Levels:
> First, Red CRITICAL for y_hat >= 0.70, denoting synthetic or malicious content that requires immediate inspection;
> Second, Orange WARNING between 0.40 and 0.70, flagging ambiguous or hybrid documents for editorial verification;
> And Third, Green NORMAL below 0.40, certifying natural human text as safe for routine publishing.
> In Slides 8 through 10, I will now detail the formal mathematical formulations for each of these three core innovations.

---

### Slide 8: Topic 1 Architecture — Dual-Stream Morphology-Aware Gated Fusion & SupCon Loss
- **Slide Title:** Topic 1 Architecture: Dual-Stream Morphology-Aware Gated Fusion & SupCon Loss
- **Subtitle:** Fusing contextual semantic representations with 83-rule FST morphological inductive bias via dynamic gating
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Figure 4 (Dual-stream gated fusion architecture diagram); Mathematical gating formulation; Grounding & SupCon callout boxes.

#### Spoken Script (English)
> Slide 8 details the mathematical formulation of Topic 1.
> As shown in Figure 4, the input sequence x is processed through two parallel feature streams:
> First, the Semantic Stream: x is tokenized via subword BPE and encoded through 12-layer KazRoBERTa to extract dense semantic representation h_sem in R^768.
> Second, the Morphological Stream: the raw tokens are simultaneously parsed by our 83-rule FST transducer, which maps inflectional affixes and morpheme transitions into morphological representation h_morph in R^256, encoded via a bidirectional LSTM.
> To combine these streams without manual tuning, we project h_morph through learned matrix W_proj into R^768, and compute a dimension-wise dynamic gate vector g = sigma(W_g [h_sem; W_proj h_morph] + b_g) in [0, 1]^d.
> The fused representation is computed as h_fused = g * h_sem + (1 - g) * (W_proj h_morph).
> Our 83-rule FST transducer is grounded in Turkic phonotactics and vowel harmony, verified across 5,000 dictionary headwords with 99.2% inflectional coverage.
> During training, we optimize the network using a combined objective of binary cross-entropy and Supervised Contrastive Loss (SupCon) in R^128, which pulls representations of human text into tight clusters while repelling synthetic artifacts.
> The dynamic gate g serves as an adaptive routing mechanism: on standard news, g averages approximately 0.68, relying primarily on semantics; but on out-of-domain slang and colloquial reviews, g automatically adjusts to leverage the invariant morphological stream, substantially mitigating domain degradation.

---

### Slide 9: Topic 2 Architecture — Robust Generalization & Adversarial Rewriting Defense
- **Slide Title:** Topic 2 Architecture: Robust Generalization & Adversarial Rewriting Defense
- **Subtitle:** Mitigating generator and domain shift with adversarial robustness and supporting sentence-preserving chunk aggregation
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 5 (Chunking & Top-K pooling pipeline); Robust Latent Generalization callout; Sentence chunking and Top-K formula box.

#### Spoken Script (English)
> Moving to Slide 9, Topic 2 addresses the scientific challenge of Robust Generalization and Adversarial Rewriting Defense.
> In real-world environments, detectors face significant distribution shifts: unseen LLM generator architectures, domain transitions, and deliberate adversarial rewriting such as paraphrasing, synonym substitution, and inflectional noise.
> Our framework tackles this on two fronts:
> First, Generator-Agnostic Latent Invariance: By anchoring representations to regular morphological transition probabilities, our model maintains robust discriminative boundaries even when lexical surface patterns are perturbed by unseen rewriters. For Paper 2, we are expanding our SupCon formulation to train directly on AI-original versus AI-paraphrased contrastive pairs.
> Second, Supporting Engineering Extension for Long Documents: To audit multi-page documents where synthetic text may be localized to a few paragraphs, we introduce our `SentencePreservingChunker` with 10 Kazakh Abbreviation Guards (protecting abbreviations like "т.б.", "ғ.", and "ж." from spurious boundary splits).
> We aggregate document risk using Dynamic Top-K Worst-Chunk Pooling (K = max(1, floor(0.25 * N))), which prevents naive mean-pooling dilution on hybrid documents up to 25,000 words while bounding peak memory under 1.4 GB.

---

### Slide 10: Topic 3 Architecture — Evidence-Grounded Kazakh Claim Verification & Trust Matrix
- **Slide Title:** Topic 3 Architecture: Evidence-Grounded Kazakh Claim Verification & Trust Matrix
- **Subtitle:** Decoupling stylistic AI generation from factual veracity via curated reference grounding and dual-risk scoring
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Figure 6 (Fact-checking pipeline & trust matrix flow); Kazakh-FEVER pilot benchmark box; Two-dimensional coordinates box.

#### Spoken Script (English)
> Slide 10 details Topic 3: our evidence-grounded factual verification architecture and Four-Quadrant Trust Matrix.
> We formalize the critical decoupling of stylistic synthetic probability from factual veracity through an orthogonal dual-axis formulation:
> The process operates in five stages:
> 1. Claim Extraction: Factual assertions are extracted from the input Kazakh text.
> 2. Morphologically Stemmed BM25 Retrieval: Candidate evidence passages are retrieved from our curated Kazakh-FEVER knowledge store of 36 encyclopedic articles across History, Law, Science, and Public Health.
> 3. 3-Way NLI Cross-Encoder: The candidate claim and retrieved evidence are evaluated using our cross-lingual NLI backbone to predict probabilities for Supported, Refuted, and Not Enough Info (NEI).
> 4. Two-Dimensional Coordinate Mapping: Every text is projected into the unit square [0, 1]^2 defined by AI generation score y_AI and factual risk R_fact. An auxiliary composite scalar risk is computed as R_Trust = alpha * y_AI + (1 - alpha) * R_fact (alpha = 0.5 default).
> 5. Epistemic Uncertainty & Four Decision Quadrants:
>    - Quadrant 1: Verified Human Fact (low AI risk, low factual risk -> Safe).
>    - Quadrant 2: Human Misinformation (low AI risk, high factual risk -> Editorial Fact-Check Alert).
>    - Quadrant 3: Accurate AI Synthesis (high AI risk, low factual risk -> Attributed AI Assistance).
>    - Quadrant 4: Hallucinatory AI Disinformation (high AI risk, high factual risk -> Critical Deceptive Alert).
> Importantly, NEI claims indicate epistemic uncertainty (insufficient reference evidence) rather than known falsehood, transforming detection into nuanced governance.

---

### Slide 11: Experimental Setup — Kazakh Adaptation of MAGE Evaluation Protocol
- **Slide Title:** Experimental Setup: Kazakh Adaptation of MAGE Evaluation Protocol
- **Subtitle:** Rigorous 2x2 matrix evaluation across seen/unseen genres and generators with cross-validation isolation
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Datasets summary table; Protocol & Rigor callout cards; Figure 7 (KDE word length density & affix count distributions).

#### Spoken Script (English)
> Entering Chapter 4 on Slide 11, we describe our experimental setup and evaluation protocol.
> To ensure reproducible and rigorous evaluation, we adapted a MAGE-style 2x2 evaluation protocol across two orthogonal axes:
> Axis 1 evaluates Domain Transfer: In-Domain formal journalism and encyclopedic text (News and Wiki) versus Out-of-Domain colloquial e-commerce consumer reviews from Kaspi.kz.
> Axis 2 evaluates Generator Transfer: In-distribution models seen during training (Sherkala-7B) versus held-out unseen generators (Qwen-2.5-7B) and web-crawled real-world AI text (Qwen Wild).
> All experiments follow 5-fold stratified cross-validation with strict document-family isolation, prompt template controls, and deduplication to prevent data leakage.
> The decision threshold is calibrated at 0.9980 to rigorously protect human authors from false-positive accusations.
> Furthermore, all algorithmic invariants are certified by our 314 automated unit and integration tests.

---

### Slide 12: Topic 1 Empirical Results — Resolving the Out-of-Domain Blindspot (+42.18 pp Gain)
- **Slide Title:** Topic 1 Empirical Results: Resolving the Out-of-Domain Blindspot (+42.18 pp Gain)
- **Subtitle:** Our Dual-Stream Morphological Gated Detector substantially mitigates domain degradation in our evaluation
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Quadrant comparison table (KazRoBERTa vs. mBERT vs. Ours); 3 Stat Cards (+42.18 pp Blindspot Resolved, 100.00% Q4 Wild Generalization, 99.85% Q1 ROC-AUC); Figure 8 (4-quadrant ROC curves).

#### Spoken Script (English)
> Slide 12 presents our primary empirical evaluation for Topic 1 on the Kazakh MAGE-style 2x2 matrix.
> Please examine the quadrant comparison table and ROC curves in Figure 8:
> In Quadrant 1 (In-Domain News, Known Generator), all models perform competitively, with our Morpho-Gated Detector achieving 99.85% ROC-AUC (mean cross-validation AUC: 99.80 +/- 0.15%, and Test Macro-F1 of 99.24%).
> The pivotal challenge arises in Quadrant 3: transferring to Out-of-Domain colloquial Kaspi reviews. Here, standard pretrained KazRoBERTa drops sharply to 57.62% AUC, and multilingual mBERT falls to 53.20%, suffering severe domain degradation due to subword BPE root fragmentation.
> In contrast, our Dual-Stream Morphology-Aware Gated Detector achieves 99.80% ROC-AUC—delivering an absolute gain of +42.18 percentage points!
> Furthermore, in Quadrant 4 (cross-domain consumer reviews generated by held-out Qwen-2.5-7B web-crawled text), our model achieves an ROC-AUC of 1.000 (100.0%), compared to 78.45% for KazRoBERTa.
> We also evaluated length-bucket behavior: while short snippets under 50 tokens exhibit slightly higher variance, our calibrated threshold maintains a human false-positive rate well under 1%.
> As shown in Figure 8, the ROC curves confirm that morphological inductive bias consistently bridges domain and generator shifts.

---

### Slide 13: Topic 2 Empirical Results — Cross-Generator Robustness & Long-Document Evaluation
- **Slide Title:** Topic 2 Empirical Results: Cross-Generator Robustness & Long-Document Evaluation
- **Subtitle:** Evaluating out-of-distribution transfer and localized synthetic paragraph detection up to 25,000 words
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 9 (Tampering detection rate & VRAM scaling); Cross-Generator Robustness card; Practical Extension & Micro-batching card.

#### Spoken Script (English)
> Slide 13 details our Topic 2 empirical results on cross-generator transfer and document-level stress testing.
> On generator transfer, our detector demonstrates strong resilience when evaluated on held-out LLMs, yielding a +3.66 pp gain in-domain and a +21.55 pp gain out-of-domain over pretrained baselines, confirming that morphological regularity bounds model sensitivity.
> Complementing this, as a practical engineering extension for document auditing, we evaluated our sliding-window chunker across 200 synthetic long-form documents ranging from 1,000 to 25,000 words, where 1 to 5 paragraphs were covertly replaced with synthetic text:
> First, Localized Tampering Recall: Our dynamic Top-K aggregation correctly localized 100% of injected synthetic paragraphs with exact character span alignment and zero boundary truncation artifacts.
> Second, Sub-Linear Micro-Batching Scalability: Inference latency scales sub-linearly—from 0.18s for 1,000 words to 4.10s for a full 25,000-word dissertation manuscript.
> Third, Memory Footprint: As depicted in Figure 9, micro-batching caps peak GPU memory under 1.4 GB throughout the entire 25,000-word evaluation, while a CPU fallback analyzes the complete document in under 12 seconds.

---

### Slide 14: Topic 3 Empirical Results — Kazakh-FEVER Pilot Verification Benchmark
- **Slide Title:** Topic 3 Empirical Results: Kazakh-FEVER Pilot Verification Benchmark
- **Subtitle:** Empirical evaluation of evidence retrieval, NLI claim verification, and joint FEVER scoring on curated pilot data
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** NLI verification metrics table; 3 Stat Cards (100.00% NLI F1, 91.67% Evidence Recall@3, 66.67% Strict Joint FEVER); Figure 10 (3-Way NLI confusion matrix).

#### Spoken Script (English)
> On Slide 14, we evaluate our factual verification pipeline on the curated Kazakh-FEVER pilot benchmark across 36 encyclopedic articles and 120 claims (40 Supports, 40 Refutes, 40 Not Enough Info).
> Our empirical findings illuminate both the strengths and current bottlenecks of low-resource fact verification:
> First, NLI Classification Fidelity: Our cross-encoder achieved 100.00% Macro-F1 across all three claim categories on the curated pilot set, as reflected in the diagonal of Figure 10.
> Second, Evidence Retrieval Recall: Morphologically stemmed BM25 achieved an Evidence Recall@3 of 91.67% across the reference knowledge store.
> Third, Honest Bottleneck Analysis on Strict Joint FEVER: Under the strict joint metric—which requires both exact gold evidence sentence retrieval and correct NLI label classification—the system achieves 66.67%.
> This transparently reveals that evidence retrieval, rather than NLI reasoning, constitutes the primary operational bottleneck in factual verification. In Paper 2, we plan to address this bottleneck by integrating multilingual dense retrieval embeddings alongside BM25.

---

### Slide 15: Topic 3 Empirical Results — Four-Quadrant Trust Matrix Validation
- **Slide Title:** Topic 3 Empirical Results: Four-Quadrant Trust Matrix Validation
- **Subtitle:** Empirical validation of dual-risk coordinates separating authentic human fact from deceptive hallucination
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 11 (2D scatter plot of Four-Quadrant Trust Matrix); 4 Quadrant callout boxes detailing decision boundaries and actions.

#### Spoken Script (English)
> Slide 15 visualizes the empirical validation of our Four-Quadrant Trust Matrix in two-dimensional risk space.
> In Figure 11, each evaluated text is plotted with AI generation probability y_AI on the horizontal axis and factual risk R_fact on the vertical axis:
> - Quadrant 1 (Green circles, bottom-left): Authentic journalistic news clusters tightly with mean AI probability of 0.08 and factual risk of 0.04. The system validates them as authentic human writing and clears them immediately.
> - Quadrant 2 (Orange triangles, top-left): Human-authored rumors and misconceptions exhibit low AI probability (0.05) but high factual risk (R_fact = 1.00). The system flags them for editorial fact-checking, preventing human misinformation from slipping through.
> - Quadrant 3 (Blue diamonds, bottom-right): Accurate LLM summaries exhibit high AI probability (0.96) but low factual risk (R_fact = 0.00). Conventional detectors would wrongly ban these, but our system recognizes them as factually sound AI assistance.
> - Quadrant 4 (Red squares, top-right): Fabricated historical claims and hallucinated legal statutes trigger high AI and high factual risk, prompting an immediate critical alert.
> This demonstrates the necessity of preserving orthogonal 2D coordinates rather than relying solely on a collapsed scalar metric.

---

### Slide 16: Comprehensive Component Ablation Studies — Isolating Key Architectural Gains
- **Slide Title:** Comprehensive Component Ablation Studies: Isolating Key Architectural Gains
- **Subtitle:** Disentangling sentence-level morphological inductive priors from document-level aggregation modules
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Ablation configurations table (5 rows vs. 4 quadrants); Figure 12 (Horizontal bar chart of ablation impacts); 2 Takeaway Cards.

#### Spoken Script (English)
> On Slide 16, we present our component ablation study, explicitly disentangling sentence-level morphological inductive priors from document-level chunk aggregation.
> As illustrated in Figure 12:
> First, The FST Morphological Inductive Prior: Removing the 83-rule FST stream results in a 42.18 percentage point drop in out-of-domain Kaspi AUC (from 99.80% down to 57.62%). This confirms that subword transformers alone lack the structural inductive bias necessary to generalize across agglutinative morphological variations.
> Second, Dynamic Gating vs. Static Concatenation: Replacing our dimension-wise dynamic gate with static concatenation degrades colloquial review AUC by 11.35 percentage points, demonstrating that adaptive stream weighting is crucial when lexical register shifts.
> Third, Supervised Contrastive Regularization: SupCon provides latent stability and tighter class separation across cross-validation splits.
> At the document level, our dynamic Top-K worst-chunk pooling prevents a 23.75% omission drop compared to naive mean-pooling on hybrid documents.
> For Paper 2, we are expanding this ablation suite to include fine-grained factor analysis (isolating stem versus affix embeddings and comparing against character-level baselines).

---

### Slide 17: System Demonstration — Interactive 4-Tab Gradio Academic Prototype
- **Slide Title:** System Demonstration: Interactive 4-Tab Gradio Academic Prototype
- **Subtitle:** Explainable AI detection and evidence-grounded verification prototype designed for academic integrity
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 13 (Dashboard interface layout preview); 4 Tab feature cards:
  - Tab 1: Detection & Explainability
  - Tab 2: Morphological FST Lab
  - Tab 3: Benchmark Methodology Browser
  - Tab 4: Kazakh-FEVER Trust Matrix

#### Spoken Script (English)
> Entering Chapter 5 on Slide 17, we demonstrate the interactive academic prototype system developed to make our research accessible for evaluation.
> To support committee review, integrity offices, and researchers, we engineered a 4-tab Gradio prototype dashboard (previewed in Figure 13):
> - Tab 1 delivers Detection & Saliency: Users input text or select presets across News, Wiki, and Kaspi reviews. The system renders an XSS-sanitized sentence attribution heatmap, dynamic gate stream weights (e.g. KazRoBERTa 68% vs FST 32%), and lexical diagnostics.
> - Tab 2 hosts the Morphological FST Lab: It displays word-by-word morpheme decomposition trees, illustrating roots, affix chains, and vowel harmony validation.
> - Tab 3 provides the Benchmark Methodology Browser: It embeds the interactive MAGE-style 2x2 matrix and empirical ablation results with full academic transparency.
> - Tab 4 implements the Kazakh-FEVER Trust Matrix: It performs real-time BM25 evidence retrieval, 3-way NLI classification, and maps the input into our Four-Quadrant decision framework.

---

### Slide 18: System Demonstration — Cloud Packaging, Hugging Face Spaces & Test Rigor
- **Slide Title:** System Demonstration: Cloud Packaging, Hugging Face Spaces & Test Rigor
- **Subtitle:** Containerized prototype deployment bundle with sub-1.2s cold start, multi-format ingestion, and 314 passing tests
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Split visual architecture:
  - Left Column: Figure 14 (Cloud Deployment Pipeline) across 4 sequential stages (Ingestion, Automated Test Suite Rigor, Containerized Serving, Interactive Prototype Serving).
  - Right Column:
    - Top Metric Cards: 314 / 314 Automated Test Suite, < 1.2s Cold-Start Latency, 3 Formats Multi-Format Ingestion.
    - Bottom Feature Box: Hugging Face Spaces Containerized Prototype Features (`hf_space/`).

#### Spoken Script (English)
> Slide 18 demonstrates our software engineering rigor and containerized prototype deployment pipeline.
> As visualized on the left in Figure 14, our pipeline is structured across four stages:
> First, Stage 1: Multi-Format Ingestion Engine. We engineered defensive stream parsers for `.txt`, `.docx`, and `.pdf` documents, guarded by 10MB memory upload limits and a 25,000-word streaming buffer cap to prevent denial-of-service or decompression attacks.
> Second, Stage 2: Automated Test Suite Rigor. Our continuous integration pipeline executes 314 automated unit, integration, and regression tests, verifying the 83-rule FST parser, chunk aggregation, and NLI fact-verification modules with 100% pass rate in under 15 seconds.
> Third, Stage 3: Containerized Serving Engine. Packaged as a lightweight Docker container for Hugging Face Spaces under Python 3.12, the prototype uses distilled model weights, ensuring a sub-1.2 second (~1s) cold-start initialization and bounded VRAM under 1.4 GB.
> Fourth, Stage 4: Interactive Prototype Serving. The platform serves the Gradio 4.x web UI, delivering sub-85ms single-sentence inference and real-time Four-Quadrant report cards.
> On the right, our 3 stat cards confirm: 314 out of 314 automated tests verified, sub-1.2s cold-start initialization, and defensive ingestion for all 3 document formats.

---

### Slide 19: Conclusion — Summary of Thesis Contributions & Writing Progress
- **Slide Title:** Conclusion: Summary of Thesis Contributions & Writing Progress
- **Subtitle:** Master's thesis completion estimated at 85%; core theoretical, empirical, and prototype milestones achieved
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Two Structured Academic Cards: Left Card (Four Primary Academic & System Contributions); Right Card (Master's Thesis Manuscript Status ~85% Complete).
  - Left Card (Four Primary Academic & System Contributions):
    1. Algorithmic Contribution: Dual-stream 83-rule FST dynamic morphological gated fusion (+42.18 pp OOD gain).
    2. Methodological & Practical Extension: Cross-generator robustness and sentence-preserving chunk aggregation with 10 Kazakh abbreviation guards.
    3. Trustworthy Verification Contribution: Evidence-grounded Kazakh-FEVER pilot benchmark (36 articles, 120 claims) and Four-Quadrant Trust Matrix.
    4. Engineering & Prototype Contribution: Reproducible 4-tab Gradio prototype, multi-format ingestion platform, and 314 passing automated tests.
  - Right Card (Master's Thesis Manuscript Status ~85% Complete):
    - Chapter 1: Introduction (100% Complete)
    - Chapter 2: Related Work (100% Complete)
    - Chapter 3: Methodology (100% Complete)
    - Chapter 4: Experiments & Robustness (100% Complete)
    - Chapter 5: Evidence-Grounded Verification (90% Complete)
    - Chapter 6: System Prototype & Roadmap (70% Complete)

#### Spoken Script (English)
> Turning to Chapter 6 on Slide 19, I summarize the primary research contributions of this thesis and report on our dissertation manuscript writing progress.
> As displayed on the left card, this thesis delivers four primary academic and system contributions:
> First, our Algorithmic Contribution: We pioneered the integration of rule-based 83-rule FST morphological representations with pretrained transformer backbones via learned dynamic gated fusion, substantially mitigating out-of-domain collapse with an absolute +42.18 percentage point gain.
> Second, our Methodological & Practical Extension: We established generator-invariant detection under unseen LLMs and engineered sentence-preserving chunking with 10 Kazakh abbreviation guards and dynamic Top-K pooling for documents up to 25,000 words under a strictly bounded 1.4 GB VRAM envelope.
> Third, our Trustworthy Verification Contribution: We constructed the first curated evidence-grounded factual verification pilot benchmark, Kazakh-FEVER (36 articles, 120 claims), and established the two-dimensional Four-Quadrant Trust Matrix, decoupling stylistic generative probability from factual veracity to safeguard against hallucinations.
> Fourth, our Engineering & Prototype Contribution: We delivered a fully reproducible, open-source 4-tab Gradio prototype and multi-format ingestion platform (.txt, .docx, .pdf), certified by 314 passing automated regression tests.
> Looking at the right card, our Master's thesis manuscript is currently estimated at 85% overall completion:
> - Chapter 1 (Introduction) is 100% complete, establishing the linguistic motivation and problem formulation.
> - Chapter 2 (Related Work) is 100% complete, surveying SOTA detectors, LLM watermarks, and fact-checking corpora.
> - Chapter 3 (Methodology) is 100% complete, detailing Topic 1 Morphological Gated Fusion and SupCon formulations.
> - Chapter 4 (Experiments & Robustness) is 100% complete, analyzing our MAGE-style 2x2 matrix, cross-generator transfer, and ablation studies.
> - Chapter 5 (Evidence-Grounded Verification) is 90% complete, documenting the Kazakh-FEVER pilot benchmark and Trust Matrix.
> - Chapter 6 (System Prototype & Roadmap) is 70% complete, documenting the Gradio platform and outlining our Paper 2 adversarial rewriting attack benchmark.
> With our core scientific milestones accomplished and our dissertation nearing completion, let us proceed to Slide 20 to review our accepted Springer LNCS publication and today's meeting agenda.

---

### Slide 20: Current Progress — LNCS Acceptance & September Meeting Agenda
- **Slide Title:** Current Progress: LNCS Acceptance & September Meeting Agenda
- **Subtitle:** Academic Paper Acceptance, September Meeting Live Demo Agenda, and Milestone Roadmap
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** 4 Sequential Milestone Cards:
  1. Milestone 1: Accepted Springer LNCS (Paper 1 formally accepted to AIST 2026; camera-ready finalized).
  2. Milestone 2: Meeting Agenda - Live System Demonstration (Interactive 4-tab Gradio platform prepared for today's meeting).
  3. Milestone 3: September Progress - Research Milestones (~85% thesis completion, 10,000+ benchmark dataset, Kazakh-FEVER 36 articles / 120 verified claims, 314 automated passing tests).
  4. Milestone 4: Next Steps - Guidance Requested from Prof. Guo (Paper 2 positioning, thesis draft review timeline).

#### Spoken Script (English)
> Slide 20 summarizes our current academic progress, our agenda for today's live demonstration, and our immediate roadmap toward final graduation:
> First, Milestone 1: I am pleased to report that Paper 1, titled "Morphologically-Grounded AI Detection in Low-Resource Kazakh", has been officially accepted for publication in Springer Lecture Notes in Computer Science (LNCS) for AIST 2026! Camera-ready revisions have been finalized, and code repositories archived. This provides international peer-reviewed validation for our dual-stream morphological gated fusion architecture.
> Second, Milestone 2: As part of our agenda for today's meeting, I have prepared a live demonstration of our 4-tab Gradio platform to showcase our prototype in action:
> - On Tab 1, we will test real-world Kazakh articles and observe real-time sentence-level heatmaps, dynamic gate weighting, and linguistic diagnostic bullets.
> - On Tab 2, we will inspect the FST Morphological Lab, demonstrating root and suffix chain decomposition on complex agglutinative words.
> - On Tab 3, we will review the interactive benchmark browser and empirical ablation results.
> - On Tab 4, we will run the Kazakh-FEVER fact verification module, demonstrating BM25 retrieval, 3-way NLI prediction, and Four-Quadrant Trust coordinates.
> Third, Milestone 3: As of September, our overall thesis progress stands at approximately 85% completion. Our 10,000-sample multi-domain corpus is curated, Kazakh-FEVER pilot benchmark annotations are finalized, and our codebase is validated by 314 passing automated tests.
> Fourth, Milestone 4: We are now positioned to finalize our second paper and complete the final chapters of the dissertation, for which I look forward to Professor Guo's strategic guidance.

---

### Slide 21: Discussion — Guidance Requests & Strategic Questions for Prof. Guo
- **Slide Title:** Discussion: Guidance Requests & Strategic Questions for Prof. Guo
- **Subtitle:** Three core strategic decisions regarding Topic 2 robustness experiments, Kazakh-FEVER expansion, and Paper 2 target venue
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Comprehensive consultation points box covering 3 strategic dimensions:
  1. Topic 2 Adversarial Rewriting Attack Benchmark Execution (Paraphrasing & synonym perturbation)
  2. Kazakh-FEVER Evidence Retrieval Expansion Strategy (Dense embeddings vs BM25)
  3. Paper 2 Framing & Target Venue Alignment (EMNLP vs LREC-COLING)

#### Spoken Script (English)
> Turning to Slide 21, I would like to solicit Professor Guo's valuable advice and strategic direction on three core questions:
> Question 1 concerns Topic 2 Adversarial Rewriting Attack Experiments for Paper 2: How should we structure the benchmark for paraphrasing, synonym substitution, and morphological noise? We recommend evaluating rule-based synonym substitution and LLM paraphrasing on held-out test splits to quantify adversarial degradation.
> Question 2 concerns Kazakh-FEVER Retrieval Expansion: Our pilot evaluation features 36 curated articles and 120 claims, where evidence retrieval was identified as the primary bottleneck (66.67% Joint FEVER vs 100% NLI Macro-F1). Should we expand to 100+ articles and benchmark dense multilingual embeddings against BM25 prior to Paper 2 submission?
> Question 3 concerns Paper 2 Target Venue and Framing: We have two natural submission trajectories. Option A is an algorithmic methodology paper targeting EMNLP (focusing on the Dual-Risk Trust Matrix and dynamic gated optimization). Option B is a resource and benchmark paper targeting LREC/COLING (highlighting the Kazakh-FEVER corpus and low-resource evaluation gap). What is Professor Guo's advice on narrative emphasis and venue alignment?

---

### Slide 22: Committee Review Comments & Responses — Addressing Expert Feedback
- **Slide Title:** Committee Review Comments & Responses: Addressing Expert Feedback
- **Subtitle:** Itemized academic revisions: Addressing internal and international committee feedback with verified empirical rigor
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Side-by-side comparison tables: Reviewer 1 (Internal Academic Committee) vs. Reviewer 2 (International Committee); itemized status badges indicating thesis chapter references.

#### Spoken Script (English)
> Slide 22 details our itemized responses and technical resolutions to the feedback provided by the examination committee:
> Addressing Reviewer 1 from the Internal Academic Committee:
> 1. On Wild Generator Generalization: We evaluated held-out Qwen-2.5-7B generated reviews in Quadrant 4, achieving 100.00% ROC-AUC, confirming morphological regularity is generator-invariant. This is fully addressed in Thesis Chapter 4.
> 2. On Long-Document Truncation: We developed `SentencePreservingChunker` with dynamic Top-K pooling, achieving 100% localization of stealth AI injections across 25,000-word documents under 1.4 GB VRAM. This is addressed in Thesis Chapters 4 and 6.
> 3. On Conflating Veracity with Style: We developed the Kazakh-FEVER pilot benchmark and Four-Quadrant Trust Matrix, decoupling AI drafting probability from factual correctness. This is addressed in Thesis Chapter 5.
> Addressing Reviewer 2 from the International Committee:
> 1. On Reproducibility and Deployment: We built a containerized prototype for Hugging Face Spaces with sub-1.2s cold start, verified by 314 passing tests. This is addressed in Thesis Chapter 6.
> 2. On Linguistic Justification: We grounded the architecture in an 83-rule FST engine, with ablation studies proving a +42.18 pp cross-domain AUC gain over pure neural baselines. This is addressed in Thesis Chapter 3.
> 3. On Societal Impact and Fair Thresholding: We calibrated our operating decision threshold to 0.9980 to eliminate false positives against human writers, backed by sentence heatmaps in the UI. This is addressed in Thesis Chapters 5 and 6.

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

### Q5: How do the three methodological innovations in Figure 3 interlock, and why is a monolithic end-to-end model insufficient?
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
