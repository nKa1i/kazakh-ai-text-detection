# Master's Thesis Dense Speaker Notes & Quick Defense Cue-Cards

**Candidate:** Daulet (大雷)  
**Supervisor:** Prof. Guo  
**Degree:** Master of Science in Computer Science and Technology  
**Institutions:** School of Computer Science, Northwestern Polytechnical University (NPU) & Al-Farabi Kazakh National University (KazNU)  
**Target Duration:** 12–15 Minutes (~30–40 seconds per slide)  
**Primary Presentation:** `AnekeshD_Progress.pptx` (23 Widescreen 16:9 Slides)  

---

## 60-Second "Elevator Pitch" (Quick Summary for Prof. Guo)

> "Respected Professor Guo and committee members: My thesis solves the acute failure of AI text detectors in agglutinative Kazakh, where standard transformers suffer a 42-percentage-point out-of-domain collapse due to subword root fragmentation. 
> We introduce three integrated innovations: 
> 1. **Morphological Gated Fusion**: An 83-rule FST transducer dynamically combined with KazRoBERTa, recovering +42.18 pp on colloquial text with 99.80% AUC.
> 2. **Robust Generalization & Long-Doc Chunking**: Adversarial robustness against unseen LLMs and a 10-guard sentence chunker with Top-K worst-chunk pooling detecting 100% of injected paragraphs up to 25,000 words.
> 3. **Kazakh-FEVER & 2D Trust Matrix**: Central Asia's first 120-claim fact-checking benchmark and orthogonal risk plane separating factual AI writing from synthetic hallucinations.
> Our Paper 1 is accepted in Springer LNCS, our prototype runs live on Gradio and Hugging Face Spaces backed by 314 automated tests, and our thesis manuscript is currently ~85% complete."

---

## Slide-by-Slide Dense Cue-Cards (Slides 01 – 23)

```
Target Timing Breakdown:
- Slides 01–06: Background & Motivation    [~3.5 mins]
- Slides 07–10: Methodological Framework   [~3.0 mins]
- Slides 11–16: Empirical Evaluations      [~4.0 mins]
- Slides 17–18: System Implementation      [~1.5 mins]
- Slides 19–23: Contributions & Next Steps [~2.5 mins]
Total Estimated Defense Talk Time: ~14.5 Minutes
```

---

### Slide 01: Title Slide & Credentials
- **Target Time:** 30 seconds `[0:30]`
- **Visual Cue:** University seals (NPU & KazNU) and thesis title.
- **Core Claim:** Introducing the first morphologically-grounded AI detection and factual verification framework for Kazakh.
- **Dense Spoken Lines:**
  > "Good morning, Professor Guo and honorable committee members. My name is Daulet. Today I present my Master's thesis progress: *Research on Morphologically-Grounded AI-Generated Text Detection and Factual Verification for Low-Resource Kazakh*, supervised by Professor Guo within our joint NPU and KazNU program."
- **Next Slide Bridge:** "Let us review our agenda."

---

### Slide 02: Table of Contents
- **Target Time:** 20 seconds `[0:50]`
- **Visual Cue:** 6 structured thesis chapters (numbered 01 to 06).
- **Core Claim:** Standard academic progression from linguistic motivation to accepted publication.
- **Dense Spoken Lines:**
  > "Our presentation follows six chapters: Background and linguistic motivation, Related Work limitations, our 3 core Methodological Innovations, Empirical Evaluations on our MAGE-style benchmark, our interactive Gradio Prototype, and our ~85% thesis completion roadmap."
- **Next Slide Bridge:** "Beginning with Chapter 1..."

---

### Slide 03: Research Background — Agglutinative Vulnerability & Synthetic Threat
- **Target Time:** 45 seconds `[1:35]`
- **Visual Cue:** Figure 1 (Word `Қазақстандықтардың` split into 7 BPE pieces vs clean FST morpheme chain).
- **Core Claim:** Subword tokenizers obliterate Kazakh root words, blinding standard AI detectors.
- **Dense Spoken Lines:**
  > "With multilingual LLMs generating fluent Kazakh, Central Asia faces an unprecedented synthetic text threat. In Kazakh, suffixes stack up to 15 layers deep. Standard tokenizers like BPE fragment authentic words like *Қазақстандықтардың* into 7 meaningless pieces, destroying the root *Қазақ*. Our 83-rule FST restores these grammatical chains across 128 morphological tags."
- **Next Slide Bridge:** "This fragmentation causes acute system failure..."

---

### Slide 04: Problem Statement — Why Standard Detectors Fail on Kazakh
- **Target Time:** 40 seconds `[2:15]`
- **Visual Cue:** Figure 2 crimson bar (-41.79% drop) and 3 stacked failure mode cards.
- **Core Claim:** Three fatal bottlenecks: Domain Collapse (-42 pp), Truncation (512 tokens), and Truth-Agnostic output.
- **Dense Spoken Lines:**
  > "Empirical audits reveal three fatal bottlenecks in existing tools: First, **Morphological Domain Collapse**: fine-tuned KazRoBERTa drops 41.79 percentage points when moving from formal news to colloquial Kaspi reviews. Second, **Document Truncation**: 512-token limits miss attacks in 20-page documents. Third, **Truth-Blindness**: detectors cannot distinguish truthful AI summaries from human fake news."
- **Next Slide Bridge:** "Looking at existing literature..."

---

### Slide 05: Related Work — SOTA Detection Paradigms
- **Target Time:** 40 seconds `[2:55]`
- **Visual Cue:** SOTA comparison table (Binoculars, Ghostbuster, Fast-DetectGPT vs Ours).
- **Core Claim:** Western detectors collapse on Turkic morphology; symbolic linguistic priors are mandatory.
- **Dense Spoken Lines:**
  > "As shown in our comparison table, perplexity ratio detectors like Binoculars drop to 52.1% AUC due to Kazakh vocabulary sparsity; API-dependent methods like Ghostbuster fail on low-resource open models; and standard fine-tuned transformers collapse to 57.6%. Our approach incorporates morphological inductive bias, achieving 99.80% AUC with zero external API dependencies."
- **Next Slide Bridge:** "Furthermore, detection alone is insufficient..."

---

### Slide 06: Related Work — LLM Hallucination Verification Gaps
- **Target Time:** 40 seconds `[3:35]`
- **Visual Cue:** Top 2-column blue banner and 3 comparative pillars (Detectors vs FEVER/VitaminC vs Kazakh-FEVER).
- **Core Claim:** Origin detection ($\hat{y}_{AI}$) cannot verify factual truth ($R_{fact}$); existing fact-checking corpora ignore Turkic languages.
- **Dense Spoken Lines:**
  > "Stylistic detectors suffer a fundamental scientific blindspot: they only output text origin, not factual truth. An LLM can write an accurate history summary, while a human can write harmful disinformation. Furthermore, benchmarks like FEVER and VitaminC are exclusively English-centric. Topic 3 bridges this gap by constructing Central Asia's first Kazakh-FEVER benchmark and Two-Dimensional Trust Matrix."
- **Next Slide Bridge:** "Now, let us examine our overall systems architecture."

---

### Slide 07: Methodological Framework (Figure 3)
- **Target Time:** 50 seconds `[4:25]`
- **Visual Cue:** Figure 3 (3 columns: Edge Ingestion $\rightarrow$ Core Engine $\rightarrow$ 3-Tier Alert Portals).
- **Core Claim:** End-to-end framework from client parsing to transformer reasoning and real-time deployment.
- **Dense Spoken Lines:**
  > "Figure 3 illustrates our comprehensive systems architecture across three columns: On the left, our Edge Subsystem ingests multi-format documents, protecting punctuation with 10 abbreviation guards and parsing morphemes with our 83-rule FST. In the center, our core engine combines semantic KazRoBERTa with morphological BiLSTM via learned dynamic gating into $h_{fused}$, optimized with Supervised Contrastive Loss. On the right, scalar probabilities map to our 3-tier alert levels: Critical (red, $\ge 0.70$), Warning (orange, $0.40–0.70$), and Normal (green, $< 0.40$)."
- **Next Slide Bridge:** "Let us examine the mathematical formulation of Topic 1."

---

### Slide 08: Topic 1 Architecture — Dual-Stream Gated Fusion & SupCon
- **Target Time:** 45 seconds `[5:10]`
- **Visual Cue:** Formula box: $g = \sigma(W_g [h_{sem}; W_{proj} h_{morph}] + b_g)$ and $h_{fused} = g \odot h_{sem} + (1 - g) \odot (W_{proj} h_{morph})$.
- **Core Claim:** Dynamic gating adaptively balances semantics ($h_{sem}$) and morphology ($h_{morph}$) per dimension.
- **Dense Spoken Lines:**
  > "In Topic 1, input text produces two streams: dense semantics $h_{sem}$ from KazRoBERTa in $\mathbb{R}^{768}$, and morphological representations $h_{morph}$ from our 83-rule FST BiLSTM projected via $W_{proj}$ into $\mathbb{R}^{768}$. A learned gate $g \in [0, 1]^d$ dynamically weights meaning versus grammar. On formal news, the gate relies on semantics ($g \approx 0.68$); on colloquial slang, it automatically increases grammatical weighting. SupCon loss pulls human latent clusters together while repelling synthetic text."
- **Next Slide Bridge:** "Moving to Topic 2..."

---

### Slide 09: Topic 2 Architecture — Robust Generalization & Chunking
- **Target Time:** 40 seconds `[5:50]`
- **Visual Cue:** Top-K pooling formula: $K = \max(1, \lfloor 0.25 \times N \rfloor)$ and sliding window diagram.
- **Core Claim:** Morphological regularity bounds sensitivity to generator shift; Top-K worst-chunk pooling prevents dilution.
- **Dense Spoken Lines:**
  > "Topic 2 tackles robust generalization against unseen generators and adversarial rewriting. Anchoring representations in morphological regularity prevents rewriters from escaping detection. For document auditing, our `SentencePreservingChunker` protects Kazakh abbreviations like *т.б.* and *ж.б.* across 256-token windows. To prevent human text from diluting localized AI paragraphs, our Top-K engine pools the worst 25% suspicious chunks."
- **Next Slide Bridge:** "Moving to Topic 3..."

---

### Slide 10: Topic 3 Architecture — Kazakh-FEVER & 2D Trust Matrix
- **Target Time:** 45 seconds `[6:35]`
- **Visual Cue:** 2D Trust Matrix quadrant diagram (4 quadrants in $[0, 1]^2$) and 5-stage verification pipeline.
- **Core Claim:** Decoupling stylistic probability ($\hat{y}_{AI}$) from factual risk ($R_{fact}$); handling epistemic uncertainty.
- **Dense Spoken Lines:**
  > "Topic 3 formalizes evidence-grounded verification. We extract claims, retrieve evidence passages via stemmed BM25 from 36 curated Kazakh Wikipedia articles, and classify claims via a 3-way NLI cross-encoder into Supported, Refuted, or Not Enough Info (NEI). Every document maps to 2D coordinates $(\hat{y}_{AI}, R_{fact})$. NEI claims represent epistemic uncertainty ($R_{fact} = 0.50$), cleanly separating human rumor (Q2) from valid AI assistance (Q3) and malicious hallucination (Q4)."
- **Next Slide Bridge:** "Let us turn to empirical results in Chapter 4."

---

### Slide 11: Experimental Setup — Kazakh MAGE Protocol
- **Target Time:** 35 seconds `[7:10]`
- **Visual Cue:** 2x2 matrix table (Seen/Unseen Domains $\times$ Seen/Unseen Generators) and KDE density plots.
- **Core Claim:** Zero data leakage across 5-fold cross-validation, certified by 314 automated regression tests.
- **Dense Spoken Lines:**
  > "Our evaluation adapts a MAGE-style 2x2 matrix across two orthogonal axes: Domain Transfer (News vs. Kaspi reviews) and Generator Transfer (Sherkala-7B vs. held-out Qwen-2.5-7B and web-crawled AI). We enforce strict 5-fold document-family isolation with zero leakage. All algorithmic invariants are certified by our 314 automated tests."
- **Next Slide Bridge:** "Here are the Topic 1 results..."

---

### Slide 12: Topic 1 Results — Resolving the Blindspot (+42.18 pp Gain)
- **Target Time:** 45 seconds `[7:55]`
- **Visual Cue:** Quadrant table, 3 stat cards (+42.18 pp gain, 100% Q4 AUC, 99.85% Q1 AUC), and ROC curves.
- **Core Claim:** Resolves the out-of-domain collapse from 57.62% up to 99.80% (+42.18 pp).
- **Dense Spoken Lines:**
  > "On in-domain news (Q1), our model achieves 99.85% ROC-AUC (Macro-F1 99.24%). In out-of-domain Kaspi reviews (Q3), baseline KazRoBERTa drops to 57.62% AUC. Our Morpho-Gated detector maintains 99.80% AUC—an absolute improvement of **+42.18 percentage points**! On held-out Qwen text (Q4), we achieve 100.0% AUC with a human false-positive rate under 2.1%."
- **Next Slide Bridge:** "Evaluating generator robustness and long documents..."

---

### Slide 13: Topic 2 Results — Generator Robustness & Chunking Scalability
- **Target Time:** 40 seconds `[8:35]`
- **Visual Cue:** Figure 9 (100% tampering recall & VRAM curve under 1.4 GB up to 25,000 words).
- **Core Claim:** Generator gain (+21.55 pp OOD); 100% localization in long documents with sub-linear latency.
- **Dense Spoken Lines:**
  > "Under generator shift, our model gains +21.55 pp out-of-domain over pretrained baselines. On our 200-document stress test up to 25,000 words, Top-K pooling achieved **100% localized tampering recall**, pinpointing injected AI paragraphs with exact spans. Latency scales sub-linearly (4.1s for 25,000 words) while capping peak GPU memory under 1.4 GB."
- **Next Slide Bridge:** "Examining our factual verification results..."

---

### Slide 14: Topic 3 Results — Kazakh-FEVER Benchmark
- **Target Time:** 40 seconds `[9:15]`
- **Visual Cue:** NLI confusion matrix (diagonal 100%) and 3 stat cards (100% NLI F1, 91.67% Recall@3, 66.67% Joint FEVER).
- **Core Claim:** 100% NLI classification; honest disclosure that BM25 retrieval is the primary bottleneck (66.67% joint FEVER).
- **Dense Spoken Lines:**
  > "On our Kazakh-FEVER pilot benchmark (36 articles, 120 claims), our NLI cross-encoder achieves 100.0% Macro-F1, and stemmed BM25 attains 91.67% Recall@3. Under strict Joint FEVER—requiring exact evidence retrieval and correct label prediction—we achieve 66.67%. This transparently demonstrates that evidence retrieval is the primary operational bottleneck, which we plan to upgrade with dense retrieval in Paper 2."
- **Next Slide Bridge:** "Visualizing the 2D Trust Matrix..."

---

### Slide 15: Topic 3 Results — Four-Quadrant Trust Matrix Validation
- **Target Time:** 40 seconds `[9:55]`
- **Visual Cue:** Figure 11 (2D scatter plot showing green, orange, blue, and red clusters).
- **Core Claim:** Empirical validation of 4 quadrants, preventing truthful AI from being banned and catching human rumors.
- **Dense Spoken Lines:**
  > "Figure 11 visualizes our 2D risk space: Quadrant 1 (green) confirms authentic news ($y_{AI}=0.08, R_{fact}=0.04$). Quadrant 2 (orange) catches human rumors with low AI probability but high factual risk. Quadrant 3 (blue) recognizes accurate AI summaries as helpful assistance rather than flagging them. Quadrant 4 (red) triggers critical alerts on deceptive hallucinations."
- **Next Slide Bridge:** "To confirm which components drive these gains..."

---

### Slide 16: Component Ablation Studies
- **Target Time:** 40 seconds `[10:35]`
- **Visual Cue:** Figure 12 horizontal bar chart and ablation table.
- **Core Claim:** Removing FST causes -42.18 pp collapse; dynamic gating outperforms static concatenation by +11.35 pp.
- **Dense Spoken Lines:**
  > "Our ablations isolate each component's contribution: Removing the 83-rule FST causes an immediate 42.18 pp drop on colloquial text. Replacing dynamic gating with static concatenation degrades AUC by 11.35 pp, proving that adaptive weighting is essential. At the document level, Top-K pooling avoids a 23.75% omission penalty compared to naive mean pooling."
- **Next Slide Bridge:** "Turning to Chapter 5: System Implementation..."

---

### Slide 17: System Demo — Interactive 4-Tab Gradio Platform
- **Target Time:** 40 seconds `[11:15]`
- **Visual Cue:** Figure 13 (Gradio dashboard preview) and 4 feature cards.
- **Core Claim:** Production-ready academic prototype with sentence attribution heatmaps and morphological trees.
- **Dense Spoken Lines:**
  > "To make our research actionable, we engineered an interactive 4-tab Gradio platform: Tab 1 provides sentence attribution heatmaps and dynamic gate weights. Tab 2 visualizes word-level FST morphological derivation trees. Tab 3 allows browsing the MAGE benchmark results. Tab 4 implements the Kazakh-FEVER Trust Matrix with real-time claim verification."
- **Next Slide Bridge:** "Looking at our cloud packaging..."

---

### Slide 18: System Demo — Cloud Packaging & 314 Passing Tests
- **Target Time:** 40 seconds `[11:55]`
- **Visual Cue:** Figure 14 deployment pipeline and 3 stat cards: 314/314 Tests, <1.2s Cold Start, 3 Formats (.txt, .docx, .pdf).
- **Core Claim:** Containerized Hugging Face Spaces bundle verified by 314 passing automated regression tests.
- **Dense Spoken Lines:**
  > "Our system is packaged as a containerized Docker prototype for Hugging Face Spaces. It features sub-1.2 second cold start, defensive multi-format parsers for .txt, .docx, and .pdf documents with a 10MB memory ceiling, sub-85ms latency, and 314 automated regression tests passing with 100% reliability."
- **Next Slide Bridge:** "Entering Chapter 6: Conclusion..."

---

### Slide 19: Conclusion — Contributions & Writing Progress (~85%)
- **Target Time:** 45 seconds `[12:40]`
- **Visual Cue:** Left card (4 Academic Contributions); Right card (Dissertation Chapters 1–6 status totaling ~85%).
- **Core Claim:** 4 major contributions; core scientific milestones achieved; manuscript estimated at ~85% completion.
- **Dense Spoken Lines:**
  > "In conclusion, this thesis delivers four primary contributions: algorithmic morphological gated fusion (+42.18 pp), robust generator generalization and Top-K chunking, Central Asia's first Kazakh-FEVER benchmark and Trust Matrix, and a verified 4-tab prototype. Our dissertation manuscript is approximately 85% complete, with Chapters 1 through 4 fully written, and Chapters 5 and 6 nearing completion."
- **Next Slide Bridge:** "Reviewing our publication and today's agenda..."

---

### Slide 20: Current Progress — Springer LNCS & Meeting Agenda
- **Target Time:** 40 seconds `[13:20]`
- **Visual Cue:** 4 milestone cards: Springer LNCS Accepted, Live Demo Prepared, September Milestones, Next Steps.
- **Core Claim:** Paper 1 accepted to AIST 2026 (Springer LNCS); interactive prototype ready for live demonstration today.
- **Dense Spoken Lines:**
  > "Our first research paper has been formally accepted for publication in Springer LNCS (AIST 2026). Our live Gradio platform is ready for demonstration today. With 10,000+ benchmark samples curated and 314 tests passing, we are on schedule for our final defense."
- **Next Slide Bridge:** "I welcome Professor Guo's guidance on three strategic questions..."

---

### Slide 21: Discussion — Strategic Consultation for Prof. Guo
- **Target Time:** 50 seconds `[14:10]`
- **Visual Cue:** 3 Strategic Guidance Cards (Adversarial Rewriting, Dense Retrieval Scaling, Paper 2 Venue).
- **Core Claim:** Seeking advisor guidance on Paper 2 experimental scope and publication timeline.
- **Dense Spoken Lines:**
  > "To maximize our research impact, I respectfully request Professor Guo's advice on three strategic points:
  > 1. For Paper 2, which adversarial rewriting attacks should we prioritize—neural paraphrasers, back-translation, or morphological perturbation?
  > 2. For Kazakh-FEVER, should we scale from 36 articles to a broader dense retrieval corpus using multilingual E5 or BGE?
  > 3. What target venue does Professor Guo recommend for Paper 2 (e.g., ACL, EMNLP, or an IEEE/ACM journal)?"
- **Next Slide Bridge:** "Finally, regarding earlier committee feedback..."

---

### Slide 22: Committee Comments & Academic References
- **Target Time:** 35 seconds `[14:45]`
- **Visual Cue:** Committee comments table showing specific thesis chapters (Chapters 3, 4, and 5).
- **Core Claim:** Every committee review comment has been resolved and integrated into the formal dissertation chapters.
- **Dense Spoken Lines:**
  > "As shown in this table, all previous committee comments have been addressed: Reviewer 1's questions on out-of-domain colloquial text and long documents are resolved in Chapters 3 and 4; Reviewer 2's inquiries regarding factual verification and threshold sensitivity are addressed in Chapter 5 with our 2D Trust Matrix."
- **Next Slide Bridge:** "This concludes my presentation."

---

### Slide 23: Closing Slide & Defense Acknowledgments
- **Target Time:** 15 seconds `[15:00]`
- **Visual Cue:** Institutional closing slide with candidate and advisor credentials.
- **Core Claim:** Expressing sincere gratitude to Professor Guo and opening the floor for discussion.
- **Dense Spoken Lines:**
  > "Thank you very much, Professor Guo and honorable committee members, for your guidance and time. I welcome your questions and look forward to our live demonstration."

---

## Defense Q&A Rapid-Response Cheat-Sheet

If Professor Guo or committee members ask these specific technical questions, use these 1–2 sentence direct answers:

1. **"Why did you use dynamic gating instead of simply concatenating the features?"**  
   *Answer:* "Concatenation forces a static weighting across all domains. Dynamic gating computes a learned weight $g \in [0, 1]^d$ per dimension, allowing the model to adaptively rely on semantics ($g \approx 0.68$) for formal news and switch to grammar on colloquial slang, yielding an 11.35 pp gain over concatenation."

2. **"Why 83 rules in your FST and not more or fewer?"**  
   *Answer:* "The 83 rules represent the complete inflectional paradigm of standard Kazakh nominal and verbal morphotactics, validated across 5,000 dictionary headwords to achieve 99.2% inflectional coverage without over-generating invalid affixes."

3. **"Why Top-K worst-chunk pooling instead of average pooling for long documents?"**  
   *Answer:* "In hybrid documents where an author injects just one or two AI paragraphs into a 20-page paper, average pooling washes out the synthetic signal. Taking the worst 25% chunks ($K = \max(1, \lfloor 0.25 N \rfloor)$) guarantees 100% tampering detection."

4. **"Why is your Strict Joint FEVER score 66.67% while NLI is 100%?"**  
   *Answer:* "Strict Joint FEVER requires both retrieving the exact gold sentence and predicting the correct NLI label. Our NLI classification is accurate, but sparse BM25 retrieval achieved 91.67% recall on evidence retrieval, identifying retrieval as the primary bottleneck to be upgraded with dense embeddings in Paper 2."

5. **"Why do you need a 2D Trust Matrix instead of a single composite risk score?"**  
   *Answer:* "A single scalar score conflates writing style with factual truth. A truthful AI summary would receive a high danger score, while human-written fake news would pass. The 2D coordinates $(\hat{y}_{AI}, R_{fact})$ keep origin and veracity orthogonal for proper governance."

6. **"What is the remaining 15% of your thesis manuscript?"**  
   *Answer:* "The remaining 15% consists of finalizing Chapter 6's documentation on adversarial rewriting benchmark experiments (paraphrasing and synonym perturbation) and scaling Kazakh-FEVER retrieval for Paper 2."
