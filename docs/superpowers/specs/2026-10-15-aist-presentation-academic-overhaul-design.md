# AIST 2026 Presentation Minimalist Modern Academic Overhaul Design

**Date**: 2026-10-15  
**Topic**: AIST 2026 Conference Oral Presentation ("De-AI" Academic Redesign)  
**Authors**: Daulet Anekesh and Irina Ualiyeva (*Al-Farabi Kazakh National University*)  
**Status**: Approved by User  
**Target File**: `presentation/aist2026_presentation.tex` -> `presentation/aist2026_presentation.pdf`  
**Desktop Deliverable**: `C:\Users\Roza\Desktop\AIST2026_Presentation_Daulet_Anekesh.pdf`

---

## 1. Problem Statement & Motivation

The current Beamer presentation exhibits telltale signs of AI-generated content:
1. **Card Overload**: Every slide wraps text and graphics in generic `tcolorbox` cards with heavy left accent borders (`borderline west={3.5pt}`) and nested pill badges, resembling a SaaS dashboard rather than a peer-reviewed scientific talk.
2. **Mechanical Bullet Syntax**: Rigid adherence to `\item \textbf{Keyword}: Explanation` formula across all slides.
3. **Redundant Takeaway Mini-Boxes**: Bottom callout boxes labeled "Headline Finding:" or "Impact:" nested underneath graphics.
4. **Squeezed 2cm Visual Plots**: Scientific charts (dotplot, token inflation, bootstrap CIs, OOD bars) are squeezed into tiny thumbnails beside cards.

The objective of this overhaul is to transform the presentation into a **Minimalist Modern ACL/NeurIPS academic aesthetic**:
- Clean white canvas with generous natural whitespace.
- Complete removal of container cards, side border stripes, and nested callout boxes.
- Large, prominent scientific figures (expanded from 2.0cm to 3.6–4.2cm).
- Scientific assertion slide titles that state key findings directly.
- Clean typographic contrast and hero numbers instead of UI pill badges.

---

## 2. Visual & Structural Specifications

### 2.1 Theme & Master Canvas
- **Aspect Ratio**: 16:9 widescreen (`aspectratio=169`, `160mm x 90mm`).
- **Background**: Pure white (`#FFFFFF`) on all 17 slides.
- **Slide Header**: Clean white background with dark navy bold title (`DarkNavy`, `#0F2042`, 14pt) directly on canvas, underlined by a subtle 0.8pt divider rule in light gray (`BorderGray`, `#D1D5DB`).
- **Slide Footline**: Subtle two-column footer in slate gray (`#6B7280`, 6.5pt) displaying `AIST 2026 | NLP Track`, author names, and frame numbers `x / 17`.
- **Zero Heavy Banners**: The dark navy top block banner is removed entirely from body slides.

### 2.2 Color Palette
- **Text Primary**: Charcoal / Off-black (`#1A1A1A`) for high contrast readability.
- **Headings & Table Headers**: Deep Academic Navy (`#0F2042`).
- **Linguistic / Scientific Accent**: Kazakh Steppe Cyan (`#008299`) for morphemes and key callouts.
- **Metric Contrast**: Muted Red (`#C53030`) for baseline errors (`58 FPs`); Muted Green (`#228B22`) for our model's improvements (`30 FPs`).
- **No Border Bars**: Completely eliminate left border lines (`borderline west`) and multi-color card framing.

### 2.3 Typography & Information Hierarchy
- **Assertion-Driven Titles**: Titles state empirical takeaways directly (e.g. *"KazRoBERTa and Hybrid FST Cut Short-Text False Positives by 43.2%"*).
- **Natural Phrasing**: Varied sentence structures, concise analytical points, and parallel lists without repetitive bold prefix keywords.
- **Hero Figures**: Sized at `3.6cm – 4.2cm` height, centered or prominent, accompanied by direct narrative annotations.

---

## 3. Slide-by-Slide Content Architecture (17 Frames)

### Slide 1: Title & Session Information
- **Title**: *Detecting AI-Generated User Reviews in Kazakh: A Study on BERT Model Performance and False Positive Reduction*
- **Layout**: Clean academic cover page on white canvas with Deep Blue title text, formal KazNU author affiliations, Springer LNCS track details, and conference session metadata (Red Hall C3, 12:45–13:10 Astana Time).

### Slide 2: Motivation & Background
- **Title**: *Generative AI in Low-Resource Languages: The Unmonitored Safety Gap*
- **Layout**: Two clean whitespace columns:
  - Left: Proliferation of Kazakh LLMs (Sherkala-7B, Qwen, LLaMA) in banking & consumer apps (Kaspi).
  - Right: Critical gap in AI text detection (95%+ English-centric; lack of agglutinative morphological modeling; severe false positive hazard).
- **Anchor**: Core objective rule stating the first empirical benchmark and morphology-grounded false-positive defense.

### Slide 3: Linguistic Bottleneck
- **Title**: *Kazakh Agglutinative Morphology Induces Severe Subword Token Fragmentation*
- **Layout**:
  - Left: Kazakh morphological hierarchy (productive clausal compounding, affix stacking, exponential lexical surface: *жаз-ба-лар-ыңыз-дан*).
  - Right: Typographical tokenization comparison. Monospace display contrasting raw subwords (*WordPiece* slicing *«Қазақстандықтардың»* into 7 noisy fragments) against *Hybrid FST* preserving the root stem `[Қазақ]` + inflectional morphemes.

### Slide 4: Operational Hazard
- **Title**: *The Short-Text Trap: Falsely Accusing Human Authors in Real-World Contexts*
- **Layout**:
  - Left: Asymmetric penalties of false alarms (merchant account bans on Kaspi, student academic integrity penalties, erosion of institutional trust).
  - Right: Baseline false alarm scale: **mBERT: 58 FPs** | **XLM-R: 67 FPs** on short reviews ($\le 60$ ch). Over 45% of mobile app reviews are $<10$ words, exhibiting extreme vulnerability to subword noise.

### Slide 5: Research Questions & Scope
- **Title**: *Research Scope: Three Core Empirical Questions*
- **Layout**: Three clean columns floating on whitespace:
  - **RQ1: Pretraining Domain**: Does conversational Kazakh pretraining outperform massively multilingual baselines?
  - **RQ2: Morphological Normalization**: Can rule-based FST universally normalize token inflation across diverse tokenizers?
  - **RQ3: False Positive Defense**: Does morphological grounding eliminate short-text false alarms under domain shift?

### Slide 6: Benchmark Dataset Construction
- **Title**: *KazAI-Detect: 12,848 Matched Human and Synthetic Reviews*
- **Layout**:
  - Left: Corpus construction details (6,424 authentic reviews from KazSAnDRA across 4 sectors; 6,424 synthetic reviews generated by Sherkala-7B with matched length and informal persona prompt controls). Total: 12,848 balanced samples.
  - Right: Stratified evaluation splits table (Training: 8,848; Held-out Test: 4,000 [1,842 short, 2,158 long]; OOD Set: 1,317 news/wiki samples).

### Slide 7: Linguistic Reality in Production Reviews
- **Title**: *Production Consumer Reviews: Three Interlocking Linguistic Challenges*
- **Layout**:
  - Top: Diagnostic authentic Kaspi review with English translation.
  - Bottom (3 Columns):
    1. *Code-Switching*: Russian loanword stems taking Kazakh suffixes (*недобросовестно*, *доставка-сы*, *каспи-ден*).
    2. *Affix Stacking*: Deep agglutinative chains (*аш-қыз-ған-дар-ың*: open + causative + past part. + plural + possessive).
    3. *Colloquial Idioms*: Informal spoken idioms (*тығып қойыпсыңдар*, *қораға кіргізіп*).

### Slide 8: Proposed Method: Hybrid FST Analyzer
- **Title**: *Hybrid FST Morphological Analyzer: Decoupling Stems from Affixes*
- **Layout**:
  - Left: Three-tier linguistic rule architecture (Tier 1: Nominal inflectional chains; Tier 2: Verbal inflections & participles; Tier 3: Code-switched Russian stems).
  - Right: Step-by-step pre-segmentation examples showing native reviews and code-switched reviews with clear morpheme boundary decomposition.

### Slide 9: Experimental Setup & Evaluated Models
- **Title**: *Experimental Regimen: Evaluated Models and Training Configuration*
- **Layout**:
  - Left: Evaluated transformer models (mBERT-base, XLM-RoBERTa-base, KazBERT, KazRoBERTa-conversational 24.8M texts).
  - Right: Regimen comparison (Pure unsegmented vs. Hybrid FST pre-segmented) and fine-tuning hyperparameters ($lr=2\times 10^{-5}$, AdamW, 3 epochs, batch size 16).

### Slide 10: Main Benchmark Results
- **Title**: *KazRoBERTa and Hybrid FST Cut Short-Text False Positives by 43.2%*
- **Layout**:
  - Left (45%): Publication-grade results table ($N=4,000$ test set) highlighting KazRoBERTa Pure (96.10% Acc) vs. FST (96.32% Acc, short FPs cut from 53 down to 30).
  - Right (55%): **Prominent 3.8cm Dotplot Chart** (`chart_combined_dotplot.png`) centered and cleanly displayed without container cards.
  - Caption: Concise scientific takeaway highlighting the 43.2% drop ($p < 10^{-5}$).

### Slide 11: Mathematical Mechanism
- **Title**: *Hybrid FST Enforces a Universal Morphemic Convergence Bound of 2.202*
- **Layout**:
  - Left (45%): Token Inflation Ratio ($T/W$) table showing mBERT exploding to 3.476 tokens/word, cut by -36.7% down to 2.202.
  - Right (55%): **Prominent 3.8cm Token Inflation Chart** (`chart_token_inflation_ratio.png`) proving universal convergence across BPE (52K) and WordPiece (119K).

### Slide 12: Statistical Significance
- **Title**: *Deterministic False Positive Reduction Confirmed by Bootstrap Resampling*
- **Layout**:
  - Left (45%): Statistical significance proofs:
    - Headline contrast: **53 $\to$ 30 False Positives (-43.21%)**
    - McNemar's paired test: $\chi^2 = 21.04$, $p = 4 \times 10^{-6}$ ($p < 10^{-5}$).
    - Bootstrap 1,000 iterations: Pure CI $[39.0, 68.0]$ vs. FST CI $[20.0, 41.0]$ with zero overlap.
  - Right (55%): **Prominent 3.8cm Bootstrap Confidence Interval Plot** (`chart_bootstrap_confidence_intervals.png`).

### Slide 13: Qualitative Interpretability
- **Title**: *Integrated Gradients: Morphological Grounding Restores Human Attribution*
- **Layout**:
  - Top: Diagnostic review *«жылдам аударымдары үшін рахмет»*.
  - Side-by-side attribution tables:
    - Pure Model (False Alarm: 84.6% AI) — subwords *ауда-ры-мда-ри* receive -1.55 AI penalty.
    - FST Model (Correct: 91.7% Human) — root stem *аудар* receives +0.48 positive human evidence.

### Slide 14: Out-of-Distribution Generalization
- **Title**: *Morphological Defense Holds Across Formal News and Wikipedia Domains*
- **Layout**:
  - Left (45%): Multi-genre OOD table ($N=1,317$). Consumer reviews (-43.8% FPR), News (-42.9% FPR), Wikipedia (-50.0% FPR).
  - Right (55%): **Prominent 3.8cm OOD Bar Chart** (`chart_ood_domain_generalization.png`).

### Slide 15: Deployment Strategy
- **Title**: *Practical Deployment: High-Throughput Screening vs. High-Precision Auditing*
- **Layout**: Two clean comparative columns:
  - **Pipeline A (High-Throughput Screening)**: Pure KazRoBERTa ($<15$ ms, zero FST dependency, high recall for web indexing).
  - **Pipeline B (High-Precision Auditing)**: Hybrid FST KazRoBERTa (-43.2% FP protection, explainable morpheme attribution for academic integrity and disputes).

### Slide 16: Conclusions & Future Directions
- **Title**: *Core Contributions and Research Roadmap Towards Fact Verification*
- **Layout**:
  - Left: 4 Core contributions (First benchmark, monolingual superiority, morphemic bound, FP reduction).
  - Right: Future roadmap (Kazakh-FEVER 3K benchmark, joint 2D Trust Matrix tracking veracity and origin).

### Slide 17: Open Science & Questions
- **Title**: *Open-Source Reproducibility and Discussion*
- **Layout**:
  - Left: Open-source resources (Hugging Face dataset, GitHub evaluation codebase, pretrained model weights).
  - Right: Prominent centered QR code with clean "Scan for Code & Data" label.
  - Bottom: Closing discussion invitation and KazNU contact emails.

---

## 4. Verification & Invariance Guardrails

1. **Exact 17 Pages Budget**: Strictly 17 frames in Beamer 16:9 (`aspectratio=169`).
2. **Zero `Overfull \vbox` / `Overfull \hbox`**: Perfect vertical clearance on every frame.
3. **Automated Unit Tests**: All 5 tests in `tests/test_aist_presentation.py` must pass cleanly.
4. **Desktop PDF Synchronization**: Final PDF compiled and copied to `C:\Users\Roza\Desktop\AIST2026_Presentation_Daulet_Anekesh.pdf`.
5. **Strict Invariance**: `aist2026/paper.tex` and `papers/kazakh_fever_conference/` remain untouched (0 diff lines).
6. **Zero Emojis**: Strictly 0 decorative emojis across all code, presentation, and notes.
