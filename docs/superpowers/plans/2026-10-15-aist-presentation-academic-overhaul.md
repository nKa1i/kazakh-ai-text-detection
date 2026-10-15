# Minimalist Modern Academic Overhaul for AIST 2026 Presentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overhaul the 17-slide AIST 2026 conference presentation (`presentation/aist2026_presentation.tex`) from an "AI SaaS dashboard" aesthetic (heavy colored cards, pill badges, nested takeaway mini-boxes) into a clean, minimalist modern ACL/NeurIPS academic presentation with pure white canvas, prominent 3.8cm hero scientific plots, assertion-driven headers, and natural whitespace.

**Architecture:** 
1. Re-architect Beamer preamble: eliminate all `tcolorbox` container cards (`accentcard`, `greencard`, `alertcard`), remove the dark navy header block from body slides, and establish a clean white top header with dark navy title text and a subtle 0.8pt divider rule.
2. Structure slides using native Beamer columns (`\begin{columns}[T]`) with direct typography, bold section subtitles, and generous whitespace.
3. Enlarge scientific plots (dotplot, token inflation, bootstrap CIs, OOD bars) from 2.0cm to **3.8cm height** as the visual anchors of empirical slides.
4. Replace rigid formulaic bullet points with natural academic presentation points, and replace pill badges with high-contrast hero metrics.

**Tech Stack:** XeLaTeX, Beamer (16:9 widescreen `aspectratio=169`), TikZ, `booktabs`, `pypdf`, Python 3 `unittest`.

## Global Constraints
- Under no circumstances shall `aist2026/paper.tex` or `papers/kazakh_fever_conference/` be touched or modified (strictly 0 diff lines).
- Strictly zero decorative emojis across all code, presentations, datasets, and manuscripts.
- Total slide count must remain strictly **17 frames / 17 PDF pages** in 16:9 widescreen format.
- Zero `Overfull \vbox` and zero internal `Overfull \hbox` across all 17 slides.
- Final deliverable PDF must be synchronized to `C:\Users\Roza\Desktop\AIST2026_Presentation_Daulet_Anekesh.pdf`.
- All automated unit tests in `tests/test_aist_presentation.py` must pass (5/5).

---

## Proposed Changes & Task Breakdown

### Task 1: Overhaul Beamer Style & Canvas Architecture

**Files:**
- Modify: `presentation/aist2026_presentation.tex:1-220`
- Test: `tests/test_aist_presentation.py`

**Interfaces:**
- Consumes: Existing font definitions (`Arial`, `Courier New`), color definitions (`DarkNavy`, `TechCyan`, `AccentGreen`, `AlertRed`, `BorderGray`, `SlateText`).
- Produces: Clean frametitle template, understated footline template, removal of card definitions, and helper macros for clean horizontal rules and metric hero callouts.

- [ ] **Step 1: Update frametitle and footline templates in `presentation/aist2026_presentation.tex`**
  - Remove the dark navy solid box header: replace with clean white background, `\color{DarkNavy}\textbf{\insertframetitle}` (14pt), followed by `\vspace{2pt}\hrule height 0.6pt color BorderGray\vspace{4pt}`.
  - Simplify footline to subtle gray text: `AIST 2026 | NLP Track \hfill Daulet Anekesh & Irina Ualiyeva | Al-Farabi KazNU \hfill \insertframenumber{} / \inserttotalframenumber`.
  - Delete `accentcard`, `greencard`, `alertcard`, `mycard`, `statpill` definitions.
  - Define minimal clean helpers: `\newcommand{\heronumber}[3]{...}` for prominent empirical metrics and `\newcommand{\subhead}[1]{{\normalsize\textbf{\color{DarkNavy}#1}\par\vspace{3pt}}}`.

- [ ] **Step 2: Compile minimal test frame with XeLaTeX**
  - Command: `xelatex -interaction=nonstopmode -halt-on-error aist2026_presentation.tex` in `presentation/`.
  - Verify Beamer compiles without missing package or macro errors.

- [ ] **Step 3: Commit Beamer architecture overhaul**
  - Command: `git add presentation/aist2026_presentation.tex && git commit -m "style(presentation): re-architect canvas to minimalist academic style with clean header and whitespace"`

---

### Task 2: Refactor Slides 1 to 5 (Problem, Typology & Operational Hazard)

**Files:**
- Modify: `presentation/aist2026_presentation.tex` (Slides 1–5)
- Test: `tests/test_aist_presentation.py`

- [ ] **Step 1: Refactor Slide 1 (Title Slide)**
  - Clean white academic title page with Deep Blue bold paper title (`\Large\bfseries\color{DarkNavy}`), author affiliations (`Al-Farabi Kazakh National University`), and Springer LNCS track & session details (Block C3 Red Hall, 12:45–13:10 Astana Time).

- [ ] **Step 2: Refactor Slide 2 (Motivation)**
  - Title: *Generative AI in Low-Resource Languages: The Unmonitored Safety Gap*
  - Two clean text columns with whitespace: Left column details the rapid rise of Kazakh LLMs (Sherkala-7B, Qwen) in banking & consumer apps; Right column details the detection gap (95%+ English-centric, high false positive hazard).
  - Bottom rule with single clear objective statement.

- [ ] **Step 3: Refactor Slide 3 (Linguistic Bottleneck)**
  - Title: *Kazakh Agglutinative Morphology Induces Severe Subword Fragmentation*
  - Left column: Morphological typology (affix stacking, clausal compounding: *жаз-ба-лар-ыңыз-дан*).
  - Right column: Clean monospace comparison box showing WordPiece slicing *«Қазақстандықтардың»* into 7 noisy subwords vs. FST preserving root stem `[Қазақ]` + inflectional morphemes.

- [ ] **Step 4: Refactor Slide 4 (Operational Hazard)**
  - Title: *The Short-Text Trap: Falsely Accusing Human Authors in Real-World Contexts*
  - Left: Asymmetric penalties of false alarms (Kaspi merchant account bans, student academic integrity penalties, erosion of trust).
  - Right: Hero metric display: **mBERT: 58 FPs** | **XLM-R: 67 FPs** on short reviews ($\le 60$ ch). Over 45% of mobile reviews are $<10$ words.

- [ ] **Step 5: Refactor Slide 5 (Research Scope)**
  - Title: *Research Scope: Three Core Empirical Questions*
  - Three clean columns with subtitle rules: RQ1 (Pretraining Domain), RQ2 (Morphemic Limit), RQ3 (False Positive Mitigation) floating on natural whitespace.

- [ ] **Step 6: Compile and verify Slides 1–5**
  - Command: `xelatex -interaction=nonstopmode aist2026_presentation.tex`
  - Render slides 1–5 to PNG: `pdftoppm -png -r 150 -f 1 -l 5 aist2026_presentation.pdf slide_test_p1_5`
  - Verify zero border spills and clean typography. Clean up test PNGs.

- [ ] **Step 7: Commit Slides 1–5**
  - Command: `git add presentation/aist2026_presentation.tex && git commit -m "style(presentation): refactor slides 1-5 to assertion-driven academic layout"`

---

### Task 3: Refactor Slides 6 to 9 (Benchmark Corpus, Linguistic Reality, Hybrid FST, Setup)

**Files:**
- Modify: `presentation/aist2026_presentation.tex` (Slides 6–9)
- Test: `tests/test_aist_presentation.py`

- [ ] **Step 1: Refactor Slide 6 (Dataset Construction)**
  - Title: *KazAI-Detect: 12,848 Matched Human and Synthetic Reviews*
  - Left: Direct bullet breakdown of authentic (KazSAnDRA) vs. synthetic (Sherkala-7B) corpus construction.
  - Right: Clean 3-line stratified evaluation split table (8,848 train, 4,000 test, 1,317 OOD).

- [ ] **Step 2: Refactor Slide 7 (Linguistic Reality)**
  - Title: *Production Consumer Reviews: Three Interlocking Linguistic Challenges*
  - Top: Authentic Kaspi review case study with clean English translation.
  - Bottom: Three distinct columns: 1. Code-Switching, 2. Affix Stacking (*аш-қыз-ған-дар-ың*), 3. Colloquial Idioms.

- [ ] **Step 3: Refactor Slide 8 (Hybrid FST Analyzer)**
  - Title: *Hybrid FST Morphological Analyzer: Decoupling Stems from Affixes*
  - Left: Three-tier rule engine (Nominal chains, Verbal inflections, Code-switched Russian loanword stems).
  - Right: Step-by-step morphemic decomposition with colored affixes and proper word spacing.

- [ ] **Step 4: Refactor Slide 9 (Experimental Framework)**
  - Title: *Experimental Regimen: Evaluated Models and Training Configuration*
  - Left: Evaluated transformer models (mBERT, XLM-R, KazBERT, KazRoBERTa).
  - Right: Pure vs. Hybrid FST regimen + fine-tuning hyperparameters ($lr=2\times 10^{-5}$, AdamW, 3 epochs, batch size 16).

- [ ] **Step 5: Compile and verify Slides 6–9**
  - Command: `xelatex -interaction=nonstopmode aist2026_presentation.tex`
  - Render slides 6–9: `pdftoppm -png -r 150 -f 6 -l 9 aist2026_presentation.pdf slide_test_p6_9`
  - Inspect PNGs. Clean up test PNGs.

- [ ] **Step 6: Commit Slides 6–9**
  - Command: `git add presentation/aist2026_presentation.tex && git commit -m "style(presentation): refactor slides 6-9 to clean academic methodology layout"`

---

### Task 4: Refactor Slides 10 to 14 (Empirical Results, Morphemic Bound, Interpretability, OOD)

**Files:**
- Modify: `presentation/aist2026_presentation.tex` (Slides 10–14)
- Test: `tests/test_aist_presentation.py`

- [ ] **Step 1: Refactor Slide 10 (Main Benchmark Results)**
  - Title: *KazRoBERTa and Hybrid FST Cut Short-Text False Positives by 43.2%*
  - Left (44%): Compact 3-line results table ($N=4,000$).
  - Right (56%): **Hero Dotplot Chart** (`chart_combined_dotplot.png`) enlarged to **3.8cm height**, centered and prominently displayed.
  - Bottom: Single-sentence scientific takeaway.

- [ ] **Step 2: Refactor Slide 11 (Mathematical Mechanism)**
  - Title: *Hybrid FST Enforces a Universal Morphemic Convergence Bound of 2.202*
  - Left (44%): Token Inflation Ratio ($T/W$) table showing mBERT exploding to 3.476, bounded to 2.202 ($p < 10^{-6}$).
  - Right (56%): **Hero Token Inflation Chart** (`chart_token_inflation_ratio.png`) enlarged to **3.8cm height**.

- [ ] **Step 3: Refactor Slide 12 (False Positive Breakthrough)**
  - Title: *Deterministic False Positive Reduction Confirmed by Bootstrap Resampling*
  - Left (44%): Hero numbers: **53 $\to$ 30 False Positives (-43.21%)**, McNemar's paired test ($\chi^2 = 21.04, p = 4 \times 10^{-6}$), and Bootstrap CI non-overlap.
  - Right (56%): **Hero Bootstrap CI Plot** (`chart_bootstrap_confidence_intervals.png`) enlarged to **3.8cm height**.

- [ ] **Step 4: Refactor Slide 13 (Qualitative Interpretability)**
  - Title: *Integrated Gradients: Morphological Grounding Restores Human Attribution*
  - Top: Diagnostic review *«жылдам аударымдары үшін рахмет»*.
  - Side-by-side attribution tables: Pure Model (-1.55 AI penalty on subwords) vs. FST Model (+0.48 human proof on stem *аудар*).

- [ ] **Step 5: Refactor Slide 14 (OOD Generalization)**
  - Title: *Morphological Defense Holds Across Formal News and Wikipedia Domains*
  - Left (44%): Multi-genre OOD table ($N=1,317$).
  - Right (56%): **Hero OOD Bar Chart** (`chart_ood_domain_generalization.png`) enlarged to **3.8cm height**.

- [ ] **Step 6: Compile and verify Slides 10–14**
  - Command: `xelatex -interaction=nonstopmode aist2026_presentation.tex`
  - Render slides 10–14: `pdftoppm -png -r 150 -f 10 -l 14 aist2026_presentation.pdf slide_test_p10_14`
  - Inspect PNGs. Clean up test PNGs.

- [ ] **Step 7: Commit Slides 10–14**
  - Command: `git add presentation/aist2026_presentation.tex && git commit -m "style(presentation): refactor empirical slides 10-14 with prominent 3.8cm hero plots"`

---

### Task 5: Refactor Slides 15 to 17 (Deployment, Conclusions, Open Science & Q&A)

**Files:**
- Modify: `presentation/aist2026_presentation.tex` (Slides 15–17)
- Test: `tests/test_aist_presentation.py`

- [ ] **Step 1: Refactor Slide 15 (Practical Deployment)**
  - Title: *Practical Deployment: High-Throughput Screening vs. High-Precision Auditing*
  - Two clean comparative columns: Pipeline A (Pure KazRoBERTa, $<15$ ms) vs. Pipeline B (Hybrid FST KazRoBERTa, -43.2% FP protection).

- [ ] **Step 2: Refactor Slide 16 (Conclusions & Roadmap)**
  - Title: *Core Contributions and Research Roadmap Towards Fact Verification*
  - Left: 4 Core contributions (Benchmark, monolingual advantage, morphemic bound, FP reduction).
  - Right: Future roadmap (Kazakh-FEVER 3K benchmark, continuous 2D Trust Matrix).

- [ ] **Step 3: Refactor Slide 17 (Open Science & Questions)**
  - Title: *Open-Source Reproducibility and Discussion*
  - Left: Open-source artifact URLs (Hugging Face dataset, GitHub codebase, pretrained model weights).
  - Right: Prominent centered QR code with clean "Scan for Code & Data" label.
  - Centered bottom: Thank you and Q&A prompt.

- [ ] **Step 4: Compile and verify Slides 15–17**
  - Command: `xelatex -interaction=nonstopmode aist2026_presentation.tex`
  - Render slides 15–17: `pdftoppm -png -r 150 -f 15 -l 17 aist2026_presentation.pdf slide_test_p15_17`
  - Inspect PNGs. Clean up test PNGs.

- [ ] **Step 5: Commit Slides 15–17**
  - Command: `git add presentation/aist2026_presentation.tex && git commit -m "style(presentation): refactor deployment, conclusions, and open science slides 15-17"`

---

### Task 6: Full Verification, Desktop Synchronization & Delivery

**Files:**
- Modify: `tests/test_aist_presentation.py`
- Output: `C:\Users\Roza\Desktop\AIST2026_Presentation_Daulet_Anekesh.pdf`

- [ ] **Step 1: Update unit test assertions in `tests/test_aist_presentation.py`**
  - Assert that all 17 frames compile without `tcolorbox` cards.
  - Assert that hero plot graphics and empirical anchors are present.
  - Assert that exactly 17 pages are generated.

- [ ] **Step 2: Run full automated test suite**
  - Command: `python -m unittest tests/test_aist_presentation.py -v`
  - Expected: 5/5 tests PASS.

- [ ] **Step 3: Render all 17 slides and visually inspect each frame**
  - Command: `pdftoppm -png -r 150 aist2026_presentation.pdf slide_final_audit`
  - Visually verify Slides 1 to 17 using `view_file` to confirm zero card borders, zero text spills, prominent 3.8cm plots, and clean academic whitespace.
  - Clean up audit PNGs.

- [ ] **Step 4: Synchronize to Desktop**
  - Command: `Copy-Item -Path "presentation\aist2026_presentation.pdf" -Destination "C:\Users\Roza\Desktop\AIST2026_Presentation_Daulet_Anekesh.pdf" -Force`

- [ ] **Step 5: Invariance Check**
  - Command: `git diff origin/main..HEAD -- aist2026/paper.tex papers/kazakh_fever_conference/`
  - Expected: strictly 0 diff lines.

- [ ] **Step 6: Commit and Push**
  - Command: `git add tests/test_aist_presentation.py presentation/aist2026_presentation.pdf && git commit -m "test(presentation): verify 17-slide minimalist academic overhaul and synchronize desktop deliverable"`
  - Command: `git push origin feat/aist2026-conference-presentation`
