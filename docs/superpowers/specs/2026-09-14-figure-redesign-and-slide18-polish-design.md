# Design Specification: Publication Figure Redesigns & Slide 18 Polish

**Author:** Daulet (大雷) / Google DeepMind Pair Programming  
**Date:** September 14, 2026  
**Target File:** `AnekeshD_Progress.pptx` & `Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx`  
**Status:** Approved by User (Brainstorming Complete)

---

## 1. Context & Motivation

During review of the thesis presentation deck (`AnekeshD_Progress.pptx`), the following visual and content issues were identified:
1. **Confusing & Duplicate Visuals:**
   - **Figure 3 (Slide 7):** Stretched across the full slide with 15 nested white text boxes and tiny bullet points, duplicating text on other slides and overwhelming viewers.
   - **Figure 4 (Slide 8):** Consisted of 5 plain text boxes containing raw mathematical formulas, duplicating the adjacent cards word-for-word rather than providing a true visual neural architecture.
   - **Figure 13 (Slide 17):** Depicted 4 vertical bullet columns that exactly duplicated the 4 slide cards, rather than looking like an authentic Gradio system interface.
   - **Figure 14 (Slide 18):** Cramped into 4 narrow columns and included internal development jargon (`Zero-emoji AST scanner`).
2. **Inappropriate Slide Metric:**
   - On Slide 18, Stat Card 3 displayed `"100% Zero Emoji Design / Strict typography compliant with academic standards"`. This was an internal agent development rule that is inappropriate and confusing to present to Professor Guo and the thesis committee.
3. **Aspect Ratio Mismatch:**
   - Figures were generated with 16:9 widescreen dimensions (`12.0 x 5.8` or `12.0 x 6.2` inches, ~2:1 aspect ratio). When placed into slide columns measuring approximately `5.85"` wide by `4.35"` high (an aspect ratio of ~1.34:1, or ~4:3), the figures suffered from severe horizontal letterboxing, small unreadable typography, and wasted vertical space.

---

## 2. Core Architectural & Visual Decisions

### 2.1 Unified Squarish Aspect Ratio Strategy (~4:3)
All redesigned figures will be generated using a **squarish aspect ratio**:
- **Dimensions:** `figsize=(7.5, 5.8)` to `(7.8, 6.0)` (aspect ratio ~1.28:1 to 1.33:1).
- **Resolution:** 300 DPI (`dpi=300, bbox_inches='tight'`).
- **Typography:** Scaled up by 25-40% compared to previous wide figures, ensuring crisp readability when viewed in PowerPoint or exported to 1080p slide images.
- **Color Palette:** Preserved institutional palette:
  - Institutional Navy (`#1E3A8A`)
  - Innovation Teal (`#0D9488`)
  - Accent Amber (`#D97706`)
  - Success Green (`#16A34A`)
  - Neutral Slate (`#0F172A`)
  - Alert Coral (`#DC2626`)
  - Card Background (`#F8FAFC`, `#EFF6FF`, `#F0FDFA`)

---

### 2.2 Slide 7 & Figure 3: Streamlined 4-Stage Methodology Pipeline
- **Slide Layout:** Converted from full-width to **Split Two-Column Layout** matching Slides 8, 17, and 18:
  - **Left Column (`left=0.60"`, `top=2.05"`, `width=5.85"`, `height=4.35"`):**
    - Embedded squarish **Figure 3** (`fig03_end_to_end_pipeline.png` / updated `fig15_methodological_innovations_framework.png`).
    - Formal Caption: *"Figure 3. Overall Methodological Innovation Framework for Kazakh AI-Generated Text Detection and Factual Verification."*
  - **Right Column (`left=6.65"`, `top=2.05"`, `width=6.08"`, `height=4.35"`):**
    - Two structured cards:
      * **Top Card (`height=2.15"`):** *Core Methodological Innovations* (Sentence-Level Dynamic Gating, Document-Level Top-K Engine, Evidence-Grounded Trust Matrix).
      * **Bottom Card (`height=2.05"`):** *Empirical & Societal Breakthroughs* (+42.18% OOD generalization, 100% hybrid tampering localization, 100% macro-F1 fact verification).
- **Figure 3 Visual Redesign:**
  - 4-Tier clean vertical flow:
    1. *Stage 1: Linguistic Inputs & Corpora:* Raw Kazakh Text, 83-Rule FST Morphological Dictionary, 36-Article Kazakh-FEVER Evidence Base.
    2. *Stage 2: Sentence-Level Dynamic Gating (Topic 1):* KazRoBERTa Semantic Stream $\oplus$ FST Affix Stream $\to$ Dynamic Gate $\mathbf{g} \to P_{\text{AI}}(s_i)$.
    3. *Stage 3: Document Chunking & Top-K (Topic 2):* 10 Abbreviation Guards $\to$ Sliding Windows $\to$ Worst-Case Top-K Pooling $\to P_{\text{doc}}$.
    4. *Stage 4: Trust Matrix & Verification (Topic 3):* Claim Extraction $\to$ BM25 Retrieval $\to$ 3-Way Cross-Encoder $\to$ 4-Quadrant Decision.
  - Large colored badges, distinct connector arrows, zero nested bullet boxes.

---

### 2.3 Figure 4 (Slide 8): Dual-Stream Neural Architecture
- **Dimensions:** `figsize=(7.5, 6.0)` (~1.25:1 squarish).
- **Visual Design:** A publication-grade neural network tensor flow diagram:
  - *Bottom (Input):* `Input Kazakh Sequence: X = [w_1, w_2, ..., w_L]`.
  - *Dual Streams:*
    - **Left Stream (Semantic, Navy/Blue):** Subword BPE Tokenizer $\to$ Pretrained KazRoBERTa (12 Transformer Layers) $\to$ Mean Pooling $\to h_{\text{sem}} \in \mathbb{R}^{768}$.
    - **Right Stream (Morphological, Teal/Green):** 83-Rule FST Parser (Stem + Suffix sequence) $\to$ Morpheme Embeddings ($d_m = 128$) $\to$ Bidirectional LSTM $\to$ Linear Projection $W_{\text{proj}} h_{\text{morph}} \in \mathbb{R}^{768}$.
  - *Fusion Core (Amber):* Dynamic Gating Unit:
    $$\mathbf{g} = \sigma(W_g [h_{\text{sem}}; W_{\text{proj}} h_{\text{morph}}] + b_g)$$
    $$h_{\text{fused}} = \mathbf{g} \odot h_{\text{sem}} + (1 - \mathbf{g}) \odot (W_{\text{proj}} h_{\text{morph}})$$
    Clear circular tensor combination nodes ($\otimes$ for gating, $\oplus$ for combination).
  - *Top Split (Heads):*
    - **Left:** Linear Classification Head $\to$ Sigmoid $\to P_{\text{AI}}(s_i) \in [0, 1]$ ($\mathcal{L}_{\text{BCE}}$).
    - **Right:** MLP Projection Head $\to$ Hyperspherical Unit Sphere $\mathbb{R}^{128}$ ($\mathcal{L}_{\text{SupCon}}$).

---

### 2.4 Figure 13 (Slide 17): Realistic Gradio UI Dashboard
- **Dimensions:** `figsize=(7.5, 6.0)` (~1.25:1 squarish).
- **Visual Design:** An authentic Gradio 4.x application wireframe/mockup:
  - *App Frame:* Dark slate header bar (`#0F172A`), title *"Kazakh AI Detection & Factual Verification Platform"*, subtitle *"Gradio 4.x Academic Web UI | Sub-85ms Inference"*.
  - *Tab Navigation:* 4 tabs with active styling:
    `[Tab 1: Detection & Explainability (Active)]`, `[Tab 2: Morphological FST Lab]`, `[Tab 3: Benchmark Methodology]`, `[Tab 4: Four-Quadrant Trust Matrix]`.
  - *Left Input & Sentence Heatmap Panel:*
    - Sample Kazakh input text area.
    - Color-highlighted sentence heatmap:
      * Sentence 1 (Human, 97.0%): Soft green background.
      * Sentence 2 (AI-generated, 98.4%): Soft red background with red underline.
    - FST decomposition callout: *"анықтайды $\to$ [анықта] (stem) + [-йд] (pres) + [-ы] (3sg)"*.
  - *Right Analytics & Decision Panel:*
    - Verdict banner: *"SUSPECTED AI GENERATED (89.2%)"* in bold red/amber.
    - Dynamic Gating Fusion Bar: Stacked bar showing KazRoBERTa 68% (Blue) vs FST Affix 32% (Green).
    - Performance metrics: Latency *74ms*, Memory *< 1.2 GB*, Fact Verification *Supported (Kazakh-FEVER)*.

---

### 2.5 Figure 14 & Slide 18 Polish: Cloud Packaging & Multi-Format Ingestion
- **Slide 18 Stat Cards Replacement:**
  - **Card 1 (`left=6.65"`, `width=1.92"`):**
    - Value: `311 / 311`
    - Title: `Automated Test Suite`
    - Subtitle: `Passing unit & regression tests with 0 regressions`
    - Accent: Green (`#16A34A`)
  - **Card 2 (`left=8.72"`, `width=1.92"`):**
    - Value: `< 1.2s`
    - Title: `Cold-Start Latency`
    - Subtitle: `Sub-second initialization with CPU/GPU dual paths`
    - Accent: Blue (`#2563EB`)
  - **Card 3 (`left=10.79"`, `width=1.94"`):**
    - Value: `3 Formats`
    - Title: `Multi-Format Ingestion`
    - Subtitle: `Defensive parsing for .txt, .docx, .pdf with 10MB memory protection`
    - Accent: Teal (`#0D9488`)
  - **Subtitle Update:**
    `"Production-ready deployment bundle with 1-click cloud launching, sub-second cold starts, and 311 passing tests"`
  - **Complete Removal:**
    Every occurrence of `"Zero Emoji Design"`, `"Zero-emoji AST scanner"`, or `"Strict typography compliant with academic standards"` is permanently excised from slides, cards, code, and diagrams.
- **Figure 14 Visual Redesign (`figsize=(7.5, 6.0)`):**
  - Modern 4-stage DevOps & Serving Flow:
    1. *Stage 1: Multi-Format Ingestion Engine:* Safe stream processing for `.txt`, `.docx`, `.pdf`, 10MB memory guards, 25,000-word safety caps.
    2. *Stage 2: Automated Test Rigor:* 311 / 311 unit & regression tests passing in < 15s, FST rule validation, zero regressions.
    3. *Stage 3: Containerized Serving:* Hugging Face Spaces Docker container, Python 3.12, cold-start < 1.2s, bounded VRAM < 1.4 GB.
    4. *Stage 4: Interactive Production Endpoints:* Gradio 4.x Web UI, sub-85ms real-time inference, defense report cards.

---

## 3. Code Modifications & Deliverables

1. **`scripts/generate_presentation_figures.py`**:
   - Redesign `generate_fig03(output_dir)`: 4:3 squarish (`figsize=(7.5, 5.8)`), streamlined 4-tier pipeline.
   - Redesign `generate_fig04(output_dir)`: 4:3 squarish (`figsize=(7.5, 6.0)`), publication-grade neural architecture.
   - Redesign `generate_fig13(output_dir)`: 4:3 squarish (`figsize=(7.5, 6.0)`), realistic Gradio UI dashboard wireframe.
   - Redesign `generate_fig14(output_dir)`: 4:3 squarish (`figsize=(7.5, 6.0)`), cloud CI/CD & deployment pipeline without emoji references.
   - Update `generate_fig15(output_dir)` if retained for reference.
2. **`scripts/build_cloned_presentation.py`**:
   - Update `populate_slide_07(slide)`: Implement split layout with squarish Figure 3 on left (`width=5.85"`, `height=4.35"`) and 2 method innovation cards on right (`width=6.08"`).
   - Update `populate_slide_18(slide)`: Update Card 1 to `311 / 311`, replace Card 3 with `3 Formats / Multi-Format Ingestion`, update subtitle.
3. **`scripts/update_presentation_methodology.py`**:
   - Update `update_slide_07(slide, fig_path)` to support split layout with squarish Figure 3 and side cards.
   - Ensure updates are applied directly to `C:\Users\Roza\Desktop\AnekeshD_Progress.pptx` and mirrored to `Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx`.
4. **`tests/test_presentation_figures.py` & `tests/test_presentation_content.py`**:
   - Verify image dimensions, file existence, and aspect ratios.
   - Verify Slide 18 card texts (311 tests, 3 formats, absence of "Zero Emoji").
   - Confirm all 311+ tests pass.
5. **Slide Exports**:
   - Re-export `ppt_images/slide_07.png`, `slide_08.png`, `slide_17.png`, `slide_18.png` using PowerPoint COM automation and verify visually.

---

## 4. Constraints & Commitments

- **Strict LaTeX Freeze:** `aist2026/paper.tex` will remain **100% untouched (0 diff lines)**.
- **Font Preservation:** All manual font adjustments made by Daulet on other slides in `AnekeshD_Progress.pptx` will be preserved intact.
- **Emoji Rule:** Zero decorative emojis in code, slides, and documentation.
- **Format:** Widescreen 16:9 presentation dimensions (13.333" x 7.500") strictly preserved.
