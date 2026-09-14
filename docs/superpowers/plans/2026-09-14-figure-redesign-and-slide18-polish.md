# Publication Figure Redesigns & Slide 18 Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign Figures 3, 4, 13, and 14 with squarish (~4:3) aspect ratios and publication-grade visual clarity, convert Slide 7 to a balanced split layout with side cards, and polish Slide 18 by replacing "Zero Emoji Design" with "3 Formats Supported (.txt · .docx · .pdf)" and updating test counts to 311.

**Architecture:** Matplotlib figure generators in `scripts/generate_presentation_figures.py` will generate 300 DPI figures using squarish dimensions (`figsize=(7.5, 5.8)` to `(7.8, 6.0)`) and publication-grade graphical schemas. `scripts/build_cloned_presentation.py` and `scripts/update_presentation_methodology.py` will embed the squarish figures into standard 5.85" x 4.35" columns on Slides 7, 8, 17, and 18, populate Slide 7's new right-column method cards, update Slide 18's stat cards, and preserve all native branding and manual user adjustments.

**Tech Stack:** Python 3.12, `matplotlib`, `python-pptx`, `win32com.client` (PowerPoint COM automation), `unittest`.

## Global Constraints
- Under no circumstances shall `aist2026/paper.tex` be modified or touched (0 diff lines).
- Zero decorative emojis across all code, slides, cards, and documentation.
- Widescreen 16:9 format (13.333" x 7.500") strictly preserved.
- Preserved native branding: KazNU logo (`Изображение 16`), NPU logo group (`LOGO组合`), and Layout 7 NPU pentagon emblem.
- Strictly preserve all manual font size edits Daulet made on `AnekeshD_Progress.pptx`.
- Explicit UTF-8 encoding (`encoding='utf-8'`) for all file operations.
- All unit and regression tests (311+) must pass cleanly.

---

### Task 1: Redesign Figures 3, 4, 13, and 14 with Squarish Aspect Ratios in `scripts/generate_presentation_figures.py`

**Files:**
- Modify: `scripts/generate_presentation_figures.py:250-454,1052-1249`
- Modify: `tests/test_presentation_figures.py:1-70`

**Interfaces:**
- Consumes: Matplotlib `patches.FancyBboxPatch`, `patches.Circle`, `patches.Arrow`, institutional color constants (`NAVY`, `TEAL`, `AMBER`, `GREEN`, `SLATE`, `CORAL`, `MUTED_GRAY`, `LIGHT_BG`, `BORDER_GRAY`).
- Produces: 300 DPI PNG files in `presentation_figures/`:
  - `fig03_end_to_end_pipeline.png` (Aspect ratio ~1.28:1, squarish 4-tier pipeline flow)
  - `fig04_morpho_gate_arch.png` (Aspect ratio ~1.25:1, squarish dual-stream neural tensor architecture)
  - `fig13_gradio_dashboard_panels.png` (Aspect ratio ~1.25:1, squarish authentic Gradio 4.x UI mockup)
  - `fig14_cloud_deployment_pipeline.png` (Aspect ratio ~1.25:1, squarish 4-stage DevOps CI/CD pipeline)

- [ ] **Step 1: Update `tests/test_presentation_figures.py` with squarish aspect ratio tests**

Update `tests/test_presentation_figures.py` to assert that Figures 3, 4, 13, and 14 exist, are non-empty, and their aspect ratio (width / height) is squarish (between 1.1:1 and 1.45:1, rather than widescreen >= 1.8:1).

```python
# In tests/test_presentation_figures.py
from PIL import Image

def test_squarish_aspect_ratios_fig03_04_13_14(self):
    target_figs = [
        "fig03_end_to_end_pipeline.png",
        "fig04_morpho_gate_arch.png",
        "fig13_gradio_dashboard_panels.png",
        "fig14_cloud_deployment_pipeline.png",
    ]
    for fig_name in target_figs:
        path = os.path.join(self.output_dir, fig_name)
        self.assertTrue(os.path.isfile(path), f"Missing {fig_name}")
        with Image.open(path) as img:
            w, h = img.size
            ratio = w / h
            self.assertGreaterEqual(ratio, 1.10, f"{fig_name} ratio {ratio:.2f} too tall")
            self.assertLessEqual(ratio, 1.48, f"{fig_name} ratio {ratio:.2f} too wide; expected squarish ~4:3")
```

- [ ] **Step 2: Run test to verify it fails on old widescreen figures**

Run:
```powershell
python -m unittest tests/test_presentation_figures.py
```
Expected: FAIL because old figures were generated with ratio ~1.9:1 to 2.0:1.

- [ ] **Step 3: Implement redesigned squarish figure generators in `scripts/generate_presentation_figures.py`**

In `scripts/generate_presentation_figures.py`:
1. **`generate_fig03(output_dir)`**:
   - `figsize=(7.5, 5.8)`, `ax.set_xlim(0, 100)`, `ax.set_ylim(0, 100)`.
   - Title: "Tripartite Research Framework: End-to-End Pipeline"
   - 4 clean vertical tiers:
     - Tier 1 (y=75-90): *Linguistic Inputs & Knowledge* (Raw Kazakh Text, 83-Rule FST Morpho-Dict, Kazakh-FEVER Corpus).
     - Tier 2 (y=52-67): *Sentence-Level Dynamic Gating* (KazRoBERTa Semantic $\oplus$ FST Morphological Stream $\to$ Dynamic Gate $\mathbf{g} \to P_{\text{AI}}(s_i)$).
     - Tier 3 (y=29-44): *Document Sliding Window & Top-K* (10 Abbreviation Guards $\to$ Sliding Windows $\to$ Worst-Case Top-K Pooling $\to P_{\text{doc}}$).
     - Tier 4 (y=6-21): *Fact-Checking Trust Matrix* (Claim Extraction $\to$ BM25 Retrieval $\to$ 3-Way NLI Cross-Encoder $\to$ 4-Quadrant Decision).
   - Clean vertical connecting arrows and distinct color badges (Navy, Blue, Green, Amber).
2. **`generate_fig04(output_dir)`**:
   - `figsize=(7.5, 6.0)`, `ax.set_xlim(0, 100)`, `ax.set_ylim(0, 100)`.
   - Title: "Dual-Stream Morphological Cross-Attention Architecture"
   - Bottom (y=4-14): Input Kazakh Sequence $X = [w_1, w_2, \dots, w_L]$ with sample sentence: "Жасанды интеллект қазақ тіліндегі мәтіндерді дәл анықтайды".
   - Dual streams (y=20-48):
     - Left (Navy/Blue): BPE Tokenizer $\to$ KazRoBERTa 12-Layer Transformer $\to$ Mean Pooling $\to h_{\text{sem}} \in \mathbb{R}^{768}$.
     - Right (Teal/Green): 83-Rule FST Parser $\to$ Morpheme Embeddings ($d_m=128$) $\to$ BiLSTM $\to$ Linear Projection $W_{\text{proj}} h_{\text{morph}} \in \mathbb{R}^{768}$.
   - Center Fusion Unit (y=56-72, Amber):
     - Dynamic Learned Gating: $\mathbf{g} = \sigma(W_g [h_{\text{sem}}; W_{\text{proj}} h_{\text{morph}}] + b_g)$.
     - Combination: $h_{\text{fused}} = \mathbf{g} \odot h_{\text{sem}} + (1 - \mathbf{g}) \odot (W_{\text{proj}} h_{\text{morph}})$.
   - Top Split Heads (y=80-94):
     - Left (Blue): Linear Head $\to$ Sigmoid $\to P_{\text{AI}}(s_i)$ ($\mathcal{L}_{\text{BCE}}$).
     - Right (Green): MLP Projection $\to$ Hypersphere $\mathbb{R}^{128}$ ($\mathcal{L}_{\text{SupCon}}$).
3. **`generate_fig13(output_dir)`**:
   - `figsize=(7.5, 6.0)`, `ax.set_xlim(0, 100)`, `ax.set_ylim(0, 100)`.
   - Realistic Gradio 4.x application window mockup:
     - Top bar (Slate `#0F172A`): "Kazakh AI Detection & Factual Verification Platform | Gradio 4.x Web UI".
     - Tab navigation: `[Tab 1: Detection & Explainability (Active)]`, `[Tab 2: Morphological FST Lab]`, `[Tab 3: Benchmark Methodology]`, `[Tab 4: Four-Quadrant Trust Matrix]`.
     - Left panel (y=10-76, x=3-54): Sample Kazakh text input with sentence heatmap:
       * Sentence 1 (Human, 97.0%): Soft green highlight.
       * Sentence 2 (AI, 98.4%): Soft red highlight.
       * Morphological decomposition box: "анықтайды $\to$ [анықта] + [йд] + [ы]".
     - Right panel (y=10-76, x=57-97): Verdict & metrics:
       * Banner: "SUSPECTED AI GENERATED (89.2%)".
       * Dynamic Gate Fusion Bar: KazRoBERTa 68% (Blue) | FST Affix 32% (Green).
       * Latency badge: "74ms", Memory: "< 1.2 GB", Fact check: "Supported".
4. **`generate_fig14(output_dir)`**:
   - `figsize=(7.5, 6.0)`, `ax.set_xlim(0, 100)`, `ax.set_ylim(0, 100)`.
   - Title: "Cloud Deployment & Production CI/CD Pipeline"
   - 4 clean pipeline stages:
     - Stage 1 (Top-Left): *Multi-Format Ingestion Engine* (Safe stream parsing for `.txt`, `.docx`, `.pdf`, 10MB memory protection, 25k-word streaming cap).
     - Stage 2 (Top-Right): *Automated Test Suite Rigor* (311 / 311 unit & regression tests, FST grammar validation, 100% CI pass in < 15s).
     - Stage 3 (Bottom-Left): *Containerized Spaces Runtime* (Docker, Python 3.12, sub-1.2s cold start, bounded < 1.4 GB VRAM).
     - Stage 4 (Bottom-Right): *Interactive Production Endpoints* (Gradio 4.x web UI, sub-85ms inference, exportable committee defense cards).
   - Zero occurrences of "Zero Emoji Design" or "Zero-emoji AST scanner".

- [ ] **Step 4: Execute generator and verify all figures pass test**

Run:
```powershell
python scripts/generate_presentation_figures.py
python -m unittest tests/test_presentation_figures.py
```
Expected: PASS with 15 figures generated at 300 DPI and squarish aspect ratios verified.

- [ ] **Step 5: Commit Task 1 changes**

```bash
git add scripts/generate_presentation_figures.py tests/test_presentation_figures.py presentation_figures/fig03_end_to_end_pipeline.png presentation_figures/fig04_morpho_gate_arch.png presentation_figures/fig13_gradio_dashboard_panels.png presentation_figures/fig14_cloud_deployment_pipeline.png
git commit -m "feat(figures): redesign Figures 3, 4, 13, and 14 with squarish aspect ratios and publication clarity"
```

---

### Task 2: Update Slide 7 Split Layout, Slide 18 Stat Cards, and Presentation Updaters

**Files:**
- Modify: `scripts/build_cloned_presentation.py:586-630,1084-1122`
- Modify: `scripts/update_presentation_methodology.py:221-272`
- Modify: `tests/test_presentation_content.py`
- Modify: `tests/test_presentation_methodology.py`

**Interfaces:**
- Consumes: Squarish `fig03_end_to_end_pipeline.png`, `fig04_morpho_gate_arch.png`, `fig13_gradio_dashboard_panels.png`, `fig14_cloud_deployment_pipeline.png` from `presentation_figures/`.
- Produces: Updated slides in `AnekeshD_Progress.pptx` and `Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx`:
  - Slide 7: Split layout with squarish Figure 3 (`width=5.85"`, `height=4.35"`) on the left and 2 method summary cards on the right.
  - Slide 18: Card 1 `311 / 311 Automated Test Suite`, Card 2 `< 1.2s Cold-Start Latency`, Card 3 `3 Formats Multi-Format Ingestion` (`.txt · .docx · .pdf`), subtitle updated to 311 passing tests, zero emoji references.

- [ ] **Step 1: Update `tests/test_presentation_content.py` to check Slide 7 split layout and Slide 18 card contents**

In `tests/test_presentation_content.py`:
- Verify Slide 7 contains Figure 3 picture AND 2 callout cards ("Core Methodological Innovations" and "Empirical & Societal Breakthroughs").
- Verify Slide 18 contains "311 / 311", "3 Formats", "Multi-Format Ingestion", and contains ZERO occurrences of "Zero Emoji".

- [ ] **Step 2: Run tests to verify they fail**

Run:
```powershell
python -m unittest tests/test_presentation_content.py tests/test_presentation_methodology.py
```
Expected: FAIL on Slide 7 cards and Slide 18 Card 3 content.

- [ ] **Step 3: Update `populate_slide_07` and `populate_slide_18` in `scripts/build_cloned_presentation.py`**

1. In `populate_slide_07(slide)`:
   - Header:
     - Title: "Research Content Overview: Comprehensive Methodological Framework"
     - Subtitle: "A unified hierarchical framework spanning sentence-level morpho-gating, document-level chunk aggregation, and evidence-grounded trust verification"
   - Figure:
     - `add_figure_with_caption(slide, "fig03_end_to_end_pipeline.png", left=0.60, top=2.05, width=5.85, height=4.35, caption_text="Figure 3. Overall Methodological Innovation Framework for Kazakh AI-Generated Text Detection and Factual Verification.")`
   - Right Column Cards:
     - Card 1 (`left=6.65, top=2.05, width=6.08, height=2.25`):
       - Title: "Three-Tier Methodological Innovations"
       - Points:
         * "1. Sentence-Level Morpho-Gating: Fuses KazRoBERTa embeddings with an 83-rule FST morphological transducer via dynamic learned gating, solving OOD domain collapse."
         * "2. Document Sliding Window & Top-K: First sentence-preserving chunker with 10 Kazakh abbreviation guards and dynamic worst-chunk Top-K aggregation up to 25k words."
         * "3. Kazakh-FEVER & Trust Matrix: First evidence-grounded fact verification corpus for Kazakh, decoupling origin detection from factual veracity via a 4-quadrant decision model."
       - Accent: NAVY_PRIMARY, bg_color: CARD_BG_BLUE
     - Card 2 (`left=6.65, top=4.45, width=6.08, height=2.35`):
       - Title: "Integrated Empirical & Societal Breakthroughs"
       - Points:
         * "- Cross-Domain Generalization: +42.18% ROC-AUC improvement on out-of-domain colloquial text over standard transformer baselines."
         * "- Stealth Tamper Localization: 100% precision in identifying isolated synthetic paragraphs inserted into long human documents."
         * "- Fact Verification Precision: 100.0% Macro-F1 across 36 verified encyclopedic topics with sub-1.4 GB bounded memory."
       - Accent: TEAL_ACCENT, bg_color: CARD_BG_WHITE
2. In `populate_slide_18(slide)`:
   - Header subtitle: "Production-ready deployment bundle with 1-click cloud launching, sub-second cold starts, and 311 passing tests"
   - Card 1: `title="Automated Test Suite"`, `value_str="311 / 311"`, `subtitle="Passing unit & regression tests with 0 regressions"`, `accent_color=GREEN_ACCENT`, `bg_color=CARD_BG_GREEN`
   - Card 2: `title="Cold-Start Latency"`, `value_str="< 1.2s"`, `subtitle="Sub-second initialization with CPU/GPU dual paths"`, `accent_color=BLUE_ACCENT`, `bg_color=CARD_BG_WHITE`
   - Card 3: `title="Multi-Format Ingestion"`, `value_str="3 Formats"`, `subtitle="Defensive parsing for .txt, .docx, .pdf with 10MB memory guards"`, `accent_color=TEAL_ACCENT`, `bg_color=CARD_BG_WHITE`
   - Callout box below cards: update text points and remove any emoji mention.

- [ ] **Step 4: Update `update_presentation_methodology.py` for Slide 7 and Slide 18**

Update `update_slide_07(slide, fig_path)` to insert squarish Figure 3 on the left and the 2 structured innovation cards on the right.
Add/update `update_slide_18(slide, fig_path)` to replace Card 3 with "3 Formats / Multi-Format Ingestion", update Card 1 to "311 / 311", and embed squarish Figure 14.
Run the presentation updater against `AnekeshD_Progress.pptx`.

- [ ] **Step 5: Run tests to verify they pass**

Run:
```powershell
python scripts/build_cloned_presentation.py
python scripts/update_presentation_methodology.py
python -m unittest tests/test_presentation_content.py tests/test_presentation_methodology.py
```
Expected: PASS on all tests.

- [ ] **Step 6: Commit Task 2 changes**

```bash
git add scripts/build_cloned_presentation.py scripts/update_presentation_methodology.py tests/test_presentation_content.py tests/test_presentation_methodology.py
git commit -m "feat(presentation): apply Slide 7 split layout and Slide 18 multi-format ingestion card"
```

---

### Task 3: High-Resolution Slide PNG Exports, Speaker Notes Sync & Full Regression

**Files:**
- Modify: `thesis_presentation_speaker_notes_prof_guo.md`
- Export: `ppt_images/slide_07.png`, `slide_08.png`, `slide_17.png`, `slide_18.png`

**Interfaces:**
- Consumes: `AnekeshD_Progress.pptx`, PowerPoint COM automation.
- Produces: Freshly rendered PNGs in `ppt_images/`, synchronized defense notes.

- [ ] **Step 1: Re-export all 23 slides to `ppt_images/`**

Run:
```powershell
python scripts/export_presentation_slides.py
```
Expected: 23 slides exported in high-resolution (1080p) to `ppt_images/`.

- [ ] **Step 2: Visually verify exported slides using view_file**

Verify:
- `ppt_images/slide_07.png`: Squarish Figure 3 on left, 2 clean method cards on right. No cluttered 15 nested bullet boxes.
- `ppt_images/slide_08.png`: Squarish Figure 4 neural network tensor flow on left, no stretched aspect ratio, large legible typography.
- `ppt_images/slide_17.png`: Squarish Figure 13 realistic Gradio UI on left, 4 distinct tab cards on right.
- `ppt_images/slide_18.png`: Squarish Figure 14 DevOps pipeline on left, Card 1 (311 / 311), Card 2 (< 1.2s), Card 3 (3 Formats Multi-Format Ingestion). Zero emoji text anywhere.

- [ ] **Step 3: Update `thesis_presentation_speaker_notes_prof_guo.md`**

Synchronize speaker notes for Slides 7 and 18:
- Slide 7: Walkthrough of the split layout (squarish 4-stage pipeline on the left, 3 methodological innovations and empirical breakthroughs on the right).
- Slide 18: Walkthrough of the deployment pipeline (multi-format document ingestion engine with 10MB memory guards, 311 automated unit and regression tests, containerized serving).

- [ ] **Step 4: Verify full test suite and paper freeze**

Run:
```powershell
git diff aist2026/paper.tex
python -m unittest discover -s tests -p "test_*.py"
```
Expected:
- `git diff aist2026/paper.tex` produces empty output (0 diff lines).
- 311+ tests pass cleanly in under 5 seconds.

- [ ] **Step 5: Commit Task 3 changes**

```bash
git add thesis_presentation_speaker_notes_prof_guo.md
git commit -m "docs(notes): synchronize defense speaker notes with squarish figures and multi-format ingestion"
```
