# Kazakh AI-Text Detector: UI Visual Redesign & Morphological Polish Design Specification

**Date:** 2026-09-11  
**Status:** Approved by User  
**Target Repository:** `kazakh-ai-text-detection`  
**Deployment Target:** Hugging Face Spaces (`nKa1i/kazakh-ai-text-detector`)

---

## 1. Overview & Context

The Kazakh AI-Text Detection application features a production-grade, dual-stream architecture (BERT contextual semantics + FST morphological structure) designed for robust cross-domain and zero-shot detection. Following user feedback on the Gradio interface, this design details the visual modernization, layout streamlining, and linguistic morphological refinements necessary to achieve a publication-grade presentation.

### Primary Objectives
1. **Direct Tab Access:** Remove the introductory hero banner so the 3 academic tabs (`Detection & Explainability`, `Morphological FST Lab`, `Benchmark & Academic Methodology`) start directly at the very top of the window.
2. **Left-Aligned Language Switcher:** Place the language switcher (`English` / `Қазақша`) immediately adjacent to "Quick Benchmark Samples:" on the left, eliminating the awkward white-space gap across the top row.
3. **Unified Executive Summary Card (Component 1):**
   - Retire the dated ASCII progress bar (`[====== 3.0% =====]`) and eliminate redundant duplicate verdict labels.
   - Replace separate raw Gradio textboxes (`prob_box`, `ratio_box`, `stats_display`) with a sleek, cohesive HTML/CSS executive card featuring a modern gradient progress bar and a 3-tile metric grid (`AI Probability`, `AI Content Ratio`, `Document Volume`).
   - Seamlessly nest dynamic linguistic explanation bullets below the metrics.
4. **Sentence-Level Visual Heatmap & Legend (Component 2):**
   - Refine the 3-tier legend into rounded pill badges with high-contrast indicator dots.
   - Enhance the heatmap container with subtle borders and smooth hover tooltips.
5. **Dynamic Fusion Gate (Component 3):**
   - Streamline the dual-stream split bar (BERT Context blue `#3b82f6` $\rightarrow$ `#2563eb` vs FST Morphology purple `#8b5cf6` $\rightarrow$ `#7c3aed`) with rounded track pills, high-contrast labels, and clear linguistic footnotes.
6. **Morphological FST Stemming & Table Polish (Component 4):**
   - Support the Kazakh agentive derivational suffix `-ші / -шы` in `AdvancedKazakhFSTAnalyzer` and `ui/sentence_analyzer.py` before possessive stripping, preventing false truncations such as `жетекші` $\rightarrow$ `жетекш` + `-і (POSS.3SG)` and ensuring correct analysis as root `жетек` + `-ші (AGENT/DERIV)`.
   - Polish table typography, subtle zebra striping, and rounded grammatical tags for both 4-column (Tab 1) and 8-column (Tab 2) decomposition tables.

---

## 2. Layout & Architectural Changes

### 2.1 Header & Top-Level Tab Structure
- In `ui/app.py` (`create_app`):
  - Remove `hero_banner = gr.HTML(render_hero_html("en"))`.
  - The application opens directly with `with gr.Tabs() as tabs:`.
  - Update `switch_ui_language`: remove the hero banner output and its corresponding Gradio component target, reducing the returned tuple count from 40 to 39 components.
- In `ui/app.py` Tab 1:
  - Refactor `quick-samples-header-row` into a compact flex container:
    ```html
    <div class="quick-samples-header-flex">
      <span class="quick-samples-title">Quick Benchmark Samples:</span>
      <div class="lang-switch-inline">...</div>
    </div>
    ```
  - Both elements align to `flex-start` with a clean 12px gap, eliminating the empty horizontal gap.

### 2.2 Component 1: Unified Executive Summary Card & Confidence Meter
- In `ui/highlighting.py`:
  - Update `render_confidence_meter(probability, verdict, lang, doc_result=None)`:
    - **Header:** Prominent status badge (`badge-human`, `badge-hybrid`, or `badge-ai`).
    - **Modern Meter:** Full-width rounded progress track with a smooth CSS gradient fill (`meter-human`, `meter-amber`, `meter-ai`) showing the percentage. No ASCII brackets.
    - **3-Tile KPI Grid:**
      - Tile 1: **AI Probability** (e.g. `3.0%`).
      - Tile 2: **AI Content Ratio** (e.g. `0.0%`).
      - Tile 3: **Document Volume** (e.g. `240 words • 14 sentences • 2 windows`).
  - In `ui/app.py`:
    - Consolidate `verdict_badge`, `confidence_meter_display`, `prob_box`, `ratio_box`, and `stats_display` into the unified executive summary display, significantly reducing visual clutter.

### 2.3 Component 2: Sentence-Level Visual Heatmap & Legend
- In `ui/app.py` / `ui/highlighting.py`:
  - `render_legend_html(lang)`:
    - Render 3 rounded pill badges:
      - `Human Written: 0% – 39.9%` (Soft green background, green border, dark green text).
      - `Borderline / Caution: 40.0% – 99.79%` (Soft amber background, amber border, dark amber text).
      - `AI-Generated: ≥ 99.80%` (Soft red background, red border, dark red text).
  - In `HEATMAP_CSS`:
    - Ensure `.kaz-heatmap-container` and `.kaz-sentence` have accessible contrast, smooth transitions, and high-readability typography across light and dark themes.

### 2.4 Component 3: Dynamic Fusion Gate
- In `ui/highlighting.py` (`render_dynamic_gate_bar`):
  - Retain the proportional dual split-bar with BERT blue (`#3b82f6` $\rightarrow$ `#2563eb`) and FST purple (`#8b5cf6` $\rightarrow$ `#7c3aed`).
  - Improve container contrast with rounded pills (`border-radius: 9999px`), crisp percentage readouts inside or above bars, and clear linguistic interpretation footnotes.

### 2.5 Component 4: Morphological FST Stemming & Decomposition Tables
- In `fst_analyzer.py` and `ui/sentence_analyzer.py`:
  - Add agentive/derivational suffix definitions:
    - `AGENT_TAGS = {"-ші": "AGENT", "-шы": "AGENT"}`
  - In stemming order: before stripping the 3rd person possessive `-і` / `-ы`, check if the word ends with `-ші` / `-шы` (or if it is a known agentive noun like `жетекші`, `жұмысшы`, `жазушы`, `оқушы`, `жүргізуші`, `сатушы`).
  - Decompose `жетекші` as:
    - Root: `жетек`
    - POS: `NOUN`
    - Affixes: `["-ші (DERIV.AGENT)"]`
- In `ui/highlighting.py`:
  - Polish `.fst-table` CSS with clean table header borders, subtle alternating row backgrounds, and rounded badges for POS and affix chains.

---

## 3. Constraints & Invariants

1. **Zero Paper Modifications:** Under no circumstances shall `aist2026/paper.tex` be touched or modified.
2. **Zero Emoji Mandate:** Strictly no decorative emojis in any UI labels, buttons, cards, legends, tables, or markdown documents.
3. **Strict XSS Sanitization:** All dynamic user text in HTML templates must pass through `html.escape(..., quote=True)`.
4. **Defensive Operation:** Must run seamlessly in offline/CPU environments without requiring active GPUs or loaded model weights.
5. **Synchronization:** All updates must be mirrored in `hf_space/` for seamless deployment to Hugging Face Spaces.

---

## 4. Verification Plan

1. **Unit Tests:**
   - Run `tests/test_ui_highlighting.py` to verify the redesigned executive KPI card, confidence meter, legend, and gate bars.
   - Run `tests/test_ui_sentence_analyzer.py` to verify morphological stemming for `жетекші` $\rightarrow$ `жетек` + `-ші (DERIV.AGENT)`.
   - Run `tests/test_ui_integration.py` to verify the 3-tab layout, header row alignment, language toggle, and zero emoji presence.
   - Run `tests/test_hf_space_bundle.py` to verify deployment bundle integrity.
2. **Full Regression Suite:**
   - Execute all unit tests across the repository.
3. **Live Hugging Face Deployment:**
   - Deploy updated bundle via `scripts/push_to_hf.py` and verify build status.
