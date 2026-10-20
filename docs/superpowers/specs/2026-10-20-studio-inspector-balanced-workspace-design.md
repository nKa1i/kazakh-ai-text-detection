# Design Spec: Studio & Inspector Balanced Workspace (Apple / Notion Light)

- **Date**: 2026-10-20
- **Status**: Approved by User
- **Author**: Antigravity Pair Programming System
- **Topic**: Studio & Inspector Balanced Workspace Architecture, Pure English UI, and Canvas Harmonization

---

## 1. Problem Statement & Motivation

Following the initial two-column scaffold, user visual feedback identified two key UX and aesthetic issues:
1. **Jarring Dark Mode Contrast & Blockiness**:
   - Because the user's OS/browser was in dark mode, Gradio automatically injected `.dark` classes that turned the page background pitch black (`#0b0f19`).
   - Custom HTML cards had hardcoded white backgrounds (`#ffffff`), producing harsh rectangular white blocks floating in a black void.
   - Elements were enclosed in nested bounding boxes with redundant duplicate confidence progress bars (one inside the summary card, one directly underneath).
2. **Severe Vertical Asymmetry (Lopsided Scrolling)**:
   - The left control column was only ~400px tall (textarea + buttons), leaving a large empty void below it.
   - The right column stretched 2,000px+ vertically with KPIs, duplicate meters, full-document sentence heatmaps, morphology tables, and verification matrices, forcing constant scrolling.
3. **Cluttered Bilingual Chrome**:
   - Navigation tabs and titles contained repetitive slash strings (`Overview & Heatmap / Шолу`, `Morphology & Slang Lab / Морфология`, `Fact Verification & Trust Matrix / Тексеру`), adding visual noise.

---

## 2. Goals & Non-Goals

### Goals
- **Enforce Consistent Light Notion Canvas**: Force `color-scheme: light !important;` and override Gradio's automatic dark mode so the background is always soft off-white (`#fbfbfa`), with pure white cards (`#ffffff`) and hairline zinc borders (`#e4e4e7`).
- **Balanced Studio & Inspector Layout (scale=5 / scale=5)**:
  - **Left Column (Document Studio)**: Houses text ingestion, curated preset cards, live typing statistics ribbon, action buttons, and the **Visual Sentence Heatmap**.
  - **Right Column (Diagnostic Inspector)**: Houses the persistent Executive Summary HUD (with a single integrated 6px confidence meter) and 3 focused diagnostic tabs.
- **Pure English UI**: Remove the Kazakh language switch toggle from the utility bar; streamline all labels, buttons, tabs, and drawer headers to clean English.
- **4 High-Value Enhancements**:
  1. *Visual Morpheme Diff Highlighting* in the Auto-FST tab (sky blue for Russian layout normalization, violet for newly induced slang roots).
  2. *Export Audit Report (.md)* button for downloading the analysis record.
  3. *Live Typing Statistics Ribbon* (words, sentences, estimated read time, morpheme complexity).
  4. *Descriptive Scenario Cards* replacing generic pill buttons.

### Non-Goals
- Modifying underlying detection models, FST grammar rules, or NLI verifier logic.
- Altering conference paper manuscripts (`aist2026/paper.tex` or `papers/kazakh_fever_conference/`).
- Removing backend support for analyzing authentic Kazakh text (the system analyzes Kazakh input; only the UI chrome is standardized to English).

---

## 3. Global Constraints
- Under no circumstances shall `aist2026/paper.tex` or `papers/kazakh_fever_conference/` be touched or modified (strictly 0 diff lines).
- Strictly zero decorative emojis across all code, datasets, comments, documentation, UI labels, and HTML templates.
- Complete parity between `ui/` and `hf_space/ui/`.
- Sub-millisecond latency on CPU (< 1.0 ms per FST parse).
- All Python file operations must explicitly specify `encoding='utf-8'`.

---

## 4. Architecture & Component Layout

### 4.1. Top Utility Header & Collapsible Drawer
- Clean title: `Kazakh AI-Generated Text Detector & Explainability Dashboard`.
- Subtitle: `Production-grade, explainable detection of AI-generated Kazakh text with sentence-level heatmaps, dynamic dual-stream gate fusion, and deep morphological FST breakdowns.`
- Top-right collapsible drawer:
  - Accordion label: `Methodology & Academic Benchmarks`.
  - Contents: Kaz-MAGE $2 \times 2$ operational risk matrix, empirical evaluation tables ($N=4,000$), dual-stream gating formula, and BibTeX citations.

### 4.2. Left Column: The Document Studio (`scale=5`)
1. **Curated Scenario Cards**:
   - `Kaspi Review [Human]`
   - `Formal News [Human]`
   - `Sherkala-7B [AI]`
   - `Slang & Russian Layout [Informal]`
2. **Primary Document Textarea**:
   - `text_input`: `lines=9`, `max_lines=16`, interactive, with subtle zinc border.
3. **Live Typing Statistics Ribbon**:
   - Subtle monospace indicator: `Words: {N} • Sentences: {S} • Est. Read Time: {T}s • Morpheme Depth: {D}x`.
4. **Document Upload Accordion**:
   - Collapsible `.txt`, `.docx`, `.pdf` ingestion.
5. **Feature Toggles**:
   - Checkboxes for `Auto-FST Slang & Normalization` and `Kazakh-FEVER Factual Trust Verification` (both default to True).
6. **Actions Row**:
   - `Run Full Analysis` (`.btn-primary-charcoal`, primary action).
   - `Export Audit (.md)` (`.btn-secondary-outline`, exports markdown report).
   - `Clear` (`.btn-text`, resets workspace).
7. **Visual Sentence Heatmap (Relocated to Left Studio)**:
   - Soft pastel sentence highlights:
     - Authentic Human (`#ecfdf5` background, `#10b981` border, `#065f46` text)
     - Borderline / Caution (`#fffbeb` background, `#f59e0b` border, `#92400e` text)
     - AI-Generated (`#fff1f2` background, `#f43f5e` border, `#9f1239` text)
   - Clicking any sentence immediately sends it to the Right Inspector for deep-dive diagnostics.

### 4.3. Right Column: The Diagnostic Inspector (`scale=5`)
1. **Persistent Top HUD**:
   - **Executive Summary Card**: Verdict badge (`[AUTHENTIC HUMAN]`, `[MACHINE-GENERATED]`, `[PARTIALLY AI]`), probability percentage, 4 KPI tiles (Words, Sentences, AI Content Volume, Lexical Coverage), and a single sleek 6px confidence meter track.
   - *Eliminates the redundant second confidence meter bar.*
2. **3 Consolidated Sub-Tabs**:
   - **Tab 1: Sentence Morphology & Fusion**:
     - Selected sentence callout box.
     - Dynamic Dual-Stream Fusion Gate split-bar ($g$ Context vs. $1-g$ Morphology).
     - Full Agglutinative Morpheme Decomposition Table (Prefix, Stem, Suffix Chain, Grammatical Function).
     - Dynamic linguistic reasoning explanation bullets.
   - **Tab 2: Auto-FST & Slang Lab**:
     - **Interactive Morpheme Visual Diff**: Side-by-side highlighting showing normalized Russian-layout letters in sky blue (`#e0f2fe`) and newly induced slang roots in soft violet (`#f3e8ff`).
     - Auto-Inductions Table (new stems, vowel harmony, coda profile, 100% guard agreement).
   - **Tab 3: Fact Verification & Trust Matrix**:
     - 2D Trust Matrix coordinate chart ($T_{\text{fact}}$ vs $T_{\text{gen}}$) with quadrant classification.
     - Atomic factual claims & Wikipedia evidence citations table.

---

## 5. CSS Canvas Enforcement (Notion Light)

```css
/* Force Pure Light Apple / Notion Canvas */
html, body, .gradio-container, .dark {
    color-scheme: light !important;
    background-color: #fbfbfa !important;
    color: #18181b !important;
}

/* Remove dark-mode overrides */
.dark .gradio-container,
.dark .hero-card,
.dark .panel-box,
.dark .workspace-left-col,
.dark .workspace-right-col {
    background-color: #fbfbfa !important;
    color: #18181b !important;
}

/* De-boxified layout */
.studio-card {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 8px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
}
```

---

## 6. Testing & Verification Plan

1. **Scaffold & Layout Verification**:
   - Update `tests/test_ui_two_column_workspace.py` to assert balanced columns (`scale=5` / `scale=5`), single primary text input, visual heatmap in left column, and right-column tabs.
2. **English-Only Audit**:
   - Assert zero bilingual slash labels (`/ Шолу`, `/ Морфология`, `/ Тексеру`) across all UI elements.
3. **Export Functionality Test**:
   - Assert `generate_audit_markdown(res)` produces a valid markdown string with verdict, confidence, sentence table, Auto-FST inductions, and FEVER evidence.
4. **Conference Paper Invariance**:
   - `git diff origin/main..HEAD -- aist2026/paper.tex papers/kazakh_fever_conference/` must be 0 diff lines.
5. **Zero Emojis Audit**:
   - Regex audit verifying 0 decorative emojis across modified files.
