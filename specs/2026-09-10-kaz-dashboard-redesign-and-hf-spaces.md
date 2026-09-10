# Kaz-Dashboard Redesign & Hugging Face Spaces Deployment Design Spec

**Date:** September 10, 2026  
**Target:** Production-Grade Academic UI & 1-Click Hugging Face Spaces Cloud Deployment  
**Status:** Approved by User  

---

## 1. Problem Statement & Motivation

The existing Kazakh AI-Text Detection interface successfully implements sentence-level heatmaps, dynamic gate visualization, and FST word breakdowns. However, based on user review and academic presentation requirements for Prof. Guo and research assistant Wang Na, several usability and aesthetic limitations exist:

1. **Visual Tone & Clutter**: Excessive emojis across headers, buttons, and badges create an informal, overly casual appearance rather than an institutional, publication-grade aesthetic.
2. **Workflow & Navigation Friction**: Presets were isolated in a collapsible accordion, forcing users to click, select, load, and scroll before running an analysis.
3. **Lack of Dynamic Plain-Language Explanations**: Numerical probabilities (e.g. `99.8% AI`) require technical interpretation. Non-specialists need concise, human-readable bullet points explaining *why* the text was flagged, grounded in real linguistic features (lexical diversity, discourse connectors, agglutinative suffix chains).
4. **Cloud Accessibility**: Reviewers and professors need a direct, zero-setup public web link (Hugging Face Spaces) rather than running Python locally.

---

## 2. Architecture & Components

```
                               ┌──────────────────────────────────────────────┐
                               │             Modernized Web UI                │
                               │        ui/app.py (Tabbed Layout)             │
                               ├──────────────────────┬───────────────────────┤
                               │ 🇰🇿 Қазақша           │ 🇬🇧 English            │
                               └──────────┬───────────┴───────────┬───────────┘
                                          │                       │
                ┌─────────────────────────┼───────────────────────┴────────────────────────┐
                ▼                         ▼                                                ▼
     Tab 1: Detection & Explain       Tab 2: Morphological FST Lab            Tab 3: Benchmark & Report
     - 6 Quick Sample Pills           - Interactive Word/Sentence Parser      - ACL 2024 MAGE 2x2 Matrix
     - Multiline Input + File Upload  - Suffix Peeling & Tagging              - Kaggle Dual T4 Benchmarks
     - Visual Confidence Meter        - Dynamic Gate Split Bar (BERT vs FST)  - Academic Methodology
     - Plain-Language Why Bullets     - Agglutinative Affix Badges            - Citation References
     - XSS-Safe Sentence Heatmap
     - Inline Sentence Inspector
                │
                ▼
     ui/linguistic_explainer.py
     - Computes Type-Token Ratio (TTR)
     - Counts formal LLM discourse markers vs colloquial tokens
     - Evaluates FST affix chain complexity
     - Generates 2-3 dynamic, grounded bullet reasons
```

### Component Details

### 1. `ui/linguistic_explainer.py` [NEW]
- **Purpose**: Computes linguistic diagnostics on the text to produce dynamic, plain-language explanations.
- **Interfaces**:
  - `generate_linguistic_explanation(text: str, doc_result: DocumentAnalysisResult, lang: str = "kz") -> List[str]`
- **Linguistic Signals Analyzed**:
  1. *Lexical Variety*: Type-Token Ratio ($\text{TTR} = \frac{|V|}{N}$). Low TTR in long text indicates synthetic repetition; high natural variation indicates human composition.
  2. *Formulaic Discourse Markers*: Frequency of typical LLM connectors (*қорытындылай келе*, *айта кету керек*, *маңызды рөл атқарады*, *осыған орай*, *жүйелі түрде*, *айтарлықтай*).
  3. *Agglutinative Suffix Density*: Average affixes per word peeled by `AdvancedKazakhFSTAnalyzer`.
  4. *Colloquial & Consumer Tokens*: Presence of natural conversational or e-commerce words (*керемет*, *жақсы*, *рахмет*, *алдым*, *ұнады*, *жеткізу*, *доставка*, *каспи*).
- **Output**: Returns a list of 2–3 concise sentences in Kazakh or English.

### 2. `ui/app.py` Redesign [MODIFY]
- **Visual Aesthetic**:
  - Strip all decorative emojis from titles, buttons, badges, and headers.
  - Clean typographic hierarchy (Inter/Roboto sans-serif, high-contrast labels, soft slate borders `1px solid #e2e8f0`).
  - Refined status badges: `[AUTHENTIC HUMAN]`, `[PARTIALLY AI / HYBRID]`, `[MACHINE-GENERATED]`.
  - Gradient confidence bar: Smooth horizontal progress bar showing probability from 0% (Human Green) to 100% (AI Red).
- **Tabbed Layout**:
  - **Tab 1: Detection & Explainability**:
    - Quick Preset Pills: 6 sleek buttons directly above the input box for 1-click loading and testing.
    - Input Textbox & File Drag-and-Drop (.txt, .docx, .pdf up to 10 MB).
    - Status Banner & Visual Confidence Progress Bar.
    - "Why This Verdict" Plain-Language Explanation Card.
    - XSS-Safe Sentence Heatmap with clean three-tier legend.
    - Inline Sentence Inspector card directly below the heatmap.
  - **Tab 2: Morphological FST Lab**:
    - Dedicated laboratory interface to analyze any Kazakh sentence or phrase.
    - Dynamic Gate Fusion Bar ($\mathbf{g}$ vs. $1 - \mathbf{g}$).
    - Agglutinative Word Breakdown Table (Root, POS, Case, Plural, Possessive, Verbal Tense/Person badges).
  - **Tab 3: Benchmark & Academic Methodology**:
    - Complete documentation of the ACL 2024 MAGE $2 \times 2$ evaluation protocol.
    - Kaggle Dual Tesla T4 benchmark comparison table.
    - Citation and thesis context for Prof. Guo and peer reviewers.

### 3. `hf_space/` Standalone Cloud Bundle [NEW]
- **Directory Structure**:
  - `hf_space/app.py`: Standalone Gradio entrypoint configured for Hugging Face Spaces.
  - `hf_space/README.md`: Space configuration metadata:
    ```yaml
    ---
    title: Kazakh AI-Text Detector & Explainability System
    emoji: 🔍
    colorFrom: blue
    colorTo: green
    sdk: gradio
    sdk_version: 6.26.0
    app_file: app.py
    pinned: false
    license: apache-2.0
    ---
    ```
  - `hf_space/requirements.txt`: Minimal lightweight dependencies (`gradio>=6.0`, `python-docx`, `pypdf`, `scikit-learn`).
  - `hf_space/kaz_mage/`: Chunker, aggregator, and document data structures.
  - `hf_space/ui/`: Highlighting, presets, sentence analyzer, and linguistic explainer.
  - `hf_space/fst_analyzer.py`: Embedded FST morphological analyzer.
- **Safety & Performance**: Defaults to the ultra-fast, defensive `OfflineHeuristicDetector` ensuring instant sub-second cold starts with zero crashes on Hugging Face's free CPU tier.

### 4. `scripts/push_to_hf.py` [NEW]
- Helper script utilizing `huggingface_hub.HfApi().upload_folder(...)` to upload the contents of `hf_space/` directly to any target Hugging Face Space (`username/space-name`).

---

## 3. Global Constraints & Verification Plan

1. **Camera-Ready Protection**: Under no circumstances shall `aist2026/paper.tex` be touched or modified.
2. **Regression Integrity**: All existing 174 test cases across 27 test files must continue to pass with 0 failures and 0 errors.
3. **Mandatory XSS Prevention**: All user-controlled text rendered into HTML spans or tables must pass through `html.escape(..., quote=True)`.
4. **Defensive Operation**: Functions 100% offline without requiring active GPUs.
5. **Unit Tests**:
   - `tests/test_ui_linguistic_explainer.py`: Tests TTR computation, formulaic marker detection, and bilingual explanation output.
   - `tests/test_hf_space_bundle.py`: Verifies syntax and import integrity of the bundled `hf_space/` files using `py_compile`.
