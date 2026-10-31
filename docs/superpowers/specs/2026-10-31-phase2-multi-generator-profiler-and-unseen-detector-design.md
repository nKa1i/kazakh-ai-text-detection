# Phase 2: Multi-Model Generator Profiling & Unseen Detector Design Specification

## Executive Summary
This specification defines the architecture, algorithmic morphosyntactic feature extraction, open-set generator attribution, generator-invariant unseen detection, empirical Leave-One-Generator-Out (LOGO) benchmark suite, and Gradio UI integration for **Phase 2: Multi-Model Generator Profiling & Unseen Detector**.

The design decouples binary AI detection from generator family profiling so that:
1. **Stage 1 (Generator-Invariant Binary Detector)** reliably flags synthetic Kazakh prose even when produced by an unseen, out-of-distribution LLM family whose surface discourse markers were never seen during calibration.
2. **Stage 2 (Open-Set Multi-Model Morphological Profiler)** extracts an 8-dimensional morphosyntactic fingerprint vector powered by `DynamicFSTOverlay` (`src/auto_fst/overlay.py`) to attribute AI-generated texts to known LLM families (`ISSAI-KazLLM-70B`, `Sherkala-8B`, `DeepSeek-R1`, `GPT-4o`, `Qwen-2.5-72B`) or explicitly flag them as **`Unseen / Out-of-Distribution LLM`** when standardized feature distance exceeds the open-set threshold $\tau_{\text{ood}}$.

---

## 1. System Architecture

```
                          Raw Kazakh Document
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
     Auto-FST v2 Normalization          Sliding-Window Chunker
   (DynamicFSTOverlay + Restorer)     (SentencePreservingChunker)
                   │                               │
                   └───────────────┬───────────────┘
                                   ▼
     ┌───────────────────────────────────────────────────────────┐
     │ STAGE 1: Generator-Invariant Binary Detector              │
     │ - Evaluates universal agglutinative & syntactic invariants│
     │   (suffix entropy, burstiness, calque/copula density,     │
     │    conversational particle asymmetry, dynamic gate g)     │
     │ - Outputs: document_ai_probability, verdict, chunks       │
     └─────────────────────────────┬─────────────────────────────┘
                                   │
          ┌────────────────────────┴────────────────────────┐
          │ If Authentic Human (P_ai < 0.50)                │ If AI / Hybrid (P_ai >= 0.50)
          ▼                                                 ▼
  [Source: Authentic Human]        ┌───────────────────────────────────────────────────┐
                                   │ STAGE 2: Open-Set Morphological Generator Profiler│
                                   │ (MultiModelGeneratorProfiler)                     │
                                   │ - Extracts 8D Morphosyntactic Fingerprint Vector  │
                                   │ - Computes standardized distance d(x, mu_k)       │
                                   │   against 5 Known Generator Family Centroids:     │
                                   │   1. ISSAI-KazLLM-70B (Sovereign Policy/Admin)    │
                                   │   2. Sherkala-8B (Sovereign Encyclopedic)         │
                                   │   3. DeepSeek-R1 (Analytical Reasoning/CoT)       │
                                   │   4. GPT-4o (Frontier Technical/Balanced)         │
                                   │   5. Qwen-2.5-72B (Multilingual Diplomatic/Macro) │
                                   │ - Open-Set Gate: if min_k d(x, mu_k) > tau_ood    │
                                   │   -> Flags "Unseen / Out-of-Distribution LLM"     │
                                   └───────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1. Multi-Model Morphological Profiler (`models/generator_profiler.py`)
- **Data Contract (`GeneratorProfileResult` dataclass)**:
  - `predicted_family: str` — Top matched family name (`"ISSAI-KazLLM-70B"`, `"Sherkala-8B"`, `"DeepSeek-R1"`, `"GPT-4o"`, `"Qwen-2.5-72B"`, `"Unseen / Out-of-Distribution LLM"`, or `"Authentic Human"`).
  - `archetype_label: str` — Descriptive category (`"Sovereign Kazakh Policy LLM"`, `"Sovereign Kazakh Encyclopedic LLM"`, `"Analytical Reasoning LLM"`, `"Frontier Multilingual LLM"`, `"Open-Weights Multilingual LLM"`, `"Unseen Cross-Lingual / OOD LLM"`, or `"Human Authorship"`).
  - `is_unseen_generator: bool` — `True` when the document is classified as AI/Hybrid and its minimum centroid distance exceeds $\tau_{\text{ood}}$.
  - `confidence: float` — Attribution confidence in $[0, 1]$.
  - `ood_distance: float` — Minimum standardized weighted Euclidean distance $d_{\min}(\mathbf{x}, \boldsymbol{\mu}_k)$ to known generator centroids.
  - `ood_threshold: float` — Calibrated open-set rejection threshold $\tau_{\text{ood}}$.
  - `family_probabilities: Dict[str, float]` — Normalized similarity distribution across the 5 known generator families.
  - `feature_vector: Dict[str, float]` — Named 8-dimensional morphosyntactic feature dictionary.
  - `dominant_traits: List[str]` — Top 2-3 human-readable morphological/syntactic signatures explaining the profile.
  - `to_dict() -> Dict[str, Any]` — JSON-serializable representation.

- **8-Dimensional Morphosyntactic Feature Vector (`extract_morphosyntactic_profile(text)`)**:
  All 8 features are computed algorithmically from `DynamicFSTOverlay` segmentation and normalized text tokens:
  1. `abstract_deriv_density`: Frequency of abstract nominal/adjectival derivational suffixes (`-лық`, `-лік`, `-дық`, `-дік`, `-тық`, `-тік`) per word.
  2. `passive_participle_density`: Frequency of passive/progressive verbal suffixes and bureaucratic converbs (`-луда`, `-луде`, `-лды`, `-лді`, `-ғандықтан`, `-гендіктен`, `-қандықтан`, `-кендіктен`, `-атын`, `-етін`) per word.
  3. `logical_reasoning_density`: Frequency of deductive/analytical reasoning stems and constructions (`көрсеткендей`, `сәйкес`, `нәтижесінде`, `кезеңінде`, `алгоритм`, `модель`, `теориялық`, `талдау`, `гипотеза`, `теңдеу`, `параметр`, `оңтайландыру`) per word.
  4. `diplomatic_macro_density`: Frequency of institutional/macroeconomic framing stems (`ынтымақтастық`, `әртараптандыру`, `тартымдылығын`, `серпін`, `алғышарты`, `кепілдік`, `инвестиция`, `айналымын`, `әріптестік`, `шалғай`) per word.
  5. `infrastructural_domain_density`: Frequency of regional/ecological/public-sector stems (`табиғи`, `ауыл`, `шаруашылығын`, `медициналық`, `энергетикалық`, `көлік`, `су`, `кәсіптік`, `мәдени`, `мұраны`, `азық-түлік`) per word.
  6. `calque_translationese_rate`: Frequency of literal cross-lingual calque constructions (`өз кезегінде`, `тұрғысынан алғанда`, `шешуші рөлге ие`, `бірі болып келеді`, `үлкен мән беріледі`, `назарға алынады`, `қол жеткізуге болады`, `жағдайлар жасалуда`) per sentence.
  7. `affix_depth_per_word`: Mean number of FST-segmented morpheme suffixes per word.
  8. `sentence_length_uniformity`: $1.0 / (1.0 + \text{CV}_{\text{sent}})$, where $\text{CV}_{\text{sent}}$ is the coefficient of variation of sentence word counts (high uniformity $\to 1.0$ in synthetic text; lower in bursty human text).

- **Leave-One-Generator-Out Support (`active_families` parameter)**:
  `MultiModelGeneratorProfiler(fst_overlay=None, ood_threshold=2.15, active_families=None)` allows restricting the known centroid bank to any subset of families (e.g., 4 seen families in a LOGO fold) so held-out generators can be tested rigorously for open-set OOD detection.

### 2.2. Generator-Invariant Unseen Detection (`models/heuristic_detector.py` & `kaz_mage/document.py`)
- **`DocumentAnalysisResult` Extension (`kaz_mage/document.py`)**:
  - Add optional field `generator_profile: Optional[Dict[str, Any]] = None` to `DocumentAnalysisResult` (and include in `to_dict()`).
- **Generator-Invariant Structural Scoring (`models/heuristic_detector.py`)**:
  - Expand `_score_chunk()` so that even if `ai_hits == 0` (or when lexical markers from a held-out generator are masked/absent), a chunk is scored as synthetic ($\ge 0.9982$) when it exhibits:
    - Zero human conversational suffixes (`-ғой`, `-қой`, `-ау`, 1st/2nd person informal endings) and zero colloquial lexical markers, AND
    - Compound formal agglutinative structure: e.g., high abstract derivational + passive/participle suffix density, calque translationese constructions, or high multi-syllabic word length ($\ge 7.6$ chars/word) combined with formal nominal/verbal FST suffix chains ($\ge 0.55$ suffixes/word).
  - Preserve strict backward compatibility with `calibrated_threshold = 0.9980`: single-copula casual human sentences (`"Бұл бірінші қазақша сөйлем. Ал бұл екінші сөйлем болып табылады."`) and all 6 `QUICK_SAMPLES`, `PRESET_SAMPLES`, and `VERIFICATION_PRESET_SAMPLES` verdicts remain unchanged.
  - Attach `result.generator_profile = self.profiler.profile_document(text, ai_probability=result.document_ai_probability, verdict=result.verdict).to_dict()` inside `predict_document()`.

### 2.3. 5-Fold LOGO + 6th Wild Unseen Benchmark (`scripts/evaluate_multi_generator_profiler.py`)
- **Dataset Construction**:
  - Uses the 5 core generator families from `scripts/build_hybrid_mixed_benchmark.py` (`ISSAI-KazLLM-70B`, `Sherkala-8B`, `DeepSeek-R1`, `GPT-4o`, `Qwen-2.5-72B`) plus a 6th held-out wild generator archetype `Llama-3.3-70B-Calque` (12 multi-sentence calque-heavy cross-lingual synthetic Kazakh passages that avoid the standard `AI_MARKERS` list) and 36 authentic human passages across both clean and Russian-layout corrupted regimes.
- **Evaluations**:
  1. **5-Fold Leave-One-Generator-Out (LOGO) + 6th Wild Unseen Binary Detection**:
     - Compares a **Closed-Set Lexical Baseline** (which relies only on seen generator transition phrases) against the **Proposed Morpho-Invariant Detector** across all 5 LOGO folds and the 6th wild family.
     - Asserts proposed detector achieves $\ge 92.0\%$ unseen recall and $\ge 0.95$ ROC-AUC across held-out families, beating the lexical baseline by $\ge 25.0$ percentage points on unseen calque/held-out generators while maintaining $\le 5.0\%$ human FPR.
  2. **Multi-Generator Attribution & Open-Set OOD Rejection**:
     - Evaluates closed-set 5-family attribution accuracy ($\ge 80.0\%$) and open-set OOD rejection recall ($\ge 85.0\%$) when a generator is held out from `active_families` or when `Llama-3.3-70B-Calque` is evaluated.
- **Generated Artifacts**:
  - `data/multi_generator_logo_results.json`
  - `data/chart_multi_generator_fingerprints_and_logo.png` (300-DPI 2-panel publication chart: Panel A = Morphosyntactic Fingerprint Heatmap across 6 Generator Families + Human; Panel B = Seen vs. Unseen LOGO Binary Recall & Open-Set OOD Detection Rate).

### 2.4. Gradio UI & `hf_space/` Integration (`ui/highlighting.py`, `ui/app.py`, `hf_space/`)
- **`render_generator_profile_card(generator_profile, lang="en")` in `ui/highlighting.py`**:
  - Renders an XSS-safe (`html.escape(..., quote=True)`), dark-charcoal HTML card displaying:
    - Predicted Generator Family badge (or `[UNSEEN / OOD GENERATOR]` badge when `is_unseen_generator=True`, or `[HUMAN AUTHORSHIP]` when verdict is human).
    - Archetype classification, attribution confidence / OOD distance metric, and top 3 family similarity bars.
    - Dominant morphosyntactic traits pills.
- **Integration in `ui/app.py` and `hf_space/ui/app.py`**:
  - `render_executive_summary_card()` accepts an optional `generator_profile: Optional[Dict[str, Any]] = None` parameter and appends `render_generator_profile_card()` when present, preserving the exact 8-tuple return signature of `handle_analyze_document()` and 63-element return signature of `switch_ui_language()`.
  - `generate_audit_markdown()` includes a `## Multi-Model Generator Fingerprint Profile` section when `generator_profile` is available.
  - Full dual-tree synchronization across `models/`, `kaz_mage/`, `ui/`, and `hf_space/`.

---

## 3. Error Handling, Anti-Overfitting & Guardrails
1. **Zero Decorative Emojis**: Strictly zero emojis in all Python code, HTML templates, figure labels, markdown reports, and git commit messages (`GEMINI.md` Rule 4).
2. **Anti-Shortcut Guardrail**: `MultiModelGeneratorProfiler` and `OfflineHeuristicDetector` must compute features strictly from morphological segmentation (`DynamicFSTOverlay`), suffix inventories, and structural syntax—never by hashing or matching benchmark sentence strings.
3. **Short / Empty Input Safety**: Empty or whitespace-only inputs return `GeneratorProfileResult(predicted_family="Awaiting Input", is_unseen_generator=False, confidence=0.0)` without division-by-zero errors.
