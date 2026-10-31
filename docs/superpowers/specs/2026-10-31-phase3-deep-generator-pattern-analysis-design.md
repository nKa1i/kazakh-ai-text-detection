# Phase 3: Deep Pattern Analysis & Comparative Generator Study Design Specification

## Executive Summary
This specification defines the architectural design, linguistic feature extraction engine, statistical pattern prober, comparative report generator, publication-grade visualization suite, and Gradio UI integration for **Phase 3: Deep Pattern Analysis & Comparative Generator Study**.

While Phase 2 established open-set attribution and Leave-One-Generator-Out (LOGO) unseen detection using an 8-dimensional centroid model, Phase 3 probes the deeper linguistic question: **Why and how do frontier and sovereign LLM families systematically diverge from authentic Kazakh human prose, and what invariant morphological/syntactic habits expose unseen generators?**

The methodology adapts insights from recent English NLP literature on machine-generated text attribution (TuringBench [Uchendu et al.], MAGE [Li et al.], Ghostbuster [Verma et al.], and RAID [Liang et al.]) and grounds them in the distinctive properties of Kazakh agglutinative morphology, finite state phonology, and syntactic calquing.

---

## 1. Theoretical Framework: Kazakh LLM Signatures

Agglutinative languages pose unique generation challenges for token-based autoregressive decoders:

1. **Morphological Chain Regularity vs. Human Burstiness**:
   - *Human Kazakh*: Displays extreme variance in morpheme chain lengths (ranging from bare monosyllabic stems to 4-5 stacked affixes: e.g. `бас-қару-шы-лар-ымыз-дың`), with high morphological entropy.
   - *LLM Kazakh*: Exhibits hyper-regular, low-variance morpheme chains (averaging 1.8-2.2 affixes per token) avoiding complex possessive-case stacking.

2. **Analytic Copula Crutches (`болып табылады`) vs. Synthetic Predication**:
   - *Human Kazakh*: Employs direct synthetic inflectional suffixes for predicates (`келеді`, `анықтады`, `жүргізілді`).
   - *LLM Kazakh*: Over-relies on bureaucratic, Russian/English-translated analytic constructions (`болып табылады`, `болып келеді`, `болып саналады`, `қызмет атқарады`).

3. **Discourse Transition Rigidity**:
   - *LLM Kazakh*: Systematically inserts stereotyped transition formulas (`Қорытындылай келе`, `Атап өткен жөн`, `Осыған байланысты`, `Сонымен қатар`) at fixed paragraph-initial coordinates with near-zero positional entropy.
   - *Human Kazakh*: Uses diverse, organic rhetorical transitions (`Демек`, `Алайда`, `Сөйтіп`, `Шындығында`).

4. **Syntactic Calquing & SOV Word Order Compliance**:
   - *Human Kazakh*: Strictly enforces Subject-Object-Verb (SOV) order, placing finite predicates at clause and sentence boundaries.
   - *LLM Kazakh*: Frequently leaks Subject-Verb-Object (SVO) structures through literal translation from English or Russian training data, placing auxiliary verbs medially before long prepositional or nominal clauses.

5. **Loan Assimilation & Vowel Harmony Violations**:
   - *Human Kazakh*: Either assimilates loanwords or selects harmonizing suffixes following the final syllable's front/back vowel status.
   - *LLM Kazakh*: Inaccurately pairs front-vowel case endings (`-ге`, `-де`) with back-vowel foreign stems, or generates unassimilated English lemmas.

---

## 2. Component Architecture

```
                           Raw Kazakh Document / Corpus
                                        │
                                        ▼
                           Auto-FST v2 Dynamic Overlay
                    (DynamicFSTOverlay + RussianLayoutNormalizer)
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │       GeneratorPatternProber              │
                  │       (models/pattern_prober.py)          │
                  │                                           │
                  │ 1. Chain Length Entropy & Variance        │
                  │ 2. Copula-to-Synthetic Predicate Ratio    │
                  │ 3. Discourse Transition Formula Density   │
                  │ 4. SOV Compliance Rate (Final-Verb Ratio) │
                  │ 5. Loanword Assimilation Ratio            │
                  │ 6. Lexical Type-Token Ratio & Burstiness  │
                  │ 7. Passive & Causative Suffix Saturation  │
                  │ 8. Sentence Length Uniformity             │
                  └─────────────────────┬─────────────────────┘
                                        │
              ┌─────────────────────────┴─────────────────────────┐
              ▼                                                   ▼
┌───────────────────────────────┐           ┌───────────────────────────────────┐
│ GeneratorFingerprintVisualizer│           │ Pattern Report & Profile Builder  │
│ (models/generator_viz.py)     │           │ (scripts/generate_pattern_report) │
│ - 8-Axis SVG Radar Charts     │           │ - Statistical Profiles (JSON)     │
│ - Deviation / Z-Score Bar Slices│          │ - Publication Markdown/LaTeX      │
│ - Zero external rendering libs│           │   Comparative Diagnostic Report   │
└─────────────┬─────────────────┘           └─────────────────┬─────────────────┘
              │                                               │
              └─────────────────────────┬─────────────────────┘
                                        ▼
                        Gradio UI & HF Spaces Integration
                       (Interactive Linguistic Inspector)
```

---

## 3. Module Specifications

### 3.1. `GeneratorPatternProber` (`models/pattern_prober.py`)
- **Inputs**: Arbitrary Kazakh document string or tokenized chunk list.
- **Methods**:
  - `probe(text: str) -> PatternProbeResult`: Computes deep morphosyntactic and syntactic metrics.
  - `compute_deviation(text: str, family: Optional[str] = None) -> Dict[str, float]`: Computes standardized z-scores relative to human baseline.
  - `explain_patterns(text: str) -> List[PatternExplanation]`: Returns human-readable diagnostic observations (e.g. `"Copula density is 2.8x higher than authentic human writing"`).
- **Extracted Metrics (`PatternProbeResult`)**:
  - `chain_len_entropy: float` — Shannon entropy of morpheme chain lengths.
  - `chain_len_variance: float` — Variance of morpheme count per word.
  - `copula_ratio: float` — Frequency of analytic copulas per 100 words.
  - `synthetic_verb_ratio: float` — Frequency of genuine synthetic finite verbs.
  - `discourse_rigidity_score: float` — Density and coordinate regularity of transition phrases.
  - `sov_compliance_score: float` — Ratio of sentences terminating in valid finite verbal suffixes.
  - `loan_assimilation_score: float` — Ratio of correctly harmonized loan stems.
  - `lexical_ttr: float` — Type-Token Ratio over sliding 100-word windows.
  - `burstiness: float` — Standard deviation of sentence length divided by mean length.

### 3.2. `GeneratorFingerprintVisualizer` (`models/generator_viz.py`)
- **Objective**: Generates clean, zero-emoji, responsive SVG radar charts without heavy external dependencies (no matplotlib or seaborn required at runtime).
- **Methods**:
  - `render_radar_chart_svg(probes: Dict[str, Dict[str, float]], labels: List[str], width: int = 500, height: int = 400) -> str`: Renders SVG containing overlaid polygons (e.g., Target Text vs. Human Baseline vs. Model Centroid).
  - `render_deviation_bars_svg(deviations: Dict[str, float], width: int = 500, height: int = 240) -> str`: Renders horizontal differential bar chart highlighting anomalous linguistic dimensions.

### 3.3. Comparative Pattern Report Suite (`scripts/generate_generator_pattern_report.py`)
- **Objective**: Evaluates samples from all 5 known families (`ISSAI-KazLLM-70B`, `Sherkala-8B`, `DeepSeek-R1`, `GPT-4o`, `Qwen-2.5-72B`), wild unseen calque generators, and authentic human texts.
- **Outputs**:
  - `data/generator_linguistic_profiles.json` — Statistical metrics (mean, std, 95% confidence intervals) per generator family.
  - `docs/reports/2026-10-31-kazakh-llm-linguistic-signatures.md` — Formal academic comparative study report containing:
    1. Cross-model morphological profile table.
    2. Primary linguistic weaknesses identified for each LLM family.
    3. Unseen detector generalization rationale.

### 3.4. Gradio UI Linguistic Fingerprint Inspector (`ui/app.py`, `hf_space/ui/app.py`)
- Adds a **"Linguistic Fingerprint & Diagnostics"** panel to the UI.
- Displays:
  - SVG Radar Chart comparing the analyzed document against the Human Baseline.
  - Anomaly Badges identifying the strongest machine signatures (e.g. "Hyper-regular chain length", "Excessive analytic copula").
  - Explanation bullet list in Kazakh / English.

---

## 4. Test Strategy & Acceptance Criteria

1. **Unit Tests**:
   - `tests/test_pattern_prober.py`: Asserts correct extraction of all 8 pattern metrics on human vs. synthetic texts; verifies z-score deviations.
   - `tests/test_generator_viz.py`: Asserts generated SVG strings are valid XML, contain polygon coordinates, and render clean without throwing exceptions.
   - `tests/test_generate_pattern_report.py`: Asserts report generation script executes with `--dry-run` and generates valid JSON and Markdown artifacts.
2. **Acceptance Criteria**:
   - Human prose displays significantly higher `chain_len_entropy` and lower `copula_ratio` than synthetic text ($p < 0.001$).
   - SVG generator produces self-contained, theme-adaptive vectors with zero external library requirements.
   - All existing tests in `tests/test_hf_space_bundle.py` and `tests/test_ui_integration.py` continue to pass with zero regressions.
   - Zero decorative emojis across all code, tests, docstrings, commits, and UI elements.
