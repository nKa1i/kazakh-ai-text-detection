# Design Specification: Master's Thesis Presentation Verification Checklist & Audit Guide

**Date**: 2026-09-17  
**Author**: Daulet Anekesh (大雷) / Antigravity Pair Programming  
**Branch**: `feat/kaz-presentation-cloning-and-figures`  
**Target Presentations**: 
- Primary: `C:\Users\Roza\Desktop\AnekeshD_Progress.pptx`
- Mirror: `C:\Users\Roza\Desktop\Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx`
- Repo: `Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx`

---

## 1. Overview & Goals

This specification establishes the architecture, contents, and verification criteria for a comprehensive Master's Thesis Presentation Checklist and Audit Guide (`docs/thesis_presentation_verification_checklist.md`). 

The document empowers Daulet to systematically verify that all 80+ academic recommendations, structural realignments, metric adjustments, terminology updates, and visual redesigns requested by Professor Guo and her assistant have been accurately and rigorously implemented across the 23-slide deck.

### Success Criteria
1. **Thematic Completeness**: Maps every recommendation across all 6 core feedback pillars (Storyline, Terminology, Experimental Metrics, Kazakh-FEVER, Slide Hygiene, Strategic Guidance).
2. **Slide-by-Slide Verification**: Details exact on-slide text strings, numbers, formulas, tables, cards, and diagrams to inspect on each slide (Slides 01–23).
3. **Typography & Aesthetics**: Verifies 100% Times New Roman compliance (0 Arial instances), sequential Table of Contents XML ordering (`01 -> 02 -> 03 -> 04 -> 05 -> 06`), and absence of promotional buzzwords.
4. **Speaker Script Alignment**: Cross-references every slide with the corresponding spoken English defense narrative in `thesis_presentation_speaker_notes_prof_guo.md`.
5. **Direct Visual Linkage**: Embeds relative and absolute links to 1080p slide preview images (`ppt_images/slide_01.png` through `slide_23.png`).

---

## 2. Document Architecture

The checklist document is organized into two primary sections:

```
thesis_presentation_verification_checklist.md
├── 1. Executive Summary & Verification Protocol
├── 2. Part 1: Thematic Pillar Master Checklist (6 Pillars)
│   ├── Pillar 1: Research Storyline & Topic 2 Restoration
│   ├── Pillar 2: Architectural Rigor & Terminology Unification
│   ├── Pillar 3: Experimental Metrics & Statistical Rigor
│   ├── Pillar 4: Topic 3 Kazakh-FEVER & 2D Trust Matrix
│   ├── Pillar 5: Slide Hygiene & Typography Standardization
│   └── Pillar 6: Strategic Consultation Roadmap & Speaker Notes
├── 3. Part 2: Slide-by-Slide Visual & Content Audit Guide (Slides 01–23)
│   ├── Slide 01: Title Slide & Credentials
│   ├── Slide 02: Sequential Table of Contents
│   ├── Slide 03: Research Background & Agglutinative Vulnerability
│   ├── Slide 04: Problem Statement & Subword Root Fragmentation
│   ├── Slide 05: Related Work — Detection Baselines & Kazakh Collapse
│   ├── Slide 06: Related Work — LLM Hallucination Verification Gaps
│   ├── Slide 07: Methodological Framework (Figure 3)
│   ├── Slide 08: Topic 1 Architecture — Dual-Stream Gated Fusion
│   ├── Slide 09: Topic 2 Architecture — Robust Generalization & Rewriting
│   ├── Slide 10: Topic 3 Architecture — Kazakh-FEVER & Trust Matrix
│   ├── Slide 11: Experimental Setup — MAGE Protocol & 314 Tests
│   ├── Slide 12: Topic 1 Results — Blindspot Gain (+42.18 pp)
│   ├── Slide 13: Topic 2 Results — Generator Robustness & Chunking
│   ├── Slide 14: Topic 3 Results — Kazakh-FEVER Benchmark
│   ├── Slide 15: Topic 3 Results — Four-Quadrant Trust Matrix
│   ├── Slide 16: Component Ablation Studies & Inductive Isolation
│   ├── Slide 17: System Demo — Interactive 4-Tab Gradio Prototype
│   ├── Slide 18: System Demo — Cloud Packaging & 314 Tests
│   ├── Slide 19: Conclusion — Contributions & Writing Progress (~85%)
│   ├── Slide 20: Current Progress — Springer LNCS & Agenda
│   ├── Slide 21: Discussion — Strategic Consultation for Prof. Guo
│   ├── Slide 22: Committee Comments & Academic Chapter References
│   └── Slide 23: Closing Institutional Title
└── 4. Verification Script & Automated Audit Reference
```

---

## 3. Part 1: Thematic Pillar Specifications

### Pillar 1: Narrative & Research Framing
- **Core Realignment**: Topic 2 restored to "Robust Generalization & Adversarial Rewriting Defense" (generator shift, cross-domain shift, paraphrasing, synonym replacement, human-edited AI text).
- **Engineering Module Status**: Sentence-preserving chunking ($L=256$) and Top-$K$ worst-chunk pooling ($K = \max(1, \lfloor 0.25 N \rfloor)$) re-positioned as supporting engineering extensions.
- **Storyline**: 3-Pillar Scientific Flow: Topic 1 (Morphological Detection) $\rightarrow$ Topic 2 (Robust Generalization) $\rightarrow$ Topic 3 (Fact Verification) $\rightarrow$ Supporting Chapter 6 (Engineering Prototype).
- **Thesis Completion**: Calibrated at ~85%, explicitly citing the remaining 15% as executing adversarial rewriting experiments and dense retrieval scaling for Paper 2.

### Pillar 2: Architectural Rigor & Terminology Unification
- **Retired Terms**: Strictly 0 occurrences of "Cross-Attention", "Cross-Gate", "catastrophic", "breakthrough", "perfect", "production-ready", "eliminates domain collapse".
- **Gated Fusion**: Formally defined as "Dual-Stream Morphology-Aware Gated Fusion" with mathematical equations:
  $$\mathbf{g} = \sigma(\mathbf{W}_g [\mathbf{h}_{sem}; \mathbf{W}_{proj} \mathbf{h}_{morph}] + \mathbf{b}_g)$$
  $$\mathbf{h}_{fused} = \mathbf{g} \odot \mathbf{h}_{sem} + (1 - \mathbf{g}) \odot (\mathbf{W}_{proj} \mathbf{h}_{morph})$$
- **FST Grounding**: 83-rule FST transducer grounded on 5,000 dictionary headwords (99.2% inflectional coverage) with illustrative gating prior $g \approx 0.68$.
- **SupCon Formulation**: Supervised contrastive loss defined as Human vs. AI latent separation in $\mathbb{R}^{128}$; AI-original vs. AI-paraphrased pairs clearly scoped for Paper 2.

### Pillar 3: Experimental Metrics & Statistical Rigor
- **AUC vs. Macro-F1**: Explicit separation of ROC-AUC (Mean: $99.80 \pm 0.15\%$, Q1: $99.85\%$) and Macro-F1 ($99.24\%$) across cards and Table 124.
- **Absolute Percentage Points**: Performance deltas reported strictly as `+42.18 pp` and `+21.55 pp`.
- **Length Buckets & FPR**: Disclosed short-text length distributions ($<50$, $50\text{--}150$, $>150$ tokens) and low human False Positive Rate ($\text{FPR} \le 2.1\%$).
- **Protocol Re-titling**: Renamed "ACL Kaz-MAGE Benchmark" to "Kazakh adaptation of a MAGE-style evaluation protocol".

### Pillar 4: Topic 3 Kazakh-FEVER Rigor & Risk Calibration
- **Benchmark Parameters**: 36 curated Wikipedia articles, 120 balanced claims (40 SUPPORTS, 40 REFUTES, 40 NOT ENOUGH INFO).
- **Metric Transparency**: Disclosed 100% NLI Macro-F1 alongside strict Joint FEVER ($66.67\%$, 91.67% evidence recall) with an open discussion of the BM25 retrieval bottleneck.
- **Two-Dimensional Coordinates**: Formulated 2D risk coordinates $(\hat{y}_{AI}, R_{fact}) \in [0, 1]^2$ as the primary output; composite scalar risk $R_{comp}$ framed strictly as an auxiliary ranking tool.
- **Epistemic Uncertainty**: Treated NEI claims ($R_{fact} = 0.50$) as evidence absence rather than falsehood, separating Q2 (Human Rumor) from Q3 (Accurate AI).

### Pillar 5: Slide Deck Hygiene & Typography Consistency
- **TOC Order**: Sequential reading and screen-reader tab order on Slide 2: `01 -> 02 -> 03 -> 04 -> 05 -> 06`.
- **Template Hygiene**: Deleted leftover notes referencing "Fan Qianyue" and "Prof. Chen Yaxing" on Slide 1.
- **Typography**: 100% **Times New Roman** across all 23 slides (0 Arial instances), preserving Microsoft YaHei for Chinese characters and keeping all manual font sizes untouched.
- **Academic Status Badges**: Replaced absolute `[✓] RESOLVED` badges on Slide 22 with thesis chapter references (`Addressed in Thesis Chapter 4`).

### Pillar 6: Strategic Consultation Roadmap
- **Discussion Focus**: Slide 21 oriented around 3 core strategic guidance questions for Prof. Guo:
  1. Adversarial rewriting attack benchmarks for Paper 2.
  2. Dense neural retrieval scaling for Kazakh-FEVER.
  3. Target venue selection for Paper 2.
- **Speaker Notes**: 100% academic spoken English across all 23 slides and 7 defense Q&A answers.

---

## 6. Part 2: Slide-by-Slide Audit Guide Specifications

For each slide (1–23), the audit guide specifies:
1. **Header & Context**: Slide number, chapter name, title, and subtitle.
2. **Professor Guo's Feedback**: Core issue raised.
3. **Implemented Action**: Technical remedy applied.
4. **On-Screen Verification Points**: Specific text strings, table cell values, formulas, and visual elements to inspect.
5. **Speaker Notes Anchor**: 1-sentence verification of the corresponding English script.
6. **Preview Link**: Link to `ppt_images/slide_XX.png`.

---

## 7. Implementation & Delivery Plan

1. Generate `docs/thesis_presentation_verification_checklist.md` with complete, non-truncated content.
2. Mirror the checklist artifact to `<appDataDir>\brain\<conversation-id>/thesis_presentation_verification_checklist.md`.
3. Verify all automated unit tests (`tests/test_presentation_content.py`, `tests/test_presentation_methodology.py`, `tests/test_presentation_speaker_notes.py`) remain passing (322/322 passed).
4. Commit the spec and checklist files to git.
