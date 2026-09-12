# Specification: Presentation Methodological Framework Restructuring & Slide 19 Restoration

**Date:** 2026-09-12  
**Author:** Daulet (大雷) / Antigravity Pair Programming  
**Target Deck:** `C:\Users\Roza\Desktop\AnekeshD_Progress.pptx` (and project mirror `Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx`)  
**Target Notes:** `thesis_presentation_speaker_notes_prof_guo.md`  

---

## 1. Context and Problem Statement

Professor Guo provided feedback on the Master's thesis presentation:
> *"You will also need to prepare framework diagrams to illustrate the methodological innovations you intend to develop."*

Initially, an overall methodological innovations framework (Figure 15) was placed on Slide 19. However, Slide 19 is structurally located under **Chapter 6 ("总结与展望" / Summary & Outlook)**. Placing a dense technical architecture diagram in the conclusion section broke the academic narrative flow and inadvertently removed the critical **Thesis Manuscript Status (~85% complete)** and **Academic Contributions Summary** that supervisors expect to see in Chapter 6.

In contrast, **Slide 7** is titled *Research Content Overview: Three Interlocking Technical Innovations* under **Chapter 3 ("研究内容与总体方案" / Methodology)**. This is the exact logical location where a comprehensive, publication-grade overall technical roadmap belongs before drilling down into individual sub-models on Slides 8, 9, and 10.

---

## 2. Goals and Non-Goals

### Goals
1. **Relocate Comprehensive Framework to Chapter 3 (Slide 7)**:
   - Upgrade Slide 7 by embedding the publication-grade Methodological Innovations Framework (`presentation_figures/fig15_methodological_innovations_framework.png`) across the full slide width (`width=12.133"`).
   - Establish Slide 7 as the master architectural roadmap for the entire thesis.
2. **Restore Slide 19 to Conclusion & Progress Summary**:
   - Re-establish Slide 19 as *Conclusion: Summary of Thesis Contributions & Writing Progress*.
   - Present the 4 primary academic contributions (Algorithmic, Methodological, Societal, Institutional) and the chapter-by-chapter thesis manuscript status (~85% complete).
   - Preserve all native institutional branding, section trackers, and manual font size edits.
3. **Retain Slide 20 Meeting Agenda & LNCS Acceptance**:
   - Keep Slide 20 focused on the Springer LNCS (AIST 2026) paper acceptance, the live 4-tab Gradio demonstration agenda for today's meeting, September progress, and next steps for Prof. Guo.
4. **Synchronize 100% English Speaker Notes**:
   - Update `thesis_presentation_speaker_notes_prof_guo.md` so Slide 7 walks through the comprehensive framework and Slide 19 walks through thesis progress and contributions.
   - Maintain 100% English-only spoken delivery with zero Chinese characters in spoken blocks.
5. **Rigorous Verification & Paper Safety**:
   - Maintain strict immutability of `aist2026/paper.tex` (0 diff lines).
   - Update and pass all automated tests (309+ passing tests).

### Non-Goals
- Adding extra slides that would disrupt the 23-slide widescreen deck structure.
- Modifying Slides 1–6, 8–18, or 21–23.
- Modifying experimental data or benchmark evaluation metrics.

---

## 3. Detailed Slide-by-Slide Design

### 3.1 Slide 7: Methodology Anchor (Chapter 3)
- **Slide Index**: 6 (Slide 7 of 23)
- **Section Tracker (`矩形 29`)**: Points to Section 3 (*研究内容与系统架构*)
- **Slide Header**:
  - Title: `Research Content Overview: Comprehensive Methodological Framework`
  - Subtitle: `A unified hierarchical framework spanning sentence-level morpho-gating, document-level chunk aggregation, and evidence-grounded trust verification`
- **Visual Elements**:
  - Full-width embedded framework diagram: `presentation_figures/fig15_methodological_innovations_framework.png`
  - Position: `left = Inches(0.60)`, `top = Inches(1.85)`, `width = Inches(12.133)`, `height = Inches(4.85)`
  - Formal caption: `Figure 3. Overall Methodological Innovation Framework for Kazakh AI-Generated Text Detection and Factual Verification.`

### 3.2 Slide 19: Conclusion Anchor (Chapter 6)
- **Slide Index**: 18 (Slide 19 of 23)
- **Section Tracker (`矩形 29`)**: Points to Section 6 (*总结与展望*)
- **Slide Header**:
  - Title: `Conclusion: Summary of Thesis Contributions & Writing Progress`
  - Subtitle: `Master's thesis completion estimated at 85%; core theoretical, empirical, and engineering milestones achieved`
- **Card 1 (Left Column - Academic Contributions)**:
  - Position: `left = Inches(0.60)`, `top = Inches(2.05)`, `width = Inches(6.65)`, `height = Inches(4.75)`
  - Title: `Four Primary Academic Contributions of the Thesis`
  - Accent Color: Institutional Navy (`#1E3A8A`), Background: Clean White (`#FFFFFF`)
  - Content:
    - *1. Algorithmic Contribution: Dual-Stream Morphological Cross-Attention*: Pioneered the integration of rule-based 83-rule FST morphological representations with transformer semantic backbones via learned dynamic gating, eliminating out-of-domain collapse (+42.18% AUC gain).
    - *2. Methodological Contribution: Long-Document Sliding Window & Dynamic Top-K*: Engineered the first sentence-preserving chunker with 10 Kazakh abbreviation guards and adaptive worst-case Top-K pooling, achieving 100% localization in hybrid documents up to 25,000 words.
    - *3. Societal Contribution: Kazakh-FEVER & Four-Quadrant Trust Matrix*: Constructed the first evidence-grounded factual verification corpus for Kazakh, establishing a dual-risk framework decoupling factual veracity from AI style.
    - *4. Institutional Impact: Open-Source Production Deployment*: Delivered a 4-tab Gradio academic dashboard and lightweight Hugging Face Spaces cloud package for academic integrity in Central Asia.
- **Card 2 (Right Column - Manuscript Status)**:
  - Position: `left = Inches(7.45)`, `top = Inches(2.05)`, `width = Inches(5.28)`, `height = Inches(4.75)`
  - Title: `Master's Thesis Manuscript Status (~85% Complete)`
  - Accent Color: Emerald Green (`#16A34A`), Background: Subtle Mint/White (`#F0FDF4`)
  - Content:
    - `Chapter 1: Introduction (100% Complete)`: Motivation, Turkic linguistic context, problem statement.
    - `Chapter 2: Related Work (100% Complete)`: SOTA detectors, LLM watermarks, fact-checking corpora.
    - `Chapter 3: Methodology (100% Complete)`: Mathematical formulations of Topics 1, 2, and 3.
    - `Chapter 4: Experiments & Benchmarks (100% Complete)`: Kaz-MAGE 2x2 matrix, Kazakh-FEVER, ablation tables.
    - `Chapter 5: System Implementation & UI (90% Complete)`: 4-tab Gradio dashboard, HF Spaces, latency profiling.
    - `Chapter 6: Conclusion & Future Outlook (70% Complete)`: Final synthesis and future research directions.

### 3.3 Slide 20: Meeting Agenda & Roadmap Anchor (Chapter 6)
- **Slide Index**: 19 (Slide 20 of 23)
- **Section Tracker (`矩形 29`)**: Points to Section 6 (*总结与展望*)
- **Slide Header**:
  - Title: `Current Progress: LNCS Acceptance & September Meeting Agenda`
  - Subtitle: `Academic paper acceptance, live system demonstration agenda, and research milestone roadmap`
- **4 Milestone Cards**:
  1. `MILESTONE 1: ACCEPTED`: Springer LNCS (AIST 2026) Paper 1 camera-ready finalized.
  2. `MILESTONE 2: MEETING AGENDA`: Live 4-tab Gradio system demonstration for today's meeting.
  3. `MILESTONE 3: SEPTEMBER PROGRESS`: Research milestones (~85% completion, 10k dataset, 303+ passing tests).
  4. `MILESTONE 4: NEXT STEPS`: Guidance requested from Prof. Guo on Paper 2 positioning and defense timeline.

---

## 4. Speaker Notes Synchronization (`thesis_presentation_speaker_notes_prof_guo.md`)

- **Slide 7 Spoken Script**:
  - Structured, articulate English walkthrough of the overall technical framework in Figure 3/15.
  - Explains the three stages: Inputs & 36-article knowledge store $\to$ Innovation 1 (Morpho-gated cross-attention) $\to$ Innovation 2 (Sentence-preserving chunking & Top-K) $\to$ Innovation 3 (Kazakh-FEVER & Four-Quadrant Trust Matrix).
  - Explicitly transitions into Slides 8, 9, and 10 for deep mathematical and algorithmic details.
- **Slide 19 Spoken Script**:
  - Structured, articulate English walkthrough of the Master's thesis writing status (~85% complete, Chapters 1–6) and the 4 core contributions.
  - Explicitly transitions into Slide 20 for Paper 1 publication, live demo agenda, and next steps.
- **Language Policy**:
  - 100% English-only across all spoken script sections.
  - Zero Chinese characters in spoken script blocks.
  - Zero decorative emojis.

---

## 5. Automated Testing & Verification Plan

1. **`tests/test_presentation_methodology.py`**:
   - Verify Slide 7 contains the embedded framework picture with width `~12.13"` and formal caption.
   - Verify Slide 19 contains the two callout cards for Academic Contributions and Manuscript Status (~85%).
   - Verify Slide 20 contains the 4 milestone cards (Springer LNCS, Gradio, September, Next Steps).
   - Verify 16:9 widescreen dimensions (`13.333" x 7.500"`) and 23 slides total.
2. **`tests/test_presentation_speaker_notes.py`**:
   - Verify all 23 slides have dedicated `#### Spoken Script (English)` blocks.
   - Verify zero Chinese characters in spoken blocks.
   - Verify Slide 7 walks through the framework and Slide 19 walks through manuscript progress.
   - Verify zero decorative emojis.
3. **Full Regression Test**:
   - Run `python -m unittest discover -s tests -p "test_*.py"` confirming all 309+ tests pass.
4. **Ground-Truth Invariant**:
   - Verify `git diff aist2026/paper.tex` is strictly empty.
