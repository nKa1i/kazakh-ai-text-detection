# Design Specification: Methodological Innovations Framework & September Meeting Progress

- **Date**: 2026-09-12
- **Topic**: Methodological Innovation Framework Diagrams & September Progress Update for Professor Guo
- **Target Presentation File**: C:\Users\Roza\Desktop\AnekeshD_Progress.pptx
- **Presenter**: 大雷 (Daulet)
- **Advisor**: 郭教授 (Prof. Guo)
- **Status**: Approved via Interview (/grill-me)

---

## 1. Objective & Motivation

Following initial review of AnekeshD_Progress.pptx, Professor Guo provided the following critical feedback:
> You will also need to prepare framework diagrams to illustrate the methodological innovations you intend to develop.

In addition, the user made selective manual font size adjustments across slides and specified that:
1. Paper 1 has been officially accepted into **Springer LNCS** (AIST 2026).
2. The timeline/roadmap on Slide 20 should be re-anchored to **September 2026**, highlighting immediate completed tasks, the live Gradio system demonstration for today's meeting, and concrete next research steps.
3. Slide 19 will be transformed into the **Overall Methodological Innovation Framework Diagram**, visually illustrating the complete end-to-end technical innovations (Sentence, Document, Fact Verification).
4. All user font adjustments in C:\Users\Roza\Desktop\AnekeshD_Progress.pptx must be strictly preserved.

---

## 2. Detailed Slide-by-Slide Design

### Slide 19: Comprehensive Methodological Innovations Framework Diagram
- **Header**:
  - Title: Methodological Innovations: Comprehensive Technical Framework
  - Subtitle: 面向低资源哈萨克语的AI生成文本检测与事实核验总体方法创新架构与技术流程
- **Visual Asset**:
  - Embedded high-resolution vector-quality diagram: presentation_figures/fig15_methodological_innovations_framework.png (300 DPI, full width ~12.13 inches, height ~4.85 inches, left=0.60 inches, top=1.85 inches).
- **Architecture Diagram Structure (4 Interlocking Stages)**:
  1. **Stage 1 — Inputs & Domain Knowledge**:
     - Raw Kazakh Text / Multi-Paragraph Ingestion
     - 83-Rule FST Morphological Dictionary (Apertium/PyDataverse Turkic roots + case/aspect/plural affixes)
     - 36-Article Curated Encyclopedic Reference Corpus (segmented sentence units)
  2. **Stage 2 — Methodological Innovation 1 (Sentence-Level Detection)**:
     - Dual-Stream Cross-Attention: Pretrained KazRoBERTa (h_sem in R^768) + FST BiLSTM Morpheme Encoder (h_morph in R^256)
     - Dynamic Learned Gating Vector: g = sigmoid(W_g [h_sem; h_morph] + b_g) in [0, 1]^d
     - Supervised Contrastive Loss (L_SupCon): Hyperspherical feature clustering across domains
     - Output: Calibrated Sentence AI Probability P_AI(s_i)
  3. **Stage 3 — Methodological Innovation 2 (Document-Level Chunking & Aggregation)**:
     - 10 Kazakh Abbreviation Lookahead Regex Guards (preventing bibliographic/administrative false splits: т.б., ж.б., ғ., жж., қ., мыс.)
     - Sentence-Preserving Sliding Window (256-word window, 1-sentence overlap stride, exact character offsets)
     - Dynamic Worst-Case Top-K Pooling: K = max(1, min(k_cfg, ceil(0.25 * M))), Score_doc = (1/K) * sum(Score_i)
     - Output: Document Verdict & Exact Character Offsets of Tampered Spans
  4. **Stage 4 — Methodological Innovation 3 (Fact-Checking & Four-Quadrant Trust Matrix)**:
     - Automated Claim Formulation -> BM25 Morphological Sentence Retrieval (Top-3 candidates)
     - 3-Way Cross-Encoder NLI (SUPPORTS, REFUTES, NOT ENOUGH INFO)
     - Dual-Risk Trust Formulation: Risk_Trust = alpha * P(AI) + (1 - alpha) * P(Refutes)
     - Output: Four-Quadrant Actionable Decision (Q1 Human Fact, Q2 Human Misinformation, Q3 AI Synthesized Fact, Q4 AI Disinformation Alert)
- **Academic Caption**:
  Figure 15. Overall Methodological Innovation Framework for Kazakh AI-Generated Text Detection and Factual Verification.

---

### Slide 20: September Progress, LNCS Acceptance & Meeting Demonstration
- **Header**:
  - Title: Current Progress: LNCS Acceptance & September Meeting Agenda
  - Subtitle: 论文录用进展、9月课题组汇报演示计划与后续工作推进安排
- **4 Proportional Academic Cards (left=0.60, 3.70, 6.80, 9.90, width=2.85, height=4.85)**:
  1. **Card 1: Springer LNCS Conference Acceptance (Paper 1)**:
     - Title: Paper 1 Accepted to Springer LNCS (AIST 2026)
     - Status: Camera-ready manuscript successfully finalized; no changes required to core publication.
     - Theoretical Contribution: Formally establishes the dual-stream morphological gating and Kaz-MAGE benchmark.
     - Acceptance Milestone: Peer-reviewed validation of the foundational research hypothesis.
  2. **Card 2: Meeting Demonstration: Interactive 4-Tab Gradio Dashboard**:
     - Title: Live System Demonstration (Meeting Agenda)
     - Status: Standalone dashboard fully prepared for live demonstration in today's / September meeting.
     - Interactive Highlights:
       - Tab 1: Live color-coded sentence explainability heatmap with linguistic discourse reasoning.
       - Tab 2: Morphological FST Lab with real-time root/affix decomposition and gating balance meter.
       - Tab 3: Kaz-MAGE tri-domain benchmark explorer and empirical ROC analysis.
       - Tab 4: Factual verification pipeline over the 36-article corpus with quadrant risk placement.
  3. **Card 3: September Research Accomplishments (Thesis Progress)**:
     - Title: September Research Milestones (~85% Complete)
     - Curated Corpus: 10,000+ benchmark texts across News, Wikipedia, and Consumer Reviews.
     - Kazakh-FEVER: Curated 36 articles and 120 verified claims; 100% NLI Macro-F1.
     - Engineering Rigor: Micro-batching (<1.4 GB VRAM on 25k words); 302 passing automated tests; containerized Hugging Face Spaces cloud package.
  4. **Card 4: Immediate Next Steps & Guidance Requested**:
     - Title: Next Steps & Guidance Requested
     - Feedback Integration: Incorporate Professor Guo's guidance on framework diagrams and chapter drafts.
     - Paper 2 Preparation: Draft manuscript on Kazakh-FEVER and Dual-Risk Trust Matrix for high-impact NLP venue.
     - Chapter Finalization: Schedule internal laboratory review for Chapters 5 & 6.

---

## 3. Preservation of Baseline Presentation & User Modifications

- The authoritative PowerPoint document is: C:\Users\Roza\Desktop\AnekeshD_Progress.pptx.
- All user modifications (font sizes, text spacing, layout adjustments on Slides 1-18, 21-23) must be preserved in place.
- On Slide 19, the existing plain text rectangles and textboxes are removed, replaced by the full-width ig15_methodological_innovations_framework.png and formal caption.
- On Slide 20, the textboxes inside the 4 milestone cards are updated with the Springer LNCS acceptance, meeting demonstration agenda, September milestones, and next steps.
- The output will be saved back to C:\Users\Roza\Desktop\AnekeshD_Progress.pptx.

---

## 4. Verification Plan

1. **Figure Generation**:
   - scripts/generate_presentation_figures.py updated with generate_fig15().
   - Generates presentation_figures/fig15_methodological_innovations_framework.png at 300 DPI (>50 KB, valid PNG header).
2. **Slide Updating**:
   - Script updates C:\Users\Roza\Desktop\AnekeshD_Progress.pptx.
   - Verified via unit test:
     - Slide 19 has embedded picture (Figure 15) with caption.
     - Slide 20 contains 'Springer LNCS', 'Gradio', 'September', 'Professor Guo'.
     - All user font sizes on other slides are preserved.
3. **PowerPoint COM Rendering**:
   - Re-export Slides 19 and 20 to PNG in ppt_images/ and visually inspect.
4. **Safety Constraint**:
   - Verify git diff aist2026/paper.tex remains 100% empty.
   - Run full regression suite (302+ tests pass).