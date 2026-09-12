# Methodological Innovations Framework & September Meeting Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Implement the Methodological Innovations Framework diagram (Figure 15) on Slide 19 and update Slide 20 with Springer LNCS acceptance, September research milestones, and the live Gradio meeting demonstration plan in C:\Users\Roza\Desktop\AnekeshD_Progress.pptx, strictly preserving all manual user font adjustments.

**Architecture:** (1) Generate high-resolution, publication-grade 300 DPI Figure 15 in scripts/generate_presentation_figures.py; (2) Implement scripts/update_presentation_methodology.py to directly load AnekeshD_Progress.pptx, replace legacy plain text shapes on Slide 19 with the full-width Figure 15, and update the 4 milestone cards on Slide 20 with Springer LNCS acceptance and the live Gradio agenda; (3) Re-export slide images to ppt_images/ via PowerPoint COM, update speaker notes, and execute full regression.

**Tech Stack:** Python 3.12, python-pptx, matplotlib, win32com.client.

## Global Constraints

- Under no circumstances shall ist2026/paper.tex be touched or modified.
- Zero decorative emojis across all slides, code, tables, and documentation.
- Widescreen 16:9 format (13.333 x 7.500 inches) strictly preserved.
- Native branding preserved: KazNU logo (Изображение 16), NPU logo group (LOGO组合), and Layout 7 NPU pentagon emblem.
- Base presentation is C:\Users\Roza\Desktop\AnekeshD_Progress.pptx (all user font size adjustments on other slides must be preserved).
- All Python file operations must explicitly specify encoding='utf-8'.

---

### Task 1: Generate Figure 15 (Comprehensive Methodological Innovations Framework)

**Files:**
- Modify: scripts/generate_presentation_figures.py
- Modify: 	ests/test_presentation_figures.py
- Output: presentation_figures/fig15_methodological_innovations_framework.png

**Interfaces:**
- Produces: ig15_methodological_innovations_framework.png (300 DPI, ~12.0 x 5.2 inches aspect ratio, institutional color palette: Navy #1E3A8A, Teal #0D9488, Amber #D97706, Slate #0F172A, Green #16A34A).
- Tests: 	ests/test_presentation_figures.py verifying Figure 15 exists, size > 50 KB, and valid PNG magic bytes header.

- [ ] **Step 1: Write the failing test for Figure 15**

Update 	ests/test_presentation_figures.py to add fig15_methodological_innovations_framework.png to EXPECTED_FIGURES:

`python
EXPECTED_FIGURES = [
    fig01_subword_vs_fst.png,
    fig02_domain_collapse.png,
    fig03_tripartite_framework.png,
    fig04_morpho_gate_arch.png,
    fig05_chunking_topk_flow.png,
    fig06_trust_matrix_pipeline.png,
    fig07_dataset_distribution.png,
    fig08_roc_curves_kaz_mage.png,
    fig09_long_doc_injection.png,
    fig10_kaz_fever_confusion_matrix.png,
    fig11_four_quadrant_trust_matrix.png,
    fig12_ablation_study_barchart.png,
    fig13_gradio_dashboard_panels.png,
    fig14_cloud_deployment_pipeline.png,
    fig15_methodological_innovations_framework.png,
]
`

- [ ] **Step 2: Run test to verify it fails**

Run: C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_presentation_figures.py
Expected: FAIL (Figure 15 does not exist).

- [ ] **Step 3: Implement generate_fig15() in scripts/generate_presentation_figures.py**

Implement generate_fig15(output_dir):
- Dimensions: igsize=(14.5, 6.2), 300 DPI.
- 4 clear horizontal columns / interconnected stages:
  1. Input & Knowledge Sources: Raw Kazakh Text, 83-Rule FST Morphological Dictionary, 36-Article Reference Corpus.
  2. Method Innovation 1 (Sentence-Level): Dual-Stream Cross-Attention (KazRoBERTa {sem}$ + FST BiLSTM {morph}$) -> Dynamic Learned Gating  \in [0, 1]^d$ -> SupCon Loss $\mathcal{L}_{SupCon}$ -> Sentence AI Score {AI}(s_i)$.
  3. Method Innovation 2 (Document-Level): 10 Kazakh Abbreviation Regex Protection Guards -> Sentence-Preserving Sliding Window (256w, 1-sent overlap) -> Dynamic Worst-Case Top-$ Pooling ( = \max(1, \min(k_{cfg}, \lceil 0.25 M \rceil))$) -> Document AI Score & Tampered Chunk Spans.
  4. Method Innovation 3 (Fact Verification): Factual Claim Extraction -> BM25 Morphological Sentence Retrieval -> 3-Way Cross-Encoder NLI -> Dual-Risk Scorer ($\text{Risk}_{Trust} = \alpha P(AI) + (1-\alpha) P(Refutes)$) -> Four-Quadrant Decision (Q1-Q4).
- Connecting arrows between stages, mathematical formula annotations, clean typography, zero emojis.
- Call generate_fig15(output_dir) in main().

- [ ] **Step 4: Run test to verify it passes**

Run:
`ash
C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe scripts/generate_presentation_figures.py
C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_presentation_figures.py
`
Expected: PASS (All 15 figures verified).

- [ ] **Step 5: Commit**

`ash
git add scripts/generate_presentation_figures.py tests/test_presentation_figures.py presentation_figures/fig15_methodological_innovations_framework.png
git commit -m feat: generate publication-grade Figure 15 methodological innovations framework
`

---

### Task 2: Implement Presentation Updater for Slides 19 and 20

**Files:**
- Create: scripts/update_presentation_methodology.py
- Create: 	ests/test_presentation_methodology.py
- Target Document: C:\Users\Roza\Desktop\AnekeshD_Progress.pptx

**Interfaces:**
- Consumes: C:\Users\Roza\Desktop\AnekeshD_Progress.pptx, presentation_figures/fig15_methodological_innovations_framework.png.
- Produces: Updated C:\Users\Roza\Desktop\AnekeshD_Progress.pptx with:
  - Slide 19: Full-width Figure 15 diagram + formal caption + updated bilingual title.
  - Slide 20: 4 updated academic milestone cards with Springer LNCS acceptance, meeting demonstration agenda, September milestones, and next research steps.
  - Preserves user font sizes and styles across Slides 1-18, 21-23.

- [ ] **Step 1: Write the failing test for Slide 19 and Slide 20 updates**

Create 	ests/test_presentation_methodology.py:

`python
import os
import unittest
from pptx import Presentation

class TestPresentationMethodology(unittest.TestCase):
    def test_slide_19_and_20_updates(self):
        ppt_path = os.path.join(os.path.expanduser(~), Desktop, AnekeshD_Progress.pptx)
        self.assertTrue(os.path.exists(ppt_path), f{ppt_path} must exist)
        prs = Presentation(ppt_path)
        self.assertEqual(len(prs.slides), 23)

        # Slide 19: Check for embedded picture Figure 15
        s19 = prs.slides[18]
        pictures = [s for s in s19.shapes if s.shape_type == 13]
        self.assertGreater(len(pictures), 0, Slide 19 must contain embedded Figure 15 picture)
        s19_text =  .join(s.text_frame.text for s in s19.shapes if s.has_text_frame)
        self.assertIn(Methodological Innovations, s19_text)
        self.assertIn(Figure 15, s19_text)

        # Slide 20: Check for Springer LNCS, Gradio, September
        s20 = prs.slides[19]
        s20_text =  .join(s.text_frame.text for s in s20.shapes if s.has_text_frame)
        self.assertIn(Springer LNCS, s20_text)
        self.assertIn(Gradio, s20_text)
        self.assertIn(September, s20_text)
        self.assertIn(Professor Guo, s20_text)

if __name__ == __main__:
    unittest.main()
`

- [ ] **Step 2: Run test to verify it fails**

Run: C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_presentation_methodology.py
Expected: FAIL (Slide 19 does not have Figure 15 picture; Slide 20 does not contain Springer LNCS).

- [ ] **Step 3: Implement scripts/update_presentation_methodology.py**

Implement the script:
1. Open C:\Users\Roza\Desktop\AnekeshD_Progress.pptx.
2. Slide 19 (Index 18):
   - Update header textbox:
     Title: Methodological Innovations: Comprehensive Technical Framework
     Subtitle: 面向低资源哈萨克语的AI生成文本检测与事实核验总体方法创新架构与技术流程
   - Delete legacy plain text rectangles and textboxes where 	op > Inches(1.2) (while protecting 灯片编号占位符 10, 直接连接符 6, 矩形 4, 矩形 29, section label textboxes).
   - Insert picture presentation_figures/fig15_methodological_innovations_framework.png:
     left = Inches(0.60), 	op = Inches(1.85), width = Inches(12.133), height = Inches(4.85).
   - Add formal caption textbox below picture:
     Figure 15. Overall Methodological Innovation Framework for Kazakh AI-Generated Text Detection and Factual Verification.
3. Slide 20 (Index 19):
   - Update header textbox:
     Title: Current Progress: LNCS Acceptance & September Meeting Agenda
     Subtitle: 论文录用进展、9月课题组汇报演示计划与后续工作推进安排
   - Update text frames in the 4 milestone cards:
     * Card 1:
       Header: MILESTONE 1: ACCEPTED\nSpringer LNCS (AIST 2026)
       Bullets:
       - Paper 1: Morphologically-Grounded AI Detection in Low-Resource Kazakh.
       - Status: Officially accepted to Springer LNCS; camera-ready finalized.
       - Contribution: Formally validates dual-stream cross-attention and Kaz-MAGE benchmark.
     * Card 2:
       Header: MILESTONE 2: MEETING AGENDA\nLive System Demonstration
       Bullets:
       - Interactive 4-tab Gradio system prepared for today's meeting demonstration.
       - Tab 1: Real-time sentence explainability heatmap and discourse reasoning.
       - Tab 2: Morphological FST Lab with live root/affix decomposition.
       - Tab 3 & 4: Kaz-MAGE benchmark explorer and Kazakh-FEVER trust matrix.
     * Card 3:
       Header: MILESTONE 3: SEPTEMBER PROGRESS\nResearch Milestones (~85%)
       Bullets:
       - Curated 10,000+ benchmark dataset across News, Wikipedia, and Consumer Reviews.
       - Kazakh-FEVER: Curated 36 articles and 120 verified claims; 100% NLI Macro-F1.
       - Engineering: Micro-batching (<1.4 GB VRAM); 302 automated passing tests.
     * Card 4:
       Header: MILESTONE 4: NEXT STEPS\nGuidance Requested from Prof. Guo
       Bullets:
       - Feedback Integration: Incorporate Professor Guo's advice on framework diagrams and thesis draft.
       - Paper 2 Preparation: Draft manuscript on Kazakh-FEVER & Four-Quadrant Trust Matrix.
       - Chapter Finalization: Schedule internal laboratory review for Chapters 5 & 6.
4. Save updated presentation directly back to C:\Users\Roza\Desktop\AnekeshD_Progress.pptx.

- [ ] **Step 4: Run test to verify it passes**

Run:
`ash
C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe scripts/update_presentation_methodology.py
C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_presentation_methodology.py
`
Expected: PASS.

- [ ] **Step 5: Commit**

`ash
git add scripts/update_presentation_methodology.py tests/test_presentation_methodology.py
git commit -m feat(presentation): embed Figure 15 on Slide 19 and update Slide 20 with LNCS acceptance and meeting demo
`

---

### Task 3: Visual Verification, Speaker Notes Synchronization & Regression

**Files:**
- Output: ppt_images/slide_19.png, ppt_images/slide_20.png
- Modify: 	hesis_presentation_speaker_notes_prof_guo.md

- [ ] **Step 1: Re-export slides via PowerPoint COM automation**

Run script to re-export Slide 19 and Slide 20 (and full deck) to ppt_images/ using win32com.client.

- [ ] **Step 2: Visually verify exported slides**

Use iew_file to inspect ppt_images/slide_19.png and ppt_images/slide_20.png to confirm:
- Figure 15 renders with sharp lines, clear labels, and no clipping on Slide 19.
- Slide 20 cards are balanced with legible text and zero overlap.
- Zero emojis.

- [ ] **Step 3: Update 	hesis_presentation_speaker_notes_prof_guo.md**

Update speaker notes for Slides 19 and 20:
- Slide 19 script: Explaining the 3 interlocking methodological innovations (Sentence-level FST cross-attention, Document-level sliding window Top-K, Fact verification Kazakh-FEVER trust matrix).
- Slide 20 script: Highlighting the Springer LNCS acceptance, introducing the live Gradio system demonstration for today's meeting, and requesting guidance from Professor Guo on Paper 2.

- [ ] **Step 4: Verify paper immutability and run full regression**

`ash
git diff aist2026/paper.tex
C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest discover -s tests -p test_*.py
`
Expected:
- git diff aist2026/paper.tex: Empty (0 lines modified).
- Full regression: 302+ tests passing.

- [ ] **Step 5: Final Commit**

`ash
git add thesis_presentation_speaker_notes_prof_guo.md docs/
git commit -m chore: synchronize speaker notes and complete visual verification for Slides 19-20
`