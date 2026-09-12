# Presentation Methodological Framework Restructuring & Slide 19 Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconfigure the Master's thesis presentation deck (`C:\Users\Roza\Desktop\AnekeshD_Progress.pptx`) by embedding the overall Methodological Innovations Framework on Slide 7 (Chapter 3: Methodology) where it logically belongs, restoring Slide 19 (Chapter 6: Conclusion) to the Thesis Contributions & Manuscript Progress Summary (~85% complete), retaining Slide 20's LNCS meeting agenda, and synchronizing 100% English-only speaker notes.

**Architecture:** Update `scripts/update_presentation_methodology.py` using `python-pptx` to programmatically update Slide 7 and Slide 19 in `AnekeshD_Progress.pptx`, mirror to `Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx`, update `thesis_presentation_speaker_notes_prof_guo.md`, and re-export slide renderings to `ppt_images/`. Everything is verified using TDD across `tests/test_presentation_methodology.py` and `tests/test_presentation_speaker_notes.py`.

**Tech Stack:** Python 3.12, `python-pptx`, `win32com.client` (PowerPoint automation), `matplotlib`, `unittest`.

## Global Constraints

- Under no circumstances shall `aist2026/paper.tex` be touched or modified (0 diff lines).
- Strictly preserve all manual font size adjustments Daulet made in `C:\Users\Roza\Desktop\AnekeshD_Progress.pptx`.
- Zero decorative emojis across all code, slide shapes, tables, speaker notes, and documentation.
- Widescreen 16:9 format (`13.333" x 7.500"`) and exactly 23 slides strictly preserved.
- Speaker notes must remain strictly 100% English-only in all spoken script sections.
- Full regression suite must pass cleanly (309+ tests).

---

### Task 1: Update Presentation Methodology Engine for Slide 7 & Slide 19

**Files:**
- Modify: `scripts/update_presentation_methodology.py`
- Test: `tests/test_presentation_methodology.py`

**Interfaces:**
- Consumes: `presentation_figures/fig15_methodological_innovations_framework.png` (300 DPI), `C:\Users\Roza\Desktop\AnekeshD_Progress.pptx`
- Produces: Updated `AnekeshD_Progress.pptx` and `Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx` with Slide 7 framework and Slide 19 progress restoration.

- [ ] **Step 1: Write the failing tests in `tests/test_presentation_methodology.py`**

Update `tests/test_presentation_methodology.py` to assert:
1. Slide 7 (index 6) has embedded framework picture with width ~12.13", header "Research Content Overview: Comprehensive Methodological Framework", and caption containing "Figure 3".
2. Slide 19 (index 18) has restored header "Conclusion: Summary of Thesis Contributions & Writing Progress" and contains the two callout boxes ("Four Primary Academic Contributions" and "Master's Thesis Manuscript Status (~85% Complete)").
3. Slide 20 (index 19) retains the 4 milestone cards (Springer LNCS, Gradio demo, September progress, Next steps).

```python
# In tests/test_presentation_methodology.py:
def test_slide_7_and_19_and_20_structure(self):
    ppt_path = os.path.join(os.path.expanduser("~"), "Desktop", "AnekeshD_Progress.pptx")
    self.assertTrue(os.path.exists(ppt_path))
    prs = Presentation(ppt_path)
    self.assertEqual(len(prs.slides), 23)

    # Slide 7 (Index 6)
    s7 = prs.slides[6]
    s7_text = " ".join(s.text_frame.text for s in s7.shapes if s.has_text_frame)
    self.assertIn("Comprehensive Methodological Framework", s7_text)
    self.assertIn("Figure 3", s7_text)
    s7_pics = [s for s in s7.shapes if s.shape_type == 13]
    self.assertGreater(len(s7_pics), 0)
    self.assertAlmostEqual(s7_pics[0].width.inches, 12.133, delta=0.1)

    # Slide 19 (Index 18)
    s19 = prs.slides[18]
    s19_text = " ".join(s.text_frame.text for s in s19.shapes if s.has_text_frame)
    self.assertIn("Summary of Thesis Contributions & Writing Progress", s19_text)
    self.assertIn("Four Primary Academic Contributions", s19_text)
    self.assertIn("Master's Thesis Manuscript Status (~85% Complete)", s19_text)
    self.assertIn("Chapter 1", s19_text)
    self.assertIn("Chapter 6", s19_text)

    # Slide 20 (Index 19)
    s20 = prs.slides[19]
    s20_text = " ".join(s.text_frame.text for s in s20.shapes if s.has_text_frame)
    self.assertIn("LNCS Acceptance & September Meeting Agenda", s20_text)
    self.assertIn("Springer LNCS", s20_text)
    self.assertIn("Gradio", s20_text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_presentation_methodology.py`  
Expected: FAIL (Slide 7 does not yet have Figure 15, and Slide 19 currently has Figure 15 instead of the progress summary).

- [ ] **Step 3: Update `scripts/update_presentation_methodology.py`**

Implement:
1. `update_slide_07(slide, fig_path)`:
   - Clear existing content shapes below `top > Inches(1.2)` (preserving tracker `矩形 29`, slide number, etc.).
   - Set title: `Research Content Overview: Comprehensive Methodological Framework`.
   - Set subtitle: `A unified hierarchical framework spanning sentence-level morpho-gating, document-level chunk aggregation, and evidence-grounded trust verification`.
   - Insert picture `fig15_methodological_innovations_framework.png` at `left=Inches(0.60)`, `top=Inches(1.85)`, `width=Inches(12.133)`, `height=Inches(4.85)`.
   - Add caption textbox: `Figure 3. Overall Methodological Innovation Framework for Kazakh AI-Generated Text Detection and Factual Verification.` at `left=Inches(0.60)`, `top=Inches(6.78)`, `width=Inches(12.133)`, `height=Inches(0.35)`.
2. `update_slide_19(slide)`:
   - Clear existing content shapes below `top > Inches(1.2)`.
   - Set title: `Conclusion: Summary of Thesis Contributions & Writing Progress`.
   - Set subtitle: `Master's thesis completion estimated at 85%; core theoretical, empirical, and engineering milestones achieved`.
   - Add Card 1 (Left, `left=Inches(0.60)`, `top=Inches(2.05)`, `width=Inches(6.65)`, `height=Inches(4.75)`):
     Title: `Four Primary Academic Contributions of the Thesis`
     Points:
     - `1. Algorithmic Contribution: Dual-Stream Morphological Cross-Attention:`
     - `   Pioneered the integration of rule-based 83-rule FST morphological representations with transformer semantic backbones via learned dynamic gating, eliminating out-of-domain collapse (+42.18% AUC gain).`
     - ``
     - `2. Methodological Contribution: Long-Document Sliding Window & Dynamic Top-K:`
     - `   Engineered the first sentence-preserving chunker with 10 Kazakh abbreviation guards and dynamic Top-K worst-chunk pooling, achieving 100% localization in hybrid documents up to 25,000 words.`
     - ``
     - `3. Societal Contribution: Kazakh-FEVER & Four-Quadrant Trust Matrix:`
     - `   Constructed the first evidence-grounded factual verification corpus for Kazakh, establishing a dual-risk framework decoupling factual veracity from AI style.`
     - ``
     - `4. Institutional Impact: Open-Source Production Deployment:`
     - `   Delivered an open-source, reproducible 4-tab Gradio system and Hugging Face Space for academic integrity in Central Asia.`
   - Add Card 2 (Right, `left=Inches(7.45)`, `top=Inches(2.05)`, `width=Inches(5.28)`, `height=Inches(4.75)`):
     Title: `Master's Thesis Manuscript Status (~85% Complete)`
     Points:
     - `Chapter 1: Introduction (100% Complete)`
     - `  - Motivation, Turkic linguistic context, problem statement.`
     - ``
     - `Chapter 2: Related Work (100% Complete)`
     - `  - SOTA detectors, LLM watermarks, fact-checking corpora.`
     - ``
     - `Chapter 3: Methodology (100% Complete)`
     - `  - Mathematical formulations of Topics 1, 2, and 3.`
     - ``
     - `Chapter 4: Experiments & Benchmarks (100% Complete)`
     - `  - Kaz-MAGE 2x2 matrix, Kazakh-FEVER, ablation tables.`
     - ``
     - `Chapter 5: System Implementation & UI (90% Complete)`
     - `  - 4-tab Gradio dashboard, HF Spaces, latency profiling.`
     - ``
     - `Chapter 6: Conclusion & Future Outlook (70% Complete)`
     - `  - Final synthesis and future research directions.`
3. Run the script to update `AnekeshD_Progress.pptx` and copy to project root mirror.

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_presentation_methodology.py`  
Expected: PASS.

- [ ] **Step 5: Commit changes**

```bash
git add scripts/update_presentation_methodology.py tests/test_presentation_methodology.py
git commit -m "feat(presentation): relocate framework to Slide 7 and restore Slide 19 progress summary"
```

---

### Task 2: Synchronize 100% English Speaker Notes

**Files:**
- Modify: `thesis_presentation_speaker_notes_prof_guo.md`
- Modify: `C:\Users\Roza\.gemini\antigravity\brain\29832fd0-dcc9-4b3c-9300-06719048089c\thesis_presentation_speaker_notes_prof_guo.md`
- Test: `tests/test_presentation_speaker_notes.py`

**Interfaces:**
- Consumes: Slide 7 framework and Slide 19 progress structure
- Produces: Fully synchronized English speaker notes for Daulet's presentation to Prof. Guo

- [ ] **Step 1: Write test assertions in `tests/test_presentation_speaker_notes.py`**

Add checks asserting that Slide 7 spoken script walks through the overall framework (Inputs, Sentence Gating, Document Top-K, Fact-Checking Trust Matrix) and Slide 19 walks through the thesis manuscript completion status (~85% complete) and contributions.

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_presentation_speaker_notes.py`  
Expected: FAIL (Slide 7 script still references the old tripartite cards, and Slide 19 still references Figure 15).

- [ ] **Step 3: Update `thesis_presentation_speaker_notes_prof_guo.md`**

Update:
1. **Slide 7 Spoken Script (English)**:
   - Walk Professor Guo through Figure 3 (the comprehensive framework) as the entry into Chapter 3.
   - Explain Stage 1 Inputs & Evidence Base $\to$ Innovation 1 Dual-stream FST Gating $\to$ Innovation 2 Sentence-preserving sliding window with Top-K pooling $\to$ Innovation 3 Kazakh-FEVER and Four-Quadrant Trust Matrix.
   - Smooth transition into Slides 8, 9, and 10.
2. **Slide 19 Spoken Script (English)**:
   - Walk Professor Guo through Chapter 6 (Summary of Thesis Contributions & Writing Progress).
   - Report ~85% overall manuscript completion with the chapter-by-chapter status (Chapters 1–4 complete, Chapter 5 at 90%, Chapter 6 at 70%).
   - Reiterate the 4 core contributions (algorithmic, methodological, societal, institutional).
   - Smooth transition into Slide 20 (Paper 1 Springer LNCS acceptance and live Gradio demo agenda).
3. Synchronize both the repository file and the artifact copy.

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_presentation_speaker_notes.py`  
Expected: PASS.

- [ ] **Step 5: Commit changes**

```bash
git add thesis_presentation_speaker_notes_prof_guo.md tests/test_presentation_speaker_notes.py
git commit -m "feat(notes): synchronize Slide 7 framework and Slide 19 progress speaker notes in English"
```

---

### Task 3: Re-export Slide Visuals & Full Regression Verification

**Files:**
- Script: `scripts/export_presentation_slides.py`
- Output: `ppt_images/slide_07.png`, `ppt_images/slide_19.png`

- [ ] **Step 1: Run slide export script**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe scripts/export_presentation_slides.py`  
Verify: `ppt_images/slide_07.png` displays Figure 15 in full width cleanly under Chapter 3, and `ppt_images/slide_19.png` displays the restored contributions and progress cards under Chapter 6.

- [ ] **Step 2: Run full regression test suite**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest discover -s tests -p "test_*.py"`  
Expected: All 309+ tests PASS with 0 errors.

- [ ] **Step 3: Confirm paper immutability invariant**

Run: `git diff aist2026/paper.tex`  
Expected: 0 diff lines (completely empty).

- [ ] **Step 4: Commit any visual validation updates**

```bash
git add docs/ tests/ scripts/
git commit -m "chore: complete slide re-export and full regression verification"
```
