# Master's Thesis Presentation Verification Checklist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a comprehensive, audit-ready verification checklist and automated verification script validating that all 80+ academic recommendations from Professor Guo are implemented across the 23-slide thesis presentation deck.

**Architecture:** A two-task structure: (1) Generate the full Markdown checklist artifact (`docs/thesis_presentation_verification_checklist.md` mirrored to brain artifacts) containing Part 1 (6 Thematic Master Pillars) and Part 2 (Slide-by-Slide Visual & Content Audit Guide for Slides 01–23); (2) Implement an automated verification script and test (`scripts/verify_presentation_against_checklist.py` and `tests/test_presentation_verification_checklist.py`) programmatically validating deck invariants against the checklist.

**Tech Stack:** Python 3.12, python-pptx, unittest, Markdown / GitHub Flavored Markdown.

## Global Constraints

- Under no circumstances shall `aist2026/paper.tex` be touched or modified (0 diff lines).
- Zero decorative emojis across all code, slides, and documentation.
- Widescreen 16:9 format (`13.333" x 7.500"`) and 23 slides strictly preserved.
- 100% Times New Roman for Latin/English typography (0 Arial).
- Presentation targets: `C:\Users\Roza\Desktop\AnekeshD_Progress.pptx` (primary) and `Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx`.
- All Python file operations must explicitly specify `encoding='utf-8'`.
- All markdown files must be completely filled out with zero "TODO", "TBD", or placeholders.

---

### Task 1: Generate Master Presentation Verification Checklist and Audit Guide

**Files:**
- Create: `docs/thesis_presentation_verification_checklist.md`
- Create: `C:\Users\Roza\.gemini\antigravity\brain\29832fd0-dcc9-4b3c-9300-06719048089c\thesis_presentation_verification_checklist.md`
- Test: `tests/test_presentation_verification_checklist.py`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-09-17-presentation-verification-checklist-design.md`, `ppt_images/slide_01.png` through `slide_23.png`, `thesis_presentation_speaker_notes_prof_guo.md`.
- Produces: `docs/thesis_presentation_verification_checklist.md` with:
  - Part 1: Thematic Master Checklist across 6 core feedback pillars.
  - Part 2: Detailed Slide-by-Slide Visual & Content Audit Guide for all 23 slides (headers, feedback addressed, exact on-slide text/numbers/formulas, speaker script anchor, and 1080p preview link).

- [ ] **Step 1: Write the failing test for checklist document completeness**

```python
# tests/test_presentation_verification_checklist.py
import os
import unittest

CHECKLIST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs",
    "thesis_presentation_verification_checklist.md"
)

class TestPresentationVerificationChecklist(unittest.TestCase):

    def test_checklist_document_exists_and_complete(self):
        self.assertTrue(os.path.isfile(CHECKLIST_PATH), f"Checklist missing at {CHECKLIST_PATH}")
        with open(CHECKLIST_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # Check size and structure
        self.assertGreater(len(content), 10000, "Checklist should be comprehensive (>10KB)")
        self.assertNotIn("TODO", content)
        self.assertNotIn("TBD", content)
        self.assertNotIn("[ ] [Placeholder", content)

        # Check 6 thematic pillars
        self.assertIn("Pillar 1: Research Storyline & Topic 2 Restoration", content)
        self.assertIn("Pillar 2: Architectural Rigor & Terminology Unification", content)
        self.assertIn("Pillar 3: Experimental Metrics & Statistical Rigor", content)
        self.assertIn("Pillar 4: Topic 3 Kazakh-FEVER & 2D Trust Matrix", content)
        self.assertIn("Pillar 5: Slide Hygiene & Typography Standardization", content)
        self.assertIn("Pillar 6: Strategic Consultation Roadmap & Speaker Notes", content)

        # Check all 23 slides are covered in Part 2
        for s_idx in range(1, 24):
            slide_tag = f"Slide {s_idx:02d}:"
            self.assertIn(slide_tag, content, f"Slide {s_idx:02d} missing from audit guide")
            self.assertIn(f"slide_{s_idx:02d}.png", content, f"Preview image link for Slide {s_idx:02d} missing")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_presentation_verification_checklist.py`  
Expected: FAIL with `AssertionError: Checklist missing at ...docs\thesis_presentation_verification_checklist.md`

- [ ] **Step 3: Write comprehensive checklist document and mirror to brain artifact**

Write the complete `docs/thesis_presentation_verification_checklist.md` adhering to the design specification:
- Executive Summary & Quick Start Audit Protocol.
- Part 1: Thematic Master Checklist (Pillars 1–6) covering all ~80 individual feedback points with `[x]` checkboxes and rationale.
- Part 2: Slide-by-Slide Visual & Content Audit Guide (Slides 01–23) with exact titles, subtitles, feedback addressed, exact text strings, formulas, tables, speaker note cross-references, and preview links.
- Copy to `C:\Users\Roza\.gemini\antigravity\brain\29832fd0-dcc9-4b3c-9300-06719048089c\thesis_presentation_verification_checklist.md`.

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_presentation_verification_checklist.py`  
Expected: PASS

- [ ] **Step 5: Commit checklist**

```bash
git add -f docs/thesis_presentation_verification_checklist.md tests/test_presentation_verification_checklist.py
git commit -m "docs(audit): create comprehensive master thesis presentation verification checklist"
```

---

### Task 2: Implement Automated Presentation Verification Checker Script

**Files:**
- Create: `scripts/verify_presentation_against_checklist.py`
- Modify: `tests/test_presentation_verification_checklist.py`

**Interfaces:**
- Consumes: `C:\Users\Roza\Desktop\AnekeshD_Progress.pptx`, `Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx`.
- Produces: CLI verification runner that executes programmatic assertions against the deck matching the checklist specifications.

- [ ] **Step 1: Expand test suite to test verification checker logic**

Add unit tests in `tests/test_presentation_verification_checklist.py` to verify:
- Function `verify_presentation_file(ppt_path)` runs and returns `(is_valid, passed_checks, failed_checks)`.
- Verifies 10 core deck invariants:
  1. Slide count == 23.
  2. Slide dimensions 13.333" x 7.500" (16:9).
  3. 0 Arial font elements (100% Times New Roman for Latin text).
  4. Slide 2 Table of Contents strictly ordered 01 to 06.
  5. Slide 6 contains Hallucination Verification, FEVER, and 2D Trust Matrix.
  6. Slide 7 contains Figure 3 full-width framework.
  7. Slide 8 contains Dual-Stream Gated Fusion formula and 83 FST rules.
  8. Slide 9 contains Robust Generalization & Adversarial Rewriting Defense.
  9. Slide 10 contains Kazakh-FEVER and Two-Dimensional Trust Matrix.
  10. Slide 18 contains 314 / 314 tests and zero emoji absence.
  11. Slide 19 contains ~85% complete and 4 contributions.
  12. Slide 20 contains Springer LNCS and Gradio demo agenda.
  13. Slide 21 contains 3 strategic consultation questions for Prof. Guo.
  14. Zero banned terms ("cross-attention", "catastrophic", "breakthrough", "perfect", "Fan Qianyue").

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_presentation_verification_checklist.py`  
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.verify_presentation_against_checklist'`

- [ ] **Step 3: Implement `scripts/verify_presentation_against_checklist.py`**

Implement the script with clean functions, CLI `argparse`, and colored terminal reporting:
- `check_slide_count_and_dimensions(prs)`
- `check_typography(prs)`
- `check_sequential_toc(prs)`
- `check_content_anchors(prs)`
- `check_banned_terms(prs)`
- `main()` checking both `C:\Users\Roza\Desktop\AnekeshD_Progress.pptx` and the repo copy.

- [ ] **Step 4: Run script and test suite to verify 100% pass**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe scripts/verify_presentation_against_checklist.py`  
Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest discover -s tests -p "test_*.py"`  
Expected: PASS (all tests passing, 0 errors).

- [ ] **Step 5: Commit script and test updates**

```bash
git add scripts/verify_presentation_against_checklist.py tests/test_presentation_verification_checklist.py
git commit -m "feat(audit): implement automated presentation verification checker script"
```
