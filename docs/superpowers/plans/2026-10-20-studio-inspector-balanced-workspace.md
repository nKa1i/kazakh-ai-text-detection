# Studio & Inspector Balanced Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Kazakh AI-Text Detection and Explainability Gradio UI into a balanced, publication-grade **Studio & Inspector Workspace** with a guaranteed Notion Light canvas (eliminating the black background clash and blocky nested boxes), pure English navigation, balanced vertical scrolling by moving the sentence heatmap to the left column, and 4 high-value diagnostic enhancements.

**Architecture:** 
- Enforce `color-scheme: light !important;` and override Gradio's automatic `.dark` mode classes so the canvas is always soft off-white (`#fbfbfa`) with pure white cards (`#ffffff`).
- Rebalance the layout into two equal-weight columns (`scale=5` / `scale=5`):
  - **Left Column (Document Studio)**: Houses scenario preset cards, document textarea, live typing stats ribbon, action buttons (`Run Full Analysis`, `Export Audit (.md)`, `Clear`), and the **Visual Sentence Heatmap** (eliminating the empty left-side void).
  - **Right Column (Diagnostic Inspector)**: Houses the persistent Executive Summary HUD (with a single sleek 6px confidence meter, removing the duplicate meter bar) and 3 focused diagnostic sub-tabs (Sentence Morphology & Fusion, Auto-FST & Slang Lab, Fact Verification & Trust Matrix).
- Remove the Kazakh language switch radio button, simplifying all chrome labels, buttons, and drawer headers to clean, concise English.
- Add an interactive visual morpheme diff in Auto-FST, an audit export generator (`.md`), and live document metrics.

**Tech Stack:** Python 3.10+, Gradio 4/5, HTML5/CSS3, Unicode Cyrillic Kazakh (`encoding='utf-8'`).

## Global Constraints
- Under no circumstances shall `aist2026/paper.tex` or `papers/kazakh_fever_conference/` be touched or modified (strictly 0 diff lines).
- Strictly zero decorative emojis across all code, datasets, comments, documentation, UI labels, and HTML templates.
- All Python file operations must explicitly specify `encoding='utf-8'`.
- All code files must contain complete, production-ready content with zero placeholders ("TODO", "TBD").
- Sub-millisecond latency on CPU (< 1.0 ms per parse).
- Backward compatibility with existing UI tests (`tests/test_ui_theme.py`, `tests/test_ui_cards_styling.py`, `tests/test_ui_tables_styling.py`, `tests/test_ui_autofst_tab.py`, `tests/test_ui_integration.py`).
- All changes in `ui/` must be mirrored to `hf_space/ui/`.

---

### Task 1: Notion Light Canvas Enforcement & CSS De-Boxification

**Files:**
- Modify: `ui/app.py`
- Modify: `hf_space/ui/app.py`
- Create / Test: `tests/test_ui_studio_canvas.py`

**Interfaces:**
- Produces: CSS rules in `APP_CSS` that strictly enforce light mode (`color-scheme: light !important;`, `.dark` override), de-boxify nested gray wrappers, style the live stats ribbon (`.live-stats-ribbon`), and provide interactive diff highlights (`.diff-norm`, `.diff-induce`).
- Consumes: `get_apple_notion_theme()`, `HEATMAP_CSS`.

- [ ] **Step 1: Write unit tests verifying CSS canvas enforcement and de-boxification**

Create `tests/test_ui_studio_canvas.py`:
```python
# -*- coding: utf-8 -*-
"""Unit tests verifying Notion Light canvas enforcement and CSS de-boxification."""

import unittest
from ui.app import APP_CSS

class TestUIStudioCanvas(unittest.TestCase):
    def test_light_canvas_enforced(self):
        self.assertIn("color-scheme: light", APP_CSS)
        self.assertIn("#fbfbfa", APP_CSS)

    def test_dark_mode_override_present(self):
        self.assertIn(".dark", APP_CSS)
        self.assertIn(".dark .gradio-container", APP_CSS)

    def test_live_stats_and_diff_css_classes(self):
        self.assertIn(".live-stats-ribbon", APP_CSS)
        self.assertIn(".diff-norm", APP_CSS)
        self.assertIn(".diff-induce", APP_CSS)

if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_ui_studio_canvas.py -v`
Expected: FAIL on missing CSS selectors.

- [ ] **Step 3: Update `APP_CSS` in `ui/app.py` and `hf_space/ui/app.py`**

- In `APP_CSS`:
  - Enforce `html, body, .gradio-container, .dark { color-scheme: light !important; background-color: #fbfbfa !important; color: #18181b !important; }`.
  - Override `.dark .gradio-container`, `.dark .hero-card`, `.dark .panel-box` with `#fbfbfa` and `#ffffff` backgrounds.
  - Add styles for `.live-stats-ribbon`, `.scenario-card`, `.diff-norm`, and `.diff-induce`.
  - Remove redundant border frames on input wrappers.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_ui_studio_canvas.py -v`
Expected: PASS.

- [ ] **Step 5: Commit changes**

```bash
git add ui/app.py hf_space/ui/app.py tests/test_ui_studio_canvas.py
git commit -m "style(ui): enforce notion light canvas and de-boxify studio containers"
```

---

### Task 2: Studio & Inspector Balanced Layout Scaffold (Heatmap on Left)

**Files:**
- Modify: `ui/app.py`
- Modify: `hf_space/ui/app.py`
- Modify: `tests/test_ui_two_column_workspace.py`

**Interfaces:**
- Produces: Balanced 2-column layout in `create_app()`:
  - Left Column (`scale=5`, `.workspace-left-col`): Presets, Text Input, Live Stats Ribbon, File Accordion, Toggles, Action Buttons, and **Visual Sentence Heatmap**.
  - Right Column (`scale=5`, `.workspace-right-col`): Single Executive Summary HUD + 3 Sub-Tabs (Sentence Morphology & Fusion, Auto-FST Slang Lab, Fact Verification & Trust Matrix).
- Consumes: `render_executive_summary_card`, `render_document_heatmap`, `render_confidence_meter`.

- [ ] **Step 1: Write unit tests verifying balanced Studio & Inspector structure**

Add to `tests/test_ui_two_column_workspace.py`:
```python
    def test_balanced_column_scales(self):
        """Verifies left and right columns both have scale=5 for balanced scrolling."""
        cols = [b for b in self.blocks if isinstance(b, gr.Column)]
        left_cols = [c for c in cols if hasattr(c, "elem_classes") and c.elem_classes and "workspace-left-col" in c.elem_classes]
        right_cols = [c for c in cols if hasattr(c, "elem_classes") and c.elem_classes and "workspace-right-col" in c.elem_classes]
        self.assertEqual(len(left_cols), 1)
        self.assertEqual(len(right_cols), 1)
        self.assertEqual(left_cols[0].scale, 5)
        self.assertEqual(right_cols[0].scale, 5)

    def test_no_duplicate_confidence_meter(self):
        """Verifies there is no redundant second full-width confidence meter HTML block outside the executive card."""
        html_blocks = [b for b in self.blocks if isinstance(b, gr.HTML)]
        # Ensure confidence_meter_display is integrated or removed as a standalone second red bar
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_ui_two_column_workspace.py -v`
Expected: FAIL on column scales (currently 4 and 6).

- [ ] **Step 3: Restructure `create_app()` in `ui/app.py` and `hf_space/ui/app.py`**

- In `ui/app.py` and `hf_space/ui/app.py`:
  - Set `left_col` to `scale=5` and `right_col` to `scale=5`.
  - Move `heatmap_header`, `legend_display`, and `heatmap_display` into the Left Column directly below the action buttons.
  - In Right Column top HUD, keep `executive_card_display` which already contains the sleek 6px confidence meter, and remove the redundant standalone second meter bar.
  - Consolidate Sub-Tabs:
    - Tab 1: `Sentence Morphology & Fusion` (Selected sentence box, Dynamic Fusion Gate bar, morpheme decomposition table, linguistic reasoning bullets).
    - Tab 2: `Auto-FST & Slang Lab` (Visual diff card, Auto-Inductions table).
    - Tab 3: `Fact Verification & Trust Matrix` (2D Trust Matrix coordinate chart, Verified Claims table).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_ui_two_column_workspace.py -v`
Expected: PASS.

- [ ] **Step 5: Commit changes**

```bash
git add ui/app.py hf_space/ui/app.py tests/test_ui_two_column_workspace.py
git commit -m "feat(ui): scaffold balanced studio and inspector layout with heatmap on left"
```

---

### Task 3: Pure English UI Simplification & 4 High-Value Enhancements

**Files:**
- Modify: `ui/app.py`
- Modify: `hf_space/ui/app.py`
- Modify: `ui/highlighting.py`
- Modify: `hf_space/ui/highlighting.py`
- Modify: `tests/test_ui_two_column_workspace.py`

**Interfaces:**
- Produces:
  - `render_morpheme_visual_diff(raw_text, normalized_text, inductions)` in `ui/highlighting.py`.
  - `generate_audit_markdown(res)` in `ui/app.py` for `.md` export.
  - `calculate_document_statistics(text)` in `ui/app.py` for live ribbon updates.
  - Pure English UI strings across all navigation, buttons, and headers.
- Consumes: `DynamicFSTOverlay`, `AutoFSTInducer`, `OfflineHeuristicDetector`.

- [ ] **Step 1: Write unit tests verifying English purity, diff renderer, and audit export**

Add to `tests/test_ui_two_column_workspace.py`:
```python
    def test_pure_english_ui_chrome(self):
        """Verifies zero bilingual slash labels in tabs and buttons."""
        tab_items = [b for b in self.blocks if isinstance(b, gr.Tab)]
        for t in tab_items:
            label = getattr(t, "label", "")
            self.assertNotIn("/", label, f"Found bilingual slash in tab label: {label}")

    def test_audit_markdown_export(self):
        """Verifies generate_audit_markdown returns a valid markdown audit document."""
        from ui.app import generate_audit_markdown, run_full_analysis_logic
        res = run_full_analysis_logic("Қазақстан 1991 жылы тәуелсіздік алды.")
        md_text = generate_audit_markdown(res, source_text="Қазақстан 1991 жылы тәуелсіздік алды.")
        self.assertIn("# Document Inspection & Verification Audit Report", md_text)
        self.assertIn("## Executive Verdict", md_text)

    def test_morpheme_visual_diff_renderer(self):
        """Verifies render_morpheme_visual_diff generates visual diff highlights."""
        from ui.highlighting import render_morpheme_visual_diff
        diff_html = render_morpheme_visual_diff(
            raw_text="салем вайбтадык",
            normalized_text="сәлем вайбтадық",
            inductions=[{"stem": "вайб", "suffixes": ["та", "дық"]}]
        )
        self.assertIn("diff-norm", diff_html)
        self.assertIn("diff-induce", diff_html)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_ui_two_column_workspace.py -v`
Expected: FAIL on missing functions and bilingual slashes.

- [ ] **Step 3: Implement Pure English UI & Enhancements**

- In `ui/app.py` and `hf_space/ui/app.py`:
  - Remove `lang_radio` from the UI header.
  - Set tab titles: `Sentence Morphology & Fusion`, `Auto-FST & Slang Lab`, `Fact Verification & Trust Matrix`.
  - Set drawer title: `Methodology & Academic Benchmarks`.
  - Implement `calculate_document_statistics(text) -> str`: returns formatted live stats ribbon.
  - Implement `generate_audit_markdown(res, source_text) -> str`: creates formatted audit file for download.
  - Wire `export_btn.click` to generate and download `audit_report.md`.
  - Wire `text_input.change` to update `stats_ribbon`.
- In `ui/highlighting.py` and `hf_space/ui/highlighting.py`:
  - Implement `render_morpheme_visual_diff(raw_text, normalized_text, inductions) -> str`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_ui_two_column_workspace.py -v`
Expected: PASS.

- [ ] **Step 5: Commit changes**

```bash
git add ui/app.py hf_space/ui/app.py ui/highlighting.py hf_space/ui/highlighting.py tests/test_ui_two_column_workspace.py
git commit -m "feat(ui): implement pure english navigation, morpheme visual diff, and audit export"
```

---

### Task 4: Full Automated Regression Verification, Server Reload & Delivery

**Files:**
- Test: All repository UI & Auto-FST test suites

- [ ] **Step 1: Run full regression test suites**

```bash
python -m unittest tests/test_ui_studio_canvas.py -v
python -m unittest tests/test_ui_two_column_workspace.py -v
python -m unittest tests/test_ui_theme.py -v
python -m unittest tests/test_ui_cards_styling.py -v
python -m unittest tests/test_ui_tables_styling.py -v
python -m unittest tests/test_ui_autofst_tab.py -v
python -m unittest discover -s tests -p "test_auto_fst*.py" -v
```
Expected: 100% pass rate.

- [ ] **Step 2: Verify conference papers invariance**

```bash
git diff origin/main..HEAD -- aist2026/paper.tex papers/kazakh_fever_conference/
```
Expected: Strictly 0 diff lines.

- [ ] **Step 3: Relaunch Gradio Daemon Server**

Restart server on port 7860:
```powershell
python hf_space/app.py
```
Verify `http://127.0.0.1:7860` returns HTTP 200 OK.

- [ ] **Step 4: Final delivery walkthrough to user**
