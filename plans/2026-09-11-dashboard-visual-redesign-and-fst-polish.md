# Dashboard Visual Redesign & FST Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernize the Kazakh AI-Text Detection UI with publication-grade executive summary cards, direct top-level tab navigation, left-aligned language switcher, refined heatmap/gate visuals, and agentive FST morphological stemming.

**Architecture:** Refactor `ui/highlighting.py` to produce a unified executive KPI card, modern gradient confidence meter (retiring ASCII brackets), and pill legends; update `ui/app.py` to launch with tabs at the very top and left-align the language switcher; enhance `fst_analyzer.py` and `ui/sentence_analyzer.py` with agentive derivational suffix `-ші/-шы` handling; sync all changes to `hf_space/`.

**Tech Stack:** Python 3.12, Gradio, PyTorch, regex, HTML5/CSS3.

## Global Constraints

- Under no circumstances shall `aist2026/paper.tex` be touched or modified.
- Zero decorative emojis in any UI labels, buttons, cards, legends, tables, or markdown documents.
- Mandatory strict XSS sanitization: all dynamic user text in HTML templates must pass through `html.escape(..., quote=True)`.
- Defensive operation in CPU/mock environments without requiring active GPUs.
- Clean UTF-8 encoding across all files.
- Maintain 100% pass rate across all existing tests.

---

### Task 1: FST Agentive Derivational Suffix Support (`-ші/-шы`)

**Files:**
- Modify: `fst_analyzer.py:120-180`
- Modify: `ui/sentence_analyzer.py:60-120`
- Test: `tests/test_ui_sentence_analyzer.py`

**Interfaces:**
- Consumes: Kazakh word strings (e.g. `жетекші`, `жұмысшы`, `жазушы`).
- Produces: Correct decomposition dictionary `{"word": "жетекші", "root": "жетек", "pos": "NOUN", "affixes": ["-ші (DERIV.AGENT)"]}` instead of false possessive stem `жетекш`.

- [ ] **Step 1: Write the failing test in `tests/test_ui_sentence_analyzer.py`**

```python
def test_agentive_suffix_stemming(self):
    from ui.sentence_analyzer import analyze_sentence_morphemes
    res = analyze_sentence_morphemes("жетекші")
    self.assertEqual(len(res), 1)
    self.assertEqual(res[0]["word"], "жетекші")
    self.assertEqual(res[0]["root"], "жетек")
    self.assertEqual(res[0]["pos"], "NOUN")
    self.assertTrue(any("AGENT" in aff or "-ші" in aff for aff in res[0]["affixes"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_ui_sentence_analyzer.py`
Expected: FAIL (root is `'жетекш'` instead of `'жетек'`).

- [ ] **Step 3: Implement agentive derivational suffix rule in `fst_analyzer.py` and `ui/sentence_analyzer.py`**

In `ui/sentence_analyzer.py`:
```python
AGENT_DERIV_TAGS = {
    "-ші": "AGENT",
    "-шы": "AGENT",
}
```
In `analyze_sentence_morphemes`:
Before stripping 3rd person possessive `-і` / `-ы`, check if the word ends with `-ші` or `-шы` and length > 4 (or base root has at least 3 chars), stripping `-ші`/`-шы` as `DERIV.AGENT` rather than stripping the trailing vowel as possessive.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_ui_sentence_analyzer.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fst_analyzer.py ui/sentence_analyzer.py tests/test_ui_sentence_analyzer.py
git commit -m "feat(fst): add agentive derivational suffix rule for words like жетекші"
```

---

### Task 2: Highlighting Engine Modernization (Executive KPI Card & Modern Meter)

**Files:**
- Modify: `ui/highlighting.py:165-270, 575-645`
- Test: `tests/test_ui_highlighting.py`

**Interfaces:**
- Consumes: `probability`, `verdict`, `lang`, `total_words`, `total_sents`, `total_windows`, `ai_ratio`.
- Produces:
  - `render_confidence_meter(probability, verdict, lang)`: Clean gradient meter without ASCII brackets.
  - `render_executive_summary_card(probability, verdict, lang, total_words, total_sents, total_windows, ai_ratio)`: Comprehensive publication-grade executive card with status badge, gradient progress bar, and 3-tile KPI grid.
  - `render_legend_html(lang)`: High-contrast 3-tier legend pills.

- [ ] **Step 1: Write failing tests in `tests/test_ui_highlighting.py`**

```python
def test_render_confidence_meter_no_ascii_brackets(self):
    html_out = render_confidence_meter(0.03, "[AUTHENTIC HUMAN]", lang="en")
    self.assertNotIn("[======", html_out)
    self.assertIn("confidence-track", html_out)
    self.assertIn("3.0%", html_out)

def test_render_executive_summary_card(self):
    from ui.highlighting import render_executive_summary_card
    html_out = render_executive_summary_card(
        probability=0.03,
        verdict="[AUTHENTIC HUMAN]",
        lang="en",
        total_words=120,
        total_sents=8,
        total_windows=1,
        ai_ratio=0.0
    )
    self.assertIn("executive-summary-card", html_out)
    self.assertIn("3.0%", html_out)
    self.assertIn("120", html_out)
    self.assertNotIn("[======", html_out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_ui_highlighting.py`
Expected: FAIL (`render_executive_summary_card` not defined / `[======` present).

- [ ] **Step 3: Implement `render_executive_summary_card` and update `render_confidence_meter` and CSS**

In `ui/highlighting.py`:
1. Remove ASCII bracket generation from `render_confidence_meter`.
2. Add `render_executive_summary_card`:
   - Verdict status badge (`badge-human`, `badge-hybrid`, `badge-ai`).
   - Rounded progress track with smooth CSS gradient fill.
   - 3-tile KPI grid (`AI Probability`, `AI Content Ratio`, `Document Volume`).
3. Refine `HEATMAP_CSS` and `APP_CSS` styling for dark/light contrast and rounded pills.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_ui_highlighting.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ui/highlighting.py tests/test_ui_highlighting.py
git commit -m "feat(ui): implement publication-grade executive summary card and modern progress meter"
```

---

### Task 3: Tabbed UI Header Streamlining & Layout Integration

**Files:**
- Modify: `ui/app.py:780-820, 1130-1360`
- Test: `tests/test_ui_integration.py`

**Interfaces:**
- Consumes: `render_executive_summary_card`, `render_legend_html`, `switch_ui_language`.
- Produces: Gradio Blocks demo starting directly with the 3 tabs at top, left-aligned language switcher next to Quick Samples, and unified executive card display.

- [ ] **Step 1: Update `tests/test_ui_integration.py`**

Test that:
- `create_app()` launches without `hero_banner`.
- `switch_ui_language()` returns 39 components (synchronously updating labels and executive card without hero banner).
- Language toggle is aligned with Quick Samples header.
- Zero decorative emojis present anywhere in layout.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_ui_integration.py`
Expected: FAIL (mismatched outputs count or missing hero_banner).

- [ ] **Step 3: Refactor `ui/app.py` layout and event handlers**

1. Remove `hero_banner = gr.HTML(render_hero_html("en"))` from `create_app`.
2. In Tab 1 header:
   Replace split columns with single left-aligned flex row:
   ```python
   with gr.Row(elem_classes=["quick-samples-header-flex"]):
       quick_samples_label = gr.Markdown(f"**{d['quick_samples_title']}**", elem_classes=["quick-samples-title"])
       lang_radio = gr.Radio(
           choices=["English", "Қазақша"],
           value="English",
           show_label=False,
           container=False,
           interactive=True,
           elem_classes=["lang-switch-inline"]
       )
   ```
3. In Tab 1 Right Column:
   Replace separate `verdict_badge`, `confidence_meter_display`, `prob_box`, `ratio_box`, and `stats_display` with `executive_card_display = gr.HTML(...)`.
4. Update `analyze_text`, `clear_all`, and `switch_ui_language` to populate `executive_card_display`.
5. Update `switch_ui_language` return tuple and event bindings (39 outputs).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_ui_integration.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ui/app.py tests/test_ui_integration.py
git commit -m "feat(ui): start tabs at top, left-align language switcher, and embed executive card"
```

---

### Task 4: Hugging Face Space Bundle Synchronization & Deployment Verification

**Files:**
- Modify: `hf_space/app.py`
- Modify: `hf_space/ui/highlighting.py`
- Modify: `hf_space/ui/sentence_analyzer.py`
- Modify: `hf_space/fst_analyzer.py`
- Test: `tests/test_hf_space_bundle.py`

**Interfaces:**
- Consumes: Updated project files.
- Produces: Synchronized `hf_space/` bundle passing all regression tests and deployable via `scripts/push_to_hf.py`.

- [ ] **Step 1: Copy updated files to `hf_space/`**

Copy `ui/app.py` -> `hf_space/app.py`, `ui/highlighting.py` -> `hf_space/ui/highlighting.py`, `ui/sentence_analyzer.py` -> `hf_space/ui/sentence_analyzer.py`, `fst_analyzer.py` -> `hf_space/fst_analyzer.py`.

- [ ] **Step 2: Run all test suites**

Run: `python -m unittest discover -s tests -p "test_*.py"`
Expected: All tests PASS with 0 errors.

- [ ] **Step 3: Commit and push**

```bash
git add hf_space/ tests/
git commit -m "chore(hf_space): sync visual redesign and morphological FST enhancements to HF Space bundle"
git push origin main
```

- [ ] **Step 4: Deploy to Hugging Face Spaces**

Run: `python scripts/push_to_hf.py --token <HF_TOKEN>`
Verify deployment status.

