# Kaz-ExplainabilityUI: Sentence-Level Highlighting & Interactive Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade, interactive explainability web dashboard in Gradio featuring document-level KPI metrics, XSS-safe sentence-level color-highlighted heatmaps, one-click benchmark demonstration samples across informal reviews, formal news, Wikipedia, and hybrid texts, and deep morphological FST word breakdown with dynamic fusion gate analysis.

**Architecture:**
- `ui/highlighting.py`: XSS-safe HTML/CSS rendering engine for sentence heatmap generation with calibrated threshold color mapping.
- `ui/file_loader.py`: Safe document loader supporting `.txt`, `.docx`, and `.pdf` with 10MB size limits and graceful fallbacks.
- `ui/presets.py`: Curated benchmark text samples (Kaspi review, formal news, Wikipedia, Sherkala AI, Qwen AI, hybrid essay).
- `ui/sentence_analyzer.py`: Sentence-level scoring and FST morphological token breakdown generator.
- `ui/app.py`: Modernized Gradio Blocks interface wiring the reactive dashboard.

**Tech Stack:** Python 3.12, Gradio, `AdvancedKazakhFSTAnalyzer`, `DocumentDetector`, HTML/CSS.

## Global Constraints
- Under no circumstances shall `aist2026/paper.tex` be touched or modified.
- Full regression test suite must pass with 0 errors (all existing 121 tests preserved).
- Mandatory XSS prevention: all text rendered in HTML components must pass through `html.escape()`.
- Defensive operation: UI must run seamlessly in CPU/mock environments without requiring active GPUs.

---

### Task 1: XSS-Safe Sentence Heatmap & HTML/CSS Highlighting Engine

**Files:**
- Create: `ui/highlighting.py`
- Test: `tests/test_ui_highlighting.py`

**Interfaces:**
- Produces:
  - `render_document_heatmap(sentences_data: List[Dict[str, Any]], calibrated_threshold: float = 0.9980) -> str`
  - `render_dynamic_gate_bar(gate_value: float) -> str`
  - `render_morpheme_table(word_breakdowns: List[Dict[str, Any]]) -> str`
  - CSS stylesheet string `HEATMAP_CSS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ui_highlighting.py
import unittest
import html
from ui.highlighting import render_document_heatmap, render_dynamic_gate_bar, render_morpheme_table

class TestUiHighlighting(unittest.TestCase):
    def test_xss_escaping_in_heatmap(self):
        malicious_input = [{"index": 0, "text": "<script>alert('xss')</script>", "ai_probability": 0.05}]
        rendered = render_document_heatmap(malicious_input)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;", rendered)

    def test_color_scale_classification(self):
        sents = [
            {"index": 0, "text": "Адам мәтіні.", "ai_probability": 0.05},
            {"index": 1, "text": "Аралық мәтін.", "ai_probability": 0.50},
            {"index": 2, "text": "Жасанды мәтін.", "ai_probability": 0.9995}
        ]
        rendered = render_document_heatmap(sents, calibrated_threshold=0.9980)
        self.assertIn("lvl-human", rendered)
        self.assertIn("lvl-amber", rendered)
        self.assertIn("lvl-ai", rendered)

    def test_dynamic_gate_bar_rendering(self):
        bar_html = render_dynamic_gate_bar(0.523)
        self.assertIn("52.3%", bar_html)
        self.assertIn("47.7%", bar_html)

    def test_morpheme_table_rendering(self):
        breakdowns = [
            {"word": "жасанды", "root": "жаса", "pos": "VERB", "affixes": ["-н (PASS)", "-ды (ADJ)"]}
        ]
        table_html = render_morpheme_table(breakdowns)
        self.assertIn("жасанды", table_html)
        self.assertIn("жаса", table_html)
        self.assertIn("-н (PASS)", table_html)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_ui_highlighting.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'ui.highlighting'`

- [ ] **Step 3: Implement `ui/highlighting.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_ui_highlighting.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/highlighting.py tests/test_ui_highlighting.py
git commit -m "feat: implement XSS-safe sentence heatmap and HTML/CSS explainability rendering"
```

---

### Task 2: Defensive File Loader & Benchmark Preset Demonstrations

**Files:**
- Create: `ui/file_loader.py`
- Create: `ui/presets.py`
- Test: `tests/test_ui_file_loader.py`
- Test: `tests/test_ui_presets.py`

**Interfaces:**
- Produces:
  - `load_document_file(file_obj_or_path: Any) -> Tuple[str, Optional[str]]` (returns `(text, error_message)`)
  - `PRESET_SAMPLES: Dict[str, Dict[str, str]]` containing 6 curated samples with metadata.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ui_file_loader.py
import unittest
import tempfile
import os
from ui.file_loader import load_document_file

class TestUiFileLoader(unittest.TestCase):
    def test_load_txt_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
            f.write("Қазақстанның болашағы жастардың қолында.")
            tmp_path = f.name
        try:
            text, err = load_document_file(tmp_path)
            self.assertIsNone(err)
            self.assertEqual(text, "Қазақстанның болашағы жастардың қолында.")
        finally:
            os.remove(tmp_path)

    def test_file_size_limit(self):
        # Create dummy file > 10MB
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="wb") as f:
            f.write(b"0" * (11 * 1024 * 1024))
            tmp_path = f.name
        try:
            text, err = load_document_file(tmp_path)
            self.assertIn("10 MB", err)
            self.assertEqual(text, "")
        finally:
            os.remove(tmp_path)
```

```python
# tests/test_ui_presets.py
import unittest
from ui.presets import PRESET_SAMPLES, get_preset_choices, get_preset_text

class TestUiPresets(unittest.TestCase):
    def test_all_six_presets_available(self):
        choices = get_preset_choices()
        self.assertEqual(len(choices), 6)
        for c in choices:
            text = get_preset_text(c)
            self.assertGreater(len(text), 20)
            self.assertIn(c, PRESET_SAMPLES)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests/test_ui_file_loader.py tests/test_ui_presets.py`
Expected: FAIL

- [ ] **Step 3: Implement `ui/file_loader.py` and `ui/presets.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests/test_ui_file_loader.py tests/test_ui_presets.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/file_loader.py ui/presets.py tests/test_ui_file_loader.py tests/test_ui_presets.py
git commit -m "feat: implement defensive file loader and 6 curated benchmark presets"
```

---

### Task 3: Sentence-Level Scoring & FST Morphological Analyzer Engine

**Files:**
- Create: `ui/sentence_analyzer.py`
- Test: `tests/test_ui_sentence_analyzer.py`

**Interfaces:**
- Consumes:
  - `SentencePreservingChunker` from `kaz_mage.chunker`
  - `AdvancedKazakhFSTAnalyzer` from `fst_analyzer`
- Produces:
  - `analyze_document_sentences(text: str, document_result: DocumentAnalysisResult, fst_analyzer=None) -> List[Dict[str, Any]]`
  - `analyze_sentence_morphemes(sentence_text: str, fst_analyzer=None) -> List[Dict[str, Any]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ui_sentence_analyzer.py
import unittest
from ui.sentence_analyzer import analyze_sentence_morphemes, analyze_document_sentences
from kaz_mage.document import DocumentAnalysisResult, DocumentChunk

class TestUiSentenceAnalyzer(unittest.TestCase):
    def test_sentence_morphemes_breakdown(self):
        breakdowns = analyze_sentence_morphemes("Қазақстанның болашағы жарқын.")
        self.assertGreater(len(breakdowns), 0)
        self.assertEqual(breakdowns[0]["word"], "Қазақстанның")
        self.assertIn("root", breakdowns[0])
        self.assertIn("affixes", breakdowns[0])

    def test_document_sentence_attribution(self):
        chunk = DocumentChunk(0, "Бұл бірінші сөйлем. Бұл екінші сөйлем.", 0, 38, 6, 2, ai_probability=0.999, gate_value=0.52, is_ai=True)
        doc_res = DocumentAnalysisResult("Machine-Generated", 0.999, 1.0, 0.998, 6, 2, 1, chunk, [chunk])
        sents = analyze_document_sentences("Бұл бірінші сөйлем. Бұл екінші сөйлем.", doc_res)
        self.assertEqual(len(sents), 2)
        self.assertAlmostEqual(sents[0]["ai_probability"], 0.999, places=3)
        self.assertEqual(sents[0]["index"], 0)
        self.assertEqual(sents[1]["index"], 1)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_ui_sentence_analyzer.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'ui.sentence_analyzer'`

- [ ] **Step 3: Implement `ui/sentence_analyzer.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_ui_sentence_analyzer.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/sentence_analyzer.py tests/test_ui_sentence_analyzer.py
git commit -m "feat: implement sentence-level scoring and FST morphological analysis engine"
```

---

### Task 4: Modernized Gradio Dashboard Integration

**Files:**
- Modify: `ui/app.py`
- Test: `tests/test_ui_integration.py`

**Interfaces:**
- Produces:
  - `create_app() -> gr.Blocks`
  - Standalone executable CLI `python ui/app.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ui_integration.py
import unittest
from ui.app import create_app

class TestUiIntegration(unittest.TestCase):
    def test_gradio_app_structure_and_components(self):
        demo = create_app(load_model=False)
        self.assertIsNotNone(demo)
        # Verify app title and block structure
        self.assertEqual(demo.title, "Kazakh AI-Text Detector & Explainability UI")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_ui_integration.py`
Expected: FAIL

- [ ] **Step 3: Implement `ui/app.py`**

Assemble the Gradio Blocks UI integrating the preset buttons, file upload handler, document heatmap, dropdown sentence inspector, and FST drilldown card.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_ui_integration.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/app.py tests/test_ui_integration.py
git commit -m "feat: modernize Gradio app with document heatmap and interactive explainability"
```

---

### Task 5: Full Regression, End-to-End Verification, Whole-Branch Review & Merge

- [ ] **Step 1: Run all UI tests**
`python -m unittest tests/test_ui_highlighting.py tests/test_ui_file_loader.py tests/test_ui_presets.py tests/test_ui_sentence_analyzer.py tests/test_ui_integration.py`
Expected: All pass.

- [ ] **Step 2: Run complete project regression suite (125+ tests)**
`python -m unittest discover -s tests -p "test_*.py"`
Expected: All pass.

- [ ] **Step 3: Dispatch whole-branch code reviewer subagent**
Verify zero regressions, XSS safety, and zero modifications to `aist2026/paper.tex`.

- [ ] **Step 4: Merge `feat/kaz-explainability-ui` into `main`**

```bash
git checkout main
git merge --no-ff feat/kaz-explainability-ui -m "Merge branch 'feat/kaz-explainability-ui' into main"
```
