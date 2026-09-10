# Kaz-Dashboard Redesign & Hugging Face Spaces Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Kazakh AI text detection UI into a sleek, academic, publication-grade dashboard with zero emoji clutter, a 3-tab layout (Detection, Morphological FST Lab, Benchmark Methodology), dynamic linguistic reasoning bullets, and a 1-click cloud deployment bundle for Hugging Face Spaces.

**Architecture:** 
- `ui/linguistic_explainer.py`: Computes lexical diversity (TTR), discourse markers, and FST suffix complexity to generate dynamic plain-language reasoning bullets.
- `ui/app.py` & `ui/highlighting.py`: Modernized Gradio Blocks UI featuring quick-sample preset pills, a horizontal confidence meter, and clean tabbed navigation with bilingual support (`Қазақша` / `English`).
- `hf_space/`: Standalone, self-contained cloud bundle with instant sub-second CPU cold start for Hugging Face Spaces.
- `scripts/push_to_hf.py`: One-command CLI upload script using `huggingface_hub`.

**Tech Stack:** Python 3.12, Gradio 6.x, `AdvancedKazakhFSTAnalyzer`, `SentencePreservingChunker`, `DocumentAggregator`, `huggingface_hub`.

## Global Constraints
- Under no circumstances shall `aist2026/paper.tex` be touched or modified.
- Full regression test suite must pass with 0 errors (all 174 existing tests preserved).
- Zero emoji clutter in UI: use clean typography, status pills (`[AUTHENTIC HUMAN]`), and subtle border styling.
- Strict XSS sanitization: all dynamic user content must pass through `html.escape(..., quote=True)`.
- Defensive operation: UI must run seamlessly in CPU/mock environments without requiring active GPUs.

---

### Task 1: Dynamic Linguistic Explainer Engine

**Files:**
- Create: `ui/linguistic_explainer.py`
- Test: `tests/test_ui_linguistic_explainer.py`

**Interfaces:**
- Produces:
  - `generate_linguistic_explanation(text: str, doc_result: Optional[DocumentAnalysisResult] = None, lang: str = "kz") -> List[str]`
  - `compute_linguistic_features(text: str) -> Dict[str, Any]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ui_linguistic_explainer.py
import unittest
from ui.linguistic_explainer import compute_linguistic_features, generate_linguistic_explanation
from kaz_mage.document import DocumentAnalysisResult, DocumentChunk

class TestLinguisticExplainer(unittest.TestCase):
    def test_compute_features_human_text(self):
        text = "Бұл менің сүйікті дүкенімнен сатып алған тауарым. Сапасы өте керемет, жеткізу жылдам болды."
        feats = compute_linguistic_features(text)
        self.assertGreater(feats["colloquial_marker_count"], 0)
        self.assertEqual(feats["formulaic_marker_count"], 0)
        self.assertGreater(feats["type_token_ratio"], 0.7)

    def test_compute_features_ai_text(self):
        text = "Қорытындылай келе, бұл құбылыс заманауи қоғам үшін ерекше маңызды рөл атқарады. Осыған орай, жүйелі түрде талдау жасау қажет."
        feats = compute_linguistic_features(text)
        self.assertGreater(feats["formulaic_marker_count"], 0)

    def test_generate_explanation_bilingual(self):
        text = "Қорытындылай келе, айта кету керек, маңызды рөл атқарады."
        chunk = DocumentChunk(0, text, 0, len(text), 7, 1, ai_probability=0.9995, is_ai=True)
        res = DocumentAnalysisResult("Machine-Generated", 0.9995, 1.0, 0.9980, 7, 1, 1, chunk, [chunk])
        
        bullets_kz = generate_linguistic_explanation(text, res, lang="kz")
        self.assertGreater(len(bullets_kz), 0)
        self.assertTrue(any("формулалық" in b.lower() or "дискурстық" in b.lower() for b in bullets_kz))

        bullets_en = generate_linguistic_explanation(text, res, lang="en")
        self.assertGreater(len(bullets_en), 0)
        self.assertTrue(any("discourse" in b.lower() or "formulaic" in b.lower() for b in bullets_en))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m unittest tests/test_ui_linguistic_explainer.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'ui.linguistic_explainer'`

- [ ] **Step 3: Implement `ui/linguistic_explainer.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_ui_linguistic_explainer.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/linguistic_explainer.py tests/test_ui_linguistic_explainer.py
git commit -m "feat: implement dynamic linguistic explanation engine"
```

---

### Task 2: Academic Aesthetic & Tabbed Dashboard Redesign

**Files:**
- Modify: `ui/app.py`
- Modify: `ui/highlighting.py`
- Modify: `tests/test_ui_integration.py`

**Interfaces:**
- Produces:
  - Redesigned `create_app(detector=None, load_model=False) -> gr.Blocks`
  - Tab 1: Detection & Explainability (Quick sample pills, input, confidence progress bar, plain-language bullets, heatmap, inline inspector).
  - Tab 2: Morphological FST Lab (interactive parser, gate split-bar, FST breakdown table).
  - Tab 3: Benchmark & Academic Methodology (ACL 2024 MAGE matrix & results table).

- [ ] **Step 1: Write integration tests for new tabs and layout**

Update `tests/test_ui_integration.py` verifying the 3 tabs, quick sample pill handlers, confidence bar calculations, and explanation bullets.

- [ ] **Step 2: Run test to verify failure on old structure**

Run: `python -m unittest tests/test_ui_integration.py`

- [ ] **Step 3: Implement `ui/app.py` redesign**

- [ ] **Step 4: Run all UI tests**

Run: `python -m unittest tests/test_ui_highlighting.py tests/test_ui_file_loader.py tests/test_ui_presets.py tests/test_ui_sentence_analyzer.py tests/test_ui_linguistic_explainer.py tests/test_ui_integration.py`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add ui/app.py ui/highlighting.py tests/test_ui_integration.py
git commit -m "feat: redesign dashboard with academic 3-tab layout, quick sample pills, and zero emoji clutter"
```

---

### Task 3: Standalone Hugging Face Spaces Cloud Package & Deployment Script

**Files:**
- Create: `hf_space/app.py`
- Create: `hf_space/README.md`
- Create: `hf_space/requirements.txt`
- Bundle: `hf_space/kaz_mage/`, `hf_space/ui/`, `hf_space/fst_analyzer.py`
- Create: `scripts/push_to_hf.py`
- Test: `tests/test_hf_space_bundle.py`

**Interfaces:**
- Produces:
  - Self-contained `hf_space/` directory ready for drag-and-drop or git push to Hugging Face Spaces.
  - `scripts/push_to_hf.py --repo-id <username/space_name> [--token <hf_token>]`

- [ ] **Step 1: Write tests for HF space bundle**

```python
# tests/test_hf_space_bundle.py
import unittest
import py_compile
import os

class TestHfSpaceBundle(unittest.TestCase):
    def test_bundle_files_exist(self):
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hf_space")
        self.assertTrue(os.path.exists(os.path.join(base_dir, "app.py")))
        self.assertTrue(os.path.exists(os.path.join(base_dir, "README.md")))
        self.assertTrue(os.path.exists(os.path.join(base_dir, "requirements.txt")))
        self.assertTrue(os.path.exists(os.path.join(base_dir, "fst_analyzer.py")))

    def test_bundle_syntax_compilation(self):
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hf_space")
        app_file = os.path.join(base_dir, "app.py")
        py_compile.compile(app_file, doraise=True)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m unittest tests/test_hf_space_bundle.py`

- [ ] **Step 3: Implement `hf_space/` package and `scripts/push_to_hf.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_hf_space_bundle.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hf_space/ scripts/push_to_hf.py tests/test_hf_space_bundle.py
git commit -m "feat: implement standalone Hugging Face Spaces cloud package and upload script"
```

---

### Task 4: Full Regression, End-to-End Verification & Merging

- [ ] **Step 1: Run complete project regression suite (175+ tests)**

`python -m unittest discover -s tests -p "test_*.py"`
Expected: All pass.

- [ ] **Step 2: Verify zero changes to `aist2026/paper.tex`**

`git diff main..HEAD aist2026/paper.tex`
Expected: Empty diff.

- [ ] **Step 3: Verify live local execution on port 7860**

Verify the server runs with the new tabbed layout and zero errors.

- [ ] **Step 4: Update Walkthrough documentation**
