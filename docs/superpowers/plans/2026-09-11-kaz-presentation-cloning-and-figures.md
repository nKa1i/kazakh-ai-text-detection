# Presentation Template Cloning & Publication Figures Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Master's thesis presentation into a visually rich, institutional slide deck matching `Aliya PPT.pptx` by directly cloning the baseline template (preserving native KazNU & NPU logos, NPU pentagon badge, and gradient highlight tracker), removing commercial watermarks, generating 14 publication-grade figures, and embedding them across 23 slides.

**Architecture:** A Python-based automation pipeline: (1) `scripts/generate_presentation_figures.py` generates 14 high-resolution (300 DPI) figures from empirical Kaz-MAGE and Kazakh-FEVER benchmark data; (2) `scripts/build_cloned_presentation.py` directly clones `C:\Users\Roza\Downloads\Aliya PPT.pptx`, removes `合作QQ` watermarks, preserves native branding, repositions the native `矩形 29` gradient highlight tracker per section, updates text/tables, and embeds the figures; (3) exports slides to PNG via PowerPoint COM for visual verification; and (4) mirrors to the user's Desktop.

**Tech Stack:** Python 3.12, `python-pptx`, `matplotlib`, `seaborn`, `numpy`, `win32com.client`.

## Global Constraints

- Under no circumstances shall `aist2026/paper.tex` be touched or modified.
- Zero decorative emojis across all slides, code, tables, and documentation.
- Widescreen 16:9 format ($13.333 \times 7.500$ inches) strictly preserved.
- Native branding preserved: KazNU logo (`Изображение 16`), NPU logo group (`LOGO组合`), and top-left NPU pentagon emblem (`Layout 7`).
- Native gradient highlight tracker (`矩形 29`) repositioned per section; commercial watermark (`合作QQ`) stripped.
- All Python file operations must explicitly specify `encoding='utf-8'`.

---

### Task 1: Generate 14 High-Impact Publication Figures

**Files:**
- Create: `scripts/generate_presentation_figures.py`
- Test: `tests/test_presentation_figures.py`
- Output directory: `presentation_figures/`

**Interfaces:**
- Produces: 14 PNG files in `presentation_figures/` at 300 DPI (`fig01_subword_vs_fst.png` through `fig14_cloud_deployment_pipeline.png`).

- [ ] **Step 1: Write the failing test for figure generation**

```python
# tests/test_presentation_figures.py
import os
import unittest

EXPECTED_FIGURES = [
    "fig01_subword_vs_fst.png",
    "fig02_domain_collapse.png",
    "fig03_tripartite_framework.png",
    "fig04_morpho_gate_arch.png",
    "fig05_chunking_topk_flow.png",
    "fig06_trust_matrix_pipeline.png",
    "fig07_dataset_distribution.png",
    "fig08_roc_curves_kaz_mage.png",
    "fig09_long_doc_injection.png",
    "fig10_kaz_fever_confusion_matrix.png",
    "fig11_four_quadrant_trust_matrix.png",
    "fig12_ablation_study_barchart.png",
    "fig13_gradio_dashboard_panels.png",
    "fig14_cloud_deployment_pipeline.png"
]

class TestPresentationFigures(unittest.TestCase):
    def test_all_figures_exist_and_valid(self):
        fig_dir = os.path.abspath("presentation_figures")
        self.assertTrue(os.path.exists(fig_dir), f"Directory {fig_dir} must exist")
        for fig_name in EXPECTED_FIGURES:
            path = os.path.join(fig_dir, fig_name)
            self.assertTrue(os.path.exists(path), f"Figure {fig_name} must exist")
            size = os.path.getsize(path)
            self.assertGreater(size, 5000, f"Figure {fig_name} must be larger than 5KB")
            with open(path, "rb") as f:
                header = f.read(8)
                self.assertEqual(header[:4], b"\x89PNG", f"Figure {fig_name} must have valid PNG header")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_presentation_figures.py`
Expected: FAIL (Directory or figures do not exist).

- [ ] **Step 3: Implement `scripts/generate_presentation_figures.py`**

Implement the generator script rendering all 14 publication-grade figures using `matplotlib` (Agg backend) with the institutional Navy (`#1E3A8A`), Teal (`#0D9488`), Amber (`#D97706`), Green (`#16A34A`), and Slate (`#0F172A`) palette:
1. `fig01_subword_vs_fst.png`: Byte fragmentation vs FST stem+affix segmentation.
2. `fig02_domain_collapse.png`: KazRoBERTa (57.62%) vs Proposed (99.80%) on reviews.
3. `fig03_tripartite_framework.png`: 3-tier system flowchart.
4. `fig04_morpho_gate_arch.png`: Dual-stream cross-attention schematic with dynamic gate $\mathbf{g}$.
5. `fig05_chunking_topk_flow.png`: Sliding window with 1-sentence overlap and Top-K aggregation.
6. `fig06_trust_matrix_pipeline.png`: BM25 retrieval + NLI + Four-Quadrant Trust Matrix flow.
7. `fig07_dataset_distribution.png`: Token length and affix density across News, Wiki, Reviews.
8. `fig08_roc_curves_kaz_mage.png`: 4-panel ROC curves for Q1, Q2, Q3, Q4.
9. `fig09_long_doc_injection.png`: Hybrid paragraph localization curve & VRAM scaling.
10. `fig10_kaz_fever_confusion_matrix.png`: Normalized 3-way NLI confusion matrix (100% F1).
11. `fig11_four_quadrant_trust_matrix.png`: 2D scatter plot (AI probability vs. factual risk).
12. `fig12_ablation_study_barchart.png`: Performance drop bar chart without FST/Gate/SupCon.
13. `fig13_gradio_dashboard_panels.png`: UI diagram of the 4 Gradio tabs.
14. `fig14_cloud_deployment_pipeline.png`: Hugging Face Spaces cloud package & test pass badge.

- [ ] **Step 4: Execute generator and verify tests pass**

Run:
```bash
C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe scripts/generate_presentation_figures.py
C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_presentation_figures.py
```
Expected: PASS (All 14 figures generated and verified).

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_presentation_figures.py tests/test_presentation_figures.py presentation_figures/
git commit -m "feat: generate 14 publication-grade figures for thesis presentation"
```

---

### Task 2: Presentation Template Cloning & Branding Engine

**Files:**
- Create: `scripts/build_cloned_presentation.py`
- Test: `tests/test_cloned_presentation.py`

**Interfaces:**
- Consumes: `C:\Users\Roza\Downloads\Aliya PPT.pptx`.
- Produces: `Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx` retaining native Slide 1 KazNU + NPU logos, watermarks, Layout 7 NPU pentagon emblem, and repositioned native `矩形 29` gradient tracker.

- [ ] **Step 1: Write the failing test for template cloning and branding**

```python
# tests/test_cloned_presentation.py
import os
import unittest
from pptx import Presentation

class TestClonedPresentation(unittest.TestCase):
    def test_cloned_presentation_branding_and_cleanliness(self):
        ppt_path = os.path.abspath("Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx")
        self.assertTrue(os.path.exists(ppt_path), f"{ppt_path} must exist")
        prs = Presentation(ppt_path)
        self.assertEqual(len(prs.slides), 23, "Must have exactly 23 slides")
        
        # Verify Slide 1 branding
        s1 = prs.slides[0]
        s1_names = [s.name for s in s1.shapes]
        self.assertIn("Изображение 16", s1_names, "KazNU logo must be preserved on Slide 1")
        self.assertIn("LOGO组合", s1_names, "NPU logo group must be preserved on Slide 1")
        self.assertIn("校徽打底", s1_names, "Watermark must be preserved on Slide 1")
        
        # Verify watermark removal across all slides
        for idx, slide in enumerate(prs.slides):
            for shape in slide.shapes:
                if shape.has_text_frame:
                    self.assertNotIn("243001978", shape.text_frame.text,
                                     f"Slide {idx+1} must not contain commercial QQ watermark")
                    self.assertNotIn("合作QQ", shape.text_frame.text,
                                     f"Slide {idx+1} must not contain QQ watermark text")
        
        # Verify content slides have native Layout 7 with NPU pentagon emblem
        for idx in range(2, 22):
            slide = prs.slides[idx]
            self.assertEqual(slide.slide_layout.name, "1_标题和内容",
                             f"Slide {idx+1} must use Layout 7 with top-left NPU pentagon badge")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_cloned_presentation.py`
Expected: FAIL (Baseline file not yet cloned/rebuilt with these criteria).

- [ ] **Step 3: Implement `scripts/build_cloned_presentation.py`**

Implement cloning logic:
1. Open `C:\Users\Roza\Downloads\Aliya PPT.pptx`.
2. Slide 1:
   - Preserve `Изображение 16` (KazNU logo), `LOGO组合` (NPU logo group), `校徽打底`, `图书馆照片`.
   - Update `主标题` to:
     `Research on Morphologically-Grounded AI-Generated Text Detection and Factual Verification for Low-Resource Kazakh`
   - Add subtitle: `面向低资源哈萨克语的形态感知AI生成文本检测与事实核验研究`.
   - Update `汇报人` to: `汇报人：大雷 (Daulet)      导师：郭教授 (Prof. Guo)      2026年9月`.
   - Remove `合作QQ： 243001978`.
3. Slide 2:
   - Preserve `海天苑` background photo, `背景色块`, `目录英文`, `打底色块`.
   - Update 6 numbered TOC entries with clean English + Chinese subtitles.
   - Remove `合作QQ： 243001978`.
4. Slides 3–22:
   - Preserve the top bar `矩形 4`, section text boxes, divider line `直接连接符 6`, slide numbers.
   - Reposition native `矩形 29` (gradient highlight block) according to section index:
     - Slides 3–4 (Background): `left = 1.30"`, `width = 1.82"`
     - Slides 5–6 (Related Work): `left = 3.12"`, `width = 1.66"`
     - Slides 7–10 (Content): `left = 4.78"`, `width = 1.81"`
     - Slides 11–16 (Experiments): `left = 6.59"`, `width = 2.15"`
     - Slides 17–18 (System Demo): `left = 6.59"`, `width = 2.15"`
     - Slides 19–21 (Conclusion & Future Work): `left = 8.92"`, `width = 1.95"`
     - Slide 22 (Comments & Responses): `left = 10.83"`, `width = 2.00"`
   - Clear legacy poultry farm shapes below `y = 1.2"`.
5. Slide 23:
   - Preserve `长安校区` photo and framing.
   - Remove `合作QQ： 243001978`.
   - Update text: `Thank You for Your Attention!` / `请郭老师批评指正`.

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe scripts/build_cloned_presentation.py
C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_cloned_presentation.py
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_cloned_presentation.py tests/test_cloned_presentation.py
git commit -m "feat: clone presentation template with native branding and watermark excision"
```

---

### Task 3: Content Population, Comparison Tables & Figure Embedding across All 23 Slides

**Files:**
- Modify: `scripts/build_cloned_presentation.py`
- Test: `tests/test_presentation_content.py`

**Interfaces:**
- Consumes: 14 PNG files in `presentation_figures/`.
- Produces: Complete 23-slide deck with embedded figures, comparative tables, stat callout cards, and committee responses.

- [ ] **Step 1: Write the failing test for slide content and figures**

```python
# tests/test_presentation_content.py
import os
import unittest
from pptx import Presentation

class TestPresentationContent(unittest.TestCase):
    def test_figures_and_tables_embedded_correctly(self):
        ppt_path = os.path.abspath("Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx")
        prs = Presentation(ppt_path)
        
        # Verify slides that must have embedded pictures
        fig_slides = [3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        for s_num in fig_slides:
            slide = prs.slides[s_num - 1]
            pictures = [s for s in slide.shapes if s.shape_type == 13] # MSO_SHAPE_TYPE.PICTURE
            self.assertGreater(len(pictures), 0, f"Slide {s_num} must have at least one embedded picture figure")
        
        # Verify slides that must have tables
        table_slides = [5, 11, 12, 14, 16]
        for s_num in table_slides:
            slide = prs.slides[s_num - 1]
            tables = [s for s in slide.shapes if s.has_table]
            self.assertGreater(len(tables), 0, f"Slide {s_num} must contain an academic table")
        
        # Verify Slide 22 Committee comments
        s22 = prs.slides[21]
        text_s22 = "".join(s.text_frame.text for s in s22.shapes if s.has_text_frame)
        self.assertIn("Reviewer 1", text_s22)
        self.assertIn("Reviewer 2", text_s22)
        self.assertIn("100.00% AUC", text_s22)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_presentation_content.py`
Expected: FAIL (Content and figures not yet populated across all slides).

- [ ] **Step 3: Update `scripts/build_cloned_presentation.py` to populate all slides and embed figures**

Implement content rendering and figure insertion for all content slides:
- Slide 3: Background narrative + `fig01_subword_vs_fst.png` + 4 stat cards.
- Slide 4: Challenges narrative + `fig02_domain_collapse.png` + 3 failure mode cards.
- Slide 5: SOTA comparison table + 3 key takeaways.
- Slide 6: International benchmarks analysis cards (FEVER, VitaminC, SciFact).
- Slide 7: Tripartite overview cards + `fig03_tripartite_framework.png`.
- Slide 8: Topic 1 mathematical cards + `fig04_morpho_gate_arch.png`.
- Slide 9: Topic 2 chunking cards + `fig05_chunking_topk_flow.png`.
- Slide 10: Topic 3 verification cards + `fig06_trust_matrix_pipeline.png`.
- Slide 11: Datasets table + `fig07_dataset_distribution.png`.
- Slide 12: Kaz-MAGE 2x2 matrix table + `fig08_roc_curves_kaz_mage.png` + stat cards.
- Slide 13: Hybrid document stress cards + `fig09_long_doc_injection.png`.
- Slide 14: Kazakh-FEVER NLI table + `fig10_kaz_fever_confusion_matrix.png` + stat cards.
- Slide 15: Four-Quadrant cards + `fig11_four_quadrant_trust_matrix.png`.
- Slide 16: Component ablation table + `fig12_ablation_study_barchart.png`.
- Slide 17: Gradio dashboard capability cards + `fig13_gradio_dashboard_panels.png`.
- Slide 18: Cloud deployment cards + `fig14_cloud_deployment_pipeline.png`.
- Slide 19: Summary of contributions + dissertation status tracker (~85%).
- Slide 20: 4 roadmap milestone cards (AIST 2026 camera-ready, Paper 2, pre-defense, defense).
- Slide 21: Discussion points card for Prof. Guo.
- Slide 22: Committee comments response table (Reviewer 1 & Reviewer 2 with checkmarks `[✓]`).
- Slide 23: Closing layout with candidate & advisor metadata.

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe scripts/build_cloned_presentation.py
C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_presentation_content.py
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_cloned_presentation.py tests/test_presentation_content.py
git commit -m "feat: populate all 23 slides with academic content, tables, and 14 figures"
```

---

### Task 4: High-Resolution Rendering Validation & Desktop Delivery

**Files:**
- Output directory: `ppt_images/`
- Target: `C:\Users\Roza\Desktop\Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx`
- Artifact: `thesis_presentation_speaker_notes_prof_guo.md`

- [ ] **Step 1: Export all 23 slides to PNG via PowerPoint COM automation**

Run:
```bash
C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -c "
import os, win32com.client
ppt_app = win32com.client.Dispatch('PowerPoint.Application')
prs_path = os.path.abspath('Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx')
out_dir = os.path.abspath('ppt_images')
deck = ppt_app.Presentations.Open(prs_path, WithWindow=False)
deck.SaveAs(out_dir, 17)
deck.Close()
ppt_app.Quit()
print('Exported 23 slides to:', out_dir)
"
```

- [ ] **Step 2: Inspect exported slide images for visual excellence**

Inspect slides (especially Slide 1, 3, 5, 8, 11, 12, 14, 15, 17, 22, 23) using `view_file` to confirm:
- KazNU and NPU logos look sharp and positioned correctly on Slide 1.
- No `合作QQ` watermark remains.
- All figures render clearly with legible labels and captions.
- Zero overlapping shapes or truncated text boxes.

- [ ] **Step 3: Mirror the presentation to Desktop**

Run:
```bash
copy Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx C:\Users\Roza\Desktop\Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx
```

- [ ] **Step 4: Verify paper integrity & run full regression**

Verify `aist2026/paper.tex` was untouched:
```bash
git diff aist2026/paper.tex
```
Expected: Empty diff.

Run full test suite:
```bash
C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest discover -s tests -p "test_*.py"
```
Expected: 288+ tests passing.

- [ ] **Step 5: Final Commit**

```bash
git add scripts/ tests/ docs/
git commit -m "chore: complete presentation cloning, visual figure integration, and delivery"
```
