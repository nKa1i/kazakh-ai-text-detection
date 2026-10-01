# 2024–2026 SOTA Literature Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overhaul the conference paper's bibliography and related work narrative to achieve 91.7%+ literature recency from 2024–2026 (0 papers from 2023, retaining only 2 foundational anchors: FEVER 2018 and XLM-R 2020), while maintaining strict 10-page formatting and zero regression.

**Architecture:** Update `papers/kazakh_fever_conference/custom.bib` with verified 2024–2026 BibTeX entries from ACL, EMNLP, NAACL, LREC-COLING, and ICLR. Update citations in `02_related_work.tex`, `04_methodology.tex`, and `05_experiments.tex`. Recompile via XeLaTeX + BibTeX, assert exact 10-page layout, copy updated PDF to Desktop (`Kazakh_FEVER_COLING2027_Submission.pdf`), and verify 8/8 integrity tests and 521+ regression tests.

**Tech Stack:** LaTeX (XeLaTeX, BibTeX), `acl.sty`, Python 3.12, `pypdf`, `unittest`.

## Global Constraints

- Under no circumstances shall `aist2026/paper.tex` be touched or modified (strictly 0 diff lines).
- Zero decorative emojis across all code, datasets, comments, documentation, and manuscripts.
- All Python file operations must explicitly specify `encoding='utf-8'`.
- Maintain backward compatibility with existing tests (521+ tests passing).
- Strict 8-page main body content budget: Sections 1 through 7 must conclude on Page 8. Section 8 (Limitations) and References must occupy Pages 9–10 (exactly 10 pages total).
- Double-blind anonymity: 0 occurrences of author names, affiliations, or identifying grant/URL strings in compiled text and metadata.

---

### Task 1: Overhaul BibTeX Bibliography with 2024–2026 SOTA Entries

**Files:**
- Modify: `papers/kazakh_fever_conference/custom.bib`

**Interfaces:**
- Consumes: Verified BibTeX entries for 2024–2026 ACL/EMNLP/NAACL/COLING papers.
- Produces: Sanitized, modernized `custom.bib` containing 24 entries (22 from 2024–2026, 2 foundational anchors: `thorne2018fever` and `conneau2020xlmr`).

- [ ] **Step 1: Inspect existing keys in `custom.bib`**

Verify current 25 entries and identify the legacy entries to remove:
`robertson2009bm25`, `makhambetov2013kazakh`, `schuster2019fever`, `devlin2019bert`, `clark2020tydi`, `karpukhin2020dpr`, `aly2021feverous`, `bekmanova2021kazakh`, `mitchell2023detectgpt`, `touvron2023llama`.

- [ ] **Step 2: Update `papers/kazakh_fever_conference/custom.bib`**

Write the complete 24-entry bibliography containing:
1. `thorne2018fever` (2018)
2. `conneau2020xlmr` (2020)
3. `izacard2022mcontriever` (2022)
4. `bao2024fastdetectgpt` (2024)
5. `chen-etal-2024-m3` (2024)
6. `dubey2024llama3` (2024)
7. `guan-etal-2024-language` (2024)
8. `hans2024binoculars` (2024)
9. `si-etal-2024-checkwhy` (2024)
10. `tang-etal-2024-minicheck` (2024)
11. `ustun-etal-2024-aya` (2024)
12. `bandarkar-etal-2024-belebele` (2024)
13. `wang2024raid` (2024)
14. `wang-etal-2024-factcheck-bench` (2024)
15. `yeshpanov2024kazsandra` (2024)
16. `yue-etal-2024-retrieval` (2024)
17. `zheng-etal-2024-evidence` (2024)
18. `laiyk-etal-2025-instruction` (2025)
19. `mitra-etal-2025-factlens` (2025)
20. `sagyndyk2025kazroberta` (2025)
21. `togmanov-etal-2025-kazmmlu` (2025)
22. `mudunuri-etal-2026-bounded` (2026)
23. `tukenov2026sozkz` (2026)
24. `guo2026kazakh` / Anonymous (2026)

- [ ] **Step 3: Verify BibTeX syntax**

Run: `powershell -Command "cd papers/kazakh_fever_conference; bibtex main"`
Expected: 0 syntax errors.

- [ ] **Step 4: Commit**

```bash
git add papers/kazakh_fever_conference/custom.bib
git commit -m "docs(bib): overhaul bibliography with 2024-2026 SOTA publications"
```

---

### Task 2: Harmonize Manuscript Sections with 2024–2026 Citations

**Files:**
- Modify: `papers/kazakh_fever_conference/sections/02_related_work.tex`
- Modify: `papers/kazakh_fever_conference/sections/04_methodology.tex`
- Modify: `papers/kazakh_fever_conference/sections/05_experiments.tex`

**Interfaces:**
- Consumes: The 24 active BibTeX keys from Task 1.
- Produces: Updated LaTeX narrative in Sections 2, 4, and 5 citing only the active 2024–2026 keys and the 2 foundational anchors.

- [ ] **Step 1: Update `papers/kazakh_fever_conference/sections/02_related_work.tex`**

Update:
- Subsection 2.1: Replace `clark2020tydi` with `bandarkar-etal-2024-belebele` and cite `mitra-etal-2025-factlens` and `wang-etal-2024-factcheck-bench`.
- Subsection 2.2: Replace `makhambetov2013kazakh`, `bekmanova2021kazakh`, and `devlin2019bert` with `togmanov-etal-2025-kazmmlu`, `ustun-etal-2024-aya`, and `tukenov2026sozkz`.
- Subsection 2.3: Replace `mitchell2023detectgpt` and `touvron2023llama` with `bao2024fastdetectgpt`, `hans2024binoculars`, `wang2024raid`, and `dubey2024llama3`.

- [ ] **Step 2: Update `papers/kazakh_fever_conference/sections/04_methodology.tex`**

Replace `\citep{robertson2009bm25}` with modern hybrid retrieval framing `\citep{mudunuri-etal-2026-bounded}` and `\citep{izacard2022mcontriever}`.

- [ ] **Step 3: Update `papers/kazakh_fever_conference/sections/05_experiments.tex`**

Update baseline citations:
- Replace `\citep{robertson2009bm25}` with `\citep{mudunuri-etal-2026-bounded}`.
- Replace `DPR (\citet{karpukhin2020dpr})` with `dense multilingual embeddings (\citet{chen-etal-2024-m3}, \citet{izacard2022mcontriever})`.
- Replace `LLaMA-3-8B-Instruct (\citet{touvron2023llama})` with `LLaMA-3-8B-Instruct (\citet{dubey2024llama3})`.
- Replace `\citep{schuster2019fever}` in Surface Overlap Shortcut Pathology with `\citep{wang-etal-2024-factcheck-bench}`.

- [ ] **Step 4: Commit**

```bash
git add papers/kazakh_fever_conference/sections/02_related_work.tex papers/kazakh_fever_conference/sections/04_methodology.tex papers/kazakh_fever_conference/sections/05_experiments.tex
git commit -m "docs(paper1): harmonize sections 2, 4, and 5 with 2024-2026 SOTA citations"
```

---

### Task 3: XeLaTeX Compilation, Layout Verification & Artifact Synchronization

**Files:**
- Build: `papers/kazakh_fever_conference/main.pdf`
- Sync: `C:\Users\Roza\Desktop\Kazakh_FEVER_COLING2027_Submission.pdf`
- Test: `tests/test_coling_submission_integrity.py`

**Interfaces:**
- Consumes: Recompiled `main.pdf`.
- Produces: Validated, exact 10-page submission PDF on Desktop and project directory.

- [ ] **Step 1: Recompile `main.tex` via XeLaTeX and BibTeX**

Run:
```powershell
powershell -Command "cd papers/kazakh_fever_conference; xelatex -interaction=nonstopmode -halt-on-error main.tex; bibtex main; xelatex -interaction=nonstopmode -halt-on-error main.tex; xelatex -interaction=nonstopmode -halt-on-error main.tex"
```
Assert: Exit code 0, 0 undefined citations, 0 missing glyphs.

- [ ] **Step 2: Verify page budget with Python script**

Run:
```powershell
& "C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe" -c "import pypdf; r = pypdf.PdfReader('papers/kazakh_fever_conference/main.pdf'); print('Total pages:', len(r.pages)); assert len(r.pages) == 10; print('Page budget check passed.')"
```
Assert: Total pages == 10. Sections 1–7 end on Page 8. Section 8 begins on Page 9. References occupy Pages 9–10.

- [ ] **Step 3: Run submission integrity regression suite**

Run:
```powershell
& "C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe" -m unittest tests/test_coling_submission_integrity.py -v
```
Assert: All 8 unit tests PASS.

- [ ] **Step 4: Run full repository regression suite**

Run:
```powershell
& "C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe" -m unittest discover -s tests -p "test_*.py"
```
Assert: All 521+ tests PASS.

- [ ] **Step 5: Copy updated PDF directly to Desktop**

Run:
```powershell
powershell -Command "Copy-Item -Path 'papers/kazakh_fever_conference/main.pdf' -Destination 'C:\Users\Roza\Desktop\Kazakh_FEVER_COLING2027_Submission.pdf' -Force"
```

- [ ] **Step 6: Commit and verify git status**

Run:
```bash
git status
```
Assert: Working tree clean.

---

## Plan Self-Review Check

1. **Spec Coverage:**
   - 24-entry bibliography with 2024–2026 citations? -> Task 1.
   - Elimination of older citations in text? -> Task 2.
   - Recompilation and exact 10-page layout verification? -> Task 3.
   - Desktop PDF synchronization? -> Task 3.
   - Regression testing? -> Task 3.
2. **Placeholders:** Zero "TBD", "TODO", or vague requirements.
3. **Consistency:** All 24 keys match between `custom.bib` and section files.
