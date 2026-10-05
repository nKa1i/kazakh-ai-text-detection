# ARR Author Checklist Compliance and Supervisor Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor conference manuscript Section 8 into explicit `Limitations` and `Ethical Considerations` sections for automated `aclpubcheck` compliance, update supervisor details to Prof. Bin Guo across all submission documents and dissertation front matter, and verify end-to-end regression integrity.

**Architecture:** Split `08_limitations.tex` into two discrete top-level sections conforming to ARR guidelines without expanding the 10-page document budget. Update OpenReview submission brief metadata and thesis front matter to reflect Prof. Bin Guo's verified profile (`~Bin_Guo3`, `guob@nwpu.edu.cn`). Extend the automated test suite with anonymity and heading checks, and compile both documents cleanly.

**Tech Stack:** XeLaTeX, BibTeX, Python 3.12 (unittest, pypdf), Git.

## Global Constraints
- Under no circumstances shall `aist2026/paper.tex` be touched or modified (strictly 0 diff lines).
- Strictly zero decorative emojis across all code, datasets, comments, documentation, and manuscripts.
- All Python file operations must explicitly specify `encoding='utf-8'`.
- All code files and manuscripts must contain complete, publication-ready academic content with zero placeholders ("TODO", "TBD").
- Maintain backward compatibility with existing tests (521/521 tests passing).

---

### Task 1: Manuscript Section 8 Heading Restructuring & Compilation

**Files:**
- Modify: `papers/kazakh_fever_conference/sections/08_limitations.tex`
- Output: `papers/kazakh_fever_conference/main.pdf`
- Mirror: `C:\Users\Roza\Desktop\Kazakh_FEVER_COLING2027_Submission.pdf`

**Interfaces:**
- Consumes: Existing paragraphs in `08_limitations.tex`.
- Produces: Conforming two-section structure (`\section{Limitations}` and `\section{Ethical Considerations}`) compiled into a strict 10-page PDF with Sections 1–7 concluding on Page 8.

- [ ] **Step 1: Edit `08_limitations.tex`**
  Split the heading into:
  ```latex
  \section{Limitations}
  \label{sec:limitations}

  \paragraph{Linguistic and Typological Scope}
  ...
  \paragraph{Evidence Corpus and Temporal Freshness}
  ...
  \paragraph{Computational and Environmental Resource Bounds}
  ...

  \section{Ethical Considerations}
  \label{sec:ethics}

  \paragraph{Ethical Considerations and \mbox{Responsible Deployment}}
  ...
  ```

- [ ] **Step 2: Recompile the conference paper**
  Run XeLaTeX + BibTeX pipeline inside `papers/kazakh_fever_conference/`:
  ```bash
  xelatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  xelatex -interaction=nonstopmode -halt-on-error main.tex
  xelatex -interaction=nonstopmode -halt-on-error main.tex
  ```

- [ ] **Step 3: Verify page layout and geometry**
  Run Python validation script asserting:
  - Page count is exactly 10 pages.
  - Section 7 (Conclusion) ends on Page 8.
  - Section 8 (`Limitations`) begins on Page 9.
  - Page size is exactly A4 (`595.28 x 841.89 pt`).

- [ ] **Step 4: Update Desktop submission PDF**
  Copy `papers/kazakh_fever_conference/main.pdf` to `C:\Users\Roza\Desktop\Kazakh_FEVER_COLING2027_Submission.pdf`.

- [ ] **Step 5: Commit changes**
  ```bash
  git add papers/kazakh_fever_conference/sections/08_limitations.tex
  git commit -m "style(paper1): decouple Limitations and Ethical Considerations sections for ARR checklist compliance"
  ```

---

### Task 2: OpenReview Submission Brief Author & Service Contributor Update

**Files:**
- Modify: `docs/coling_2027_arr_submission_brief.md`

**Interfaces:**
- Consumes: Prof. Bin Guo's OpenReview profile (`~Bin_Guo3`) and NWPU institutional email (`guob@nwpu.edu.cn`).
- Produces: Fully synchronized, copy-paste ready OpenReview submission metadata and Sustainable Reviewing service contributor nomination.

- [ ] **Step 1: Update Author 3 entry in `docs/coling_2027_arr_submission_brief.md`**
  Replace references to Zhijiang Guo with:
  - Name: **Bin Guo**
  - OpenReview Account / ID: `~Bin_Guo3`
  - OpenReview Profile URL: `https://openreview.net/profile?id=~Bin_Guo3`
  - Primary Institutional Email: `guob@nwpu.edu.cn`
  - Affiliation: School of Computer Science, Northwestern Polytechnical University, Xi'an, Shaanxi, China

- [ ] **Step 2: Update Sustainable Reviewing Service Contributor details**
  Update the designated service contributor in Section 4 and Section 7 of the brief to **Prof. Bin Guo** (`~Bin_Guo3`, `guob@nwpu.edu.cn`).

- [ ] **Step 3: Commit changes**
  ```bash
  git add -f docs/coling_2027_arr_submission_brief.md
  git commit -m "docs: update supervisor profile and service contributor to Prof. Bin Guo in ARR submission brief"
  ```

---

### Task 3: Master's Thesis Front Matter Supervisor Alignment

**Files:**
- Modify: `thesis/frontmatter/title_page.tex:28-30`
- Modify: `thesis/frontmatter/acknowledgments.tex:3`

**Interfaces:**
- Consumes: Academic supervisor title and affiliation.
- Produces: Correct dissertation title page and acknowledgements acknowledging Prof. Bin Guo.

- [ ] **Step 1: Update `thesis/frontmatter/title_page.tex`**
  Change:
  ```latex
  \textbf{Academic Supervisor:}\\
  {\large Prof. Zhijiang Guo} \\
  {\small Northwestern Polytechnical University} \\[0.5cm]
  ```
  to:
  ```latex
  \textbf{Academic Supervisor:}\\
  {\large Prof. Bin Guo} \\
  {\small Northwestern Polytechnical University} \\[0.5cm]
  ```

- [ ] **Step 2: Update `thesis/frontmatter/acknowledgments.tex`**
  Change:
  ```latex
  First and foremost, I would like to express my deepest and most sincere gratitude to my academic supervisor, \textbf{Professor Zhijiang Guo}, for his indispensable mentorship...
  ```
  to:
  ```latex
  First and foremost, I would like to express my deepest and most sincere gratitude to my academic supervisor, \textbf{Professor Bin Guo}, for his indispensable mentorship...
  ```

- [ ] **Step 3: Recompile thesis and verify dissertation compilation**
  Run XeLaTeX inside `thesis/` and run `tests/test_dissertation_compilation.py`.

- [ ] **Step 4: Commit changes**
  ```bash
  git add thesis/frontmatter/title_page.tex thesis/frontmatter/acknowledgments.tex
  git commit -m "docs(thesis): update academic supervisor to Prof. Bin Guo in dissertation front matter"
  ```

---

### Task 4: Submission Integrity Regression Test Suite & Verification

**Files:**
- Modify: `tests/test_coling_submission_integrity.py`

**Interfaces:**
- Consumes: Updated manuscript, anonymity list, and section headers.
- Produces: 100% passing automated test suite with zero regression.

- [ ] **Step 1: Update `tests/test_coling_submission_integrity.py`**
  - Add `"Bin"` to `PROHIBITED_ANONYMITY_TERMS`.
  - Update `test_limitations_section_present` to verify both:
    - `\section{Limitations}`
    - `\section{Ethical Considerations}`
  - Keep check for required subsection titles:
    - `Linguistic and Typological Scope`
    - `Evidence Corpus and Temporal Freshness`
    - `Computational and Environmental Resource Bounds`
    - `Ethical Considerations and Responsible Deployment`

- [ ] **Step 2: Run submission integrity test**
  ```bash
  python -m unittest tests/test_coling_submission_integrity.py -v
  ```

- [ ] **Step 3: Run full repository regression test suite**
  ```bash
  python -m unittest discover -s tests -p "test_*.py"
  ```

- [ ] **Step 4: Verify invariance of `aist2026/paper.tex`**
  ```bash
  git diff origin/main..HEAD -- aist2026/paper.tex
  ```
  Must output strictly 0 lines.

- [ ] **Step 5: Commit changes**
  ```bash
  git add tests/test_coling_submission_integrity.py
  git commit -m "test(paper1): update submission integrity tests for decoupled Limitations sections and Prof. Bin Guo anonymity"
  ```
