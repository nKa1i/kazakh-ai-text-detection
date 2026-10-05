# Design Specification: ARR Author Checklist Compliance and Supervisor Update

**Date**: 2026-10-05  
**Topic**: ARR Author Checklist Strict Heading Compliance & Academic Supervisor Alignment to Prof. Bin Guo  
**Target Venue**: ACL Rolling Review (ARR) October 2026 Cycle / COLING 2027  
**Status**: Approved for Implementation Planning  

---

## 1. Context and Motivation

Following an exhaustive audit of the official ACL Rolling Review (ARR) guidance pages:
- [ARR Author Checklist](https://aclrollingreview.org/authorchecklist)
- [ARR Author Guidelines](https://aclrollingreview.org/authors)

Two critical items must be addressed to ensure 100% compliance with automated desk-check filters (`aclpubcheck`) and administrative verification:
1. **Section Heading Alignment (Approach 1)**: ARR guidelines explicitly require:
   > *"The paper has a section titled “Limitations” placed as follows: ACL formatting guidelines... If you have an optional ethics section, we recommend that it is titled ‘Ethical considerations’."*
   The current manuscript combines these into `\section{Limitations and Ethical Considerations}`. We will decouple this into `\section{Limitations}` and `\section{Ethical Considerations}`.
2. **Academic Supervisor Identity Alignment**: The Chinese supervisor at Northwestern Polytechnical University (NWPU) is confirmed as **Professor Bin Guo** (OpenReview profile: `~Bin_Guo3`, email: `guob@nwpu.edu.cn`). This update applies to the ARR submission brief, the nominated service contributor profile, the master's thesis front matter, and the automated anonymity integrity tests.

---

## 2. Detailed Technical Design

### 2.1 Component 1: Conference Manuscript Restructuring (`papers/kazakh_fever_conference/`)
- **File**: `papers/kazakh_fever_conference/sections/08_limitations.tex`
- **Changes**:
  - Replace `\section{Limitations and Ethical Considerations}` with `\section{Limitations}`.
  - Retain existing paragraphs:
    - `\paragraph{Linguistic and Typological Scope}`
    - `\paragraph{Evidence Corpus and Temporal Freshness}`
    - `\paragraph{Computational and Environmental Resource Bounds}`
  - Insert `\section{Ethical Considerations}`.
  - Under `\section{Ethical Considerations}`, format paragraph as:
    `\paragraph{Responsible Deployment and \mbox{Decision Support}}` (or retain exact wording emphasizing human accountability, confidence scores, and citation grounding).
- **Compilation & Verification**:
  - Recompile `main.tex` using `xelatex` + `bibtex` inside `papers/kazakh_fever_conference/`.
  - Verify total page count remains strictly **10 pages**.
  - Verify Sections 1 through 7 end strictly on **Page 8**.
  - Verify `\section{Limitations}` begins at the top of **Page 9**, followed by `\section{Ethical Considerations}` and references on Pages 9–10.
  - Copy freshly compiled `main.pdf` to `C:\Users\Roza\Desktop\Kazakh_FEVER_COLING2027_Submission.pdf`.

### 2.2 Component 2: OpenReview Submission Brief (`docs/coling_2027_arr_submission_brief.md`)
- **Changes**:
  - Update Author 3:
    - Name: **Bin Guo**
    - Role: Senior Supervisor & Principal Investigator
    - Primary OpenReview Account / ID: `~Bin_Guo3` (`https://openreview.net/profile?id=~Bin_Guo3`)
    - Primary Institutional Email: `guob@nwpu.edu.cn`
    - Affiliation: School of Computer Science, Northwestern Polytechnical University, Xi'an, Shaanxi, China
  - Update Sustainable Reviewing Service Contributor Nomination:
    - Primary Nominated Reviewer: **Professor Bin Guo** (`~Bin_Guo3`, `guob@nwpu.edu.cn`).
    - Note requirement to complete the ARR reviewer registration form within 48 hours of the October 12 AoE deadline.

### 2.3 Component 3: Master's Thesis Front Matter Alignment (`thesis/frontmatter/`)
- **Files**:
  - `thesis/frontmatter/title_page.tex`:
    - Line 29: Update `Academic Supervisor: Prof. Zhijiang Guo` to `Academic Supervisor: Prof. Bin Guo`.
  - `thesis/frontmatter/acknowledgments.tex`:
    - Line 3: Update gratitude from `Professor Zhijiang Guo` to `Professor Bin Guo`.
- **Compilation & Verification**:
  - Recompile `thesis/main.tex` with `xelatex` and verify clean build.

### 2.4 Component 4: Submission Integrity Regression Test Suite (`tests/test_coling_submission_integrity.py`)
- **Changes**:
  - Update `PROHIBITED_ANONYMITY_TERMS`:
    - Add `"Bin"` (and retain `"Guo"`, `"Zhijiang"`, `"Daulet"`, `"Anekesh"`, `"Ualiyeva"`, `"KazNU"`, `"NWPU"`, `"nKa1i"`).
  - Update `test_limitations_section_present`:
    - Assert `\section{Limitations}` exists in `08_limitations.tex`.
    - Assert `\section{Ethical Considerations}` exists in `08_limitations.tex`.
  - Run full test suite (`python -m unittest discover -s tests -p "test_*.py"`).

---

## 3. Strict Invariant Constraints
1. **Baseline Invariance**: `aist2026/paper.tex` must have strictly **0 diff lines**.
2. **Zero Decorative Emojis**: Strictly zero decorative emojis across all code, datasets, comments, and manuscripts.
3. **UTF-8 Encoding**: All file reads and writes must explicitly enforce `encoding='utf-8'`.
4. **Test Pass Rate**: 100% passing tests (521/521).
