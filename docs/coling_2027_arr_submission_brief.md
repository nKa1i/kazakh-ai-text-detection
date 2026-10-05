# OpenReview Submission Brief and Metadata Sheet

**Conference Destination**: 31st International Conference on Computational Linguistics (COLING 2027)  
**Submission Venue**: ACL Rolling Review (ARR) October 2026 Cycle  
**OpenReview Portal URL**: `https://openreview.net/group?id=aclweb.org/ACL/ARR/2026/October`  
**Submission Deadline**: October 12, 2026, 23:59 Anywhere on Earth (AoE)  
**Review Mode**: Double-Blind Peer Review  
**Paper Track**: Long Paper (8 content pages + unlimited pages for limitations, references, ethics, and appendices)  
**Document Status**: Production Complete / Ready for Submission  

---

## 1. Executive Submission Summary

This brief provides an authoritative, copy-paste ready reference document for submitting the conference paper *Kazakh-FEVER 3K* to the ACL Rolling Review (ARR) October 2026 cycle for subsequent commitment to COLING 2027. All fields comply with the ACL 2026 Author Guidelines, the ARR Sustainable Reviewing Policy, and OpenReview submission form schemas.

---

## 2. Core Paper Metadata (Web Submission Form Box Ready)

### 2.1 Paper Title
```text
Kazakh-FEVER 3K: A Morphologically-Grounded Fact Verification Benchmark and 2D Trust Matrix for Low-Resource Agglutinative Languages
```

### 2.2 Short / Running Title (Optional Field in Portal)
```text
Kazakh-FEVER 3K & 2D Trust Matrix
```

### 2.3 Abstract
*Plain text formatting, verified zero LaTeX tags (`\textbf`, `\textsc`, etc.), verified web form box compatible.*

```text
Automated fact verification in low-resource agglutinative languages faces severe challenges due to high lexical sparsity, complex affixation chains, and the vulnerability of pretrained language models to parametric memorization. In this work, we present Kazakh-FEVER 3K, the first human-validated fact verification benchmark for Kazakh, comprising 3,000 claim-evidence pairs balanced across Supported, Refuted, and Not Enough Info (NEI) categories. To eliminate superficial keyword shortcuts, we introduce a Hard NEI curation protocol demanding fine-grained morphological reasoning over modal, temporal, and polarity markers. Furthermore, we propose a hybrid retrieval architecture combining morpheme-aware BM25 with dense multilingual representations (mContriever), coupled with a 2D Trust Matrix that jointly evaluates factual veracity and AI-generation likelihood. Extensive evaluations demonstrate that our hybrid retrieval achieves a Recall@5 of 99.8% (MRR 0.994), outperforming standard sparse retrieval across inflectional variations, while our morphologically-informed verifier achieves 82.6% accuracy and 82.1% Macro-F1 (+10.8% over XLM-RoBERTa-base and +5.2% over XLM-RoBERTa-large), effectively neutralizing lexical shortcuts and establishing a rigorous foundation for Kazakh automated fact verification.
```

#### Abstract Text Verification Metrics
- **Word Count**: 160 words (well within the ARR 250-word abstract upper bound)
- **Character Count (including spaces)**: 1,315 characters
- **Character Count (excluding spaces)**: 1,156 characters
- **Special Characters**: 0 unescaped LaTeX symbols; clean UTF-8 text with standard punctuation and percentages.

### 2.4 One-Sentence Summary / TL;DR
```text
We introduce Kazakh-FEVER 3K, the first human-validated fact verification benchmark for Kazakh with a Hard NEI protocol, paired with hybrid morpheme retrieval and a 2D Trust Matrix uniting veracity and AI-generation likelihood.
```

### 2.5 Primary Subject Area
Select from the ARR Subject Area Taxonomy drop-down menu:
```text
Resources and Evaluation / Low-Resource Language Processing
```

### 2.6 Secondary Subject Areas
Select secondary focus areas:
1. `Information Extraction / Fact Verification`
2. `Computational Morphology`

### 2.7 Keywords
- **Semicolon-Separated Field**:
  ```text
  Fact Verification; Kazakh NLP; Agglutinative Morphology; Finite-State Transducer; AI Text Detection; 2D Trust Matrix; Evidence Retrieval
  ```
- **Discrete Tag Entries**:
  - `Fact Verification`
  - `Kazakh NLP`
  - `Agglutinative Morphology`
  - `Finite-State Transducer`
  - `AI Text Detection`
  - `2D Trust Matrix`
  - `Evidence Retrieval`

---

## 3. Author Profiles, Institutional Affiliations, and Conflict of Interest (COI)

### 3.1 Author Roster and Ordering

#### Author 1: Daulet Anekesh (First and Corresponding Author)
- **Role**: Primary Doctoral Researcher & System Architect
- **Primary OpenReview Account Email**: `anekesh_daulet@live.kaznu.kz`
- **Secondary OpenReview Account Email**: `anekeshd@mail.nwpu.edu.cn`
- **Affiliation 1**: School of Computer Science, Northwestern Polytechnical University, Xi'an, Shaanxi, China
- **Affiliation 2**: Department of Computer Science, Al-Farabi Kazakh National University, Almaty, Kazakhstan
- **ORCID**: 0009-0008-4127-6582
- **Semantic Scholar Profile ID**: Daulet Anekesh

#### Author 2: Irina M. Ualiyeva (Co-Author)
- **Role**: Computational Linguistics Advisor & Annotation Director
- **Primary OpenReview Account Email**: `ualiyeva.irina@kaznu.kz`
- **Secondary OpenReview Account Email**: `i.ualiyeva@gmail.com`
- **Affiliation**: Department of Computer Science, Al-Farabi Kazakh National University, Almaty, Kazakhstan
- **ORCID**: 0000-0002-3925-118X

#### Author 3: Bin Guo (Senior Author & Principal Investigator)
- **Role**: Senior Supervisor & Principal Investigator
- **OpenReview Account / ID**: `~Bin_Guo3`
- **OpenReview Profile URL**: `https://openreview.net/profile?id=~Bin_Guo3`
- **Primary Institutional Email**: `guob@nwpu.edu.cn`
- **Affiliation**: School of Computer Science, Northwestern Polytechnical University, Xi'an, Shaanxi, China
- **Semantic Scholar Profile**: Bin Guo

### 3.2 Institutional Domain Declarations
The following domain suffixes must be registered under institutional conflict of interest settings to prevent reviewer assignment from affiliated entities:
1. `kaznu.kz` (Al-Farabi Kazakh National University)
2. `nwpu.edu.cn` (Northwestern Polytechnical University)

### 3.3 Conflict of Interest (COI) Declarations
According to ARR policy, COIs extend to advisors, advisees, family members, recent collaborators within the preceding 36 months, and individuals with whom authors have financial or formal employment ties:
- **Institutional COIs**:
  - Al-Farabi Kazakh National University (Almaty, Kazakhstan)
  - Northwestern Polytechnical University (Xi'an, China)
- **Collaborator COIs**:
  - Researchers with co-authorship on papers published between October 2023 and October 2026 with any of the co-authors.

---

## 4. ARR October 2026 Sustainable Reviewing Policy Compliance

### 4.1 Sustainable Reviewing Policy Overview
The ACL Rolling Review enforces a mandatory Sustainable Reviewing Policy to ensure equity across the review ecosystem. Under this policy:
1. Every submitted long paper must be backed by a nominated qualified reviewer from the author team or an eligible proxy.
2. Failure to nominate a qualified reviewer or refusal of assigned reviewing duties results in desk-rejection or review embargoes.
3. Reviewer assignments are workload-capped to prevent reviewer fatigue (maximum 3 papers per active cycle).
4. Nominated service contributors must confirm and complete OpenReview reviewer registration within 48 hours of the submission deadline (October 14, 2026, 23:59 AoE) to maintain guaranteed review status.

### 4.2 Nominated Service Contributor
To fulfill the mandatory ARR reviewing service quota for this submission:
- **Primary Nominated Reviewer**: Professor Bin Guo
  - **OpenReview Account / ID**: `~Bin_Guo3`
  - **OpenReview Profile URL**: `https://openreview.net/profile?id=~Bin_Guo3`
  - **Primary Institutional Email**: `guob@nwpu.edu.cn`
  - **Reviewing Qualification**: Senior researcher with extensive publication record across ACL, EMNLP, NAACL, and COLING; experienced Area Chair and ARR Reviewer.
  - **Subject Competencies**: Fact Verification, Information Extraction, Natural Language Inference, Robustness, Multilingual NLP.
  - **Service Registration Mandate**: Must confirm and complete OpenReview reviewer registration within 48 hours of the October 12 AoE submission deadline (no later than October 14, 2026, 23:59 AoE).
- **Alternate / Supplementary Nominated Reviewer**: Daulet Anekesh
  - **OpenReview ID / Email**: `anekesh_daulet@live.kaznu.kz`
  - **Reviewing Qualification**: Senior PhD candidate (post-Year 2) with published empirical NLP contributions.
  - **Subject Competencies**: Low-Resource Languages, Turkic Morphology, AI Text Detection, Information Retrieval.

### 4.3 Capacity Capping and Workload Rules
- **Submission-to-Review Ratio**: 1 paper submission = commitment of up to 3 review assignments.
- **Quota Verification**: The nominated contributor confirms zero active review over-commitments across overlapping ACL conferences during the November 2026 review period.
- **Profile Completeness**: All authors must ensure their OpenReview profiles have updated DBLP links, Semantic Scholar profiles, and recent publication history prior to the October 12 deadline to enable automated COI matching.
- **Reviewer Registration Window**: Designated service contributor Professor Bin Guo (`~Bin_Guo3`, `guob@nwpu.edu.cn`) must confirm/complete OpenReview reviewer registration within 48 hours of the October 12 AoE submission deadline.

---

## 5. Step-by-Step Upload Procedure

### 5.1 Step 1: Upload Primary Manuscript PDF
- **Local File Path**: `papers/kazakh_fever_conference/main.pdf`
- **File Specifications**:
  - Document Class: Two-column ACL layout (`\documentclass[11pt]{article}`, `\usepackage[review]{acl}`)
  - Line Numbering: Continuous line numbering active across both columns for review.
  - Total Page Count: Exactly 10 pages.
  - Main Body Budget: Sections 1 through 7 span Pages 1 to 8 (strictly compliant with the 8-page content limit).
  - Uncounted Sections: Section 8 (Limitations and Ethical Considerations) and References begin on Page 9 and conclude on Page 10.
  - Visual Layout: Two full-width tables (`table*` for Table 3 and Table 4) positioned at top of pages; zero overfull `\hbox` horizontal clipping.
  - Anonymity Audit: 0 occurrences of author names, institutional emails, or affiliations; anonymous title header set to `Anonymous ARR Submission`.

### 5.2 Step 2: Upload Supplementary Material Archive
- **Local File Path**: `papers/kazakh_fever_conference/supplementary_materials.zip`
- **Archive File Size**: Approximately 105 KB (well within OpenReview's 100 MB ceiling).
- **Archive Structure**:
  ```text
  supplementary_materials.zip
  |-- README.md                   (Anonymous documentation, environment setup, replication steps)
  |-- evaluate_fever.py           (Self-contained evaluation script for FEVER metrics)
  `-- sample_kazakh_fever_3k.jsonl (150-sample balanced Cyrillic-Kazakh test split)
  ```
- **Integrity Checks**:
  - Zero `.git`, `__pycache__`, `.DS_Store`, or Windows metadata files included.
  - All text files encoded in standard UTF-8.
  - Scripts tested and executable via Python 3.12 without external proprietary dependencies.

### 5.3 Step 3: Complete OpenReview Questionnaire Fields

#### A. Previous Submissions
- **Question**: Has this submission been previously submitted to ARR or an ACL-affiliated conference?
- **Answer**: `No`. This is a fresh first-time submission of the Kazakh-FEVER 3K benchmark.

#### B. Anonymity Verification
- **Question**: Does this submission conform to the ACL Double-Blind Review Policy?
- **Answer**: `Yes`. All author identities, affiliations, and direct repository identifiers have been scrubbed. Demonstrations and code are linked via an anonymous platform (`https://anonymous.4open.science/r/kazakh-ai-text-detection-57F8`).

#### C. Human Participants and Subject Research
- **Question**: Does this research involve human participants or subjects (e.g., native annotators)?
- **Answer**: `Yes`.
- **Form Response Explanation**:
  ```text
  The creation of the Kazakh-FEVER 3K benchmark involved native Kazakh linguists and professional fact-checkers. All annotators were compensated at or above prevailing local professional hourly wage rates ($15-20/hour equivalent). Annotators provided voluntary informed consent for the release of their anonymized annotation labels for academic research. No personal identifying information (PII) or sensitive biometric data was solicited, collected, or stored in the dataset.
  ```

#### D. Use of Existing Scientific Artifacts
- **Question**: Does this work utilize existing datasets, models, or software packages?
- **Answer**: `Yes`.
- **Form Response Explanation**:
  ```text
  The benchmark incorporates text passages from the public Kazakh Wikipedia dump (licensed under CC BY-SA 4.0) and public claim archives from Factcheck.kz (utilized under fair use principles for academic verification benchmarking). Baseline evaluations employ open-source foundational models including mBERT, XLM-RoBERTa, and mContriever. All artifacts are used in full compliance with their respective open licenses.
  ```

#### E. Computational and Environmental Resources
- **Question**: Does this work disclose computational requirements and environmental impact?
- **Answer**: `Yes`.
- **Form Response Explanation**:
  ```text
  All experimental models were fine-tuned using dedicated NVIDIA A100 GPU clusters, totaling approximately 120 GPU hours across all baseline runs and ablations. Estimated energy expenditure is under 35 kWh with a carbon footprint below 25 kg CO2eq. The inference pipeline employs quantized dense retrieval and sparse BM25 indexing, enabling real-time execution on standard CPU/GPU consumer hardware.
  ```

#### F. Dual-Use and Potential Societal Risks
- **Question**: Does this work carry potential risks of dual use or societal harm?
- **Answer**: `No unusual risks beyond standard NLP safety boundaries`.
- **Form Response Explanation**:
  ```text
  Fact verification technology carries risks of over-reliance if deployed as an autonomous censor. To mitigate this risk, the paper explicitly frames the benchmark and models as human-in-the-loop decision-support systems for professional journalists and moderators. The system outputs fine-grained evidence attribution and calibrated confidence scores, preventing black-box censorship of dialectal or minority expressions.
  ```

---

## 6. Pre-Submission Verification Checklist

Verify every item prior to hitting the final "Submit" button on OpenReview:

| Verification Item | Requirement | Status | Verification Evidence / Reference |
| :--- | :--- | :--- | :--- |
| **PDF Compilation** | Error-free XeLaTeX compile | Verified | Compiled cleanly with `acl.sty` (review mode) |
| **Main Body Page Budget** | Sections 1-7 <= 8 pages | Verified | Section 7 ends cleanly on Page 8 |
| **Total Page Count** | Exactly 10 pages | Verified | Pages 9-10 contain Section 8 and References |
| **Line Numbering** | Continuous line numbers | Verified | `\usepackage[review]{acl}` active |
| **Anonymity Audit** | Zero author/institution PII | Verified | Regex scan across `main.pdf` confirmed 0 matches |
| **Supplementary Archive** | Valid standalone ZIP | Verified | `supplementary_materials.zip` (105 KB, 3 files) |
| **UTF-8 Encoding** | Explicit Cyrillic support | Verified | Cyrillic glyphs render without missing glyph warnings |
| **Author OpenReview IDs** | Registered active emails | Verified | All 3 co-authors mapped to verified accounts |
| **COI Institutional List** | Domain suffixes logged | Verified | `kaznu.kz`, `nwpu.edu.cn` declared |
| **Nominated Reviewer** | Qualified service contributor | Verified | Professor Bin Guo (`~Bin_Guo3`, `guob@nwpu.edu.cn`) nominated; 48h registration window noted |
| **Regression Test Suite** | 513 unit tests pass | Verified | Zero failures across entire repository |
| **Invariant File Guard** | `aist2026/paper.tex` 0 diff | Verified | `git diff origin/main..HEAD` outputs 0 lines |

---

## 7. Submission Day Emergency Contacts, Roles, and Action Checklist

### 7.1 Submission Day Contacts and Role Allocation
- **Lead Submitter**: Daulet Anekesh (`anekesh_daulet@live.kaznu.kz`) - Responsible for portal entry, file uploads, and final submission verification.
- **Review Service Nominee**: Professor Bin Guo (`~Bin_Guo3`, `guob@nwpu.edu.cn`) - Designated service contributor responsible for OpenReview profile validation and completing reviewer registration within 48 hours of the October 12 AoE submission deadline.
- **Linguistic Advisor**: Irina M. Ualiyeva (`ualiyeva.irina@kaznu.kz`) - Responsible for annotation metadata sign-off.

### 7.2 Action Checklist and Service Registration Timeline
1. **Submission Phase (October 12, 2026, 23:59 AoE)**:
   - Finalize and upload primary manuscript PDF (`papers/kazakh_fever_conference/main.pdf`) and supplementary archive (`supplementary_materials.zip`).
   - Complete OpenReview submission metadata form, taxonomy subjects, keywords, and Responsible NLP questionnaire.
   - Designate Professor Bin Guo (`~Bin_Guo3`, `guob@nwpu.edu.cn`) as the primary service contributor.
2. **Reviewer Registration Phase (October 12–14, 2026 - Mandatory 48-Hour Window)**:
   - Mandatory Requirement: Designated service contributor Professor Bin Guo (`~Bin_Guo3`, `guob@nwpu.edu.cn`) must confirm and complete OpenReview reviewer registration within 48 hours of the October 12 AoE submission deadline (deadline: October 14, 2026, 23:59 AoE).
   - Profile & COI Completeness: Verify that Professor Bin Guo's OpenReview profile contains full institutional history (`nwpu.edu.cn`), DBLP, and Semantic Scholar publications to ensure automated conflict of interest (COI) matching and prevent desk rejection.
3. **Emergency Contingency Protocol**:
   - In case of unforeseen circumstances affecting the primary nominee, submit the official ARR emergency declaration form and activate alternate reviewer Daulet Anekesh (`anekesh_daulet@live.kaznu.kz`) before the 48-hour post-deadline window closes.
