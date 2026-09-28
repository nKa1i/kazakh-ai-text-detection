"""Automated Submission Integrity Regression Suite for COLING 2027 / ARR October.

Validates that papers/kazakh_fever_conference/main.pdf compiles cleanly under XeLaTeX
with an exact 10-page layout (Sections 1-7 ending on or before Page 8), zero missing glyphs,
zero undefined citations or references, strict double-blind anonymity, complete
Limitations and Ethical Considerations section, valid supplementary materials archive,
and zero regressions or modifications against the invariant aist2026/paper.tex baseline.
"""

import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest
import zipfile

import pypdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFERENCE_DIR = PROJECT_ROOT / "papers" / "kazakh_fever_conference"
SECTIONS_DIR = CONFERENCE_DIR / "sections"
MAIN_TEX = CONFERENCE_DIR / "main.tex"
MAIN_PDF = CONFERENCE_DIR / "main.pdf"
MAIN_LOG = CONFERENCE_DIR / "main.log"
MAIN_AUX = CONFERENCE_DIR / "main.aux"
SUPPLEMENTARY_ZIP = CONFERENCE_DIR / "supplementary_materials.zip"
AIST_PAPER_TEX = PROJECT_ROOT / "aist2026" / "paper.tex"

PROHIBITED_ANONYMITY_TERMS = [
    "Daulet",
    "Anekesh",
    "Ualiyeva",
    "Zhijiang",
    "Guo",
    "KazNU",
    "NWPU",
    "nKa1i",
]

BANNED_ZIP_ARTIFACTS = [
    ".git",
    ".gitignore",
    ".ds_store",
    "__pycache__",
    ".pyc",
    ".pyo",
    "thumbs.db",
    "desktop.ini",
]

REQUIRED_ZIP_FILES = [
    "README.md",
    "evaluate_fever.py",
    "sample_kazakh_fever_3k.jsonl",
]

REQUIRED_LIMITATIONS_SUBSECTIONS = [
    "Linguistic and Typological Scope",
    "Evidence Corpus and Temporal Freshness",
    "Computational and Environmental Resource Bounds",
    "Ethical Considerations and Responsible Deployment",
]


class TestColingSubmissionIntegrity(unittest.TestCase):
    """Test suite verifying end-to-end submission integrity for COLING 2027 ARR cycle."""

    def test_compilation_success(self):
        """Assert main.pdf exists, is non-empty, and verified compilation status in main.log."""
        self.assertTrue(
            MAIN_PDF.is_file(),
            f"Expected compiled conference paper PDF at {MAIN_PDF}",
        )
        self.assertGreater(
            MAIN_PDF.stat().st_size,
            0,
            f"Compiled conference paper PDF at {MAIN_PDF} is empty",
        )

        self.assertTrue(
            MAIN_LOG.is_file(),
            f"Expected LaTeX compilation log at {MAIN_LOG}",
        )
        log_content = MAIN_LOG.read_text(encoding="utf-8", errors="ignore")

        # Verify output written statement in LaTeX log
        self.assertRegex(
            log_content,
            r"Output written on .*main\.pdf \(\d+ pages\)",
            f"main.log does not confirm completed PDF output in {MAIN_LOG}",
        )

        # Assert zero fatal compilation errors
        self.assertNotIn(
            "Fatal error occurred",
            log_content,
            f"Fatal error detected in {MAIN_LOG}",
        )
        self.assertNotIn(
            "! Emergency stop",
            log_content,
            f"Emergency stop detected in {MAIN_LOG}",
        )

    def test_page_budget(self):
        """Assert Sections 1-7 end on or before Page 8, and total page count is exactly 10 pages."""
        self.assertTrue(MAIN_PDF.is_file(), f"PDF file not found: {MAIN_PDF}")

        reader = pypdf.PdfReader(str(MAIN_PDF))
        total_pages = len(reader.pages)
        self.assertEqual(
            total_pages,
            10,
            f"COLING 2027 ARR paper must be exactly 10 pages, found {total_pages}",
        )

        # Confirm 10 pages recorded in main.log
        if MAIN_LOG.is_file():
            log_content = MAIN_LOG.read_text(encoding="utf-8", errors="ignore")
            self.assertIn(
                "Output written on main.pdf (10 pages)",
                log_content,
                "main.log must record exactly 10 output pages",
            )

        # Page 1 must contain Title and Section 1 Introduction
        page_1_text = reader.pages[0].extract_text() or ""
        self.assertIn("1 Introduction", page_1_text)

        # Page 8 must contain Section 7 Conclusion and conclude main body
        page_8_text = reader.pages[7].extract_text() or ""
        self.assertIn("7 Conclusion", page_8_text)

        # Section 8 Limitations and References must NOT start on Pages 1-8
        for page_idx in range(8):
            page_text = reader.pages[page_idx].extract_text() or ""
            self.assertNotIn(
                "8 Limitations and Ethical Considerations",
                page_text,
                f"Section 8 appeared prematurely on Page {page_idx + 1}",
            )

        # Page 9 must contain Section 8 Limitations and start References
        page_9_text = reader.pages[8].extract_text() or ""
        self.assertIn(
            "8 Limitations and Ethical Considerations",
            page_9_text,
            "Section 8 must begin on Page 9",
        )
        self.assertIn(
            "References",
            page_9_text,
            "References must begin on Page 9 following Section 8",
        )

        # Page 10 must contain the continuation of References
        page_10_text = reader.pages[9].extract_text() or ""
        self.assertGreater(
            len(page_10_text.strip()),
            0,
            "Page 10 must contain valid continuation content",
        )

        # Verify section labels in main.aux if present
        if MAIN_AUX.is_file():
            aux_content = MAIN_AUX.read_text(encoding="utf-8", errors="ignore")
            # \newlabel{sec:conclusion}{{7}{8}}
            conclusion_match = re.search(r"\\newlabel\{sec:conclusion\}\{\{7\}\{(\d+)\}\}", aux_content)
            if conclusion_match:
                conclusion_page = int(conclusion_match.group(1))
                self.assertLessEqual(
                    conclusion_page,
                    8,
                    f"Section 7 Conclusion page ({conclusion_page}) exceeds 8-page content budget",
                )

            # \newlabel{sec:limitations}{{8}{9}}
            limitations_match = re.search(r"\\newlabel\{sec:limitations\}\{\{8\}\{(\d+)\}\}", aux_content)
            if limitations_match:
                limitations_page = int(limitations_match.group(1))
                self.assertGreaterEqual(
                    limitations_page,
                    9,
                    f"Section 8 Limitations page ({limitations_page}) should be on Page 9 or later",
                )

    def test_no_missing_glyphs(self):
        """Scan papers/kazakh_fever_conference/main.log asserting 0 occurrences of 'Missing character:'."""
        self.assertTrue(MAIN_LOG.is_file(), f"Log file not found: {MAIN_LOG}")
        log_content = MAIN_LOG.read_text(encoding="utf-8", errors="ignore")

        missing_character_lines = [
            line.strip()
            for line in log_content.splitlines()
            if "Missing character:" in line
        ]
        self.assertEqual(
            len(missing_character_lines),
            0,
            f"Found {len(missing_character_lines)} missing character warning(s) in main.log:\n"
            + "\n".join(missing_character_lines[:10]),
        )

    def test_no_undefined_references(self):
        """Assert 0 undefined citations and 0 undefined references in main.log."""
        self.assertTrue(MAIN_LOG.is_file(), f"Log file not found: {MAIN_LOG}")
        log_content = MAIN_LOG.read_text(encoding="utf-8", errors="ignore")

        undefined_citations = [
            line.strip()
            for line in log_content.splitlines()
            if "LaTeX Warning:" in line and "Citation" in line and "undefined" in line
        ]
        self.assertEqual(
            len(undefined_citations),
            0,
            f"Found {len(undefined_citations)} undefined citation warning(s) in main.log:\n"
            + "\n".join(undefined_citations[:10]),
        )

        undefined_references = [
            line.strip()
            for line in log_content.splitlines()
            if "LaTeX Warning:" in line and "Reference" in line and "undefined" in line
        ]
        self.assertEqual(
            len(undefined_references),
            0,
            f"Found {len(undefined_references)} undefined reference warning(s) in main.log:\n"
            + "\n".join(undefined_references[:10]),
        )

    def test_double_blind_anonymity(self):
        """Extract text across all pages of main.pdf and verify strictly 0 occurrences of prohibited terms."""
        self.assertTrue(MAIN_PDF.is_file(), f"PDF file not found: {MAIN_PDF}")
        reader = pypdf.PdfReader(str(MAIN_PDF))

        violations = []
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            for term in PROHIBITED_ANONYMITY_TERMS:
                if term.lower() in page_text.lower():
                    violations.append(
                        f"Page {page_idx + 1}: found prohibited term '{term}'"
                    )

        self.assertEqual(
            len(violations),
            0,
            f"Double-blind anonymity violations found in main.pdf ({len(violations)}):\n"
            + "\n".join(violations),
        )

        # Verify main.tex author block is scrubbed
        if MAIN_TEX.is_file():
            main_text = MAIN_TEX.read_text(encoding="utf-8", errors="ignore")
            self.assertIn("Anonymous ARR Submission", main_text)
            self.assertIn("Affiliation scrubbed for double-blind review", main_text)

    def test_limitations_section_present(self):
        """Assert Section 8 (Limitations and Ethical Considerations) is present and non-empty."""
        limitations_file = SECTIONS_DIR / "08_limitations.tex"
        self.assertTrue(
            limitations_file.is_file(),
            f"Expected limitations section file at {limitations_file}",
        )
        self.assertGreater(
            limitations_file.stat().st_size,
            0,
            f"Limitations section file at {limitations_file} is empty",
        )

        content = limitations_file.read_text(encoding="utf-8")
        self.assertIn(
            "\\section{Limitations and Ethical Considerations}",
            content,
            "08_limitations.tex missing '\\section{Limitations and Ethical Considerations}'",
        )

        # Verify main.tex inputs the limitations section
        main_content = MAIN_TEX.read_text(encoding="utf-8")
        self.assertTrue(
            "\\input{sections/08_limitations.tex}" in main_content or "\\input{sections/08_limitations}" in main_content,
            "main.tex must input 'sections/08_limitations'",
        )

        # Verify all 4 required subsections are present in LaTeX source
        for subsection in REQUIRED_LIMITATIONS_SUBSECTIONS:
            self.assertIn(
                subsection,
                content,
                f"Required subsection '{subsection}' missing from 08_limitations.tex",
            )

        # Verify Section 8 appears in compiled main.pdf
        if MAIN_PDF.is_file():
            reader = pypdf.PdfReader(str(MAIN_PDF))
            full_text = " ".join(page.extract_text() or "" for page in reader.pages)
            clean_text = " ".join(re.sub(r"\d+", " ", full_text).split())
            self.assertIn(
                "Limitations and Ethical Considerations",
                clean_text,
                "Limitations and Ethical Considerations not found in main.pdf text",
            )
            for subsection in REQUIRED_LIMITATIONS_SUBSECTIONS:
                self.assertIn(
                    subsection,
                    clean_text,
                    f"Subsection '{subsection}' not found in compiled main.pdf",
                )

    def test_supplementary_zip_validity(self):
        """Assert supplementary_materials.zip exists, is valid uncorrupted ZIP, and sanitized."""
        self.assertTrue(
            SUPPLEMENTARY_ZIP.is_file(),
            f"Expected supplementary archive at {SUPPLEMENTARY_ZIP}",
        )
        self.assertGreater(
            SUPPLEMENTARY_ZIP.stat().st_size,
            0,
            f"Supplementary archive at {SUPPLEMENTARY_ZIP} is empty",
        )

        with zipfile.ZipFile(str(SUPPLEMENTARY_ZIP), "r") as zf:
            bad_file = zf.testzip()
            self.assertIsNone(
                bad_file,
                f"Supplementary archive CRC check failed on member: {bad_file}",
            )

            namelist = zf.namelist()
            for required_file in REQUIRED_ZIP_FILES:
                self.assertIn(
                    required_file,
                    namelist,
                    f"Required file '{required_file}' missing from supplementary ZIP",
                )
                info = zf.getinfo(required_file)
                self.assertGreater(
                    info.file_size,
                    0,
                    f"File '{required_file}' in supplementary ZIP is empty",
                )

            # Check zero git / cache / OS artifacts
            for member_name in namelist:
                member_lower = member_name.lower()
                for artifact in BANNED_ZIP_ARTIFACTS:
                    self.assertNotIn(
                        artifact,
                        member_lower,
                        f"Banned artifact pattern '{artifact}' found in ZIP entry '{member_name}'",
                    )

            # Check zero prohibited anonymity strings in file names or contents
            for member_name in namelist:
                member_lower = member_name.lower()
                for term in PROHIBITED_ANONYMITY_TERMS:
                    self.assertNotIn(
                        term.lower(),
                        member_lower,
                        f"Prohibited term '{term}' found in ZIP filename '{member_name}'",
                    )

                raw_bytes = zf.read(member_name)
                member_text = raw_bytes.decode("utf-8", errors="ignore")
                for term in PROHIBITED_ANONYMITY_TERMS:
                    self.assertNotIn(
                        term.lower(),
                        member_text.lower(),
                        f"Prohibited term '{term}' found in contents of '{member_name}' in ZIP",
                    )

            # Validate sample_kazakh_fever_3k.jsonl structure
            jsonl_bytes = zf.read("sample_kazakh_fever_3k.jsonl")
            jsonl_lines = [
                line.strip()
                for line in jsonl_bytes.decode("utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(
                len(jsonl_lines),
                150,
                f"sample_kazakh_fever_3k.jsonl must contain at least 150 samples, found {len(jsonl_lines)}",
            )

            expected_fields = {"id", "claim", "evidence_text", "gold_label", "evidence_doc_id", "distortion_type"}
            observed_labels = set()
            for line_idx, line_str in enumerate(jsonl_lines):
                item = json.loads(line_str)
                missing = expected_fields - set(item.keys())
                self.assertEqual(
                    len(missing),
                    0,
                    f"Line {line_idx + 1} in sample_kazakh_fever_3k.jsonl missing fields: {missing}",
                )
                observed_labels.add(item["gold_label"])

            self.assertEqual(
                observed_labels,
                {"SUPPORTED", "REFUTED", "NOT ENOUGH INFO"},
                f"Unexpected labels in sample_kazakh_fever_3k.jsonl: {observed_labels}",
            )

    def test_aist_paper_invariance(self):
        """Verify that aist2026/paper.tex is strictly invariant with 0 git diff lines against origin/main."""
        self.assertTrue(
            AIST_PAPER_TEX.is_file(),
            f"Expected invariant baseline paper at {AIST_PAPER_TEX}",
        )
        self.assertGreater(
            AIST_PAPER_TEX.stat().st_size,
            0,
            f"Invariant baseline paper at {AIST_PAPER_TEX} is empty",
        )

        git_bin = shutil.which("git")
        self.assertIsNotNone(git_bin, "git executable not found on system PATH")

        result = subprocess.run(
            ["git", "diff", "origin/main..HEAD", "--", "aist2026/paper.tex"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"git diff failed with code {result.returncode}:\n{result.stderr}",
        )
        self.assertEqual(
            result.stdout.strip(),
            "",
            f"aist2026/paper.tex has unexpected git diff against origin/main:\n{result.stdout}",
        )


if __name__ == "__main__":
    unittest.main()
