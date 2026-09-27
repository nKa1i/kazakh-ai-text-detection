"""Automated Compilation Regression Test Suite for Master's Thesis Dissertation.

Validates that thesis/main.tex, all 6 chapters, and all 3 formal appendices
compile cleanly under XeLaTeX + BibTeX with zero missing glyphs, zero undefined
citations or references, zero placeholders, and page count >= 60.
"""

from pathlib import Path
import os
import re
import shutil
import subprocess
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THESIS_DIR = PROJECT_ROOT / "thesis"
FRONTMATTER_DIR = THESIS_DIR / "frontmatter"
CHAPTERS_DIR = THESIS_DIR / "chapters"
APPENDICES_DIR = THESIS_DIR / "appendices"

REQUIRED_THESIS_FILES = [
    THESIS_DIR / "main.tex",
    THESIS_DIR / "references.bib",
    FRONTMATTER_DIR / "title_page.tex",
    FRONTMATTER_DIR / "abstract.tex",
    FRONTMATTER_DIR / "acknowledgments.tex",
    FRONTMATTER_DIR / "abbreviations.tex",
    CHAPTERS_DIR / "01_introduction.tex",
    CHAPTERS_DIR / "02_related_work.tex",
    CHAPTERS_DIR / "03_linguistic_foundation.tex",
    CHAPTERS_DIR / "04_morpho_contrastive_detection.tex",
    CHAPTERS_DIR / "05_factual_verification_and_fever.tex",
    CHAPTERS_DIR / "06_system_explainability_and_conclusion.tex",
    APPENDICES_DIR / "appendix_a_fst_grammar.tex",
    APPENDICES_DIR / "appendix_b_fever_annotation.tex",
    APPENDICES_DIR / "appendix_c_error_cases.tex",
]

BANNED_PLACEHOLDERS = re.compile(r"\b(TODO|TBD|FIXME)\b", re.IGNORECASE)

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
    "]+",
    flags=re.UNICODE,
)


class TestDissertationCompilation(unittest.TestCase):
    """Regression test suite for thesis dissertation compilation and structural integrity."""

    def test_all_required_files_exist(self):
        """Assert main.tex, references.bib, all 4 frontmatter, 6 chapters, and 3 appendices exist."""
        for file_path in REQUIRED_THESIS_FILES:
            self.assertTrue(
                file_path.is_file(),
                f"Required dissertation file does not exist: {file_path}",
            )

    def test_no_placeholders_in_thesis(self):
        """Assert zero occurrences of TODO, TBD, or FIXME across all thesis files."""
        for root, _, files in os.walk(THESIS_DIR):
            for fname in files:
                if fname.endswith((".tex", ".bib")):
                    p = Path(root) / fname
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    matches = BANNED_PLACEHOLDERS.findall(content)
                    self.assertEqual(
                        len(matches),
                        0,
                        f"Found banned placeholder(s) {matches} in {p}",
                    )

    def test_no_decorative_emojis_in_thesis(self):
        """Assert strictly zero decorative emojis across all .tex and .bib thesis files."""
        for root, _, files in os.walk(THESIS_DIR):
            for fname in files:
                if fname.endswith((".tex", ".bib")):
                    p = Path(root) / fname
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    emojis_found = EMOJI_PATTERN.findall(content)
                    self.assertEqual(
                        len(emojis_found),
                        0,
                        f"Decorative emojis found in {p}: {emojis_found}",
                    )

    def test_aist2026_invariance(self):
        """Verify that aist2026/paper.tex exists and git diff against origin/main is empty."""
        aist_file = PROJECT_ROOT / "aist2026" / "paper.tex"
        self.assertTrue(aist_file.is_file(), "aist2026/paper.tex must exist")
        git_bin = shutil.which("git")
        if git_bin:
            result = subprocess.run(
                ["git", "diff", "origin/main..HEAD", "--", "aist2026/paper.tex"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, f"git diff failed with error: {result.stderr}")
            self.assertEqual(
                result.stdout.strip(),
                "",
                f"aist2026/paper.tex has unexpected git diff against origin/main: {result.stdout}",
            )

    def test_xelatex_compilation_and_pdf_integrity(self):
        """Assert XeLaTeX compilation succeeds with exit code 0, producing >= 60 page PDF."""
        xelatex_bin = shutil.which("xelatex")
        if not xelatex_bin:
            self.skipTest("xelatex not found on PATH; skipping compilation execution.")

        main_tex = THESIS_DIR / "main.tex"
        main_pdf = THESIS_DIR / "main.pdf"
        main_log = THESIS_DIR / "main.log"

        # Determine if compilation is required
        # If PDF or BBL missing, or if any source is newer than PDF, recompile
        needs_compile = not main_pdf.is_file() or not (THESIS_DIR / "main.bbl").is_file()
        if not needs_compile:
            pdf_mtime = main_pdf.stat().st_mtime
            for f in REQUIRED_THESIS_FILES:
                if f.stat().st_mtime > pdf_mtime:
                    needs_compile = True
                    break

        if needs_compile:
            cmd = ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
            res = subprocess.run(cmd, cwd=str(THESIS_DIR), capture_output=True, text=True, timeout=180)
            self.assertEqual(
                res.returncode,
                0,
                f"xelatex compilation failed with code {res.returncode}:\n{res.stdout[-1500:]}",
            )

            bibtex_bin = shutil.which("bibtex")
            if bibtex_bin:
                res_bib = subprocess.run(
                    ["bibtex", "main"],
                    cwd=str(THESIS_DIR),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(res_bib.returncode, 0, f"bibtex failed:\n{res_bib.stdout}")

                subprocess.run(cmd, cwd=str(THESIS_DIR), capture_output=True, text=True, timeout=180)
                subprocess.run(cmd, cwd=str(THESIS_DIR), capture_output=True, text=True, timeout=180)

        self.assertTrue(main_pdf.is_file(), f"Expected compiled PDF at {main_pdf}")

        # Multi-layer check for page count >= 60
        num_pages = 0
        try:
            import pypdf
            reader = pypdf.PdfReader(str(main_pdf))
            num_pages = len(reader.pages)
        except ImportError:
            pass

        if num_pages == 0:
            pdfinfo_bin = shutil.which("pdfinfo")
            if pdfinfo_bin:
                info_res = subprocess.run(
                    [pdfinfo_bin, str(main_pdf)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                match = re.search(r"Pages:\s+(\d+)", info_res.stdout)
                if match:
                    num_pages = int(match.group(1))

        if num_pages == 0 and main_log.is_file():
            log_text = main_log.read_text(encoding="utf-8", errors="ignore")
            match = re.search(r"Output written on main\.pdf \((\d+) pages\)", log_text)
            if match:
                num_pages = int(match.group(1))

        self.assertGreaterEqual(
            num_pages,
            60,
            f"Dissertation PDF must have at least 60 pages, but found {num_pages}",
        )

    def test_log_warnings_and_integrity(self):
        """Assert thesis/main.log has 0 missing characters, 0 undefined citations, 0 undefined refs."""
        main_log = THESIS_DIR / "main.log"
        self.assertTrue(main_log.is_file(), f"Expected log file at {main_log}")

        log_content = main_log.read_text(encoding="utf-8", errors="ignore")

        # 1. Missing characters
        missing_chars = [
            line for line in log_content.splitlines()
            if "Missing character: There is no" in line
        ]
        self.assertEqual(
            len(missing_chars),
            0,
            f"Found missing character warnings in main.log ({len(missing_chars)}):\n"
            + "\n".join(missing_chars[:10]),
        )

        # 2. Undefined references
        undefined_refs = [
            line for line in log_content.splitlines()
            if "Reference" in line and "undefined" in line
        ]
        self.assertEqual(
            len(undefined_refs),
            0,
            f"Found undefined reference warnings in main.log ({len(undefined_refs)}):\n"
            + "\n".join(undefined_refs[:10]),
        )

        # 3. Undefined citations
        undefined_cites = [
            line for line in log_content.splitlines()
            if "Citation" in line and "undefined" in line
        ]
        self.assertEqual(
            len(undefined_cites),
            0,
            f"Found undefined citation warnings in main.log ({len(undefined_cites)}):\n"
            + "\n".join(undefined_cites[:10]),
        )

    def test_main_aux_links_all_chapters_and_appendices(self):
        """Assert thesis/main.aux exists and includes all 6 chapters and 3 appendices."""
        main_aux = THESIS_DIR / "main.aux"
        self.assertTrue(main_aux.is_file(), f"Expected aux file at {main_aux}")

        aux_content = main_aux.read_text(encoding="utf-8", errors="ignore")

        expected_inputs = [
            "frontmatter/title_page.aux",
            "frontmatter/abstract.aux",
            "frontmatter/acknowledgments.aux",
            "frontmatter/abbreviations.aux",
            "chapters/01_introduction.aux",
            "chapters/02_related_work.aux",
            "chapters/03_linguistic_foundation.aux",
            "chapters/04_morpho_contrastive_detection.aux",
            "chapters/05_factual_verification_and_fever.aux",
            "chapters/06_system_explainability_and_conclusion.aux",
            "appendices/appendix_a_fst_grammar.aux",
            "appendices/appendix_b_fever_annotation.aux",
            "appendices/appendix_c_error_cases.aux",
        ]

        for expected in expected_inputs:
            self.assertIn(
                expected,
                aux_content,
                f"Expected main.aux to link component: {expected}",
            )


if __name__ == "__main__":
    unittest.main()
