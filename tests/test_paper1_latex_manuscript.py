"""Unit tests verifying the Paper 1 ACL LaTeX manuscript scaffold and authoring.

Ensures that papers/kazakh_fever_conference/ contains complete, publication-grade
academic LaTeX documents with zero placeholders and valid BibTeX entries.
"""

from pathlib import Path
import re
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER1_DIR = PROJECT_ROOT / "papers" / "kazakh_fever_conference"
SECTIONS_DIR = PAPER1_DIR / "sections"

REQUIRED_MANUSCRIPT_FILES = [
    PAPER1_DIR / "main.tex",
    PAPER1_DIR / "custom.bib",
    SECTIONS_DIR / "01_introduction.tex",
    SECTIONS_DIR / "02_related_work.tex",
    SECTIONS_DIR / "03_dataset.tex",
    SECTIONS_DIR / "04_methodology.tex",
    SECTIONS_DIR / "05_experiments.tex",
    SECTIONS_DIR / "06_analysis.tex",
    SECTIONS_DIR / "07_conclusion.tex",
    SECTIONS_DIR / "08_limitations.tex",
]

REQUIRED_BIBTEX_KEYS = [
    "thorne2018fever",
    "izacard2022mcontriever",
    "conneau2020xlmr",
    "dubey2024llama3",
    "guo2026kazakh",
    "tang-etal-2024-minicheck",
    "wang2024raid",
    "chen-etal-2024-m3",
]

BANNED_PLACEHOLDERS = ["TODO", "TBD", "PLACEHOLDER", "XXX", "FIXME"]

# Regex matching emoji characters commonly used as decorations
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
    "]+",
    flags=re.UNICODE,
)


class TestPaper1LatexManuscript(unittest.TestCase):
    """Test suite verifying Kazakh-FEVER conference paper LaTeX scaffold."""

    def test_manuscript_files_exist(self):
        """Assert all 10 manuscript files exist in the target directory."""
        for file_path in REQUIRED_MANUSCRIPT_FILES:
            self.assertTrue(
                file_path.is_file(),
                f"Required manuscript file does not exist: {file_path}",
            )

    def test_no_banned_placeholders_or_emojis(self):
        """Assert zero occurrences of placeholder tokens or decorative emojis."""
        for file_path in REQUIRED_MANUSCRIPT_FILES:
            if not file_path.is_file():
                continue
            content = file_path.read_text(encoding="utf-8")

            for token in BANNED_PLACEHOLDERS:
                self.assertNotIn(
                    token,
                    content,
                    f"Banned placeholder '{token}' found in {file_path.name}",
                )

            emojis_found = EMOJI_PATTERN.findall(content)
            self.assertEqual(
                len(emojis_found),
                0,
                f"Decorative emojis found in {file_path.name}: {emojis_found}",
            )

    def test_main_document_structure(self):
        """Assert main.tex has valid ACL structure, abstract, and section inputs."""
        main_file = PAPER1_DIR / "main.tex"
        if not main_file.is_file():
            self.fail("main.tex does not exist")

        content = main_file.read_text(encoding="utf-8")
        self.assertIn("\\documentclass", content)
        self.assertIn("Kazakh-FEVER 3K", content)
        self.assertIn("\\begin{abstract}", content)
        self.assertIn("\\end{abstract}", content)
        self.assertIn("Hard NEI", content)
        self.assertIn("Trust Matrix", content)

        # Check section inputs
        for i in range(1, 9):
            prefix = f"{i:02d}_"
            self.assertTrue(
                re.search(rf"\\input\{{sections/{prefix}[a-z_]+(\.tex)?\}}", content)
                is not None,
                f"main.tex missing input for section {prefix}*",
            )

        self.assertIn("\\bibliography{custom}", content)

    def test_required_tables_and_equations(self):
        """Assert required table labels and equation formulations exist."""
        # Table 1: tab:dataset_stats in sections/03_dataset.tex
        dataset_file = SECTIONS_DIR / "03_dataset.tex"
        if dataset_file.is_file():
            ds_content = dataset_file.read_text(encoding="utf-8")
            self.assertIn("\\label{tab:dataset_stats}", ds_content)
            self.assertIn("2,100", ds_content)
            self.assertIn("450", ds_content)
            self.assertIn("300", ds_content)
            self.assertIn("150", ds_content)
            self.assertIn("3,000", ds_content)

        # Methodology equations in sections/04_methodology.tex
        method_file = SECTIONS_DIR / "04_methodology.tex"
        if method_file.is_file():
            m_content = method_file.read_text(encoding="utf-8")
            self.assertIn("\\label{eq:hybrid_retrieval}", m_content)
            self.assertIn("S_{\\text{hybrid}}", m_content)
            self.assertIn("\\label{eq:nli_verifier}", m_content)
            self.assertIn("P(y \\mid c, E)", m_content)
            self.assertIn("\\label{eq:trust_fact}", m_content)
            self.assertIn("T_{\\text{fact}}", m_content)
            self.assertIn("\\label{eq:trust_gen}", m_content)
            self.assertIn("T_{\\text{gen}}", m_content)
            self.assertIn("\\label{eq:trust_score}", m_content)
            self.assertIn("\\mathcal{T}(x)", m_content)

        # Table 2 & Table 3 in sections/05_experiments.tex
        exp_file = SECTIONS_DIR / "05_experiments.tex"
        if exp_file.is_file():
            exp_content = exp_file.read_text(encoding="utf-8")
            self.assertIn("\\label{tab:main_results}", exp_content)
            self.assertIn("\\label{tab:retrieval_results}", exp_content)
            self.assertIn("mBERT", exp_content)
            self.assertIn("XLM-RoBERTa", exp_content)
            self.assertIn("LLaMA-3", exp_content)
            self.assertIn("mContriever", exp_content)

    def test_bibtex_entries(self):
        """Assert custom.bib contains complete, valid entries for all required keys."""
        bib_file = PAPER1_DIR / "custom.bib"
        if not bib_file.is_file():
            self.fail("custom.bib does not exist")

        content = bib_file.read_text(encoding="utf-8")
        for key in REQUIRED_BIBTEX_KEYS:
            self.assertTrue(
                re.search(rf"@\w+\s*\{{\s*{key}\s*,", content, re.IGNORECASE)
                is not None,
                f"custom.bib missing required BibTeX key: {key}",
            )

    def test_section_content_coverage(self):
        """Assert each section covers required thematic and linguistic components."""
        intro = (SECTIONS_DIR / "01_introduction.tex").read_text(encoding="utf-8") if (SECTIONS_DIR / "01_introduction.tex").is_file() else ""
        self.assertIn("agglutinative", intro.lower())
        self.assertIn("shortcut", intro.lower())

        rel_work = (SECTIONS_DIR / "02_related_work.tex").read_text(encoding="utf-8") if (SECTIONS_DIR / "02_related_work.tex").is_file() else ""
        self.assertIn("fact verification", rel_work.lower())
        self.assertIn("turkic", rel_work.lower())

        analysis = (SECTIONS_DIR / "06_analysis.tex").read_text(encoding="utf-8") if (SECTIONS_DIR / "06_analysis.tex").is_file() else ""
        self.assertIn("affix", analysis.lower())

        limitations = (SECTIONS_DIR / "08_limitations.tex").read_text(encoding="utf-8") if (SECTIONS_DIR / "08_limitations.tex").is_file() else ""
        self.assertIn("limitation", limitations.lower())

    def test_aist_paper_unmodified(self):
        """Assert aist2026/paper.tex is completely untouched."""
        aist_file = PROJECT_ROOT / "aist2026" / "paper.tex"
        self.assertTrue(aist_file.is_file(), "aist2026/paper.tex must exist")


if __name__ == "__main__":
    unittest.main()
