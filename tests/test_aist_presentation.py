"""
Automated unit test suite verifying the integrity, page count, and empirical anchors
of the AIST 2026 conference presentation: Detecting AI-Generated User Reviews in Kazakh.
"""

import os
import re
import unittest
from pathlib import Path


class TestAISTPresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.tex_path = cls.repo_root / "presentation" / "aist2026_presentation.tex"
        cls.pdf_path = cls.repo_root / "presentation" / "aist2026_presentation.pdf"
        cls.desktop_pdf_path = Path("C:/Users/Roza/Desktop/AIST2026_Presentation_Daulet_Anekesh.pdf")

    def test_tex_source_exists_and_valid(self):
        self.assertTrue(self.tex_path.is_file(), f"Missing {self.tex_path}")
        content = self.tex_path.read_text(encoding="utf-8")
        self.assertGreater(len(content), 10000, "Presentation TeX source is unexpectedly short")
        self.assertIn("\\documentclass[aspectratio=169, 10pt]{beamer}", content)
        self.assertIn("Detecting AI-Generated User Reviews in Kazakh", content)

    def test_zero_placeholders_and_emojis(self):
        content = self.tex_path.read_text(encoding="utf-8")
        self.assertNotIn("TODO", content)
        self.assertNotIn("TBD", content)
        emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
        emojis_found = emoji_pattern.findall(content)
        self.assertEqual(len(emojis_found), 0, f"Found decorative emojis: {emojis_found}")

    def test_key_empirical_anchors_present(self):
        content = self.tex_path.read_text(encoding="utf-8")
        anchors = [
            "KazAI-Detect",
            "KazSAnDRA",
            "Sherkala-7B",
            "KazRoBERTa",
            "96.32\\%",
            "2.202",
            "43.2",
            "12,848",
            "4,000",
            "1,842",
            "McNemar",
            "Bootstrap",
            "12:45--13:10 Astana",
        ]
        for anchor in anchors:
            self.assertIn(anchor, content, f"Missing empirical anchor: {anchor}")

    def test_pdf_exists_and_exact_page_count(self):
        import pypdf
        self.assertTrue(self.pdf_path.is_file(), f"Missing {self.pdf_path}")
        self.assertGreater(self.pdf_path.stat().st_size, 100000, "PDF file size too small")
        reader = pypdf.PdfReader(str(self.pdf_path))
        self.assertEqual(len(reader.pages), 17, f"Expected exactly 17 pages, found {len(reader.pages)}")

    def test_desktop_pdf_synchronized(self):
        self.assertTrue(self.desktop_pdf_path.is_file(), f"Missing {self.desktop_pdf_path}")
        repo_pdf_size = self.pdf_path.stat().st_size
        desktop_pdf_size = self.desktop_pdf_path.stat().st_size
        self.assertEqual(
            repo_pdf_size,
            desktop_pdf_size,
            f"Desktop PDF size ({desktop_pdf_size}) != Repo PDF size ({repo_pdf_size})"
        )


if __name__ == "__main__":
    unittest.main()
