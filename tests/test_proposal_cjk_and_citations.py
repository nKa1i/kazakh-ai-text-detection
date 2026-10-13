"""Automated regression suite for NPU proposal CJK line breaking and citation integrity.

Validates that proposal/main.pdf compiles cleanly under XeLaTeX with zero missing
glyphs, zero undefined citations, complete Chinese requirements text on Page 2,
and sufficient main body word count (at least 3500 words).
"""

import os
import re
import unittest
import pypdf


class TestProposalCJKAndCitations(unittest.TestCase):
    """Regression test suite for NPU Thesis Proposal CJK wrapping and citations."""

    def setUp(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.pdf_path = os.path.join(self.base_dir, "proposal", "main.pdf")
        self.log_path = os.path.join(self.base_dir, "proposal", "main.log")

    def test_pdf_exists(self):
        """Verify proposal/main.pdf exists."""
        self.assertTrue(os.path.exists(self.pdf_path), "proposal/main.pdf does not exist")

    def test_no_missing_characters(self):
        """Verify zero missing character warnings in proposal/main.log."""
        with open(self.log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_content = f.read()
        missing = re.findall(r"Missing character:.*", log_content)
        self.assertEqual(len(missing), 0, f"Found missing characters in main.log: {missing}")

    def test_no_undefined_citations(self):
        """Verify zero undefined citation warnings in proposal/main.log."""
        with open(self.log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_content = f.read()
        undefined_citations = re.findall(r"LaTeX Warning: Citation .* undefined", log_content)
        self.assertEqual(len(undefined_citations), 0, f"Undefined citations: {undefined_citations}")

    def test_word_count(self):
        """Verify main body text (Page 3 onwards) contains at least 3500 words."""
        reader = pypdf.PdfReader(self.pdf_path)
        total_text = ""
        # Pages 3 to end are main body (0-indexed pages 2 onwards)
        for page in reader.pages[2:]:
            text = page.extract_text()
            if text:
                total_text += " " + text
        words = len(total_text.split())
        self.assertGreaterEqual(words, 3500, f"Main body word count {words} is less than 3500")

    def test_cjk_page_wrapping(self):
        """Verify Page 2 extracted text contains all Chinese requirements text."""
        reader = pypdf.PdfReader(self.pdf_path)
        page2_text = reader.pages[1].extract_text()
        self.assertIn("研究生学位论文选题报告的要求", page2_text)
        self.assertIn("硕士生的选题报告内容应包括文献综述", page2_text)
        self.assertIn("本表可以打印或用钢笔认真填写", page2_text)

    def test_empty_reservation_box(self):
        """Verify final page contains Reservation label but pre-filled candidate text is absent."""
        reader = pypdf.PdfReader(self.pdf_path)
        last_page_text = reader.pages[-1].extract_text()
        self.assertIn("Reservation", last_page_text, "Reservation label missing on final page")
        self.assertNotIn(
            "The research program detailed in this title selection report reflects original",
            last_page_text,
            "Reservation box must not contain pre-filled text",
        )


    def test_section1_and_section2_structure(self):
        """Verify Sections 1 and 2 follow the approved student proposal headings."""
        reader = pypdf.PdfReader(self.pdf_path)
        body_text = ""
        for page in reader.pages[2:]:
            body_text += page.extract_text() + "\n"
        self.assertIn("1 Research Background", body_text)
        self.assertIn("1.1 Relevance and Context", body_text)
        self.assertIn("1.2 Challenges of Existing", body_text)
        self.assertIn("1.3 Potential of", body_text)
        self.assertIn("2 Related Work", body_text)
        self.assertIn("2.1 Design and Architecture", body_text)
        self.assertIn("2.4 Main Research Points", body_text)


if __name__ == "__main__":
    unittest.main()

