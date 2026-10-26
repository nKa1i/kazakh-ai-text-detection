# -*- coding: utf-8 -*-
"""
Automated regression test suite for Master's Proposal Defense Presentation.
Validates 5-element structure, 21 slides, branding, metadata, and figures.
"""

import os
import unittest
from pptx import Presentation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PPTX_PATH = os.path.join(PROJECT_ROOT, "Kazakh_AI_Detection_Thesis_Proposal_Defense_Oct29.pptx")
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop", "Kazakh_AI_Detection_Thesis_Proposal_Defense_Oct29.pptx")


class TestProposalDefensePresentation(unittest.TestCase):
    """Test suite for Master's Proposal Defense Presentation."""

    def setUp(self):
        self.assertTrue(os.path.exists(PPTX_PATH), f"Presentation must exist at {PPTX_PATH}")
        self.prs = Presentation(PPTX_PATH)

    def test_slide_count_is_21(self):
        """Verify the presentation has exactly 21 slides."""
        self.assertEqual(len(self.prs.slides), 21, f"Expected 21 slides, got {len(self.prs.slides)}")

    def test_desktop_mirror_exists(self):
        """Verify Desktop mirror exists and has non-zero size."""
        self.assertTrue(os.path.exists(DESKTOP_PATH), f"Desktop mirror must exist at {DESKTOP_PATH}")
        self.assertGreater(os.path.getsize(DESKTOP_PATH), 50000)

    def test_cover_slide_metadata(self):
        """Verify Slide 1 contains correct candidate and defense metadata."""
        slide1 = self.prs.slides[0]
        full_text = " ".join(shp.text for shp in slide1.shapes if shp.has_text_frame)
        self.assertIn("大雷", full_text)
        self.assertIn("郭斌", full_text)
        self.assertIn("2026", full_text)
        self.assertIn("2025291043", full_text)

    def test_no_commercial_watermarks(self):
        """Verify all commercial watermarks (243001978 / 合作QQ) are excised."""
        for s_idx, slide in enumerate(self.prs.slides):
            for shp in slide.shapes:
                if shp.has_text_frame:
                    self.assertNotIn("243001978", shp.text, f"Watermark found on slide {s_idx + 1}")
                    self.assertNotIn("合作QQ", shp.text, f"Watermark found on slide {s_idx + 1}")

    def _extract_all_text(self, shapes):
        """Recursively extracts text from shapes and group shapes."""
        texts = []
        for shp in shapes:
            if shp.has_text_frame:
                texts.append(shp.text)
            elif shp.shape_type == 6:  # Group shape
                texts.append(self._extract_all_text(shp.shapes))
        return " ".join(texts)

    def test_toc_slide_5_elements(self):
        """Verify Slide 2 (TOC) lists all 5 canonical sections."""
        slide2 = self.prs.slides[1]
        full_text = self._extract_all_text(slide2.shapes)
        self.assertIn("研究背景", full_text)
        self.assertTrue("相关工作" in full_text or "国内外研究现状" in full_text)
        self.assertIn("研究内容", full_text)
        self.assertTrue("研究创新" in full_text or "科学创新" in full_text)
        self.assertTrue("进展安排" in full_text or "进度安排" in full_text)

    def test_closing_slide_thanks(self):
        """Verify Slide 21 contains closing thanks and guidance request."""
        slide21 = self.prs.slides[20]
        full_text = " ".join(shp.text for shp in slide21.shapes if shp.has_text_frame)
        self.assertTrue("批评指正" in full_text or "THANKS" in full_text or "Thank You" in full_text)


if __name__ == "__main__":
    unittest.main()
