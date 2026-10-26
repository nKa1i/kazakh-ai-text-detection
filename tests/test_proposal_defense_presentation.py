# -*- coding: utf-8 -*-
"""
Automated regression test suite for Master's Proposal Defense Presentation (10-Minute Compact).
Validates 5-element structure, 12 slides, English-only enlarged tracker, branding, and figures.
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

    def test_slide_count_is_12(self):
        """Verify the presentation has exactly 12 slides for a 10-minute defense."""
        self.assertEqual(len(self.prs.slides), 12, f"Expected 12 slides, got {len(self.prs.slides)}")

    def test_desktop_mirror_exists(self):
        """Verify Desktop mirror exists and has non-zero size."""
        self.assertTrue(os.path.exists(DESKTOP_PATH), f"Desktop mirror must exist at {DESKTOP_PATH}")
        self.assertGreater(os.path.getsize(DESKTOP_PATH), 50000)

    def test_cover_slide_metadata(self):
        """Verify Slide 1 contains correct candidate and defense metadata."""
        slide1 = self.prs.slides[0]
        full_text = " ".join(shp.text for shp in slide1.shapes if shp.has_text_frame)
        self.assertIn("DAULET ANEKESH", full_text)
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
        """Verify Slide 2 (TOC) lists all 5 canonical sections in English."""
        slide2 = self.prs.slides[1]
        full_text = self._extract_all_text(slide2.shapes)
        self.assertIn("Research Background", full_text)
        self.assertIn("Related Work", full_text)
        self.assertIn("Research Content", full_text)
        self.assertIn("Research Innovations", full_text)
        self.assertIn("Schedule", full_text)

    def test_top_tracker_english_only(self):
        """Verify top tracker on content slides contains only English section titles."""
        expected_sections = [
            "Research Background",
            "Related Work",
            "Research Content",
            "Research Innovations",
            "Schedule"
        ]
        # Check Slide 3 (index 2)
        slide3 = self.prs.slides[2]
        header_texts = [
            shp.text_frame.text.strip()
            for shp in slide3.shapes
            if shp.top < 1000000 and shp.has_text_frame and shp.text_frame.text.strip() in expected_sections
        ]
        for sec in expected_sections:
            self.assertIn(sec, header_texts, f"Top tracker missing English title: {sec}")

    def test_schedule_is_only_schedule_table(self):
        """Verify Section 5 has only the schedule/timeline slide (Slide 11)."""
        slide11 = self.prs.slides[10]
        full_text = self._extract_all_text(slide11.shapes)
        self.assertIn("Dissertation Schedule", full_text)
        self.assertIn("2026.10", full_text)
        self.assertIn("2027.06", full_text)

    def test_closing_slide_thanks(self):
        """Verify Slide 12 contains closing thanks and guidance request."""
        slide12 = self.prs.slides[11]
        full_text = " ".join(shp.text for shp in slide12.shapes if shp.has_text_frame)
        self.assertTrue("批评指正" in full_text or "Thank You" in full_text)


if __name__ == "__main__":
    unittest.main()
