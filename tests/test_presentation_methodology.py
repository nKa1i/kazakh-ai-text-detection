# -*- coding: utf-8 -*-
"""
Tests for Slide 7, 19, and 20 updates in AnekeshD_Progress.pptx.
Validates:
- Embedding of Figure 3 (Methodological Innovations Framework) on Slide 7 across full width.
- Restoration of Slide 19 with Four Primary Academic Contributions and Manuscript Status (~85%).
- Retention of Slide 20 with Springer LNCS acceptance and September meeting agenda.
- Protection of native branding, section highlight trackers, and navigation shapes.
- Preservation of 16:9 widescreen dimensions (13.333 x 7.500 inches) and 23 total slides.
"""

import os
import unittest
from pptx import Presentation
from pptx.util import Inches


class TestPresentationMethodology(unittest.TestCase):
    def test_slide_methodology_and_progress_updates(self):
        ppt_path = os.path.join(os.path.expanduser("~"), "Desktop", "AnekeshD_Progress.pptx")
        self.assertTrue(os.path.exists(ppt_path), f"{ppt_path} must exist")
        prs = Presentation(ppt_path)
        self.assertEqual(len(prs.slides), 23)

        # Widescreen 16:9 check
        self.assertAlmostEqual(prs.slide_width.inches, 13.333, places=2)
        self.assertAlmostEqual(prs.slide_height.inches, 7.500, places=2)

        # ---------------------------------------------------------------------
        # Slide 7 (Index 6): Master Methodological Framework
        # ---------------------------------------------------------------------
        s7 = prs.slides[6]
        pictures_s7 = [s for s in s7.shapes if s.shape_type == 13]
        self.assertGreater(len(pictures_s7), 0, "Slide 7 must contain embedded framework picture")

        pic_s7 = pictures_s7[0]
        self.assertAlmostEqual(pic_s7.left.inches, 0.60, delta=0.05)
        self.assertAlmostEqual(pic_s7.top.inches, 1.85, delta=0.05)
        self.assertAlmostEqual(pic_s7.width.inches, 12.133, delta=0.05)
        self.assertAlmostEqual(pic_s7.height.inches, 4.85, delta=0.05)

        s7_text = " ".join(s.text_frame.text for s in s7.shapes if s.has_text_frame)
        self.assertIn("Comprehensive Methodological Framework", s7_text)
        self.assertIn("Figure 3", s7_text)

        # Slide 7: Check protected shapes
        s7_names = [s.name for s in s7.shapes]
        self.assertIn("灯片编号占位符 10", s7_names)
        self.assertIn("直接连接符 6", s7_names)
        self.assertIn("矩形 4", s7_names)
        self.assertIn("矩形 29", s7_names)

        # ---------------------------------------------------------------------
        # Slide 19 (Index 18): Conclusion — Contributions & Writing Progress
        # ---------------------------------------------------------------------
        s19 = prs.slides[18]
        s19_text = " ".join(s.text_frame.text for s in s19.shapes if s.has_text_frame)
        self.assertIn("Summary of Thesis Contributions & Writing Progress", s19_text)
        self.assertIn("Four Primary Academic Contributions", s19_text)
        self.assertIn("Master's Thesis Manuscript Status (~85% Complete)", s19_text)
        for ch_num in range(1, 7):
            self.assertIn(f"Chapter {ch_num}", s19_text)

        # Slide 19 should NOT contain Figure 15 picture anymore
        pictures_s19 = [s for s in s19.shapes if s.shape_type == 13]
        self.assertEqual(len(pictures_s19), 0, "Slide 19 must not have picture shapes")

        # Slide 19: Check protected shapes
        s19_names = [s.name for s in s19.shapes]
        self.assertIn("灯片编号占位符 10", s19_names)
        self.assertIn("直接连接符 6", s19_names)
        self.assertIn("矩形 4", s19_names)
        self.assertIn("矩形 29", s19_names)

        # ---------------------------------------------------------------------
        # Slide 20 (Index 19): Meeting Agenda & LNCS Acceptance
        # ---------------------------------------------------------------------
        s20 = prs.slides[19]
        s20_text = " ".join(s.text_frame.text for s in s20.shapes if s.has_text_frame)
        self.assertIn("LNCS Acceptance & September Meeting Agenda", s20_text)
        self.assertIn("Springer LNCS", s20_text)
        self.assertIn("Gradio", s20_text)


if __name__ == "__main__":
    unittest.main()
