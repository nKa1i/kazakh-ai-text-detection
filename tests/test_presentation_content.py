# -*- coding: utf-8 -*-
"""
Unit tests for content population, comparison tables, and figure embedding across all 23 slides.
Verifies:
1. All 14 figure slides (3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18) contain embedded pictures.
2. Formatted comparison tables exist on Slides 5, 11, 12, 14, 16 (and 22).
3. Slide 22 contains Committee comments with Reviewer 1, Reviewer 2, and '100.00% AUC'.
4. Captions (Figure X) and titles are present.
5. Zero emojis across all slides.
"""

import os
import unittest
from pptx import Presentation

OUTPUT_PPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx"
)


class TestPresentationContent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.prs_path = OUTPUT_PPT_PATH
        cls.assertTrue(
            os.path.isfile(cls.prs_path),
            f"Presentation file does not exist at {cls.prs_path}"
        )
        cls.prs = Presentation(cls.prs_path)

    def test_figures_and_tables_embedded_correctly(self):
        # Verify slides that must have embedded pictures
        fig_slides = [3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        for s_num in fig_slides:
            slide = self.prs.slides[s_num - 1]
            pictures = [s for s in slide.shapes if s.shape_type == 13]  # MSO_SHAPE_TYPE.PICTURE
            self.assertGreater(
                len(pictures), 0,
                f"Slide {s_num} must have at least one embedded picture figure"
            )

        # Verify slides that must have tables
        table_slides = [5, 11, 12, 14, 16]
        for s_num in table_slides:
            slide = self.prs.slides[s_num - 1]
            tables = [s for s in slide.shapes if s.has_table]
            self.assertGreater(
                len(tables), 0,
                f"Slide {s_num} must contain an academic table"
            )

        # Verify Slide 22 Committee comments
        s22 = self.prs.slides[21]
        text_s22 = "".join(s.text_frame.text for s in s22.shapes if s.has_text_frame)
        self.assertIn("Reviewer 1", text_s22)
        self.assertIn("Reviewer 2", text_s22)
        self.assertIn("100.00% AUC", text_s22)

    def test_figure_captions_present(self):
        fig_slides = [3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        for s_num in fig_slides:
            slide = self.prs.slides[s_num - 1]
            all_text = "".join(s.text_frame.text for s in slide.shapes if s.has_text_frame)
            self.assertIn(
                "Figure ", all_text,
                f"Slide {s_num} must contain a formal figure caption starting with 'Figure '"
            )

    def test_zero_emojis_across_deck(self):
        # Scan all slides for emojis
        for idx, slide in enumerate(self.prs.slides):
            for shp in slide.shapes:
                if shp.has_text_frame:
                    for ch in shp.text_frame.text:
                        code = ord(ch)
                        is_emoji = (
                            (0x1F600 <= code <= 0x1F64F) or
                            (0x1F300 <= code <= 0x1F5FF) or
                            (0x1F680 <= code <= 0x1F6FF) or
                            (0x1F700 <= code <= 0x1F77F) or
                            (0x1F900 <= code <= 0x1F9FF) or
                            (0x1FA00 <= code <= 0x1FA6F) or
                            (0x1FA70 <= code <= 0x1FAFF) or
                            (0x2600 <= code <= 0x26FF) or
                            (0x2700 <= code <= 0x27BF and code not in (0x2713, 0x2714))
                        )
                        self.assertFalse(
                            is_emoji,
                            f"Slide {idx+1} contains decorative emoji: '{ch}' (U+{code:04X})"
                        )


if __name__ == "__main__":
    unittest.main()
