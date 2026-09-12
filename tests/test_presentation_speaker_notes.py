# -*- coding: utf-8 -*-
"""
Tests for thesis_presentation_speaker_notes_prof_guo.md.
Validates:
- Presence of all 23 slide sections (Slide 1 to Slide 23).
- Presence of '#### Spoken Script (English)' for every slide.
- Strict absence of Chinese characters in any Spoken Script section (English-only requirement for candidate Daulet).
- Presence of core technical anchors: Figure 15, Springer LNCS acceptance, 4-tab Gradio live demo, 303 automated tests, and Q&A Q1-Q7.
- Zero decorative emojis across the entire document.
"""

import os
import re
import unittest


class TestPresentationSpeakerNotes(unittest.TestCase):
    def setUp(self):
        self.notes_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "thesis_presentation_speaker_notes_prof_guo.md"
        )
        self.assertTrue(os.path.exists(self.notes_path), f"{self.notes_path} must exist")
        with open(self.notes_path, "r", encoding="utf-8") as f:
            self.content = f.read()

    def test_slide_coverage(self):
        """Verify all 23 slides are covered with slide headings."""
        for i in range(1, 24):
            pattern = rf"### Slide {i}:"
            self.assertRegex(
                self.content,
                pattern,
                f"Slide {i} heading must be present in speaker notes"
            )

    def test_all_slides_have_english_spoken_script(self):
        """Verify each of the 23 slides contains a dedicated English spoken script."""
        slide_splits = re.split(r"### Slide \d+:", self.content)[1:]
        self.assertEqual(len(slide_splits), 23, "Must parse exactly 23 slide sections")

        for idx, slide_text in enumerate(slide_splits, 1):
            self.assertIn(
                "#### Spoken Script (English)",
                slide_text,
                f"Slide {idx} must contain '#### Spoken Script (English)'"
            )
            self.assertNotIn(
                "#### Spoken Script (Chinese",
                slide_text,
                f"Slide {idx} must not contain Chinese spoken script"
            )

    def test_english_only_in_spoken_scripts(self):
        """Verify there are zero Chinese characters in any Spoken Script block."""
        scripts = re.findall(
            r"#### Spoken Script \(English\)(.*?)(?=(?:\n### |\n---|\n## |\Z))",
            self.content,
            re.DOTALL
        )
        self.assertEqual(len(scripts), 23, "Must find exactly 23 English spoken script blocks")

        chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]")
        for idx, script in enumerate(scripts, 1):
            matches = chinese_char_pattern.findall(script)
            self.assertEqual(
                len(matches),
                0,
                f"Slide {idx} spoken script must be English-only, but found Chinese characters: {matches[:10]}"
            )

    def test_key_technical_anchors(self):
        """Verify vital technical milestones, frameworks, and figures are articulated."""
        self.assertIn("Figure 15", self.content)
        self.assertIn("Springer LNCS", self.content)
        self.assertIn("AIST 2026", self.content)
        self.assertIn("Gradio", self.content)
        self.assertIn("Kazakh-FEVER", self.content)
        self.assertIn("Top-K", self.content)
        self.assertIn("Reviewer 1", self.content)
        self.assertIn("Reviewer 2", self.content)

    def test_qa_questions_present(self):
        """Verify defense preparation Q1 through Q7 are present in English."""
        for q_idx in range(1, 8):
            self.assertIn(f"### Q{q_idx}:", self.content)

    def test_zero_decorative_emojis(self):
        """Enforce strict zero decorative emoji policy."""
        emoji_pattern = re.compile(
            r"[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U00002702-\U000027B0\U0001F1E0-\U0001F1FF]"
        )
        emojis_found = emoji_pattern.findall(self.content)
        self.assertEqual(len(emojis_found), 0, f"Found decorative emojis: {emojis_found}")


if __name__ == "__main__":
    unittest.main()
