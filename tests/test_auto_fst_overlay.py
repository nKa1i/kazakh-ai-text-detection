"""Unit tests for DynamicFSTOverlay, longest-match sorting, and JSON persistence."""

import os
import unittest
import tempfile
from src.auto_fst.overlay import DynamicFSTOverlay

class TestDynamicFSTOverlay(unittest.TestCase):
    def setUp(self):
        self.overlay = DynamicFSTOverlay()

    def test_longest_match_first_sorting(self):
        # Prevent "чат" from shadowing "чатбот"
        self.overlay.register_root("чат")
        self.overlay.register_root("чатбот")
        segmented = self.overlay.analyze_and_segment("чатботтың жұмысы")
        self.assertIn("чатбот -тың", segmented)
        self.assertNotIn("чат -боттың", segmented)

    def test_dynamic_slang_registration(self):
        raw = "Біз кеше қатты вайбтадық"
        before = self.overlay.analyze_and_segment(raw)
        self.assertNotIn("вайб -та -дық", before)

        self.overlay.register_root("вайб")
        after = self.overlay.analyze_and_segment(raw)
        self.assertIn("вайб -та -дық", after)

    def test_russian_layout_particle_handling(self):
        raw = "Ол әлі жұмыста турсынгой"
        segmented = self.overlay.analyze_and_segment(raw)
        self.assertTrue("тур -сын -гой" in segmented or "тұр -сың -ғой" in segmented)

    def test_json_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name
        try:
            self.overlay.register_root("донат", {"harmony": "hard"})
            self.overlay.save_to_json(temp_path)

            new_overlay = DynamicFSTOverlay()
            count = new_overlay.load_from_json(temp_path)
            self.assertEqual(count, 1)
            self.assertIn("донат", new_overlay.loanword_roots)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()
