"""Unit tests for RussianLayoutNormalizer and suffix equivalences."""

import unittest
from src.auto_fst.normalizer import RussianLayoutNormalizer, SUFFIX_EQUIVALENCES

class TestRussianLayoutNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = RussianLayoutNormalizer()

    def test_high_frequency_dictionary_normalization(self):
        self.assertEqual(self.normalizer.normalize_word("турсынгой"), "тұрсыңғой")
        self.assertEqual(self.normalizer.normalize_word("жаксы"), "жақсы")
        self.assertEqual(self.normalizer.normalize_word("салем"), "сәлем")
        self.assertEqual(self.normalizer.normalize_word("калайсын"), "қалайсың")

    def test_case_preservation(self):
        self.assertEqual(self.normalizer.normalize_word("Салем"), "Сәлем")
        self.assertEqual(self.normalizer.normalize_word("ЖАКСЫ"), "ЖАҚСЫ")

    def test_text_normalization(self):
        text = "салем калайсын, жумысын жаксы болып жатырма?"
        normalized = self.normalizer.normalize_text(text)
        self.assertIn("сәлем", normalized)
        self.assertIn("жақсы", normalized)
        self.assertIn("қалайсың", normalized)
        self.assertIn("жұмысың", normalized)

    def test_russian_layout_artifact_detection(self):
        self.assertTrue(self.normalizer.has_russian_layout_artifacts("турсынгой"))
        self.assertTrue(self.normalizer.has_russian_layout_artifacts("жаксы"))
        self.assertFalse(self.normalizer.has_russian_layout_artifacts("тұрсыңғой"))
        self.assertFalse(self.normalizer.has_russian_layout_artifacts("алма"))

    def test_suffix_equivalences_structure(self):
        self.assertIn("ған", SUFFIX_EQUIVALENCES)
        self.assertIn("ган", SUFFIX_EQUIVALENCES["ған"])
        self.assertIn("сың", SUFFIX_EQUIVALENCES)
        self.assertIn("сын", SUFFIX_EQUIVALENCES["сың"])

if __name__ == '__main__':
    unittest.main()
