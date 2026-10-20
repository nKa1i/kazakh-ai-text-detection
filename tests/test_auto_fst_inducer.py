"""Unit tests for AutoFSTInducer and phonetic profiling."""

import unittest
from src.auto_fst.inducer import AutoFSTInducer, InductionResult

class TestAutoFSTInducer(unittest.TestCase):
    def setUp(self):
        self.inducer = AutoFSTInducer()

    def test_slang_verbalizer_induction(self):
        res = self.inducer.induce_stem("вайбтаған")
        self.assertIsNotNone(res)
        self.assertEqual(res.stem, "вайб")
        self.assertIn("та", res.suffixes)
        self.assertIn("ған", res.suffixes)
        self.assertEqual(res.harmony, "hard")
        self.assertEqual(res.coda, "voiced")

    def test_soft_slang_induction(self):
        res = self.inducer.induce_stem("чиллдедік")
        self.assertIsNotNone(res)
        self.assertEqual(res.stem, "чилл")
        self.assertEqual(res.harmony, "soft")
        self.assertEqual(res.coda, "voiced")

    def test_voiceless_coda_induction(self):
        res = self.inducer.induce_stem("донаттау")
        self.assertIsNotNone(res)
        self.assertEqual(res.stem, "донат")
        self.assertEqual(res.coda, "voiceless")

    def test_noun_possessive_case_induction(self):
        res = self.inducer.induce_stem("крашыма")
        self.assertIsNotNone(res)
        self.assertEqual(res.stem, "краш")
        self.assertIn("ым", res.suffixes)
        self.assertIn("а", res.suffixes)

    def test_russian_layout_suffix_slang_induction(self):
        # 'вайбтаган' typed on Russian keyboard with 'г' instead of 'ғ'
        res = self.inducer.induce_stem("вайбтаган")
        self.assertIsNotNone(res)
        self.assertEqual(res.stem, "вайб")
        self.assertIn("та", res.suffixes)
        self.assertIn("ған", res.suffixes)

    def test_gibberish_and_short_word_rejection(self):
        self.assertIsNone(self.inducer.induce_stem("12345"))
        self.assertIsNone(self.inducer.induce_stem("ббб"))
        self.assertIsNone(self.inducer.induce_stem("asdfgh"))
        self.assertIsNone(self.inducer.induce_stem(""))

if __name__ == '__main__':
    unittest.main()
