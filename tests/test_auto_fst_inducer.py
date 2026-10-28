"""Unit tests for AutoFSTInducer, recursive slot parsing, and corpus-level paradigm induction."""

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

    def test_multi_stack_suffix_chain_induction(self):
        # 3-stack suffix: донат + тар + ымыз + дан
        res = self.inducer.induce_stem("донаттарымыздан")
        self.assertIsNotNone(res)
        self.assertEqual(res.stem, "донат")
        self.assertEqual(res.suffixes, ["тар", "ымыз", "дан"])

    def test_mixed_script_hyphenated_token_induction(self):
        res_ai = self.inducer.induce_stem("AI-дың")
        self.assertIsNotNone(res_ai)
        self.assertEqual(res_ai.stem, "AI")
        self.assertEqual(res_ai.suffixes, ["дың"])

        res_gpt = self.inducer.induce_stem("ChatGPT-мен")
        self.assertIsNotNone(res_gpt)
        self.assertEqual(res_gpt.stem, "ChatGPT")
        self.assertEqual(res_gpt.suffixes, ["мен"])

    def test_corpus_level_paradigm_induction(self):
        texts = [
            "Бүгін кешке достармен қатты вайбтадық.",
            "Мына ойынның вайбы өте ерекше болды."
        ]
        inductions = self.inducer.induce_from_corpus(texts, min_paradigm_size=2)
        stems = [ind.stem for ind in inductions]
        self.assertIn("вайб", stems)
        vib_ind = next(ind for ind in inductions if ind.stem == "вайб")
        self.assertGreaterEqual(vib_ind.confidence, 0.99)


if __name__ == '__main__':
    unittest.main()
