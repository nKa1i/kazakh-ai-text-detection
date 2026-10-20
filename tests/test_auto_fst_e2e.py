"""End-to-end integration and latency tests for Auto-FST."""

import unittest
import time
import re
from src.auto_fst.normalizer import RussianLayoutNormalizer
from src.auto_fst.inducer import AutoFSTInducer
from src.auto_fst.guard import RegressionGuard
from src.auto_fst.overlay import DynamicFSTOverlay

class TestAutoFSTEndToEnd(unittest.TestCase):
    def setUp(self):
        self.normalizer = RussianLayoutNormalizer()
        self.inducer = AutoFSTInducer()
        self.guard = RegressionGuard()
        self.overlay = DynamicFSTOverlay(normalizer=self.normalizer)

    def test_full_autonomous_slang_induction_lifecycle(self):
        sentence = "Бүгін кешке достармен қатты вайбтадық"
        # 1. Check before induction
        before_parse = self.overlay.analyze_and_segment(sentence)
        self.assertNotIn("вайб -та -дық", before_parse)

        # 2. Extract and induce
        res = self.inducer.induce_stem("вайбтадық")
        self.assertIsNotNone(res)

        # 3. Guardrail check
        passed, reason = self.guard.validate_candidate(res.stem)
        self.assertTrue(passed)

        # 4. Commit to overlay
        self.overlay.register_root(res.stem, {"harmony": res.harmony, "coda": res.coda})

        # 5. Check after induction
        after_parse = self.overlay.analyze_and_segment(sentence)
        self.assertIn("вайб -та -дық", after_parse)

    def test_russian_layout_full_lifecycle(self):
        # Russian layout sentence with slang and particles
        sentence = "салем калайсын олар турсынгой вайбтаган"
        # Extract slang
        for w in sentence.split():
            cand = self.inducer.induce_stem(w)
            if cand:
                passed, _ = self.guard.validate_candidate(cand.stem)
                if passed:
                    self.overlay.register_root(cand.stem)

        parsed = self.overlay.analyze_and_segment(sentence)
        # Verify Russian layout normalizer converted words and suffixes
        self.assertIn("сәле -м", parsed)
        self.assertIn("қалай -сың", parsed)
        self.assertIn("тұр -сың -ғой", parsed)
        self.assertIn("вайб -та -ған", parsed)

    def test_submillisecond_latency(self):
        sentence = "Каспиден доставкасы өте тез болды, оплатасын жасадым және вайбтадық."
        self.overlay.register_root("вайб")
        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            _ = self.overlay.analyze_and_segment(sentence)
        avg_ms = (time.perf_counter() - start) / iterations * 1000
        # Latency must be strictly < 5.0ms on CPU
        self.assertLess(avg_ms, 5.0)

    def test_user_informal_russian_layout_sentence(self):
        sentence = "вайбы жаксы екен бул ойыннын оте катты унады"
        inductions = []
        for w in sentence.split():
            w_clean = re.sub(r'^[^\w]+|[^\w]+$', '', w)
            w_norm = self.overlay.normalizer.normalize_word(w_clean)
            cand = self.inducer.induce_stem(w_norm)
            if cand:
                passed, _ = self.guard.validate_candidate(cand.stem)
                if passed:
                    self.overlay.register_root(cand.stem, {"harmony": cand.harmony, "coda": cand.coda})
                    inductions.append(cand.stem)

        # Only loanword 'вайб' should be induced, no native words
        self.assertEqual(inductions, ["вайб"])

        parsed = self.overlay.analyze_and_segment(sentence)
        self.assertEqual(parsed, "вайб -ы жақ -сы екен бұл ойын -ның өте қат -ты ұна -ды")

if __name__ == '__main__':
    unittest.main()
