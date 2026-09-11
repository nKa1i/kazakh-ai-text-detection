# -*- coding: utf-8 -*-
"""
tests/test_claim_extractor.py: Unit tests for Kazakh Atomic Claim Extractor.
"""

import unittest
from verification.claim_extractor import KazakhClaimExtractor


class TestKazakhClaimExtractor(unittest.TestCase):
    """Verifies proposition extraction, hedge stripping, clause splitting, and nominal predicates."""

    def setUp(self):
        self.extractor = KazakhClaimExtractor()

    def test_single_atomic_factual_sentence(self):
        text = "Қазақстан 1991 жылы тәуелсіздік алды."
        claims = self.extractor.extract_claims(text)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].text, "Қазақстан 1991 жылы тәуелсіздік алды.")
        self.assertTrue(claims[0].is_verifiable)

    def test_strip_discourse_hedges(self):
        """'Меніңше, Абай 1845 жылы туған.' should strip 'Меніңше,' and capitalize 'Абай'."""
        text = "Меніңше, Абай 1845 жылы туған."
        claims = self.extractor.extract_claims(text)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].text, "Абай 1845 жылы туған.")
        self.assertTrue(claims[0].is_verifiable)

    def test_nominal_zero_copula_predicate(self):
        """Sentences with dash copula '—' must be recognized as verifiable propositions."""
        text = "Астана қаласы — Қазақстанның елордасы."
        claims = self.extractor.extract_claims(text)
        self.assertEqual(len(claims), 1)
        self.assertIn("Астана қаласы", claims[0].text)
        self.assertTrue(claims[0].is_verifiable)

    def test_compound_sentence_clause_splitting(self):
        """Compound sentence joined by 'және' should be decomposed into two distinct claims."""
        text = "Қазақ хандығы 1465 жылы құрылды және оның негізін Керей мен Жәнібек қалады."
        claims = self.extractor.extract_claims(text)
        self.assertEqual(len(claims), 2)
        self.assertIn("1465 жылы құрылды", claims[0].text)
        self.assertIn("Керей мен Жәнібек қалады", claims[1].text)

    def test_filter_pure_subjective_opinions(self):
        """Pure customer reviews and sentiments without factual entities should be non-verifiable or filtered."""
        text = "Маған бұл фильм өте қатты ұнады! Керемет туынды екен."
        claims = self.extractor.extract_claims(text)
        # Should not extract checkable encyclopedic claims
        verifiable = [c for c in claims if c.is_verifiable]
        self.assertEqual(len(verifiable), 0)

    def test_empty_and_short_inputs(self):
        self.assertEqual(self.extractor.extract_claims(""), [])
        self.assertEqual(self.extractor.extract_claims("   "), [])
        self.assertEqual(self.extractor.extract_claims("Иә."), [])


if __name__ == "__main__":
    unittest.main()
