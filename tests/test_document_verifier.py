# -*- coding: utf-8 -*-
"""
tests/test_document_verifier.py: Integration tests for TrustworthyDocumentVerifier facade.
"""

import unittest
from verification.verifier import TrustworthyDocumentVerifier


class TestTrustworthyDocumentVerifier(unittest.TestCase):
    """Verifies end-to-end orchestration of AI detection and factual verification."""

    def setUp(self):
        self.verifier = TrustworthyDocumentVerifier()

    def test_verify_authentic_historical_document(self):
        text = "Қазақстан Республикасы 1991 жылы 16 желтоқсанда өз тәуелсіздігін ресми түрде жариялады."
        res = self.verifier.verify(text)
        self.assertIn(res.quadrant_verdict, ["Verified Human Fact", "Accurate AI Synthesis"])
        self.assertGreater(res.total_claims, 0)
        self.assertGreaterEqual(res.supported_count, 1)
        self.assertEqual(res.refuted_count, 0)
        self.assertLessEqual(res.trust_risk, 0.50)

    def test_verify_false_statement_document(self):
        text = "Қазақстан 1999 жылы тәуелсіздік алды."
        res = self.verifier.verify(text)
        self.assertGreater(res.total_claims, 0)
        self.assertGreaterEqual(res.refuted_count, 1)
        self.assertIn(res.quadrant_verdict, ["Human Misinformation", "Hallucinatory AI Disinformation"])
        self.assertGreaterEqual(res.factual_risk, 0.50)

    def test_verify_empty_document(self):
        res = self.verifier.verify("")
        self.assertEqual(res.total_claims, 0)
        self.assertEqual(res.quadrant_verdict, "Verified Human Fact")
        self.assertEqual(res.trust_risk, 0.0)


if __name__ == "__main__":
    unittest.main()
