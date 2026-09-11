# -*- coding: utf-8 -*-
"""
tests/test_trust_scorer.py: Unit tests for DualRiskTrustScorer and Four-Quadrant Matrix.
"""

import unittest
from verification.evidence import AtomicClaim, ClaimVerificationResult
from verification.trust_scorer import DualRiskTrustScorer


class TestDualRiskTrustScorer(unittest.TestCase):
    """Verifies factual penalty aggregation, mathematical risk formula, and four-quadrant matrix."""

    def setUp(self):
        self.scorer = DualRiskTrustScorer(alpha=0.5)

    def test_quadrant_1_verified_human_fact(self):
        """Low AI risk (0.05) + All claims SUPPORTED (Fact risk 0.0) -> Verified Human Fact."""
        c = AtomicClaim(claim_id="c1", text="Абай 1845 жылы туған.")
        claims = [
            ClaimVerificationResult(claim=c, verdict="SUPPORTED", confidence=0.95)
        ]
        res = self.scorer.score(doc_text="Абай 1845 жылы туған.", ai_risk=0.05, claims=claims)
        self.assertEqual(res.quadrant_verdict, "Verified Human Fact")
        self.assertAlmostEqual(res.factual_risk, 0.0)
        self.assertAlmostEqual(res.trust_risk, 0.025)
        self.assertEqual(res.supported_count, 1)

    def test_quadrant_2_human_misinformation(self):
        """Low AI risk (0.10) + REFUTED claim (Fact risk 1.0) -> Human Misinformation."""
        c = AtomicClaim(claim_id="c2", text="Қазақстан 1999 жылы тәуелсіздік алды.")
        claims = [
            ClaimVerificationResult(claim=c, verdict="REFUTED", confidence=0.90)
        ]
        res = self.scorer.score(doc_text="Қазақстан 1999 жылы тәуелсіздік алды.", ai_risk=0.10, claims=claims)
        self.assertEqual(res.quadrant_verdict, "Human Misinformation")
        self.assertAlmostEqual(res.factual_risk, 1.0)
        self.assertAlmostEqual(res.trust_risk, 0.55)
        self.assertEqual(res.refuted_count, 1)

    def test_quadrant_3_accurate_ai_synthesis(self):
        """High AI risk (0.95) + All claims SUPPORTED (Fact risk 0.0) -> Accurate AI Synthesis."""
        c = AtomicClaim(claim_id="c3", text="Астана — Қазақстанның елордасы.")
        claims = [
            ClaimVerificationResult(claim=c, verdict="SUPPORTED", confidence=0.90)
        ]
        res = self.scorer.score(doc_text="Астана — Қазақстанның елордасы.", ai_risk=0.95, claims=claims)
        self.assertEqual(res.quadrant_verdict, "Accurate AI Synthesis")
        self.assertAlmostEqual(res.factual_risk, 0.0)
        self.assertAlmostEqual(res.trust_risk, 0.475)

    def test_quadrant_4_hallucinatory_ai_disinformation(self):
        """High AI risk (0.98) + REFUTED claim (Fact risk 1.0) -> Hallucinatory AI Disinformation."""
        c = AtomicClaim(claim_id="c4", text="Алматы — 2024 жылдан бері елорда.")
        claims = [
            ClaimVerificationResult(claim=c, verdict="REFUTED", confidence=0.92)
        ]
        res = self.scorer.score(doc_text="Алматы — 2024 жылдан бері елорда.", ai_risk=0.98, claims=claims)
        self.assertEqual(res.quadrant_verdict, "Hallucinatory AI Disinformation")
        self.assertAlmostEqual(res.factual_risk, 1.0)
        self.assertAlmostEqual(res.trust_risk, 0.99)

    def test_not_enough_info_penalty(self):
        """NEI gives 0.25 penalty."""
        c = AtomicClaim(claim_id="c5", text="Белгісіз факт.")
        claims = [
            ClaimVerificationResult(claim=c, verdict="NOT ENOUGH INFO", confidence=0.50)
        ]
        res = self.scorer.score(doc_text="Белгісіз факт.", ai_risk=0.20, claims=claims)
        self.assertAlmostEqual(res.factual_risk, 0.25)
        self.assertAlmostEqual(res.trust_risk, 0.5 * 0.20 + 0.5 * 0.25)
        self.assertEqual(res.nei_count, 1)


if __name__ == "__main__":
    unittest.main()
