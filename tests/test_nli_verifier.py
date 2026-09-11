# -*- coding: utf-8 -*-
"""
tests/test_nli_verifier.py: Unit tests for NLI Claim Verifier.
"""

import unittest
from verification.evidence import AtomicClaim, EvidencePassage
from verification.nli_verifier import NLIClaimVerifier


class TestNLIClaimVerifier(unittest.TestCase):
    """Verifies SUPPORTED, REFUTED (temporal conflict, negation conflict), and NOT ENOUGH INFO."""

    def setUp(self):
        self.verifier = NLIClaimVerifier()

    def test_supported_historical_fact(self):
        """Claim: 'Қазақстан 1991 жылы тәуелсіздік алды.' vs Evidence of 1991 independence -> SUPPORTED."""
        claim = AtomicClaim(claim_id="c1", text="Қазақстан 1991 жылы тәуелсіздік алды.")
        evidence = [
            EvidencePassage(
                passage_id="wiki_01",
                title="Қазақстан тәуелсіздігі",
                text="Қазақстан Республикасы 1991 жылы 16 желтоқсанда өз тәуелсіздігін жариялады.",
                similarity_score=0.92,
                matched_stems=["қазақстан", "1991", "жыл", "тәуелсіздік"]
            )
        ]
        result = self.verifier.verify_claim(claim, evidence)
        self.assertEqual(result.verdict, "SUPPORTED")
        self.assertGreaterEqual(result.confidence, 0.70)
        self.assertIn("1991", result.explanation)

    def test_refuted_temporal_date_conflict(self):
        """Claim: 'Қазақстан 1998 жылы тәуелсіздік алды.' vs Evidence of 1991 -> REFUTED."""
        claim = AtomicClaim(claim_id="c2", text="Қазақстан 1998 жылы тәуелсіздік алды.")
        evidence = [
            EvidencePassage(
                passage_id="wiki_01",
                title="Қазақстан тәуелсіздігі",
                text="Қазақстан Республикасы 1991 жылы 16 желтоқсанда өз тәуелсіздігін жариялады.",
                similarity_score=0.85,
                matched_stems=["қазақстан", "жыл", "тәуелсіздік"]
            )
        ]
        result = self.verifier.verify_claim(claim, evidence)
        self.assertEqual(result.verdict, "REFUTED")
        self.assertIn("1998", result.explanation)
        self.assertIn("1991", result.explanation)

    def test_refuted_polar_negation(self):
        """Claim: 'Астана Қазақстанның астанасы емес.' vs Evidence 'Астана — Қазақстанның елордасы' -> REFUTED."""
        claim = AtomicClaim(claim_id="c3", text="Астана қаласы Қазақстанның астанасы емес.")
        evidence = [
            EvidencePassage(
                passage_id="wiki_02",
                title="Астана қаласы",
                text="Астана қаласы — Қазақстанның елордасы және бас қаласы болып табылады.",
                similarity_score=0.88,
                matched_stems=["астана", "қала", "қазақстан"]
            )
        ]
        result = self.verifier.verify_claim(claim, evidence)
        self.assertEqual(result.verdict, "REFUTED")
        self.assertIn("терістеу", result.explanation.lower())

    def test_not_enough_info_unrelated_passage(self):
        """Claim about Mars vs Kazakhstan history evidence -> NOT ENOUGH INFO."""
        claim = AtomicClaim(claim_id="c4", text="Марс ғаламшарында су қоры табылды.")
        evidence = [
            EvidencePassage(
                passage_id="wiki_01",
                title="Қазақстан тәуелсіздігі",
                text="Қазақстан 1991 жылы тәуелсіздік алды.",
                similarity_score=0.05,
                matched_stems=[]
            )
        ]
        result = self.verifier.verify_claim(claim, evidence)
        self.assertEqual(result.verdict, "NOT ENOUGH INFO")

    def test_empty_evidence_gives_nei(self):
        claim = AtomicClaim(claim_id="c5", text="Кез келген пікір.")
        result = self.verifier.verify_claim(claim, [])
        self.assertEqual(result.verdict, "NOT ENOUGH INFO")


if __name__ == "__main__":
    unittest.main()
