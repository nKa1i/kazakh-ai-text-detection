# -*- coding: utf-8 -*-
"""
tests/test_ui_verification_2d_matrix.py: Tests for 2D Continuous Trust Matrix & Morphological Claim Verification.
"""

import unittest
from verification.evidence import (
    AtomicClaim,
    ClaimVerificationResult,
    DocumentTrustResult,
    EvidencePassage,
)


class TestVerificationEvidence2D(unittest.TestCase):
    """Test unit for 2D trust matrix coordinates and morphological features in evidence dataclasses."""

    def test_claim_verification_result_has_morphological_features(self):
        """Verify initialization, default empty dict, explicit dict passing, and to_dict() serialization."""
        claim = AtomicClaim(claim_id="c1", text="Астана — елорда.")

        # Test default value
        default_res = ClaimVerificationResult(
            claim=claim,
            verdict="SUPPORTED",
            confidence=0.95
        )
        self.assertEqual(default_res.morphological_features, {})
        self.assertIn("morphological_features", default_res.to_dict())
        self.assertEqual(default_res.to_dict()["morphological_features"], {})

        # Test explicit dictionary passing
        custom_features = {
            "negation_mismatch": 0.0,
            "fst_jaccard": 0.88,
            "case_agreement": 1.0
        }
        res = ClaimVerificationResult(
            claim=claim,
            verdict="SUPPORTED",
            confidence=0.95,
            morphological_features=custom_features
        )
        self.assertEqual(res.morphological_features["fst_jaccard"], 0.88)
        self.assertIn("morphological_features", res.to_dict())
        self.assertEqual(res.to_dict()["morphological_features"], custom_features)

    def test_document_trust_result_has_continuous_coordinates(self):
        """Verify t_fact, t_gen, composite_trust, and quadrant_code default values, explicit initialization, and serialization."""
        # Test defaults
        default_doc = DocumentTrustResult(
            doc_text="Test",
            ai_risk=0.1,
            factual_risk=0.2,
            trust_risk=0.15,
            quadrant_verdict="Verified Human Fact"
        )
        self.assertEqual(default_doc.t_fact, 0.0)
        self.assertEqual(default_doc.t_gen, 0.0)
        self.assertEqual(default_doc.composite_trust, 0.0)
        self.assertEqual(default_doc.quadrant_code, "Q1")
        default_dict = default_doc.to_dict()
        self.assertEqual(default_dict["t_fact"], 0.0)
        self.assertEqual(default_dict["t_gen"], 0.0)
        self.assertEqual(default_dict["composite_trust"], 0.0)
        self.assertEqual(default_dict["quadrant_code"], "Q1")

        # Test explicit initialization
        doc_res = DocumentTrustResult(
            doc_text="Test Document",
            ai_risk=0.1,
            factual_risk=0.0,
            trust_risk=0.05,
            quadrant_verdict="Verified Human Fact",
            t_fact=0.85,
            t_gen=0.90,
            composite_trust=0.8753,
            quadrant_code="Q1"
        )
        self.assertEqual(doc_res.t_fact, 0.85)
        self.assertEqual(doc_res.t_gen, 0.90)
        self.assertEqual(doc_res.quadrant_code, "Q1")
        self.assertAlmostEqual(doc_res.composite_trust, 0.8753, places=3)

        doc_dict = doc_res.to_dict()
        self.assertEqual(doc_dict["t_fact"], 0.85)
        self.assertEqual(doc_dict["t_gen"], 0.90)
        self.assertEqual(doc_dict["composite_trust"], 0.8753)
        self.assertEqual(doc_dict["quadrant_code"], "Q1")


if __name__ == "__main__":
    unittest.main()
