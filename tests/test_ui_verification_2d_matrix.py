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


class TestTrustworthyDocumentVerifier2D(unittest.TestCase):
    """Test unit for TrustworthyDocumentVerifier dual verifier modes and 2D trust scoring."""

    def test_verifier_computes_2d_trust_coordinates(self):
        """Verify verifier returns continuous 2D coordinates and valid quadrant assignment."""
        from verification.verifier import TrustworthyDocumentVerifier

        verifier = TrustworthyDocumentVerifier(verifier_mode="morpho")
        text = "Астана — Қазақстанның елордасы."
        res = verifier.verify(text)

        self.assertIn(res.quadrant_code, ["Q1", "Q2", "Q3", "Q4"])
        self.assertGreaterEqual(res.t_fact, -1.0)
        self.assertLessEqual(res.t_fact, 1.0)
        self.assertGreaterEqual(res.t_gen, 0.0)
        self.assertLessEqual(res.t_gen, 1.0)
        self.assertGreaterEqual(res.composite_trust, 0.0)
        self.assertLessEqual(res.composite_trust, 1.0)
        self.assertGreater(len(res.claims), 0)
        self.assertIn("fst_root_jaccard", res.claims[0].morphological_features)
        self.assertIn("fst_jaccard", res.claims[0].morphological_features)

    def test_verifier_supports_baseline_mode(self):
        """Verify verifier operates correctly in baseline mode via init and runtime override."""
        from verification.verifier import TrustworthyDocumentVerifier

        # Mode set at initialization
        verifier_base = TrustworthyDocumentVerifier(verifier_mode="baseline")
        text = "Астана — Қазақстанның елордасы."
        res_base = verifier_base.verify(text)
        self.assertIsNotNone(res_base)
        self.assertIn(res_base.quadrant_code, ["Q1", "Q2", "Q3", "Q4"])
        self.assertGreaterEqual(res_base.t_fact, -1.0)
        self.assertLessEqual(res_base.t_fact, 1.0)
        self.assertGreaterEqual(res_base.t_gen, 0.0)
        self.assertLessEqual(res_base.t_gen, 1.0)

        # Runtime mode override
        verifier_morpho = TrustworthyDocumentVerifier(verifier_mode="morpho")
        res_override = verifier_morpho.verify(text, verifier_mode="baseline")
        self.assertIsNotNone(res_override)
        self.assertIn(res_override.quadrant_code, ["Q1", "Q2", "Q3", "Q4"])

    def test_morphological_features_attached_to_claims(self):
        """Verify 16-feature morphological diagnostic dictionary is attached to verified claims."""
        from verification.verifier import TrustworthyDocumentVerifier

        verifier = TrustworthyDocumentVerifier(verifier_mode="morpho")
        text = "Қазақстан 1999 жылы тәуелсіздік алды."
        res = verifier.verify(text)

        self.assertGreater(len(res.claims), 0)
        features = res.claims[0].morphological_features
        self.assertIsInstance(features, dict)

        required_keys = [
            "claim_negation",
            "evidence_negation",
            "directional_negation_mismatch",
            "calendar_year_conflict",
            "number_mismatch",
            "has_claim_evidential",
            "has_evidence_evidential",
            "has_modal_necessity",
            "has_modal_possibility",
            "fst_root_jaccard",
            "surface_token_overlap",
            "lexical_fst_divergence",
            "subject_proper_noun_match",
            "case_suffix_alignment",
            "claim_length_norm",
            "evidence_length_norm",
        ]
        for key in required_keys:
            self.assertIn(key, features, f"Missing required morphological feature key: {key}")

        self.assertEqual(features["calendar_year_conflict"], 1.0)
        self.assertIsInstance(features["has_claim_evidential"], bool)
        self.assertIsInstance(features["fst_root_jaccard"], float)

    def test_empty_document_2d_coordinates(self):
        """Verify empty text verification returns clean zeroed coordinates and Q1 verdict."""
        from verification.verifier import TrustworthyDocumentVerifier

        verifier = TrustworthyDocumentVerifier(verifier_mode="morpho")
        res = verifier.verify("")

        self.assertEqual(res.total_claims, 0)
        self.assertEqual(res.t_fact, 0.0)
        self.assertEqual(res.t_gen, 1.0)
        self.assertEqual(res.quadrant_code, "Q1")
        self.assertEqual(res.quadrant_verdict, "Verified Human Fact")
        self.assertEqual(res.trust_risk, 0.0)


if __name__ == "__main__":
    unittest.main()

