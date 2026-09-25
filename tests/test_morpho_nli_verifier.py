# -*- coding: utf-8 -*-
"""
Unit tests for Morphological NLI Cross-Encoder Architecture and Feature Extractor.
Tests models.morpho_nli_verifier components:
- MorphologicalAffixExtractor (16-dimensional feature vector)
- OfflineHeuristicNLIVerifier (calibrated rule-weighted verifier)
- MorphoNLIVerifier (cross-encoder architecture with defensive fallback)
"""

import unittest
from typing import List, Dict, Any

from models.morpho_nli_verifier import (
    MorphologicalAffixExtractor,
    MorphoNLIVerifier,
    OfflineHeuristicNLIVerifier,
)


class TestMorphologicalAffixExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = MorphologicalAffixExtractor()

    def test_feature_vector_dimension_and_types(self):
        claim = "Қазақстан 1991 жылы өз тәуелсіздігін жариялады."
        evidence = "1991 жылғы 16 желтоқсанда Қазақстан өз тәуелсіздігі туралы заң қабылдады."
        features = self.extractor.extract_features(claim, evidence)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 16)
        for idx, val in enumerate(features):
            self.assertIsInstance(val, float, f"Feature index {idx} must be float, got {type(val)}")

    def test_extract_affix_features_negation_mismatch(self):
        claim = "Қазақстан 1991 жылы тәуелсіздік алған жоқ."
        evidence = "Қазақстан 1991 жылы 16 желтоқсанда өз тәуелсіздігін ресми түрде жариялады."
        features = self.extractor.extract_features(claim, evidence)
        self.assertEqual(len(features), 16)
        # Index 0: Claim negation
        self.assertEqual(features[0], 1.0)
        # Index 1: Evidence negation
        self.assertEqual(features[1], 0.0)
        # Index 2: Directional negation mismatch (XOR)
        self.assertEqual(features[2], 1.0)

    def test_negation_agreement_both_affirmative_or_both_negated(self):
        # Both affirmative
        claim_aff = "Астана — елорда."
        ev_aff = "Астана қаласы — бас қала."
        feat_aff = self.extractor.extract_features(claim_aff, ev_aff)
        self.assertEqual(feat_aff[0], 0.0)
        self.assertEqual(feat_aff[1], 0.0)
        self.assertEqual(feat_aff[2], 0.0)

        # Both negated
        claim_neg = "Бұл мәлімет шындыққа жанаспайды."
        ev_neg = "Келтірілген ақпарат шын емес."
        feat_neg = self.extractor.extract_features(claim_neg, ev_neg)
        self.assertEqual(feat_neg[0], 1.0)
        self.assertEqual(feat_neg[1], 1.0)
        self.assertEqual(feat_neg[2], 0.0)

    def test_extract_affix_features_temporal_conflict(self):
        claim = "Қазақстан 1998 жылы тәуелсіздік алды."
        evidence = "Қазақстан 1991 жылы өз тәуелсіздігін жариялады."
        features = self.extractor.extract_features(claim, evidence)
        # Index 3: Temporal calendar year conflict
        self.assertEqual(features[3], 1.0)

        # Same year -> no conflict
        claim_same = "Қазақстан 1991 жылы тәуелсіздік алды."
        feat_same = self.extractor.extract_features(claim_same, evidence)
        self.assertEqual(feat_same[3], 0.0)

    def test_numerical_quantity_mismatch(self):
        claim = "Жобаға 500 адам қатысты."
        evidence = "Жобаға 200 адам қатысып, жұмыс жасады."
        features = self.extractor.extract_features(claim, evidence)
        # Index 4: Numerical quantity mismatch (500 is in claim but not evidence)
        self.assertEqual(features[4], 1.0)

        # Matching numbers
        claim_match = "Жобаға 200 адам қатысты."
        feat_match = self.extractor.extract_features(claim_match, evidence)
        self.assertEqual(feat_match[4], 0.0)

    def test_evidential_markers(self):
        claim = "Ол жиналысқа барыпты."
        evidence = "Ол жиналысқа келді."
        features = self.extractor.extract_features(claim, evidence)
        # Index 5: Evidential marker in claim
        self.assertEqual(features[5], 1.0)
        # Index 6: Evidential marker in evidence
        self.assertEqual(features[6], 0.0)

        evidence_evid = "Ол жиналысқа келмеген көрінеді, басқа жаққа кетіпті."
        feat_evid = self.extractor.extract_features(claim, evidence_evid)
        self.assertEqual(feat_evid[6], 1.0)

    def test_modal_markers(self):
        claim_nec = "Жұмысты дер кезінде орындау керек."
        ev_nec = "Тапсырманы аяқтау тиіс."
        feat_nec = self.extractor.extract_features(claim_nec, ev_nec)
        # Index 7: Modal necessity marker
        self.assertEqual(feat_nec[7], 1.0)

        claim_poss = "Бұл нәтиже мүмкін емес, бірақ ықтимал шешім бар."
        feat_poss = self.extractor.extract_features(claim_poss, ev_nec)
        # Index 8: Modal possibility marker
        self.assertEqual(feat_poss[8], 1.0)

    def test_fst_and_surface_overlap_and_divergence(self):
        claim = "Астана — Қазақстанның елордасы."
        evidence = "Астана қаласы — Қазақстанның елордасы болып табылады."
        features = self.extractor.extract_features(claim, evidence)
        # Index 9: FST root overlap > 0.0
        self.assertGreater(features[9], 0.0)
        # Index 10: Surface overlap > 0.0
        self.assertGreater(features[10], 0.0)
        # Index 11: Lexical-FST divergence equals abs(surface - fst)
        self.assertAlmostEqual(features[11], abs(features[10] - features[9]), places=4)

    def test_subject_proper_noun_match(self):
        claim = "Астана қаласы 1997 жылы астана болды."
        evidence = "Астана — Қазақстан мемлекетінің елордасы."
        features = self.extractor.extract_features(claim, evidence)
        # Index 12: Subject proper noun match
        self.assertEqual(features[12], 1.0)

        evidence_mismatch = "Алматы қаласы — үлкен мегаполис."
        feat_mismatch = self.extractor.extract_features(claim, evidence_mismatch)
        self.assertEqual(feat_mismatch[12], 0.0)

    def test_case_agreement_alignment(self):
        claim = "Қазақстанның астанасы Астана."
        evidence = "Қазақстанның тарихы терең."
        features = self.extractor.extract_features(claim, evidence)
        # Index 13: Case agreement alignment score (Қазақстанның has genitive case in both)
        self.assertGreater(features[13], 0.0)

    def test_normalized_lengths(self):
        claim = "Бір екі үш төрт бес"
        evidence = "Бір екі үш төрт бес алты жеті сегіз тоғыз он"
        features = self.extractor.extract_features(claim, evidence)
        # Index 14: Normalized claim length min(1.0, 5 / 30.0)
        self.assertAlmostEqual(features[14], 5.0 / 30.0, places=4)
        # Index 15: Normalized evidence length min(1.0, 10 / 100.0)
        self.assertAlmostEqual(features[15], 10.0 / 100.0, places=4)

    def test_edge_cases_empty_or_whitespace_strings(self):
        features = self.extractor.extract_features("", "")
        self.assertEqual(len(features), 16)
        for val in features:
            self.assertEqual(val, 0.0)

        feat_ws = self.extractor.extract_features("   ", "\t\n")
        self.assertEqual(len(feat_ws), 16)
        for val in feat_ws:
            self.assertEqual(val, 0.0)


class TestOfflineHeuristicNLIVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = OfflineHeuristicNLIVerifier()

    def test_prediction_supported(self):
        pred = self.verifier.predict_pair(
            claim="Астана — Қазақстанның елордасы.",
            evidence="Астана қаласы — Қазақстанның елордасы болып табылады."
        )
        self.assertIsInstance(pred, dict)
        self.assertEqual(pred["label"], "SUPPORTED")
        self.assertGreater(pred["confidence"], 0.5)
        self.assertIn("probabilities", pred)
        self.assertIn("features", pred)
        self.assertEqual(len(pred["features"]), 16)

    def test_prediction_refutes_temporal_conflict(self):
        pred_ref = self.verifier.predict_pair(
            claim="Қазақстан өз тәуелсіздігін 1998 жылы жариялаған.",
            evidence="Қазақстан 1991 жылы 16 желтоқсанда өз тәуелсіздігін ресми түрде жариялады."
        )
        self.assertEqual(pred_ref["label"], "REFUTES")
        self.assertGreater(pred_ref["confidence"], 0.5)
        self.assertGreater(pred_ref["probabilities"]["REFUTES"], pred_ref["probabilities"]["SUPPORTED"])

    def test_prediction_refutes_negation_mismatch(self):
        pred_neg = self.verifier.predict_pair(
            claim="Қазақстан 1991 жылы тәуелсіздік алған жоқ.",
            evidence="Қазақстан 1991 жылы 16 желтоқсанда өз тәуелсіздігін ресми түрде жариялады."
        )
        self.assertEqual(pred_neg["label"], "REFUTES")
        self.assertGreater(pred_neg["confidence"], 0.5)

    def test_prediction_not_enough_info_unrelated(self):
        pred_nei = self.verifier.predict_pair(
            claim="Марс планетасында өмір бар.",
            evidence="Астана қаласы — Қазақстанның елордасы болып табылады."
        )
        self.assertEqual(pred_nei["label"], "NOT_ENOUGH_INFO")

    def test_predict_batch(self):
        pairs = [
            ("Астана — Қазақстанның елордасы.", "Астана қаласы — Қазақстанның елордасы болып табылады."),
            ("Қазақстан өз тәуелсіздігін 1998 жылы жариялаған.", "Қазақстан 1991 жылы 16 желтоқсанда өз тәуелсіздігін ресми түрде жариялады."),
            ("Марс планетасында өмір бар.", "Астана қаласы — Қазақстанның елордасы болып табылады."),
        ]
        results = self.verifier.predict_batch(pairs)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["label"], "SUPPORTED")
        self.assertEqual(results[1]["label"], "REFUTES")
        self.assertEqual(results[2]["label"], "NOT_ENOUGH_INFO")

    def test_empty_input_edge_case(self):
        pred = self.verifier.predict_pair("", "")
        self.assertEqual(pred["label"], "NOT_ENOUGH_INFO")


class TestMorphoNLIVerifier(unittest.TestCase):
    def test_instantiation_and_prediction(self):
        model = MorphoNLIVerifier(hidden_dim=768, morpho_dim=16, num_classes=3)
        self.assertEqual(model.num_classes, 3)
        self.assertEqual(model.morpho_dim, 16)

        pred = model.predict_pair(
            claim="Астана — Қазақстанның елордасы.",
            evidence="Астана қаласы — Қазақстанның елордасы болып табылады."
        )
        self.assertIsInstance(pred, dict)
        self.assertIn("label", pred)
        self.assertIn(pred["label"], ["SUPPORTED", "REFUTES", "NOT_ENOUGH_INFO"])
        self.assertIn("probabilities", pred)
        self.assertIn("features", pred)


if __name__ == "__main__":
    unittest.main()
