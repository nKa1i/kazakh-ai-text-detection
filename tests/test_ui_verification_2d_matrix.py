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


class TestUIVerification2DMatrix(unittest.TestCase):
    """Test unit for 2D Cartesian Plane visualization, summary card, and morphological drawer rendering."""

    def test_render_2d_trust_matrix_plane_en(self):
        """Verify SVG elements, pastel quadrants, coordinate mapping, and XSS sanitization in English."""
        from ui.highlighting import render_2d_trust_matrix_plane

        # Standard coordinate rendering
        html_out = render_2d_trust_matrix_plane(t_fact=0.8, t_gen=0.9, quadrant="Q1", lang="en")
        self.assertIn("<svg", html_out)
        self.assertIn('viewBox="0 0 460 320"', html_out)
        self.assertIn("Verified Human Fact", html_out)
        self.assertIn("Human Misinformation", html_out)
        self.assertIn("Accurate AI Synthesis", html_out)
        self.assertIn("Hallucinatory AI Disinformation", html_out)

        # Coordinate point calculation: cx = 200 + 0.8 * 160 = 328.0; cy = 260 - 0.9 * 200 = 80.0
        self.assertIn('cx="328.0"', html_out)
        self.assertIn('cy="80.0"', html_out)
        self.assertIn('r="6.5"', html_out)
        self.assertIn('r="12"', html_out)
        self.assertIn('opacity="0.3"', html_out)

        # HUD badge elements
        self.assertIn("T_fact", html_out)
        self.assertIn("T_gen", html_out)
        self.assertIn("T(x)", html_out)
        self.assertIn("Q1", html_out)

        # Coordinate clamping checks: cx clamped to [40, 360], cy clamped to [60, 260]
        clamped_high = render_2d_trust_matrix_plane(t_fact=2.5, t_gen=-1.0, quadrant="Q3", lang="en")
        self.assertIn('cx="360.0"', clamped_high)
        self.assertIn('cy="260.0"', clamped_high)

        clamped_low = render_2d_trust_matrix_plane(t_fact=-2.5, t_gen=2.0, quadrant="Q2", lang="en")
        self.assertIn('cx="40.0"', clamped_low)
        self.assertIn('cy="60.0"', clamped_low)

        # XSS sanitization check
        xss_out = render_2d_trust_matrix_plane(t_fact=0.0, t_gen=0.5, quadrant="<script>alert('xss')</script>", lang="en")
        self.assertNotIn("<script>", xss_out)
        self.assertIn("&lt;script&gt;", xss_out)

    def test_render_2d_trust_matrix_plane_kz(self):
        """Verify Kazakh localized axes and pastel quadrant labels."""
        from ui.highlighting import render_2d_trust_matrix_plane

        html_kz = render_2d_trust_matrix_plane(t_fact=-0.5, t_gen=0.2, quadrant="Q4", lang="kz")
        self.assertIn("<svg", html_kz)
        self.assertIn('viewBox="0 0 460 320"', html_kz)

        # Localized quadrants
        self.assertIn("Расталған ақиқат", html_kz)
        self.assertIn("Адам қателігі", html_kz)
        self.assertIn("Нақты AI синтезі", html_kz)
        self.assertIn("AI дезинформация", html_kz)

        # Localized axis labels
        self.assertIn("Деректік ақиқаттық", html_kz)
        self.assertIn("Адам жазған / AI қаупі", html_kz)
        self.assertIn("Q4", html_kz)

    def test_standardized_cartesian_quadrant_mappings(self):
        """
        Verify all 4 quadrants follow standard Cartesian counter-clockwise convention:
        Q1 (Top-Right): Verified Human Fact (T_fact > 0, T_gen >= 0.5)
        Q2 (Top-Left): Human Misinformation (T_fact <= 0, T_gen >= 0.5)
        Q3 (Bottom-Left): Hallucinatory AI Disinformation (T_fact <= 0, T_gen < 0.5)
        Q4 (Bottom-Right): Accurate AI Synthesis (T_fact > 0, T_gen < 0.5)
        """
        from ui.highlighting import render_2d_trust_matrix_plane

        # Q1: Upper-Right
        q1_html = render_2d_trust_matrix_plane(t_fact=0.8, t_gen=0.8, quadrant="", lang="en")
        self.assertIn("Q1 (Verified Human Fact)", q1_html)

        # Q2: Upper-Left
        q2_html = render_2d_trust_matrix_plane(t_fact=-0.8, t_gen=0.8, quadrant="", lang="en")
        self.assertIn("Q2 (Human Misinformation)", q2_html)

        # Q3: Lower-Left
        q3_html = render_2d_trust_matrix_plane(t_fact=-0.8, t_gen=0.2, quadrant="", lang="en")
        self.assertIn("Q3 (Hallucinatory AI Disinformation)", q3_html)

        # Q4: Lower-Right
        q4_html = render_2d_trust_matrix_plane(t_fact=0.8, t_gen=0.2, quadrant="", lang="en")
        self.assertIn("Q4 (Accurate AI Synthesis)", q4_html)

    def test_render_trust_summary_card_contains_2d_matrix(self):
        """Verify summary card embeds the 2D Cartesian SVG plane for both None and verified inputs."""
        from ui.highlighting import render_trust_summary_card

        # None input (waiting state) embeds neutral 2D plane at (0.0, 0.5)
        html_none = render_trust_summary_card(None, lang="en")
        self.assertIn("Awaiting Verification", html_none)
        self.assertIn("<svg", html_none)
        self.assertIn('viewBox="0 0 460 320"', html_none)
        self.assertIn('cx="200.0"', html_none)
        self.assertIn('cy="160.0"', html_none)

        # Valid DocumentTrustResult
        doc_res = DocumentTrustResult(
            doc_text="Астана — Қазақстан астанасы.",
            ai_risk=0.10,
            factual_risk=0.00,
            trust_risk=0.05,
            quadrant_verdict="Verified Human Fact",
            t_fact=0.80,
            t_gen=0.90,
            composite_trust=0.8510,
            quadrant_code="Q1"
        )

        html_valid = render_trust_summary_card(doc_res, lang="en")
        self.assertIn("<svg", html_valid)
        self.assertIn('viewBox="0 0 460 320"', html_valid)
        self.assertIn("Verified Human Fact", html_valid)
        self.assertIn("lvl-verified-human", html_valid)
        self.assertIn("T_fact", html_valid)
        self.assertIn("T_gen", html_valid)
        self.assertIn("85.1%", html_valid)

    def test_render_claims_table_with_morpho_drawer(self):
        """Verify claims table renders HTML5 <details> morphological drawer with 16 diagnostic features."""
        from ui.highlighting import render_claims_verification_table

        claim = AtomicClaim(
            claim_id="c1",
            text="Қазақстан 1991 жылы тәуелсіздік алды.",
            source_sentence="Қазақстан 1991 жылы тәуелсіздік алды."
        )
        evidence = EvidencePassage(
            passage_id="ev1",
            title="Қазақстан тарихы",
            text="Қазақстан Республикасы 1991 жылы 16 желтоқсанда тәуелсіздігін жариялады.",
            matched_stems=["қазақстан", "1991", "тәуелсіздік"]
        )
        features = {
            "claim_negation": 0.0,
            "evidence_negation": 0.0,
            "directional_negation_mismatch": 0.0,
            "calendar_year_conflict": 0.0,
            "number_mismatch": 0.0,
            "has_claim_evidential": False,
            "has_evidence_evidential": False,
            "has_modal_necessity": False,
            "has_modal_possibility": False,
            "fst_root_jaccard": 0.725,
            "surface_token_overlap": 0.60,
            "lexical_fst_divergence": 0.125,
            "subject_proper_noun_match": 1.0,
            "case_suffix_alignment": 0.85,
            "claim_length_norm": 0.2,
            "evidence_length_norm": 0.4,
        }
        res = ClaimVerificationResult(
            claim=claim,
            verdict="SUPPORTED",
            confidence=0.92,
            evidence=[evidence],
            explanation="Дерек толық расталған.",
            morphological_features=features
        )

        html_en = render_claims_verification_table([res], lang="en")
        self.assertIn("<details", html_en)
        self.assertIn('class="morpho-drawer"', html_en)
        self.assertIn("<summary", html_en)
        self.assertIn("Morphological Alignment & Evidence Diagnostics", html_en)
        self.assertIn("Directional Negation Mismatch", html_en)
        self.assertIn("Calendar Year Conflict", html_en)
        self.assertIn("Number/Quantity Divergence", html_en)
        self.assertIn("Evidential Markers", html_en)
        self.assertIn("Epistemic Modals", html_en)
        self.assertIn("FST Root Jaccard", html_en)
        self.assertIn("72.5%", html_en)
        self.assertIn("Surface Token Overlap", html_en)
        self.assertIn("Lexical-FST Divergence", html_en)
        self.assertIn("Hard NEI Risk", html_en)
        self.assertIn("Case Suffix Alignment", html_en)
        self.assertIn("morpho-stem-highlight", html_en)

        # Kazakh localization check
        html_kz = render_claims_verification_table([res], lang="kz")
        self.assertIn("Морфологиялық сәйкестік және дәлелдемелер", html_kz)


class TestUIDashboardWiring2D(unittest.TestCase):
    """Test unit for Tab 4 Gradio UI Dashboard wiring, model toggle, and bilingual localization."""

    def test_handle_verify_document_with_model_selection(self):
        """Verifies handle_verify_document executes both 'morpho' and 'baseline' modes cleanly."""
        from ui.app import handle_verify_document

        text = "Қазақстан 1991 жылы тәуелсіздік алды."

        # Test Ours (Morpho)
        card_html_ours, table_html_ours = handle_verify_document(
            text=text,
            lang_choice="en",
            model_mode="morpho"
        )
        self.assertIn("<svg", card_html_ours)
        self.assertIn("morpho-drawer", table_html_ours)

        # Test Baseline
        card_html_base, table_html_base = handle_verify_document(
            text=text,
            lang_choice="en",
            model_mode="baseline"
        )
        self.assertIn("<svg", card_html_base)
        self.assertIn("claims-table", table_html_base)

    def test_switch_ui_language_updates_verification_components(self):
        """Verifies switch_ui_language produces valid updates for both English and Kazakh, including model radio."""
        from ui.app import switch_ui_language, I18N

        # Test English
        en_outputs = switch_ui_language("en")
        self.assertIsInstance(en_outputs, tuple)
        en_radio_update = None
        for item in en_outputs:
            if isinstance(item, dict) and "choices" in item:
                if I18N["en"]["verify_model_ours"] in item.get("choices", []):
                    en_radio_update = item
                    break
        self.assertIsNotNone(en_radio_update, "Model radio update missing in English outputs")
        self.assertEqual(en_radio_update.get("label"), "Verification Engine")
        self.assertEqual(en_radio_update.get("value"), "Ours (Hybrid + Morpho)")
        self.assertIn("Baseline (Surface Only)", en_radio_update.get("choices", []))

        # Test Kazakh
        kz_outputs = switch_ui_language("kz")
        self.assertIsInstance(kz_outputs, tuple)
        kz_radio_update = None
        for item in kz_outputs:
            if isinstance(item, dict) and "choices" in item:
                if I18N["kz"]["verify_model_ours"] in item.get("choices", []):
                    kz_radio_update = item
                    break
        self.assertIsNotNone(kz_radio_update, "Model radio update missing in Kazakh outputs")
        self.assertEqual(kz_radio_update.get("label"), "Тексеру механизмі")
        self.assertEqual(kz_radio_update.get("value"), "Біздің модель (Гибрид + Морфо)")
        self.assertIn("Базалық модель (Беткі қабат)", kz_radio_update.get("choices", []))

    def test_build_ui_scaffolds_verification_model_radio(self):
        """Verifies build_ui constructs the UI with the model radio without errors."""
        import gradio as gr
        from ui.app import build_ui

        demo = build_ui(load_model=False)
        self.assertIsNotNone(demo)
        self.assertIsInstance(demo, gr.Blocks)

    def test_gradio_event_dispatch_switch_ui_language(self):
        """
        Verifies that len(switch_ui_language('en')) == 49 and that
        switch_ui_language can be executed through Gradio event dispatch
        without raising ValueError.
        """
        import asyncio
        from ui.app import build_ui, switch_ui_language

        # 1. Output length verification for English and Kazakh
        en_outputs = switch_ui_language("en")
        self.assertIsInstance(en_outputs, tuple)
        self.assertEqual(len(en_outputs), 49)

        kz_outputs = switch_ui_language("kz")
        self.assertIsInstance(kz_outputs, tuple)
        self.assertEqual(len(kz_outputs), 49)

        # 2. Build demo and locate the registered event handler for switch_ui_language
        demo = build_ui(load_model=False)
        target_fn = next(
            f for f in demo.fns.values()
            if hasattr(f, "fn") and f.fn == switch_ui_language
        )
        self.assertEqual(len(target_fn.outputs), 49)

        # 3. Execute through Gradio event dispatch without raising ValueError
        res = asyncio.run(demo.call_function(target_fn, ["English"]))
        self.assertIn("prediction", res)
        self.assertEqual(len(res["prediction"]), 49)

        # Full Gradio process_api invocation check
        api_res = asyncio.run(demo.process_api(target_fn, ["English"]))
        self.assertIn("data", api_res)
        self.assertEqual(len(api_res["data"]), 49)


if __name__ == "__main__":
    unittest.main()


