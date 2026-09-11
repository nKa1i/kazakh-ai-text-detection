# -*- coding: utf-8 -*-
"""
tests/test_ui_verification.py: Unit tests for Factual Verification UI Components.
Tests XSS sanitization, Four-Quadrant trust card rendering, and claims table formatting.
"""

import unittest
from verification.evidence import (
    AtomicClaim,
    EvidencePassage,
    ClaimVerificationResult,
    DocumentTrustResult
)
from ui.highlighting import (
    render_trust_summary_card,
    render_claims_verification_table
)


class TestUIVerificationComponents(unittest.TestCase):
    """Verifies HTML/CSS rendering of trust summary cards and verification tables."""

    def setUp(self):
        self.claim1 = AtomicClaim(
            claim_id="c1",
            text="Қазақстан 1991 жылы тәуелсіздік алды.",
            source_sentence="Қазақстан 1991 жылы тәуелсіздік алды.",
            start_char=0,
            end_char=38
        )
        self.ev1 = EvidencePassage(
            passage_id="wiki_01",
            title="Қазақстан тәуелсіздігі",
            text="Қазақстан Республикасы 1991 жылы 16 желтоқсанда өз тәуелсіздігін жариялады.",
            source_url="https://kk.wikipedia.org/wiki/Independence",
            similarity_score=0.88,
            matched_stems=["қазақстан", "1991", "тәуелсіздік"]
        )
        self.res1 = ClaimVerificationResult(
            claim=self.claim1,
            verdict="SUPPORTED",
            confidence=0.92,
            evidence=[self.ev1],
            explanation="Ақпарат толық расталды."
        )

        self.doc_res = DocumentTrustResult(
            doc_text="Қазақстан 1991 жылы тәуелсіздік алды.",
            ai_risk=0.15,
            factual_risk=0.00,
            trust_risk=0.075,
            quadrant_verdict="Verified Human Fact",
            claims=[self.res1],
            total_claims=1,
            supported_count=1,
            refuted_count=0,
            nei_count=0
        )

    def test_render_trust_summary_card_content(self):
        """Verifies trust card renders quadrant verdict, risk percentages, and claim counts."""
        html_en = render_trust_summary_card(self.doc_res, lang="en")
        self.assertIn("Verified Human Fact", html_en)
        self.assertIn("15.0%", html_en)  # AI Risk
        self.assertIn("0.0%", html_en)   # Fact Risk
        self.assertIn("7.5%", html_en)   # Trust Risk
        self.assertIn("Supported: 1", html_en)

        # Bilingual Kazakh check
        html_kz = render_trust_summary_card(self.doc_res, lang="kz")
        self.assertIn("Расталған ақиқат", html_kz)

    def test_render_trust_summary_card_quadrants(self):
        """Tests rendering across all 4 quadrants of the Trust Matrix."""
        quadrants = [
            ("Verified Human Fact", "lvl-verified-human"),
            ("Human Misinformation", "lvl-human-misinfo"),
            ("Accurate AI Synthesis", "lvl-ai-synthesis"),
            ("Hallucinatory AI Disinformation", "lvl-ai-disinfo")
        ]
        for q_name, css_class in quadrants:
            dummy_res = DocumentTrustResult(
                doc_text="Тест",
                ai_risk=0.8,
                factual_risk=0.8,
                trust_risk=0.8,
                quadrant_verdict=q_name,
                claims=[],
                total_claims=0,
                supported_count=0,
                refuted_count=0,
                nei_count=0
            )
            out = render_trust_summary_card(dummy_res, lang="en")
            self.assertIn(q_name, out)
            self.assertIn(css_class, out)

    def test_render_claims_verification_table_basic(self):
        """Verifies claims table displays claims, verdicts, evidence citations, and explanations."""
        html_tbl = render_claims_verification_table([self.res1], lang="en")
        self.assertIn("Қазақстан 1991 жылы тәуелсіздік алды.", html_tbl)
        self.assertIn("SUPPORTED", html_tbl)
        self.assertIn("Қазақстан тәуелсіздігі", html_tbl)
        self.assertIn("92.0%", html_tbl)
        self.assertIn("Ақпарат толық расталды.", html_tbl)

    def test_render_claims_verification_table_xss_safety(self):
        """Strictly tests XSS sanitization across malicious claim texts and passage titles."""
        malicious_claim = AtomicClaim(
            claim_id="cx",
            text="<script>alert('xss')</script> Мәтін",
            source_sentence="<script>alert('xss')</script> Мәтін"
        )
        malicious_ev = EvidencePassage(
            passage_id="px",
            title="<img src=x onerror=alert('img_xss')>",
            text="<b onmouseover=alert('b_xss')>Қауіпті мәтін</b>",
            similarity_score=0.5
        )
        malicious_res = ClaimVerificationResult(
            claim=malicious_claim,
            verdict="REFUTED",
            confidence=0.8,
            evidence=[malicious_ev],
            explanation="<script>evil()</script>"
        )

        html_tbl = render_claims_verification_table([malicious_res], lang="en")
        self.assertNotIn("<script>", html_tbl)
        self.assertNotIn("<img src=x", html_tbl)
        self.assertNotIn("<b onmouseover=", html_tbl)
        self.assertIn("&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;", html_tbl)
        self.assertIn("&lt;img src=x onerror=alert(&#x27;img_xss&#x27;)&gt;", html_tbl)

    def test_empty_results_safe_rendering(self):
        """Verifies empty result inputs do not crash and render polite fallback messages."""
        card = render_trust_summary_card(None, lang="en")
        self.assertIn("Awaiting Verification", card)

        tbl = render_claims_verification_table([], lang="en")
        self.assertIn("No atomic claims extracted or verified yet.", tbl)


if __name__ == "__main__":
    unittest.main()
