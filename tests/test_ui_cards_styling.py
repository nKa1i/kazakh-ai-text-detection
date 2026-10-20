# -*- coding: utf-8 -*-
"""
tests/test_ui_cards_styling.py: Unit tests for Minimalist Apple / Notion Card Redesign,
Confidence Meters, and Heatmap styling.

Verifies:
- HEATMAP_CSS soft pastel palettes, 1px hairline borders, and typography.
- render_confidence_meter sleek Apple 6px track, pill status badges, and color tiers.
- render_executive_summary_card Notion callout card with 4-metric grid and monospace font.
- render_dynamic_gate_bar 8px dual-split bar with floating header labels.
- render_trust_summary_card monospace coordinate badge and soft quadrant pills.
- render_autofst_comparison_card monospace font, .latency-pill, and .autofst-card hook.
- Zero decorative emojis across all HTML outputs and CSS.
- Exact parity between ui/highlighting.py and hf_space/ui/highlighting.py.
"""

import os
import re
import unittest

from ui.highlighting import (
    HEATMAP_CSS,
    render_document_heatmap,
    render_confidence_meter,
    render_executive_summary_card,
    render_dynamic_gate_bar,
    render_trust_summary_card,
    render_autofst_comparison_card,
)
import hf_space.ui.highlighting as hf_highlighting

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE
)


class TestUICardsStyling(unittest.TestCase):
    """Test suite verifying Apple / Notion aesthetic in UI cards, confidence meters, and heatmap."""

    def test_heatmap_css_soft_pastel_and_typography(self):
        """Verifies HEATMAP_CSS contains Apple / Notion styles and pastel tiers."""
        # Container styles
        self.assertIn("#ffffff", HEATMAP_CSS)
        self.assertIn("#e4e4e7", HEATMAP_CSS)
        self.assertIn("0 1px 3px rgba(0, 0, 0, 0.03)", HEATMAP_CSS)
        self.assertIn("Inter", HEATMAP_CSS)

        # Soft pastel tiers
        self.assertIn("#ecfdf5", HEATMAP_CSS)  # lvl-human bg
        self.assertIn("#10b981", HEATMAP_CSS)  # lvl-human border
        self.assertIn("#065f46", HEATMAP_CSS)  # lvl-human text

        self.assertIn("#fffbeb", HEATMAP_CSS)  # lvl-amber bg
        self.assertIn("#f59e0b", HEATMAP_CSS)  # lvl-amber border
        self.assertIn("#92400e", HEATMAP_CSS)  # lvl-amber text

        self.assertIn("#fff1f2", HEATMAP_CSS)  # lvl-ai bg
        self.assertIn("#f43f5e", HEATMAP_CSS)  # lvl-ai border
        self.assertIn("#9f1239", HEATMAP_CSS)  # lvl-ai text

    def test_render_confidence_meter_apple_style(self):
        """Verifies render_confidence_meter returns sleek 6px track and pill badge."""
        html_meter = render_confidence_meter(0.9985, "[MACHINE-GENERATED]", lang="en")
        self.assertIn("confidence-track", html_meter)
        self.assertIn("confidence-fill", html_meter)
        self.assertIn("meter-ai", html_meter)
        self.assertIn("confidence-verdict-tag", html_meter)
        self.assertIn("99.9%", html_meter)

        # Check CSS definitions for confidence track
        self.assertIn("height: 6px", HEATMAP_CSS)
        self.assertIn("border-radius: 9999px", HEATMAP_CSS)
        self.assertIn("#f4f4f5", HEATMAP_CSS)

    def test_render_executive_summary_card_4_metric_grid(self):
        """Verifies render_executive_summary_card displays 4-metric grid with monospace values."""
        card_html = render_executive_summary_card(
            probability=0.03,
            verdict="Authentic Human",
            lang="en",
            total_words=150,
            total_sents=10,
            total_windows=1,
            ai_ratio=0.0,
            lexical_coverage=0.985
        )
        self.assertIn("executive-summary-card", card_html)
        self.assertIn("executive-kpi-grid", card_html)
        self.assertIn("SFMono-Regular", HEATMAP_CSS)
        self.assertIn("150", card_html)
        self.assertIn("3.0%", card_html)
        self.assertIn("98.5%", card_html)
        self.assertIn("Verdict", card_html)
        self.assertIn("Confidence", card_html)
        self.assertIn("Words", card_html)
        self.assertIn("Lexical Coverage", card_html)

        # Kazakh localization
        card_kz = render_executive_summary_card(
            probability=0.95,
            verdict="Machine-Generated",
            lang="kz",
            total_words=200,
            total_sents=12,
            total_windows=2,
            ai_ratio=0.85
        )
        self.assertIn("Үкім", card_kz)
        self.assertIn("Сенімділік", card_kz)
        self.assertIn("Сөз көлемі", card_kz)
        self.assertIn("Лексикалық қамту", card_kz)

    def test_render_dynamic_gate_bar_sleek_height(self):
        """Verifies render_dynamic_gate_bar renders 8px bar with floating labels."""
        gate_html = render_dynamic_gate_bar(0.65, lang="en")
        self.assertIn("gate-bar-container", gate_html)
        self.assertIn("gate-stream-bert", gate_html)
        self.assertIn("gate-stream-fst", gate_html)
        self.assertIn("65.0%", gate_html)
        self.assertIn("35.0%", gate_html)
        self.assertIn("gate-bar-header", gate_html)

        # Check CSS height definition
        self.assertIn("height: 8px", HEATMAP_CSS)

    def test_render_trust_summary_card_notion_style(self):
        """Verifies render_trust_summary_card renders monospace coordinates and soft tints."""
        from types import SimpleNamespace
        doc_res = SimpleNamespace(
            ai_risk=0.05,
            factual_risk=0.02,
            trust_risk=0.04,
            quadrant_verdict="Verified Human Fact",
            total_claims=3,
            supported_count=3,
            refuted_count=0,
            nei_count=0,
            t_fact=0.85,
            t_gen=0.92,
            composite_trust=0.88,
            quadrant_code="Q1"
        )
        trust_html = render_trust_summary_card(doc_res, lang="en")
        self.assertIn("trust-card-wrapper", trust_html)
        self.assertIn("T_fact", trust_html)
        self.assertIn("T_gen", trust_html)
        self.assertIn("lvl-verified-human", trust_html)

    def test_render_autofst_comparison_card_monospace_and_latency_chip(self):
        """Verifies render_autofst_comparison_card features monospace font and latency chip."""
        card_html = render_autofst_comparison_card(
            before_text="вайбтадық",
            after_text="вайб -та -дық",
            latency_ms=0.04,
            lang="en"
        )
        self.assertIn("autofst-card", card_html)
        self.assertIn("latency-pill", card_html)
        self.assertIn("0.04", card_html)
        self.assertIn("SFMono-Regular", card_html)
        self.assertIn("вайб -та -дық", card_html)

    def test_zero_emojis_across_all_cards(self):
        """MANDATORY ZERO EMOJI AUDIT: Ensures zero emojis in all card renders and HEATMAP_CSS."""
        self.assertEqual(EMOJI_PATTERN.findall(HEATMAP_CSS), [])

        samples = [
            render_confidence_meter(0.12, "Human", "en"),
            render_confidence_meter(0.88, "AI", "kz"),
            render_executive_summary_card(0.05, "Human", "en", 100, 5, 1, 0.0),
            render_executive_summary_card(0.95, "AI", "kz", 200, 10, 2, 0.9),
            render_dynamic_gate_bar(0.5, "en"),
            render_dynamic_gate_bar(0.7, "kz"),
            render_autofst_comparison_card("test", "test", 0.04, "en"),
            render_trust_summary_card(None, "en"),
            render_trust_summary_card(None, "kz")
        ]
        for s in samples:
            emojis = EMOJI_PATTERN.findall(s)
            self.assertEqual(emojis, [], f"Found forbidden emoji in output: {emojis}")

    def test_exact_parity_between_ui_and_hf_space(self):
        """Verifies ui/highlighting.py and hf_space/ui/highlighting.py match exactly."""
        ui_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ui",
            "highlighting.py"
        )
        hf_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "hf_space",
            "ui",
            "highlighting.py"
        )
        with open(ui_path, "r", encoding="utf-8") as f:
            ui_content = f.read()
        with open(hf_path, "r", encoding="utf-8") as f:
            hf_content = f.read()
        self.assertEqual(
            ui_content,
            hf_content,
            "ui/highlighting.py and hf_space/ui/highlighting.py must have exact parity."
        )


if __name__ == "__main__":
    unittest.main()
