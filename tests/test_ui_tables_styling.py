# -*- coding: utf-8 -*-
"""
tests/test_ui_tables_styling.py: Unit tests for Minimalist Notion Database Table Styling,
Tag Pills, and Publication-Grade Data Presentation.

Verifies:
- render_fst_decomposition_table renders .notion-table-wrapper, .notion-table, .tag-pill,
  and supports 8-column schema with muted dash for empty values.
- render_claims_verification_table renders .notion-table-wrapper, .notion-table,
  .verdict-pill (.pill-supports, .pill-refutes, .pill-nei), and .tag-pill.
- render_autofst_inductions_table renders .notion-table-wrapper, .notion-table, .tag-pill,
  and retains .autofst-table and .status-badge for backward compatibility.
- Zero decorative emojis across all table HTML outputs and CSS.
- Exact parity between ui/highlighting.py and hf_space/ui/highlighting.py.
"""

import os
import re
import unittest

from ui.highlighting import (
    HEATMAP_CSS,
    render_fst_decomposition_table,
    render_claims_verification_table,
    render_autofst_inductions_table,
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


class MockClaim:
    """Lightweight mock atomic claim for verification table testing."""
    def __init__(self, text="Қазақстан 1991 жылы тәуелсіздік алды."):
        self.text = text


class MockEvidence:
    """Lightweight mock evidence passage."""
    def __init__(self, text="Қазақстан 1991 жылы тәуелсіздігін жариялады.", title="Wikipedia"):
        self.text = text
        self.title = title
        self.source_url = "https://kk.wikipedia.org"
        self.matched_stems = ["қазақстан", "1991"]


class MockVerificationResult:
    """Lightweight mock verification result."""
    def __init__(self, verdict="SUPPORTS", conf=0.95):
        self.claim = MockClaim()
        self.verdict = verdict
        self.confidence = conf
        self.evidence = [MockEvidence()]
        self.explanation = "Historical record confirmed."
        self.morphological_features = {
            "fst_root_jaccard": 0.85,
            "surface_token_overlap": 0.70,
            "lexical_fst_divergence": 0.15,
            "case_suffix_alignment": 0.90,
            "directional_negation_mismatch": 0.0,
            "calendar_year_conflict": 0.0,
            "number_mismatch": 0.0,
        }


class TestUITablesStyling(unittest.TestCase):
    """Test suite verifying Notion Database styling across publication tables."""

    def test_heatmap_css_contains_notion_table_classes(self):
        """Verifies HEATMAP_CSS defines .notion-table-wrapper, .notion-table, .tag-pill, and .verdict-pill."""
        self.assertIn(".notion-table-wrapper", HEATMAP_CSS)
        self.assertIn(".notion-table", HEATMAP_CSS)
        self.assertIn(".tag-pill", HEATMAP_CSS)
        self.assertIn(".verdict-pill", HEATMAP_CSS)
        self.assertIn(".pill-supports", HEATMAP_CSS)
        self.assertIn(".pill-refutes", HEATMAP_CSS)
        self.assertIn(".pill-nei", HEATMAP_CSS)

    def test_fst_decomposition_table_notion_styling(self):
        """Verifies render_fst_decomposition_table outputs Notion table wrapper, table, and tag pills."""
        sample_breakdowns = [
            {
                "word": "балаларға",
                "root": "бала",
                "pos": "NOUN",
                "plural": "-лар",
                "possessive": "—",
                "case": "-ға",
                "tense": "—",
                "person": "—",
                "suffix_chain": "-лар -> -ға",
            },
            {
                "word": "келдім",
                "root": "кел",
                "pos": "VERB",
                "plural": "—",
                "possessive": "—",
                "case": "—",
                "tense": "-ді",
                "person": "-м",
                "suffix_chain": "-ді -> -м",
            }
        ]

        html_en = render_fst_decomposition_table(sample_breakdowns, lang="en")
        self.assertIn("notion-table-wrapper", html_en)
        self.assertIn("notion-table", html_en)
        self.assertIn("fst-table", html_en)
        self.assertIn("tag-pill", html_en)
        self.assertIn("балаларға", html_en)
        self.assertIn("бала", html_en)
        self.assertIn("NOUN", html_en)
        self.assertIn("-лар", html_en)
        self.assertIn("-ға", html_en)
        self.assertIn("Word", html_en)
        self.assertIn("Root", html_en)
        self.assertIn("Possessive", html_en)
        # Empty cells should have muted dash
        self.assertIn("#a1a1aa", html_en)

        # Kazakh localization
        html_kz = render_fst_decomposition_table(sample_breakdowns, lang="kz")
        self.assertIn("notion-table", html_kz)
        self.assertIn("Сөз (Word)", html_kz)
        self.assertIn("Түбір (Root)", html_kz)
        self.assertIn("Сөз табы (POS)", html_kz)
        self.assertIn("Көптік (Plural)", html_kz)
        self.assertIn("Тәуелдік (Possessive)", html_kz)
        self.assertIn("Септік (Case)", html_kz)

    def test_fst_decomposition_table_empty(self):
        """Verifies empty data handling in FST table."""
        empty_html = render_fst_decomposition_table([], lang="en")
        self.assertIn("No morphological decomposition data available", empty_html)
        empty_kz = render_fst_decomposition_table([], lang="kz")
        self.assertIn("Морфологиялық талдау деректері жоқ", empty_kz)

    def test_claims_verification_table_notion_styling(self):
        """Verifies render_claims_verification_table renders Notion styling and verdict pills."""
        results = [
            MockVerificationResult(verdict="SUPPORTS", conf=0.96),
            MockVerificationResult(verdict="REFUTES", conf=0.88),
            MockVerificationResult(verdict="NOT ENOUGH INFO", conf=0.50),
        ]

        html_out = render_claims_verification_table(results, lang="en")
        self.assertIn("notion-table-wrapper", html_out)
        self.assertIn("notion-table", html_out)
        self.assertIn("claims-table", html_out)
        self.assertIn("verdict-pill", html_out)
        self.assertIn("pill-supports", html_out)
        self.assertIn("pill-refutes", html_out)
        self.assertIn("pill-nei", html_out)
        self.assertIn("SUPPORTS", html_out)
        self.assertIn("REFUTES", html_out)
        self.assertIn("NOT ENOUGH INFO", html_out)
        self.assertIn("morpho-drawer", html_out)
        self.assertIn("tag-pill", html_out)

    def test_claims_verification_table_empty(self):
        """Verifies empty claims handling."""
        empty_html = render_claims_verification_table([], lang="en")
        self.assertIn("No atomic claims extracted", empty_html)

    def test_autofst_inductions_table_notion_styling(self):
        """Verifies render_autofst_inductions_table renders Notion styling, stem pill, and guard status."""
        inductions = [
            {
                "stem": "вайб",
                "harmony": "hard",
                "coda": "voiced",
                "suffixes": ["та", "дық"],
                "guard": "Passed (100% agreement)",
            },
            {
                "stem": "кринж",
                "harmony": "soft",
                "coda": "voiced",
                "suffixes": ["де", "ді"],
                "guard": "Passed (100% agreement)",
            }
        ]

        html_out = render_autofst_inductions_table(inductions, lang="en")
        self.assertIn("notion-table-wrapper", html_out)
        self.assertIn("notion-table", html_out)
        self.assertIn("autofst-table", html_out)
        self.assertIn("tag-pill", html_out)
        self.assertIn("status-badge", html_out)
        self.assertIn("вайб", html_out)
        self.assertIn("кринж", html_out)
        self.assertIn("Passed (100% agreement)", html_out)
        self.assertIn("Жуан (Hard)", html_out)
        self.assertIn("Жіңішке (Soft)", html_out)

    def test_autofst_inductions_table_empty(self):
        """Verifies empty inductions handling."""
        empty_html = render_autofst_inductions_table([], lang="en")
        self.assertIn("No new slang stems required", empty_html)

    def test_zero_emojis_across_all_tables(self):
        """MANDATORY ZERO EMOJI AUDIT: Ensures zero emojis in all table renders."""
        sample_breakdowns = [{
            "word": "сынақ",
            "root": "сын",
            "pos": "NOUN",
            "plural": "—",
            "possessive": "—",
            "case": "-ақ",
            "tense": "—",
            "person": "—",
            "suffix_chain": "-ақ",
        }]
        sample_claims = [MockVerificationResult()]
        sample_inductions = [{
            "stem": "тест",
            "harmony": "hard",
            "coda": "voiceless",
            "suffixes": ["тер"],
            "guard": "Passed (100% agreement)",
        }]

        rendered_outputs = [
            render_fst_decomposition_table(sample_breakdowns, lang="en"),
            render_fst_decomposition_table(sample_breakdowns, lang="kz"),
            render_claims_verification_table(sample_claims, lang="en"),
            render_claims_verification_table(sample_claims, lang="kz"),
            render_autofst_inductions_table(sample_inductions, lang="en"),
            render_autofst_inductions_table(sample_inductions, lang="kz"),
            HEATMAP_CSS,
        ]

        for idx, out in enumerate(rendered_outputs):
            emojis_found = EMOJI_PATTERN.findall(out)
            self.assertEqual(
                emojis_found,
                [],
                f"Found forbidden emoji in output index {idx}: {emojis_found}"
            )

    def test_exact_parity_between_ui_and_hf_space(self):
        """Verifies ui/highlighting.py and hf_space/ui/highlighting.py match exactly."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ui_path = os.path.join(base_dir, "ui", "highlighting.py")
        hf_path = os.path.join(base_dir, "hf_space", "ui", "highlighting.py")

        with open(ui_path, "r", encoding="utf-8") as f:
            ui_content = f.read()

        with open(hf_path, "r", encoding="utf-8") as f:
            hf_content = f.read()

        self.assertEqual(
            ui_content,
            hf_content,
            "ui/highlighting.py and hf_space/ui/highlighting.py must be byte-for-byte identical."
        )


if __name__ == "__main__":
    unittest.main()
