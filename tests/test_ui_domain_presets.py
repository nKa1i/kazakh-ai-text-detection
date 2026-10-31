# -*- coding: utf-8 -*-
"""
Unit tests for Phase 4 Gradio UI Multi-Genre Presets & Domain Audit Badge:
- Verifies all 5 domain preset samples in ui/presets.py load cleanly and produce expected verdicts
- Verifies handle_domain_preset_change returns text and bilingual genre audit metadata
- Verifies run_full_analysis_logic populates detected_domain and domain_badge
- Verifies generate_audit_markdown includes Phase 4 Cross-Domain Robustness section
- Verifies backward compatibility of handle_analyze_document (8-tuple) and switch_ui_language (63-tuple)
- Verifies zero decorative emojis across all presets and UI outputs
"""

import os
import re
import unittest

from models.heuristic_detector import OfflineHeuristicDetector
from ui.app import (
    generate_audit_markdown,
    handle_analyze_document,
    handle_domain_preset_change,
    run_full_analysis_logic,
    switch_ui_language,
)
from ui.presets import (
    DOMAIN_PRESET_SAMPLES,
    PRESET_SAMPLES,
    get_domain_preset_choices,
    get_domain_preset_metadata,
    get_domain_preset_text,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\ufe0f]")


class TestUIDomainPresets(unittest.TestCase):
    def setUp(self):
        self.detector = OfflineHeuristicDetector(calibrated_threshold=0.9980)

    def test_five_domain_presets_schema_and_expected_verdicts(self):
        choices = get_domain_preset_choices()
        self.assertEqual(len(choices), 5)
        self.assertEqual(len(DOMAIN_PRESET_SAMPLES), 5)

        expected_keys = {
            "news",
            "academic",
            "official_legal",
            "fiction_literature",
            "social_casual",
        }
        found_keys = set()

        for title in choices:
            text = get_domain_preset_text(title)
            meta = get_domain_preset_metadata(title)
            self.assertTrue(len(text.split()) >= 18)
            self.assertIn("domain_key", meta)
            self.assertIn("expected_verdict", meta)
            found_keys.add(meta["domain_key"])

            res = self.detector.predict_document(text)
            self.assertEqual(
                res.verdict,
                meta["expected_verdict"],
                msg=f"Preset '{title}' expected '{meta['expected_verdict']}', got '{res.verdict}'",
            )
            self.assertIsNone(EMOJI_RE.search(text))
            self.assertIsNone(EMOJI_RE.search(meta.get("description", "")))

        self.assertEqual(found_keys, expected_keys)
        # Ensure original 6 presets remain untouched
        self.assertEqual(len(PRESET_SAMPLES), 6)

    def test_handle_domain_preset_change_bilingual(self):
        first_choice = get_domain_preset_choices()[0]
        text_en, md_en = handle_domain_preset_change(first_choice, lang_choice="English")
        text_kz, md_kz = handle_domain_preset_change(first_choice, lang_choice="Қазақша")

        self.assertEqual(text_en, text_kz)
        self.assertIn("Genre:", md_en)
        self.assertIn("Жанр:", md_kz)
        self.assertIsNone(EMOJI_RE.search(md_en))
        self.assertIsNone(EMOJI_RE.search(md_kz))

    def test_run_full_analysis_and_audit_markdown_include_domain_badge(self):
        sample_choice = get_domain_preset_choices()[1]
        sample_text = get_domain_preset_text(sample_choice)

        res = run_full_analysis_logic(
            text=sample_text,
            autofst_enabled=True,
            verifier_enabled=True,
            current_lang="English",
            detector=self.detector,
        )
        self.assertIn("detected_domain", res)
        self.assertIn("domain_badge", res)
        self.assertIn(res["detected_domain"], ("news", "academic", "official_legal", "fiction_literature", "social_casual"))
        self.assertIn("FPR&lt;=5% Verified", res["executive_card"])

        audit_md = generate_audit_markdown(res, source_text=sample_text)
        self.assertIn("Cross-Domain Robustness & Genre Audit (Phase 4)", audit_md)
        self.assertIn(res["domain_badge"]["badge_text"], audit_md)
        self.assertIsNone(EMOJI_RE.search(audit_md))

    def test_backward_compatible_signatures(self):
        sample_text = get_domain_preset_text(get_domain_preset_choices()[0])
        out_tuple = handle_analyze_document(sample_text, detector=self.detector, lang_choice="English")
        self.assertEqual(len(out_tuple), 8)

        lang_updates = switch_ui_language("Қазақша")
        self.assertEqual(len(lang_updates), 63)


if __name__ == "__main__":
    unittest.main()
