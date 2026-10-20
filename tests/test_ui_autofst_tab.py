# -*- coding: utf-8 -*-
"""Unit tests for Tab 5 (Auto-FST & Slang Normalizer) in the Gradio Dashboard."""

import unittest
import re
from hf_space.ui.app import create_app, switch_ui_language, I18N
from hf_space.ui.highlighting import render_autofst_comparison_card, render_autofst_inductions_table
from hf_space.auto_fst import DynamicFSTOverlay, AutoFSTInducer, RussianLayoutNormalizer, RegressionGuard


class TestUIAutoFSTTab(unittest.TestCase):
    def test_rendering_comparison_card_empty(self):
        html_card = render_autofst_comparison_card("", "", 0.0, lang="en")
        self.assertIn("Analysis comparison will appear here", html_card)

    def test_rendering_comparison_card_with_content(self):
        html_card = render_autofst_comparison_card(
            before_text="вайбтадық",
            after_text="вайб -та -дық",
            latency_ms=0.12,
            lang="kk"
        )
        self.assertIn("вайб -та -дық", html_card)
        self.assertIn("0.12", html_card)

    def test_rendering_inductions_table(self):
        inductions = [{
            "stem": "вайб",
            "harmony": "hard",
            "coda": "voiced",
            "suffixes": ["та", "дық"],
            "guard": "Passed (100% agreement)"
        }]
        html_table = render_autofst_inductions_table(inductions, lang="en")
        self.assertIn("вайб", html_table)
        self.assertIn("Жуан (Hard)", html_table)
        self.assertIn("-та -дық", html_table)

    def test_gradio_app_builds_with_tab5(self):
        # Instantiate Gradio demo without loading PyTorch weights
        demo = create_app(detector=None, verifier=None, load_model=False)
        self.assertIsNotNone(demo)

    def test_bilingual_language_switch_tab5(self):
        # Test English switch
        en_res = switch_ui_language("English")
        self.assertEqual(len(en_res), 63)
        self.assertIn("Adaptive Neuro-Symbolic Morphology", en_res[49])
        self.assertIn("Russian Layout", en_res[52]["value"])

        # Test Kazakh switch
        kz_res = switch_ui_language("Қазақша")
        self.assertEqual(len(kz_res), 63)
        self.assertIn("Бейімделгіш Нейро-Символдық Морфология", kz_res[49])
        self.assertIn("Орыс пернетақтасы", kz_res[52]["value"])

    def test_zero_emojis_in_tab5(self):
        emoji_pattern = re.compile(
            r"[\U00010000-\U0010ffff"
            r"\u2600-\u27bf"
            r"\ud83c[\ud000-\udfff]"
            r"\ud83d[\ud000-\udfff]"
            r"\ud83e[\ud000-\udfff]"
            r"\u200d\u23fe\u231a\u231b\u23e9-\u23ec\u23f0\u23f3\u25fd\u25fe\u2614\u2615\u2648-\u2653\u267f\u2693\u26a1\u26aa\u26ab\u26bd\u26be\u26c4\u26c5\u26ce\u26d4\u26ea\u26f2\u26f3\u26f5\u26fa\u26fd\u2702\u2705\u2708\u2709\u270a-\u270d\u270f\u2712\u2714\u2716\u271d\u2721\u2728\u2733\u2734\u2744\u2747\u274c\u274e\u2753-\u2755\u2757\u2763\u2764\u2795-\u2797\u27a1\u27b0\u27bf\u2934\u2935\u2b05-\u2b07\u2b1b\u2b1c\u2b50\u2b55]"
        )
        for lang_key in ["en", "kz"]:
            d = I18N[lang_key]
            for k, v in d.items():
                if "autofst" in k or "tab_5" in k:
                    if isinstance(v, str):
                        self.assertFalse(
                            emoji_pattern.search(v),
                            f"Decorative emoji found in I18N[{lang_key}][{k}]: {v}"
                        )


if __name__ == '__main__':
    unittest.main()
