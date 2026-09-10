# -*- coding: utf-8 -*-
"""
tests/test_ui_linguistic_explainer.py: Unit tests for ui/linguistic_explainer.py
"""

import re
import unittest
from typing import List

from kaz_mage.document import DocumentAnalysisResult, DocumentChunk
from ui.linguistic_explainer import (
    compute_linguistic_features,
    generate_linguistic_explanation,
)


def contains_emoji(text: str) -> bool:
    """Detects emojis in text to enforce zero-decorative-emoji constraint."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return bool(emoji_pattern.search(text))


class TestLinguisticExplainer(unittest.TestCase):
    """Test suite for Kazakh linguistic feature extractor and dynamic explainer."""

    def test_compute_features_human_text(self):
        """Computes features on authentic colloquial consumer review text."""
        text = "Бұл менің сүйікті дүкенімнен сатып алған тауарым. Сапасы өте керемет, жеткізу жылдам болды."
        feats = compute_linguistic_features(text)
        
        self.assertGreater(feats["colloquial_marker_count"], 0)
        self.assertEqual(feats["formulaic_marker_count"], 0)
        self.assertGreater(feats["type_token_ratio"], 0.7)
        self.assertEqual(feats["total_words"], 13)
        self.assertEqual(feats["unique_words"], 13)
        self.assertGreater(feats["avg_word_length"], 4.0)
        self.assertGreaterEqual(feats["avg_suffix_count"], 0.0)
        self.assertIsInstance(feats["colloquial_markers_found"], list)
        self.assertIsInstance(feats["formulaic_markers_found"], list)
        self.assertTrue(any(m in feats["colloquial_markers_found"] for m in ["керемет", "сапасы", "өте", "жылдам"]))

    def test_compute_features_ai_text(self):
        """Computes features on machine-generated text with formal discourse connectors."""
        text = "Қорытындылай келе, бұл құбылыс заманауи қоғам үшін ерекше маңызды рөл атқарады. Осыған орай, жүйелі түрде талдау жасау қажет."
        feats = compute_linguistic_features(text)
        
        self.assertGreater(feats["formulaic_marker_count"], 0)
        self.assertIn("қорытындылай келе", [m.lower() for m in feats["formulaic_markers_found"]])
        self.assertEqual(feats["colloquial_marker_count"], 0)
        self.assertGreater(feats["type_token_ratio"], 0.7)
        self.assertGreater(feats["total_words"], 10)
        self.assertGreater(feats["avg_word_length"], 4.0)
        self.assertGreaterEqual(feats["avg_suffix_count"], 0.0)

    def test_compute_features_empty_and_whitespace(self):
        """Defensive handling for empty string, whitespace, and None."""
        for empty_val in ["", "   \n\t  ", None]:
            feats = compute_linguistic_features(empty_val)
            self.assertEqual(feats["total_words"], 0)
            self.assertEqual(feats["unique_words"], 0)
            self.assertEqual(feats["type_token_ratio"], 0.0)
            self.assertEqual(feats["formulaic_marker_count"], 0)
            self.assertEqual(feats["formulaic_markers_found"], [])
            self.assertEqual(feats["colloquial_marker_count"], 0)
            self.assertEqual(feats["colloquial_markers_found"], [])
            self.assertEqual(feats["avg_word_length"], 0.0)
            self.assertEqual(feats["avg_suffix_count"], 0.0)

    def test_compute_features_punctuation_and_numeric(self):
        """Defensive handling for punctuation-only and number-only inputs."""
        punct_text = "!@#$%^&*()_+=-.,<>/?;:«»\"'—–"
        feats_punct = compute_linguistic_features(punct_text)
        self.assertEqual(feats_punct["total_words"], 0)
        self.assertEqual(feats_punct["type_token_ratio"], 0.0)

        num_text = "12345 67890 2026 3.14"
        feats_num = compute_linguistic_features(num_text)
        self.assertEqual(feats_num["total_words"], 0)
        self.assertEqual(feats_num["type_token_ratio"], 0.0)

    def test_compute_features_single_word(self):
        """Single word input should yield valid metrics without divide-by-zero."""
        text = "Қазақстан"
        feats = compute_linguistic_features(text)
        self.assertEqual(feats["total_words"], 1)
        self.assertEqual(feats["unique_words"], 1)
        self.assertEqual(feats["type_token_ratio"], 1.0)
        self.assertEqual(feats["avg_word_length"], len("Қазақстан"))

    def test_compute_features_repeated_words_low_ttr(self):
        """Repeated identical words correctly lower Type-Token Ratio."""
        text = "қайталау қайталау қайталау қайталау қайталау"
        feats = compute_linguistic_features(text)
        self.assertEqual(feats["total_words"], 5)
        self.assertEqual(feats["unique_words"], 1)
        self.assertAlmostEqual(feats["type_token_ratio"], 0.2, places=4)

    def test_generate_explanation_bilingual(self):
        """Verifies bilingual generation produces 2-4 plain-language bullets with expected terminology."""
        text = "Қорытындылай келе, айта кету керек, маңызды рөл атқарады."
        chunk = DocumentChunk(0, text, 0, len(text), 7, 1, ai_probability=0.9995, is_ai=True)
        res = DocumentAnalysisResult("Machine-Generated", 0.9995, 1.0, 0.9980, 7, 1, 1, chunk, [chunk])

        # Kazakh output
        bullets_kz = generate_linguistic_explanation(text, res, lang="kz")
        self.assertGreaterEqual(len(bullets_kz), 2)
        self.assertLessEqual(len(bullets_kz), 4)
        self.assertTrue(
            any("формулалық" in b.lower() or "дискурстық" in b.lower() for b in bullets_kz),
            f"Expected formulaic/discourse marker mention in: {bullets_kz}"
        )

        # English output
        bullets_en = generate_linguistic_explanation(text, res, lang="en")
        self.assertGreaterEqual(len(bullets_en), 2)
        self.assertLessEqual(len(bullets_en), 4)
        self.assertTrue(
            any("discourse" in b.lower() or "formulaic" in b.lower() for b in bullets_en),
            f"Expected discourse/formulaic marker mention in: {bullets_en}"
        )

    def test_generate_explanation_human_verdict(self):
        """Generates evidence-backed reasoning bullets for authentic human text."""
        text = "Бұл керемет өнім, маған өте қатты ұнады, бағасы да арзан."
        chunk = DocumentChunk(0, text, 0, len(text), 9, 1, ai_probability=0.03, is_ai=False)
        res = DocumentAnalysisResult("Authentic Human", 0.03, 0.0, 0.9980, 9, 1, 1, chunk, [chunk])

        bullets_kz = generate_linguistic_explanation(text, res, lang="kz")
        self.assertGreaterEqual(len(bullets_kz), 2)
        self.assertLessEqual(len(bullets_kz), 4)

        bullets_en = generate_linguistic_explanation(text, res, lang="en")
        self.assertGreaterEqual(len(bullets_en), 2)
        self.assertLessEqual(len(bullets_en), 4)

    def test_generate_explanation_hybrid_verdict(self):
        """Generates reasoning bullets for partially AI / hybrid text."""
        text = "Бұл керемет жаңалық. Қорытындылай келе, жүйені толық зерделеу қажет."
        chunk1 = DocumentChunk(0, text[:20], 0, 20, 3, 1, ai_probability=0.02, is_ai=False)
        chunk2 = DocumentChunk(1, text[20:], 20, len(text), 8, 1, ai_probability=0.9992, is_ai=True)
        res = DocumentAnalysisResult("Partially AI / Hybrid", 0.9992, 0.5, 0.9980, 11, 2, 2, chunk2, [chunk1, chunk2])

        bullets_kz = generate_linguistic_explanation(text, res, lang="kz")
        self.assertGreaterEqual(len(bullets_kz), 2)
        self.assertLessEqual(len(bullets_kz), 4)

        bullets_en = generate_linguistic_explanation(text, res, lang="en")
        self.assertGreaterEqual(len(bullets_en), 2)
        self.assertLessEqual(len(bullets_en), 4)

    def test_generate_explanation_no_doc_result(self):
        """Handles invocation when doc_result is None gracefully."""
        text = "Қорытындылай келе, ғылыми зерттеу нәтижелері айтарлықтай маңызды рөл атқарады."
        bullets = generate_linguistic_explanation(text, None, lang="kz")
        self.assertGreaterEqual(len(bullets), 2)
        self.assertLessEqual(len(bullets), 4)

    def test_generate_explanation_empty_text(self):
        """Defensive handling when input text is empty or whitespace."""
        for empty_text in ["", "   ", None]:
            bullets_kz = generate_linguistic_explanation(empty_text, None, lang="kz")
            self.assertGreaterEqual(len(bullets_kz), 2)
            self.assertLessEqual(len(bullets_kz), 4)

            bullets_en = generate_linguistic_explanation(empty_text, None, lang="en")
            self.assertGreaterEqual(len(bullets_en), 2)
            self.assertLessEqual(len(bullets_en), 4)

    def test_generate_explanation_dict_doc_result(self):
        """Accepts plain dictionary for doc_result without failing."""
        text = "Бұл қалыпты қазақша сөйлем."
        dict_res = {
            "verdict": "Authentic Human",
            "document_ai_probability": 0.05,
            "ai_content_ratio": 0.0,
        }
        bullets = generate_linguistic_explanation(text, dict_res, lang="en")
        self.assertGreaterEqual(len(bullets), 2)
        self.assertLessEqual(len(bullets), 4)

    def test_no_decorative_emojis_in_explanations(self):
        """Enforces zero decorative emojis constraint across various texts and settings."""
        sample_texts = [
            "Қорытындылай келе, бұл құбылыс заманауи қоғам үшін ерекше маңызды рөл атқарады.",
            "Сапасы өте керемет, сатып алған дүкенімнен тез келді.",
            "Қазақстанның тарихы тереңде жатыр.",
            "",
        ]

        for text in sample_texts:
            for lang in ["kz", "en"]:
                bullets = generate_linguistic_explanation(text, None, lang=lang)
                for b in bullets:
                    self.assertFalse(
                        contains_emoji(b),
                        f"Found decorative emoji in explanation bullet: {b}"
                    )


if __name__ == "__main__":
    unittest.main()