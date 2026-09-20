# -*- coding: utf-8 -*-
"""
tests/test_build_social_media_benchmark.py: Unit tests for Kazakh social media benchmark tooling.
"""

import os
import json
import tempfile
import unittest

from scripts.build_social_media_benchmark import (
    anonymize_social_text,
    generate_social_media_prompt,
    validate_social_record,
    main,
)


class TestBuildSocialMediaBenchmark(unittest.TestCase):
    """Unit test suite for social media benchmark anonymization, generation, and validation."""

    def test_anonymize_social_text(self):
        """Asserts @username, URLs, and phone numbers are correctly masked and whitespace cleaned."""
        raw_text = (
            "Сәлем @daulet_kz және @almaty_user! Мына сілтемені көр: "
            "https://t.me/kaz_news немесе http://test.kz/item?id=42. "
            "Сұрақтар болса +77011234567 немесе 8 (707) 123-45-67 нөміріне хабарласыңыз.   "
            "Рақмет!   \n\n\n  Күтеміз.   "
        )
        anonymized = anonymize_social_text(raw_text)

        self.assertNotIn("@daulet_kz", anonymized)
        self.assertNotIn("@almaty_user", anonymized)
        self.assertIn("@user_anon", anonymized)

        self.assertNotIn("https://t.me/kaz_news", anonymized)
        self.assertNotIn("http://test.kz/item?id=42", anonymized)
        self.assertIn("[URL]", anonymized)

        self.assertNotIn("+77011234567", anonymized)
        self.assertNotIn("8 (707) 123-45-67", anonymized)
        self.assertIn("[PHONE]", anonymized)

        self.assertNotIn("   ", anonymized)
        self.assertNotIn("\n\n\n", anonymized)

    def test_generate_social_media_prompt(self):
        """Asserts prompt contains slang, code-switching, and persona instructions."""
        prompt = generate_social_media_prompt(genre="telegram", persona="student")

        self.assertIn("telegram", prompt.lower())
        self.assertIn("student", prompt.lower())

        # Slang and conversational suffixes
        self.assertTrue(
            "-сың ғой" in prompt or "-ма екен" in prompt or "-шы" in prompt or "-ау" in prompt,
            "Prompt must contain colloquial Kazakh suffixes"
        )

        # Code-switching
        self.assertTrue(
            "донаттау" in prompt or "хайптану" in prompt or "краш" in prompt,
            "Prompt must contain code-switching examples"
        )

        # Informal orthography instructions (typing к for қ, г for ғ)
        self.assertTrue(
            "к" in prompt and "қ" in prompt and "г" in prompt and "ғ" in prompt,
            "Prompt must mention informal orthography"
        )

    def test_validate_social_record_valid(self):
        """Asserts valid records pass schema, platform, and semantic validation."""
        valid_record = {
            "id": "soc_0001",
            "text": "Сәлем достар, бүгінгі сабақ қалай өтті? @user_anon айтқандай бәрі керемет болды ғой!",
            "label": "human",
            "platform": "telegram",
        }
        self.assertTrue(validate_social_record(valid_record))

        valid_ai_record = {
            "id": "soc_0002",
            "text": "Бұл жаңа құрылғы шынымен де өте ыңғайлы екен, [URL] сілтемесі арқылы толық қарап шығуға болады.",
            "label": "ai",
            "platform": "twitter",
        }
        self.assertTrue(validate_social_record(valid_ai_record))

    def test_validate_social_record_invalid_label(self):
        """Asserts invalid labels or missing/invalid platform fail validation."""
        rec_invalid_label = {
            "id": "soc_0003",
            "text": "Бұл жазбада кемінде он сөз болуы керек екенін жақсы түсініп отырмыз достар.",
            "label": "bot",
            "platform": "telegram",
        }
        self.assertFalse(validate_social_record(rec_invalid_label))

        rec_invalid_platform = {
            "id": "soc_0004",
            "text": "Бұл жазбада кемінде он сөз болуы керек екенін жақсы түсініп отырмыз достар.",
            "label": "human",
            "platform": "facebook",
        }
        self.assertFalse(validate_social_record(rec_invalid_platform))

        rec_missing_fields = {
            "id": "soc_0005",
            "text": "Бұл жазбада кемінде он сөз болуы керек екенін жақсы түсініп отырмыз достар.",
        }
        self.assertFalse(validate_social_record(rec_missing_fields))

    def test_validate_social_record_length_bounds(self):
        """Asserts < 10 or > 120 words fail validation."""
        rec_short = {
            "id": "soc_0006",
            "text": "Бұл өте қысқа жазба болып саналады.",
            "label": "human",
            "platform": "twitter",
        }
        self.assertFalse(validate_social_record(rec_short))

        words_9 = ["сөз"] * 9
        rec_9 = {
            "id": "soc_0007",
            "text": " ".join(words_9),
            "label": "human",
            "platform": "twitter",
        }
        self.assertFalse(validate_social_record(rec_9))

        words_10 = ["сөз"] * 10
        rec_10 = {
            "id": "soc_0008",
            "text": " ".join(words_10),
            "label": "human",
            "platform": "twitter",
        }
        self.assertTrue(validate_social_record(rec_10))

        words_120 = ["сөз"] * 120
        rec_120 = {
            "id": "soc_0009",
            "text": " ".join(words_120),
            "label": "ai",
            "platform": "forum",
        }
        self.assertTrue(validate_social_record(rec_120))

        words_121 = ["сөз"] * 121
        rec_121 = {
            "id": "soc_0010",
            "text": " ".join(words_121),
            "label": "ai",
            "platform": "forum",
        }
        self.assertFalse(validate_social_record(rec_121))

    def test_validate_social_record_leaked_pii(self):
        """Asserts records containing unmasked handles, URLs, or phone numbers fail."""
        rec_leaked_handle = {
            "id": "soc_0011",
            "text": "Сәлем @daulet_kz мына жазбаны қарап шықшы, пікірің өте маңызды болып тұр.",
            "label": "human",
            "platform": "telegram",
        }
        self.assertFalse(validate_social_record(rec_leaked_handle))

        rec_leaked_url = {
            "id": "soc_0012",
            "text": "Барлық мәліметтер мына жерде қолжетімді: https://example.com/data толық оқып шығыңыздар.",
            "label": "human",
            "platform": "telegram",
        }
        self.assertFalse(validate_social_record(rec_leaked_url))

        rec_leaked_phone = {
            "id": "soc_0013",
            "text": "Егер сұрақтарыңыз болса +77011234567 нөміріне дереу хабарласуға болады достар.",
            "label": "human",
            "platform": "telegram",
        }
        self.assertFalse(validate_social_record(rec_leaked_phone))

    def test_dry_run_generation(self):
        """Asserts --dry-run produces valid records with balanced labels and platforms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "social_benchmark.jsonl")
            exit_code = main(["--dry-run", "--output", out_file])
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(out_file))

            with open(out_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line.strip()) for line in f if line.strip()]

            self.assertGreaterEqual(len(lines), 20)
            labels = set()
            platforms = set()
            for rec in lines:
                self.assertTrue(validate_social_record(rec), f"Failed validation: {rec}")
                labels.add(rec["label"])
                platforms.add(rec["platform"])

            self.assertEqual(labels, {"human", "ai"})
            self.assertEqual(platforms, {"telegram", "twitter", "forum"})


if __name__ == "__main__":
    unittest.main()
