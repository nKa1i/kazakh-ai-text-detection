# -*- coding: utf-8 -*-
"""
tests/test_fst_sentinel.py: Unit tests for AutoFSTSentinel and continuous lexicon induction.
"""

import json
import os
import tempfile
import unittest

from models.fst_sentinel import (
    AutoFSTSentinel,
    SentinelRunStats,
    passes_coda_assimilation,
)


class TestAutoFSTSentinel(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.lexicon_path = os.path.join(self.tmpdir.name, "discovered_lexicon.json")
        self.audit_path = os.path.join(self.tmpdir.name, "sentinel_audit.json")

        self.sentinel = AutoFSTSentinel(
            lexicon_path=self.lexicon_path,
            audit_path=self.audit_path,
            min_distinct_posts=3,
            min_distinct_affixes=2,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_coda_assimilation_rules(self):
        # Voiceless coda + voiceless onset -> valid
        self.assertTrue(passes_coda_assimilation("тапсырыс", "-тан"))
        self.assertTrue(passes_coda_assimilation("скрин", "-деу"))

        # Voiceless coda + voiced onset -> invalid
        self.assertFalse(passes_coda_assimilation("кітап", "-дан"))
        self.assertFalse(passes_coda_assimilation("тапсырыс", "-ден"))

    def test_ai_spam_rejection_at_gate(self):
        batch = {
            "schema_version": "1.0",
            "batch_id": "batch_ai_spam_01",
            "verified": True,
            "items": [
                {
                    "comment_id": "c_ai_1",
                    "post_id": "p_ai_1",
                    "text_anonymized": (
                        "Қорытындылай келе, цифрлық трансформация барлық мемлекеттік "
                        "жүйелер үшін маңызды рөл атқарады және айтарлықтай табыс әкеледі."
                    ),
                    "verified": True,
                    "discarded": False,
                }
            ],
        }
        stats = self.sentinel.process_batch(batch)
        self.assertEqual(stats.ai_spam_rejected, 1)
        self.assertEqual(stats.human_accepted, 0)
        self.assertEqual(len(stats.promoted_stems), 0)

    def test_unverified_or_discarded_items_ignored(self):
        batch = {
            "schema_version": "1.0",
            "batch_id": "batch_unverified",
            "verified": False,
            "items": [
                {
                    "comment_id": "c_unv_1",
                    "post_id": "p_unv_1",
                    "text_anonymized": "Каспиден тауар алдым",
                    "verified": False,
                    "discarded": False,
                },
                {
                    "comment_id": "c_disc_1",
                    "post_id": "p_disc_1",
                    "text_anonymized": "Каспиге ақша салдым",
                    "verified": True,
                    "discarded": True,
                },
            ],
        }
        stats = self.sentinel.process_batch(batch)
        self.assertEqual(stats.comments_scored, 0)
        self.assertEqual(len(self.sentinel.buffer), 0)

    def test_novel_stem_buffer_accumulation_and_promotion(self):
        # 3 distinct human comments across 3 distinct posts introducing "каспи" with diverse affixes
        batch = {
            "schema_version": "1.0",
            "batch_id": "batch_human_kaspi",
            "verified": True,
            "items": [
                {
                    "comment_id": "c_h1",
                    "post_id": "p_h1",
                    "text_anonymized": "Кеше каспиден жаңа тапсырыс бердім, өте жылдам келді.",
                    "verified": True,
                    "discarded": False,
                },
                {
                    "comment_id": "c_h2",
                    "post_id": "p_h2",
                    "text_anonymized": "Бүгін каспиге ақша аудардым, рахмет сізге.",
                    "verified": True,
                    "discarded": False,
                },
                {
                    "comment_id": "c_h3",
                    "post_id": "p_h3",
                    "text_anonymized": "Осы каспидің қосымшасы өте ыңғайлы екен.",
                    "verified": True,
                    "discarded": False,
                },
            ],
        }
        stats = self.sentinel.process_batch(batch)
        self.assertEqual(stats.human_accepted, 3)
        self.assertIn("каспи", stats.promoted_stems)

        # Verify stem is registered in overlay
        self.assertIn("каспи", self.sentinel.overlay.custom_lexicon)

        # Verify persisted JSON artifacts
        self.assertTrue(os.path.isfile(self.lexicon_path))
        with open(self.lexicon_path, "r", encoding="utf-8") as f:
            lex_data = json.load(f)
        self.assertIn("каспи", lex_data["roots"])
        entry = lex_data["roots"]["каспи"]
        self.assertEqual(entry["distinct_posts"], 3)
        self.assertGreaterEqual(entry["suffix_count"], 2)

        self.assertTrue(os.path.isfile(self.audit_path))
        with open(self.audit_path, "r", encoding="utf-8") as f:
            audit_data = json.load(f)
        self.assertIn("promoted_stems", audit_data)
        self.assertIn("каспи", audit_data["promoted_stems"])


if __name__ == "__main__":
    unittest.main()
