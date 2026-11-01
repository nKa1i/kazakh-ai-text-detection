# -*- coding: utf-8 -*-
"""
tests/test_run_threads_sentinel.py: Unit tests for ThreadsSentinel pipeline runner CLI.
"""

import json
import os
import tempfile
import unittest

from scripts.run_threads_sentinel import run_sentinel_pipeline


class TestRunThreadsSentinel(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.staged_dir = os.path.join(self.tmpdir.name, "staged")
        os.makedirs(self.staged_dir, exist_ok=True)
        self.lexicon_path = os.path.join(self.tmpdir.name, "discovered_lexicon.json")
        self.audit_path = os.path.join(self.tmpdir.name, "sentinel_audit.json")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _create_batch_file(self, filename: str, verified: bool, items: list) -> str:
        filepath = os.path.join(self.staged_dir, filename)
        data = {
            "schema_version": "1.0",
            "batch_id": filename.replace(".json", ""),
            "harvested_at": "2026-11-01T00:00:00Z",
            "verified": verified,
            "items": items,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filepath

    def test_single_verified_batch_pipeline(self):
        items = [
            {
                "comment_id": "c1",
                "post_id": "p1",
                "text_anonymized": "Кеше каспиден жаңа кітап сатып алдым, керемет.",
                "verified": True,
                "discarded": False,
            },
            {
                "comment_id": "c2",
                "post_id": "p2",
                "text_anonymized": "Бүгін каспиге төлем жасадым, бәрі дұрыс болды.",
                "verified": True,
                "discarded": False,
            },
            {
                "comment_id": "c3",
                "post_id": "p3",
                "text_anonymized": "Осы каспидің қызметі маған қатты ұнайды.",
                "verified": True,
                "discarded": False,
            },
        ]
        batch_path = self._create_batch_file("batch_test_01.json", verified=True, items=items)

        exit_code, summary = run_sentinel_pipeline(
            batch_path=batch_path,
            lexicon_path=self.lexicon_path,
            audit_path=self.audit_path,
            min_posts=3,
            min_affixes=2,
            dry_run=False,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["batches_processed"], 1)
        self.assertEqual(summary["comments_scored"], 3)
        self.assertEqual(summary["human_accepted"], 3)
        self.assertEqual(summary["ai_spam_rejected"], 0)
        self.assertIn("каспи", summary["promoted_stems"])

        # Check persisted artifacts
        self.assertTrue(os.path.isfile(self.lexicon_path))
        self.assertTrue(os.path.isfile(self.audit_path))
        with open(self.lexicon_path, "r", encoding="utf-8") as f:
            lex = json.load(f)
        self.assertIn("каспи", lex["roots"])

    def test_watch_dir_skips_unverified_batches(self):
        verified_items = [
            {
                "comment_id": "cv1",
                "post_id": "pv1",
                "text_anonymized": "Кеше дүкеннен жаңа кітап сатып алдым.",
                "verified": True,
                "discarded": False,
            }
        ]
        unverified_items = [
            {
                "comment_id": "cu1",
                "post_id": "pu1",
                "text_anonymized": "Каспиден тауар алдым.",
                "verified": False,
                "discarded": False,
            }
        ]
        self._create_batch_file("batch_01.json", verified=True, items=verified_items)
        self._create_batch_file("batch_02.json", verified=False, items=unverified_items)

        exit_code, summary = run_sentinel_pipeline(
            watch_dir=self.staged_dir,
            lexicon_path=self.lexicon_path,
            audit_path=self.audit_path,
            dry_run=False,
        )

        self.assertEqual(exit_code, 0)
        # Only the verified batch should be scored
        self.assertEqual(summary["batches_processed"], 1)
        self.assertEqual(summary["skipped_unverified_batches"], 1)
        self.assertEqual(summary["comments_scored"], 1)

    def test_dry_run_does_not_mutate_disk(self):
        items = [
            {
                "comment_id": "cd1",
                "post_id": "pd1",
                "text_anonymized": "Каспиден ақша аудардым.",
                "verified": True,
                "discarded": False,
            }
        ]
        batch_path = self._create_batch_file("batch_dry.json", verified=True, items=items)

        exit_code, summary = run_sentinel_pipeline(
            batch_path=batch_path,
            lexicon_path=self.lexicon_path,
            audit_path=self.audit_path,
            dry_run=True,
        )

        self.assertEqual(exit_code, 0)
        self.assertFalse(os.path.exists(self.lexicon_path))
        self.assertFalse(os.path.exists(self.audit_path))


if __name__ == "__main__":
    unittest.main()
