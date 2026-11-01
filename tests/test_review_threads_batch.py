# -*- coding: utf-8 -*-
"""
tests/test_review_threads_batch.py: Unit tests for Human-in-the-Loop Threads review CLI.
"""

import json
import os
import tempfile
import unittest

from scripts.review_threads_batch import (
    load_batch,
    save_batch_atomic,
    triage_batch_non_interactive,
    get_batch_summary,
)


class TestReviewThreadsBatch(unittest.TestCase):
    def setUp(self):
        self.sample_batch = {
            "schema_version": "1.0",
            "batch_id": "batch_20261101T003500Z",
            "source_query": "https://www.threads.net/@user/post/C_abc123",
            "harvested_at": "2026-11-01T00:35:00Z",
            "verified": False,
            "items": [
                {
                    "comment_id": "th_1",
                    "post_id": "C_abc123",
                    "text_anonymized": "Каспиден скрин жасап жібердім @user_anon",
                    "verified": False,
                    "discarded": False,
                    "reviewed_at": None,
                },
                {
                    "comment_id": "th_2",
                    "post_id": "C_abc123",
                    "text_anonymized": "Бұл спам мәтін немесе бот хабарламасы [URL]",
                    "verified": False,
                    "discarded": False,
                    "reviewed_at": None,
                },
            ],
        }

    def test_load_and_save_batch_atomic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "batch_test.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.sample_batch, f)

            loaded = load_batch(file_path)
            self.assertEqual(loaded["batch_id"], "batch_20261101T003500Z")
            self.assertFalse(loaded["verified"])

            loaded["verified"] = True
            save_batch_atomic(file_path, loaded)

            with open(file_path, "r", encoding="utf-8") as f:
                reloaded = json.load(f)
            self.assertTrue(reloaded["verified"])

    def test_triage_batch_non_interactive_approve_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "batch_test.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.sample_batch, f)

            updated = triage_batch_non_interactive(
                file_path,
                approve_all=True,
            )
            self.assertTrue(updated["verified"])
            for item in updated["items"]:
                self.assertTrue(item["verified"])
                self.assertFalse(item["discarded"])
                self.assertIsNotNone(item["reviewed_at"])

    def test_triage_batch_discard_specific_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "batch_test.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.sample_batch, f)

            updated = triage_batch_non_interactive(
                file_path,
                discard_ids=["th_2"],
                approve_remaining=True,
            )
            self.assertTrue(updated["verified"])
            # Item 1 approved
            self.assertTrue(updated["items"][0]["verified"])
            self.assertFalse(updated["items"][0]["discarded"])

            # Item 2 discarded
            self.assertFalse(updated["items"][1]["verified"])
            self.assertTrue(updated["items"][1]["discarded"])

    def test_get_batch_summary(self):
        summary = get_batch_summary(self.sample_batch)
        self.assertEqual(summary["total_items"], 2)
        self.assertEqual(summary["verified_count"], 0)
        self.assertEqual(summary["discarded_count"], 0)
        self.assertEqual(summary["pending_count"], 2)


if __name__ == "__main__":
    unittest.main()
