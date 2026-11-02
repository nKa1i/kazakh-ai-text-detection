# -*- coding: utf-8 -*-
"""
tests/test_sentinel_daemon.py: Unit tests for ThreadsSentinel background automation daemon.
"""

import json
import os
import tempfile
import threading
import time
import unittest

from models.sentinel_daemon import SentinelDaemon


class TestSentinelDaemon(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.watch_dir = os.path.join(self.tmpdir.name, "staged")
        self.inbox_dir = os.path.join(self.tmpdir.name, "inbox")
        os.makedirs(self.watch_dir, exist_ok=True)
        os.makedirs(self.inbox_dir, exist_ok=True)

        self.lexicon_path = os.path.join(self.tmpdir.name, "discovered_lexicon.json")
        self.audit_path = os.path.join(self.tmpdir.name, "sentinel_audit.json")

        self.daemon = SentinelDaemon(
            watch_dir=self.watch_dir,
            inbox_dir=self.inbox_dir,
            lexicon_path=self.lexicon_path,
            audit_path=self.audit_path,
            min_posts=3,
            min_affixes=2,
            poll_interval=0.1,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def _create_staged_batch(self, filename: str, verified: bool, items: list, processed: bool = False) -> str:
        filepath = os.path.join(self.watch_dir, filename)
        data = {
            "schema_version": "1.0",
            "batch_id": filename.replace(".json", ""),
            "harvested_at": "2026-11-02T00:00:00Z",
            "verified": verified,
            "sentinel_processed": processed,
            "items": items,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filepath

    def test_daemon_run_once_processes_verified_batch(self):
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
                "text_anonymized": "Бүгін каспиге төлем жасадым, рахмет сізге.",
                "verified": True,
                "discarded": False,
            },
            {
                "comment_id": "c3",
                "post_id": "p3",
                "text_anonymized": "Осы каспидің қосымшасы өте ыңғайлы екен.",
                "verified": True,
                "discarded": False,
            },
        ]
        batch_path = self._create_staged_batch("batch_test_01.json", verified=True, items=items)

        # First run_once: should process and mark sentinel_processed
        summary = self.daemon.run_once()
        self.assertEqual(summary["batches_processed"], 1)
        self.assertEqual(summary["comments_scored"], 3)
        self.assertIn("каспи", summary["promoted_stems"])

        with open(batch_path, "r", encoding="utf-8") as f:
            updated_batch = json.load(f)
        self.assertTrue(updated_batch.get("sentinel_processed", False))
        self.assertIn("processed_at", updated_batch)

        # Second run_once: batch is already processed, should skip it
        summary_2 = self.daemon.run_once()
        self.assertEqual(summary_2["batches_processed"], 0)
        self.assertEqual(summary_2["comments_scored"], 0)

    def test_daemon_skips_unverified_batches(self):
        items = [
            {
                "comment_id": "c_unv",
                "post_id": "p_unv",
                "text_anonymized": "Каспиден тауар алдым.",
                "verified": False,
                "discarded": False,
            }
        ]
        batch_path = self._create_staged_batch("batch_unv.json", verified=False, items=items)

        summary = self.daemon.run_once()
        self.assertEqual(summary["batches_processed"], 0)
        self.assertEqual(summary["unverified_batches_waiting"], 1)

        with open(batch_path, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
        self.assertFalse(batch_data.get("sentinel_processed", False))

    def test_daemon_inbox_ingestion_creates_staged_batch(self):
        # Drop raw text comments into inbox
        inbox_file = os.path.join(self.inbox_dir, "new_comments.txt")
        raw_text = (
            "Кеше каспиден зат алдым номерім +7 701 123 4567 жазды.\n"
            "Ватсапқа жіберіңіз http://example.com/item\n"
        )
        with open(inbox_file, "w", encoding="utf-8") as f:
            f.write(raw_text)

        created_batches = self.daemon.process_inbox()
        self.assertEqual(len(created_batches), 1)
        staged_path = created_batches[0]
        self.assertTrue(os.path.isfile(staged_path))

        with open(staged_path, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
        self.assertFalse(batch_data["verified"])
        self.assertEqual(len(batch_data["items"]), 2)
        # Check PII anonymization in staged batch
        self.assertIn("[PHONE]", batch_data["items"][0]["text_anonymized"])
        self.assertIn("[URL]", batch_data["items"][1]["text_anonymized"])

        # Inbox file should be archived or removed
        self.assertFalse(os.path.isfile(inbox_file))

    def test_daemon_thread_lifecycle(self):
        # Test start and clean stop
        stop_event = threading.Event()
        worker = threading.Thread(target=self.daemon.run_loop, kwargs={"stop_event": stop_event})
        worker.start()
        time.sleep(0.15)
        self.assertTrue(worker.is_alive())
        stop_event.set()
        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
