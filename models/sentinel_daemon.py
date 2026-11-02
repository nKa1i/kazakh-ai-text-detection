# -*- coding: utf-8 -*-
"""
models/sentinel_daemon.py: Background automation daemon for ThreadsSentinel.

Monitors staged batch directories and drop inbox, automatically ingesting
verified Human-in-the-Loop batches and triggering FST lexicon induction.
"""

import datetime
import glob
import json
import os
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

from models.fst_sentinel import AutoFSTSentinel
from src.scrapers.threads_harvester import anonymize_social_text, create_staging_batch


class SentinelDaemon:
    """
    Lightweight background watcher that automates the ThreadsSentinel pipeline.
    """

    def __init__(
        self,
        watch_dir: str = "data/threads_staged",
        inbox_dir: str = "data/sentinel_inbox",
        lexicon_path: str = "data/discovered_lexicon.json",
        audit_path: str = "data/sentinel_audit.json",
        min_posts: int = 3,
        min_affixes: int = 2,
        poll_interval: float = 5.0,
        sentinel: Optional[AutoFSTSentinel] = None,
    ) -> None:
        self.watch_dir = watch_dir
        self.inbox_dir = inbox_dir
        self.lexicon_path = lexicon_path
        self.audit_path = audit_path
        self.min_posts = int(min_posts)
        self.min_affixes = int(min_affixes)
        self.poll_interval = float(poll_interval)

        os.makedirs(self.watch_dir, exist_ok=True)
        os.makedirs(self.inbox_dir, exist_ok=True)

        self.sentinel = sentinel or AutoFSTSentinel(
            lexicon_path=self.lexicon_path,
            audit_path=self.audit_path,
            min_distinct_posts=self.min_posts,
            min_distinct_affixes=self.min_affixes,
        )

    def process_inbox(self) -> List[str]:
        """
        Scans inbox_dir for raw .txt files, anonymizes lines into staged batches,
        and clears the ingested inbox files.

        Returns:
            List of created staged batch file paths.
        """
        created_batches: List[str] = []
        if not os.path.isdir(self.inbox_dir):
            return created_batches

        txt_files = sorted(glob.glob(os.path.join(self.inbox_dir, "*.txt")))
        for filepath in txt_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"[DAEMON WARNING] Failed to read inbox file {filepath}: {e}", file=sys.stderr)
                continue

            if not lines:
                try:
                    os.remove(filepath)
                except OSError:
                    pass
                continue

            items: List[Dict[str, Any]] = []
            base_id = os.path.splitext(os.path.basename(filepath))[0]
            for idx, raw_line in enumerate(lines, start=1):
                clean_text = anonymize_social_text(raw_line)
                items.append(
                    {
                        "comment_id": f"inbox_{base_id}_{idx}",
                        "post_id": f"inbox_{base_id}",
                        "text_anonymized": clean_text,
                        "verified": False,
                        "discarded": False,
                    }
                )

            staged_path, _ = create_staging_batch(
                source_query=f"inbox://{os.path.basename(filepath)}",
                raw_items=items,
                output_dir=self.watch_dir,
            )
            created_batches.append(staged_path)

            try:
                os.remove(filepath)
            except OSError as e:
                print(f"[DAEMON WARNING] Failed to remove processed inbox file {filepath}: {e}", file=sys.stderr)

        return created_batches

    def run_once(self) -> Dict[str, Any]:
        """
        Executes a single sweep over staged batch directory and inbox.

        Returns:
            Summary dictionary of the execution sweep.
        """
        # Process any raw text inbox drops first
        self.process_inbox()

        batches_processed = 0
        unverified_batches_waiting = 0
        comments_scored = 0
        ai_spam_rejected = 0
        human_accepted = 0
        promoted_stems: List[str] = []
        guard_rejections: List[Any] = []

        if not os.path.isdir(self.watch_dir):
            return {
                "batches_processed": 0,
                "unverified_batches_waiting": 0,
                "comments_scored": 0,
                "ai_spam_rejected": 0,
                "human_accepted": 0,
                "promoted_stems": [],
                "pending_buffer_size": len(self.sentinel.buffer),
                "guard_rejections": [],
            }

        batch_files = sorted(glob.glob(os.path.join(self.watch_dir, "batch_*.json")))
        for filepath in batch_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    batch_data = json.load(f)
            except Exception as e:
                print(f"[DAEMON WARNING] Failed to read batch {filepath}: {e}", file=sys.stderr)
                continue

            # Skip already ingested batches
            if batch_data.get("sentinel_processed", False):
                continue

            # Check verification gate
            if not batch_data.get("verified", False):
                unverified_batches_waiting += 1
                continue

            # Process verified batch
            stats = self.sentinel.process_batch(batch_data)
            batches_processed += 1
            comments_scored += stats.comments_scored
            ai_spam_rejected += stats.ai_spam_rejected
            human_accepted += stats.human_accepted
            promoted_stems.extend(stats.promoted_stems)
            guard_rejections.extend(stats.guard_rejections)

            # Mark batch as processed atomically
            batch_data["sentinel_processed"] = True
            batch_data["processed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=os.path.dirname(filepath), prefix="batch_proc_", suffix=".json"
            )
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, filepath)

        return {
            "batches_processed": batches_processed,
            "unverified_batches_waiting": unverified_batches_waiting,
            "comments_scored": comments_scored,
            "ai_spam_rejected": ai_spam_rejected,
            "human_accepted": human_accepted,
            "promoted_stems": promoted_stems,
            "pending_buffer_size": len(self.sentinel.buffer),
            "guard_rejections": guard_rejections,
        }

    def run_loop(
        self,
        stop_event: Optional[threading.Event] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        """
        Runs the continuous daemon polling loop until stop_event is set or interrupted.

        Args:
            stop_event: Optional threading event to signal loop termination.
            max_iterations: Optional limit on loop iterations (useful for tests).
        """
        print(f"[DAEMON START] Watching '{self.watch_dir}' and inbox '{self.inbox_dir}'")
        print(f"[DAEMON CONFIG] Poll interval: {self.poll_interval}s, Min posts: {self.min_posts}, Min affixes: {self.min_affixes}")

        iterations = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            if max_iterations is not None and iterations >= max_iterations:
                break

            try:
                summary = self.run_once()
                if summary["batches_processed"] > 0:
                    print(
                        f"[DAEMON EVENT] Processed {summary['batches_processed']} batch(es) | "
                        f"Comments: {summary['comments_scored']} | "
                        f"AI Spam Blocked: {summary['ai_spam_rejected']} | "
                        f"Human Accepted: {summary['human_accepted']} | "
                        f"Promoted: {summary['promoted_stems']}"
                    )
            except Exception as e:
                print(f"[DAEMON ERROR] Unexpected loop exception: {e}", file=sys.stderr)

            iterations += 1
            if stop_event is not None:
                stop_event.wait(self.poll_interval)
            else:
                try:
                    time.sleep(self.poll_interval)
                except KeyboardInterrupt:
                    print("\n[DAEMON STOP] Interrupted by user. Exiting cleanly.")
                    break
