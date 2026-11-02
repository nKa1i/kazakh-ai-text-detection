# -*- coding: utf-8 -*-
"""
scripts/run_threads_sentinel.py: Command-line runner for ThreadsSentinel pipeline.

Executes continuous AI spam filtering, HITL batch verification checks, and
phonotactically validated root stem induction into the dynamic FST lexicon.
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.fst_sentinel import AutoFSTSentinel, SentinelRunStats
from src.scrapers.threads_harvester import (
    harvest_threads_url,
    harvest_threads_search,
    harvest_threads_user_posts,
    harvest_threads_bulk,
)


def format_summary_table(summary: Dict[str, Any]) -> str:
    """Formats pipeline run statistics as an ASCII text table without emojis."""
    lines = [
        "=" * 60,
        "                THREADS SENTINEL AUDIT REPORT",
        "=" * 60,
        f"Batches Processed:          {summary['batches_processed']}",
        f"Skipped (Unverified):       {summary['skipped_unverified_batches']}",
        f"Comments Scored:            {summary['comments_scored']}",
        f"AI Spam Blocked:            {summary['ai_spam_rejected']}",
        f"Human Comments Accepted:    {summary['human_accepted']}",
        f"Novel Stems Promoted:       {len(summary['promoted_stems'])} {summary['promoted_stems']}",
        f"Pending Candidate Buffer:   {summary['pending_buffer_size']}",
        f"Regression Guard Blocks:    {len(summary['guard_rejections'])}",
        "=" * 60,
    ]
    return "\n".join(lines)


def run_sentinel_pipeline(
    batch_path: Optional[str] = None,
    watch_dir: Optional[str] = None,
    url: Optional[str] = None,
    search_query: Optional[str] = None,
    user: Optional[str] = None,
    limit: int = 15,
    lexicon_path: str = "data/discovered_lexicon.json",
    audit_path: str = "data/sentinel_audit.json",
    min_posts: int = 3,
    min_affixes: int = 2,
    dry_run: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    """
    Runs the Threads Sentinel pipeline over a single batch, watched directory, or query.

    Args:
        batch_path: Specific staged batch JSON file to process.
        watch_dir: Directory containing staged batch JSON files.
        url: Optional Threads URL to harvest immediately.
        search_query: Optional search keyword to discover and harvest posts in bulk.
        user: Optional Threads username to discover and harvest recent posts.
        limit: Max posts to crawl when using search or user discovery.
        lexicon_path: File path to save/load dynamic FST discovered lexicon.
        audit_path: File path to save/load sentinel run audit log.
        min_posts: Minimum distinct post appearances required for promotion.
        min_affixes: Minimum distinct grammatical affixes required for promotion.
        dry_run: If True, executes evaluation without persisting updates.

    Returns:
        Tuple of (exit_code, summary_dict).
    """
    target_dir = watch_dir or "data/threads_staged"

    # 1. Bulk keyword search harvest if specified
    if search_query:
        print(f"[SEARCH CRAWLER] Searching Threads for '{search_query}' (limit={limit})...")
        res = harvest_threads_search(query=search_query, limit=limit, output_dir=target_dir)
        if res.success and res.batch_file_path:
            print(f"[SEARCH HARVESTED] Discovered posts staged to: {res.batch_file_path}")
            print(f"[COMMENTS] Captured {res.comments_count} total comments (non-Kazakh auto-filtered).")
            print("[NOTICE] Staged batch requires HITL verification before ingestion:")
            print(f"         python scripts/review_threads_batch.py {res.batch_file_path}")
        else:
            print(f"[SEARCH FAILED] {res.error_message}", file=sys.stderr)
        if not batch_path and not watch_dir:
            return 0, {
                "batches_processed": 0,
                "skipped_unverified_batches": 1 if res.success else 0,
                "comments_scored": 0,
                "ai_spam_rejected": 0,
                "human_accepted": 0,
                "promoted_stems": [],
                "pending_buffer_size": 0,
                "guard_rejections": [],
            }

    # 2. Bulk user timeline harvest if specified
    if user:
        clean_user = user.strip().lstrip("@")
        print(f"[USER CRAWLER] Crawling recent posts for @{clean_user} (limit={limit})...")
        discovered_urls = harvest_threads_user_posts(clean_user, limit=limit)
        if discovered_urls:
            res = harvest_threads_bulk(discovered_urls, output_dir=target_dir)
            if res.success and res.batch_file_path:
                print(f"[USER HARVESTED] Crawled {len(discovered_urls)} posts, staged to: {res.batch_file_path}")
                print(f"[COMMENTS] Captured {res.comments_count} total comments (non-Kazakh auto-filtered).")
                print("[NOTICE] Staged batch requires HITL verification before ingestion:")
                print(f"         python scripts/review_threads_batch.py {res.batch_file_path}")
            else:
                print(f"[USER HARVEST FAILED] {res.error_message}", file=sys.stderr)
        else:
            print(f"[USER HARVEST] No recent public posts found for @{clean_user}", file=sys.stderr)
        if not batch_path and not watch_dir:
            return 0, {
                "batches_processed": 0,
                "skipped_unverified_batches": 1 if (user and discovered_urls) else 0,
                "comments_scored": 0,
                "ai_spam_rejected": 0,
                "human_accepted": 0,
                "promoted_stems": [],
                "pending_buffer_size": 0,
                "guard_rejections": [],
            }

    # 3. Harvest single URL if specified
    if url:
        res = harvest_threads_url(url, output_dir=target_dir)
        if res.success and res.batch_file_path:
            print(f"[HARVESTED] URL staged to {res.batch_file_path}")
            print("[NOTICE] Staged batch requires HITL verification before ingestion:")
            print(f"         python scripts/review_threads_batch.py {res.batch_file_path}")
        else:
            print(f"[HARVEST FAILED] {res.error_message}", file=sys.stderr)
        if not batch_path and not watch_dir:
            return 0, {
                "batches_processed": 0,
                "skipped_unverified_batches": 1 if res.success else 0,
                "comments_scored": 0,
                "ai_spam_rejected": 0,
                "human_accepted": 0,
                "promoted_stems": [],
                "pending_buffer_size": 0,
                "guard_rejections": [],
            }

    # 2. Determine batch files to process
    batch_files: List[str] = []
    if batch_path:
        if not os.path.isfile(batch_path):
            print(f"[ERROR] Batch file not found: {batch_path}", file=sys.stderr)
            return 1, {
                "batches_processed": 0,
                "skipped_unverified_batches": 0,
                "comments_scored": 0,
                "ai_spam_rejected": 0,
                "human_accepted": 0,
                "promoted_stems": [],
                "pending_buffer_size": 0,
                "guard_rejections": [],
            }
        batch_files.append(batch_path)
    else:
        search_dir = watch_dir or "data/threads_staged"
        if os.path.isdir(search_dir):
            batch_files = sorted(glob.glob(os.path.join(search_dir, "batch_*.json")))

    if not batch_files:
        print("[INFO] No batch files found to process.")
        return 0, {
            "batches_processed": 0,
            "skipped_unverified_batches": 0,
            "comments_scored": 0,
            "ai_spam_rejected": 0,
            "human_accepted": 0,
            "promoted_stems": [],
            "pending_buffer_size": 0,
            "guard_rejections": [],
        }

    # 3. Initialize Sentinel
    sentinel = AutoFSTSentinel(
        lexicon_path=lexicon_path,
        audit_path=audit_path,
        min_distinct_posts=min_posts,
        min_distinct_affixes=min_affixes,
        dry_run=dry_run,
    )

    batches_processed = 0
    skipped_unverified_batches = 0
    comments_scored = 0
    ai_spam_rejected = 0
    human_accepted = 0
    promoted_stems: List[str] = []
    guard_rejections: List[Tuple[str, str]] = []

    # 4. Iterate over batch files
    for filepath in batch_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                batch_data = json.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to load {filepath}: {e}", file=sys.stderr)
            continue

        if not batch_data.get("verified", False):
            batch_id = batch_data.get("batch_id", os.path.basename(filepath))
            print(f"[SKIP] Batch {batch_id} is unverified. Run review_threads_batch.py first.")
            skipped_unverified_batches += 1
            continue

        stats = sentinel.process_batch(batch_data)
        batches_processed += stats.batches_processed
        comments_scored += stats.comments_scored
        ai_spam_rejected += stats.ai_spam_rejected
        human_accepted += stats.human_accepted
        promoted_stems.extend(stats.promoted_stems)
        guard_rejections.extend(stats.guard_rejections)

    summary = {
        "batches_processed": batches_processed,
        "skipped_unverified_batches": skipped_unverified_batches,
        "comments_scored": comments_scored,
        "ai_spam_rejected": ai_spam_rejected,
        "human_accepted": human_accepted,
        "promoted_stems": promoted_stems,
        "pending_buffer_size": len(sentinel.buffer),
        "guard_rejections": guard_rejections,
    }

    report = format_summary_table(summary)
    print(report)

    if dry_run:
        print("[DRY-RUN] No changes were written to disk.")

    return 0, summary


def main() -> int:
    """CLI entrypoint for ThreadsSentinel runner."""
    parser = argparse.ArgumentParser(
        description="ThreadsSentinel: Autonomous Kazakh AI Spam Gate and FST Lexicon Inducer"
    )
    parser.add_argument(
        "--batch-path",
        type=str,
        default=None,
        help="Path to specific staged batch JSON file.",
    )
    parser.add_argument(
        "--watch-dir",
        type=str,
        default=None,
        help="Directory containing staged batch JSON files (default: data/threads_staged).",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional Threads URL to harvest immediately.",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Search query to discover and crawl Kazakh posts in bulk (e.g. 'қазақша', 'каспи').",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="Threads username to crawl recent posts in bulk (e.g. 'tengrinewskz').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Maximum posts to crawl for search or user discovery (default: 15).",
    )
    parser.add_argument(
        "--lexicon-path",
        type=str,
        default="data/discovered_lexicon.json",
        help="Path to discovered lexicon JSON (default: data/discovered_lexicon.json).",
    )
    parser.add_argument(
        "--audit-path",
        type=str,
        default="data/sentinel_audit.json",
        help="Path to sentinel audit JSON (default: data/sentinel_audit.json).",
    )
    parser.add_argument(
        "--min-posts",
        type=int,
        default=3,
        help="Minimum distinct post appearances required for promotion (default: 3).",
    )
    parser.add_argument(
        "--min-affixes",
        type=int,
        default=2,
        help="Minimum distinct grammatical affixes required for promotion (default: 2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute pipeline without modifying lexicon or audit files.",
    )

    args = parser.parse_args()
    exit_code, _ = run_sentinel_pipeline(
        batch_path=args.batch_path,
        watch_dir=args.watch_dir,
        url=args.url,
        search_query=args.search,
        user=args.user,
        limit=args.limit,
        lexicon_path=args.lexicon_path,
        audit_path=args.audit_path,
        min_posts=args.min_posts,
        min_affixes=args.min_affixes,
        dry_run=args.dry_run,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
