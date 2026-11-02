# -*- coding: utf-8 -*-
"""
scripts/run_sentinel_daemon.py: CLI entry point for ThreadsSentinel background automation daemon.

Continuously watches data/threads_staged/ and data/sentinel_inbox/ for verified
comments and automatically ingests valid Kazakh stems into the dynamic FST lexicon.
"""

import argparse
import sys

from models.sentinel_daemon import SentinelDaemon


def main() -> int:
    """CLI entry point for running the background sentinel daemon."""
    parser = argparse.ArgumentParser(
        description="ThreadsSentinel: Background Directory Watcher & Auto-FST Lexicon Ingestion Daemon"
    )
    parser.add_argument(
        "--watch-dir",
        type=str,
        default="data/threads_staged",
        help="Directory containing staged batch JSON files (default: data/threads_staged).",
    )
    parser.add_argument(
        "--inbox-dir",
        type=str,
        default="data/sentinel_inbox",
        help="Directory for dropping raw .txt comment files (default: data/sentinel_inbox).",
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
        help="Path to sentinel audit log JSON (default: data/sentinel_audit.json).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--min-posts",
        type=int,
        default=3,
        help="Minimum distinct post appearances required for stem promotion (default: 3).",
    )
    parser.add_argument(
        "--min-affixes",
        type=int,
        default=2,
        help="Minimum distinct grammatical affixes required for stem promotion (default: 2).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Execute a single sweep and exit immediately.",
    )

    args = parser.parse_args()

    daemon = SentinelDaemon(
        watch_dir=args.watch_dir,
        inbox_dir=args.inbox_dir,
        lexicon_path=args.lexicon_path,
        audit_path=args.audit_path,
        min_posts=args.min_posts,
        min_affixes=args.min_affixes,
        poll_interval=args.poll_interval,
    )

    if args.once:
        summary = daemon.run_once()
        print("=" * 60)
        print("           THREADS SENTINEL DAEMON SWEEP (ONCE)")
        print("=" * 60)
        print(f"Batches Processed:          {summary['batches_processed']}")
        print(f"Unverified Waiting:         {summary['unverified_batches_waiting']}")
        print(f"Comments Scored:            {summary['comments_scored']}")
        print(f"AI Spam Blocked:            {summary['ai_spam_rejected']}")
        print(f"Human Accepted:             {summary['human_accepted']}")
        print(f"Novel Stems Promoted:       {len(summary['promoted_stems'])} {summary['promoted_stems']}")
        print(f"Pending Candidate Buffer:   {summary['pending_buffer_size']}")
        print("=" * 60)
        return 0

    daemon.run_loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
