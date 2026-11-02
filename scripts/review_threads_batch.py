# -*- coding: utf-8 -*-
"""
scripts/review_threads_batch.py: Human-in-the-Loop triage CLI for staged Threads batches.
Allows reviewing, approving, and discarding comments before sending to Auto-FST Sentinel.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional


def load_batch(file_path: str) -> Dict[str, Any]:
    """Loads a staged Threads batch JSON file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Staged batch file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_batch_atomic(file_path: str, data: Dict[str, Any]) -> None:
    """Atomically writes batch data using a temporary file replacement."""
    target_dir = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(target_dir, exist_ok=True)

    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=target_dir, delete=False, encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            tmp_file = f.name
        os.replace(tmp_file, file_path)
    finally:
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass


def get_batch_summary(data: Dict[str, Any]) -> Dict[str, int]:
    """Computes triage status counts for a batch."""
    items = data.get("items", [])
    verified = sum(1 for item in items if item.get("verified") and not item.get("discarded"))
    discarded = sum(1 for item in items if item.get("discarded"))
    pending = len(items) - (verified + discarded)
    return {
        "total_items": len(items),
        "verified_count": verified,
        "discarded_count": discarded,
        "pending_count": max(0, pending),
    }


def triage_batch_non_interactive(
    file_path: str,
    approve_all: bool = False,
    discard_ids: Optional[List[str]] = None,
    approve_remaining: bool = False,
) -> Dict[str, Any]:
    """
    Applies non-interactive triage decisions to a staged batch.

    Args:
        file_path: Path to staged batch JSON.
        approve_all: If True, approves all items.
        discard_ids: List of comment_ids to mark as discarded.
        approve_remaining: If True, approves items not in discard_ids.

    Returns:
        Updated batch dictionary.
    """
    data = load_batch(file_path)
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    discard_set = set(discard_ids or [])

    any_approved = False
    for item in data.get("items", []):
        cid = item.get("comment_id")
        if approve_all:
            item["verified"] = True
            item["discarded"] = False
            item["reviewed_at"] = now_iso
            any_approved = True
        elif cid in discard_set:
            item["verified"] = False
            item["discarded"] = True
            item["reviewed_at"] = now_iso
        elif approve_remaining:
            item["verified"] = True
            item["discarded"] = False
            item["reviewed_at"] = now_iso
            any_approved = True

    # Mark batch as verified if at least one item was approved
    if any_approved or approve_all:
        data["verified"] = True

    save_batch_atomic(file_path, data)
    return data


def triage_batch_interactive(file_path: str) -> Dict[str, Any]:
    """Runs interactive terminal triage loop for a staged batch."""
    data = load_batch(file_path)
    items = data.get("items", [])
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    print(f"\n--- Staged Batch Review: {data.get('batch_id')} ---")
    print(f"Source: {data.get('source_query')}")
    print(f"Total Comments: {len(items)}\n")

    any_approved = False
    approve_rest = False

    for idx, item in enumerate(items):
        if item.get("verified") or item.get("discarded"):
            if item.get("verified"):
                any_approved = True
            continue

        cid = item.get("comment_id")
        text = item.get("text_anonymized", "")
        lang = str(item.get("lang") or "und").upper()
        print(f"[{idx + 1}/{len(items)}] ID: {cid} | Language: [{lang}]")
        print(f"Text: {text}")

        if approve_rest:
            item["verified"] = True
            item["discarded"] = False
            item["reviewed_at"] = now_iso
            any_approved = True
            continue

        while True:
            choice = input("Action [a]pprove, [d]iscard, [s]kip, [all] approve all remaining, [q]uit: ").strip().lower()
            if choice == "a":
                item["verified"] = True
                item["discarded"] = False
                item["reviewed_at"] = now_iso
                any_approved = True
                break
            elif choice == "d":
                item["verified"] = False
                item["discarded"] = True
                item["reviewed_at"] = now_iso
                break
            elif choice == "s":
                break
            elif choice == "all":
                item["verified"] = True
                item["discarded"] = False
                item["reviewed_at"] = now_iso
                any_approved = True
                approve_rest = True
                break
            elif choice == "q":
                print("Exiting review early.")
                save_batch_atomic(file_path, data)
                return data
            else:
                print("Invalid option. Please enter a, d, s, all, or q.")

    if any_approved:
        data["verified"] = True

    save_batch_atomic(file_path, data)
    summary = get_batch_summary(data)
    print("\nReview complete:")
    print(f"  Verified: {summary['verified_count']}")
    print(f"  Discarded: {summary['discarded_count']}")
    print(f"  Pending: {summary['pending_count']}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-in-the-Loop Threads Staging Reviewer")
    parser.add_argument("batch_path", type=str, help="Path to staged batch JSON file")
    parser.add_argument("--status", action="store_true", help="Print summary status without modifying")
    parser.add_argument("--approve-all", action="store_true", help="Approve all comments non-interactively")
    parser.add_argument("--discard-ids", type=str, default="", help="Comma-separated comment IDs to discard")
    args = parser.parse_args()

    if args.status:
        data = load_batch(args.batch_path)
        summary = get_batch_summary(data)
        print(f"Batch: {data.get('batch_id')} | Status: {'Verified' if data.get('verified') else 'Pending'}")
        print(f"  Total: {summary['total_items']}, Verified: {summary['verified_count']}, Discarded: {summary['discarded_count']}, Pending: {summary['pending_count']}")
        return

    if args.approve_all:
        triage_batch_non_interactive(args.batch_path, approve_all=True)
        print(f"Batch {args.batch_path} marked as approved.")
        return

    discard_list = [x.strip() for x in args.discard_ids.split(",") if x.strip()]
    if discard_list:
        triage_batch_non_interactive(args.batch_path, discard_ids=discard_list)
        print(f"Discarded IDs {discard_list} in {args.batch_path}.")
        return

    # Default to interactive triage
    triage_batch_interactive(args.batch_path)


if __name__ == "__main__":
    main()
