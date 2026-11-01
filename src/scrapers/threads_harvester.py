# -*- coding: utf-8 -*-
"""
src/scrapers/threads_harvester.py: Stateless public Meta Threads harvester & staging batch creator.
Fetches public Threads SSR HTML, extracts post and comment nodes, masks PII, and writes immutable staging batches.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.build_social_media_benchmark import anonymize_social_text

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "kk,ru;q=0.9,en;q=0.8",
    "Sec-Fetch-Site": "same-origin",
}


@dataclass
class ThreadsHarvestResult:
    """Result container for a Threads harvesting operation."""
    success: bool
    batch_file_path: Optional[str] = None
    batch_id: Optional[str] = None
    comments_count: int = 0
    error_message: Optional[str] = None


def _find_posts_in_dict(d: Any, collected: List[Dict[str, Any]]) -> None:
    """Recursively walks nested JSON looking for Threads post nodes with captions."""
    if isinstance(d, dict):
        if "post" in d and isinstance(d["post"], dict):
            p = d["post"]
            post_id = str(p.get("id") or p.get("pk") or "")
            caption = p.get("caption")
            text = ""
            if isinstance(caption, dict):
                text = caption.get("text", "")
            elif isinstance(caption, str):
                text = caption
            if text and text.strip():
                collected.append({
                    "comment_id": f"th_{post_id}" if post_id else f"th_{len(collected)+1}",
                    "post_id": post_id,
                    "text": text.strip(),
                })
        for v in d.values():
            _find_posts_in_dict(v, collected)
    elif isinstance(d, list):
        for item in d:
            _find_posts_in_dict(item, collected)


def extract_comments_from_html(
    html_str: str,
    default_post_id: str = "",
) -> List[Dict[str, Any]]:
    """
    Extracts comment and post nodes from Threads public HTML/SSR payload.
    Masks PII via anonymize_social_text.

    Args:
        html_str: Raw HTML source of the public Threads page.
        default_post_id: Fallback post ID if node identifier is absent.

    Returns:
        List of dicts with keys: comment_id, post_id, text_anonymized.
    """
    raw_nodes: List[Dict[str, Any]] = []

    # 1. Parse JSON-LD blocks (<script type="application/ld+json">)
    json_ld_matches = re.findall(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html_str,
        re.DOTALL | re.IGNORECASE,
    )
    for block in json_ld_matches:
        try:
            data = json.loads(block.strip())
            # Main post body
            if "articleBody" in data:
                pid = str(data.get("identifier") or default_post_id or "post_main")
                raw_nodes.append({
                    "comment_id": f"th_{pid}_main",
                    "post_id": pid,
                    "text": data["articleBody"],
                })
            # Nested comments
            comments = data.get("comment", [])
            if isinstance(comments, list):
                for idx, c in enumerate(comments):
                    if isinstance(c, dict):
                        cid = str(c.get("identifier") or f"th_c_{idx+1}")
                        text = str(c.get("text") or "")
                        if text.strip():
                            raw_nodes.append({
                                "comment_id": cid,
                                "post_id": str(data.get("identifier") or default_post_id or ""),
                                "text": text.strip(),
                            })
        except Exception:
            continue

    # 2. Parse SSR JSON data (<script type="application/json" data-sjs> or data-sjs)
    ssr_matches = re.findall(
        r'<script[^>]*data-sjs[^>]*>(.*?)</script>',
        html_str,
        re.DOTALL | re.IGNORECASE,
    )
    for block in ssr_matches:
        try:
            data = json.loads(block.strip())
            _find_posts_in_dict(data, raw_nodes)
        except Exception:
            continue

    # 3. Fallback: OpenGraph description if nothing else found
    if not raw_nodes:
        og_match = re.search(
            r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\'](.*?)["\']',
            html_str,
            re.IGNORECASE,
        )
        if og_match:
            desc = og_match.group(1).strip()
            if desc:
                raw_nodes.append({
                    "comment_id": f"th_{default_post_id or 'main'}_og",
                    "post_id": default_post_id or "post_main",
                    "text": desc,
                })

    # Deduplicate and anonymize PII
    seen_texts = set()
    cleaned_comments: List[Dict[str, Any]] = []

    for item in raw_nodes:
        raw_text = item.get("text", "").strip()
        if not raw_text or raw_text in seen_texts:
            continue
        seen_texts.add(raw_text)

        anonymized = anonymize_social_text(raw_text)
        cleaned_comments.append({
            "comment_id": str(item.get("comment_id") or f"th_{len(cleaned_comments)+1}"),
            "post_id": str(item.get("post_id") or default_post_id or ""),
            "text_anonymized": anonymized,
        })

    return cleaned_comments


def create_staging_batch(
    source_query: str,
    raw_items: List[Dict[str, Any]],
    output_dir: str = "data/threads_staged",
) -> Tuple[str, Dict[str, Any]]:
    """
    Creates an immutable staging batch file matching the ThreadsSentinel data contract.

    Args:
        source_query: Source post URL or username query.
        raw_items: List of raw items with keys (comment_id, post_id, text/text_anonymized).
        output_dir: Staging destination folder.

    Returns:
        Tuple of (batch_file_path, batch_data_dict).
    """
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc)
    timestamp_str = now.strftime("%Y%m%dT%H%M%SZ")
    batch_id = f"batch_{timestamp_str}"
    batch_file = os.path.join(output_dir, f"{batch_id}.json")

    items: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        cid = str(item.get("comment_id") or f"th_{idx+1}")
        pid = str(item.get("post_id") or "")
        text_anon = item.get("text_anonymized")
        if not text_anon:
            text_anon = anonymize_social_text(str(item.get("text") or ""))

        items.append({
            "comment_id": cid,
            "post_id": pid,
            "text_anonymized": text_anon,
            "verified": False,
            "discarded": False,
            "reviewed_at": None,
        })

    batch_data = {
        "schema_version": "1.0",
        "batch_id": batch_id,
        "source_query": source_query,
        "harvested_at": now.isoformat(),
        "verified": False,
        "items": items,
    }

    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, indent=2, ensure_ascii=False)

    return os.path.abspath(batch_file), batch_data


def harvest_threads_url(
    url: str,
    output_dir: str = "data/threads_staged",
    timeout_sec: float = 15.0,
) -> ThreadsHarvestResult:
    """
    Public stateless scraper fetching a Threads post or profile without authentication.

    Args:
        url: Public Threads URL (e.g. https://www.threads.net/@user/post/C_abc123).
        output_dir: Staging directory for batch JSON files.
        timeout_sec: HTTP request timeout in seconds.

    Returns:
        ThreadsHarvestResult with status, batch path, and comment count.
    """
    try:
        with httpx.Client(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=timeout_sec) as client:
            resp = client.get(url)

        if resp.status_code != 200:
            return ThreadsHarvestResult(
                success=False,
                error_message=f"HTTP {resp.status_code}: {resp.text[:120]}",
            )

        post_id_match = re.search(r'/post/([A-Za-z0-9_\-]+)', url)
        default_post_id = post_id_match.group(1) if post_id_match else "post_unknown"

        comments = extract_comments_from_html(resp.text, default_post_id=default_post_id)
        if not comments:
            return ThreadsHarvestResult(
                success=False,
                error_message="No post or comment nodes could be parsed from public HTML",
            )

        batch_path, batch_data = create_staging_batch(
            source_query=url,
            raw_items=comments,
            output_dir=output_dir,
        )

        return ThreadsHarvestResult(
            success=True,
            batch_file_path=batch_path,
            batch_id=batch_data["batch_id"],
            comments_count=len(comments),
        )

    except Exception as exc:
        return ThreadsHarvestResult(
            success=False,
            error_message=f"Network or parsing exception: {str(exc)}",
        )
