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
import socket
import sys
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.scrapers.language_filter import detect_comment_language

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
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "kk,ru,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
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

    # 4. Parse text / plaintext fields in script tags (for modern Threads SSR bundles)
    script_matches = re.findall(r'<script[^>]*>(.*?)</script>', html_str, re.DOTALL)
    text_pattern = re.compile(r'"(?:text|plaintext)"\s*:\s*"((?:\\.|[^"\\])+)"')
    for script_content in script_matches:
        for m in text_pattern.findall(script_content):
            try:
                val = json.loads(f'"{m}"').strip()
            except Exception:
                val = m.strip()

            if len(val) < 3 or val.lower() in ("true", "false", "null", "none", "center", "inherit", "auto"):
                continue
            if not re.search(r'[a-zA-Zа-яёА-ЯЁәғқңөұүһіӘҒҚҢӨҰҮҺІ]', val):
                continue
            raw_nodes.append({
                "comment_id": f"th_{default_post_id or 'node'}_{len(raw_nodes)+1}",
                "post_id": default_post_id or "post_main",
                "text": val,
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
    filter_non_kazakh: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Creates an immutable staging batch file matching the ThreadsSentinel data contract.

    Args:
        source_query: Source post URL or username query.
        raw_items: List of raw items with keys (comment_id, post_id, text/text_anonymized).
        output_dir: Staging destination folder.
        filter_non_kazakh: If True, automatically marks non-Kazakh (e.g. Russian)
            comments as discarded to shield human review and FST induction.

    Returns:
        Tuple of (batch_file_path, batch_data_dict).
    """
    if not os.path.isabs(output_dir):
        resolved_dir = os.path.join(_PROJECT_ROOT, output_dir)
    else:
        resolved_dir = output_dir

    os.makedirs(resolved_dir, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc)
    timestamp_str = now.strftime("%Y%m%dT%H%M%SZ")
    batch_id = f"batch_{timestamp_str}"
    batch_file = os.path.join(resolved_dir, f"{batch_id}.json")

    items: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        cid = str(item.get("comment_id") or f"th_{idx+1}")
        pid = str(item.get("post_id") or "")
        text_anon = item.get("text_anonymized")
        if not text_anon:
            text_anon = anonymize_social_text(str(item.get("text") or ""))

        # Run language identification filter
        lang, conf, reason = detect_comment_language(text_anon)

        already_verified = bool(item.get("verified", False))
        already_discarded = bool(item.get("discarded", False))

        discarded = already_discarded
        discard_reason = item.get("discard_reason")

        if filter_non_kazakh and not already_verified and not already_discarded:
            if lang not in ("kk", "mixed"):
                discarded = True
                discard_reason = f"auto_filtered_{lang}"

        items.append({
            "comment_id": cid,
            "post_id": pid,
            "text_anonymized": text_anon,
            "lang": lang,
            "lang_confidence": conf,
            "verified": already_verified,
            "discarded": discarded,
            "discard_reason": discard_reason,
            "reviewed_at": item.get("reviewed_at"),
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
    filter_non_kazakh: bool = True,
    proxy: Optional[str] = None,
    timeout_sec: float = 15.0,
) -> ThreadsHarvestResult:
    """
    Public stateless scraper fetching a Threads post or profile without authentication.

    Args:
        url: Public Threads URL (e.g. https://www.threads.net/@user/post/C_abc123).
        output_dir: Staging directory for batch JSON files.
        filter_non_kazakh: If True, auto-discards non-Kazakh comments.
        proxy: Optional HTTP/HTTPS proxy URL (e.g. http://127.0.0.1:7890).
        timeout_sec: HTTP request timeout in seconds.

    Returns:
        ThreadsHarvestResult with status, batch path, and comment count.
    """
def _resolve_proxy(proxy: Optional[str] = None) -> Optional[str]:
    """Resolves proxy configuration with environment and local 7890 fallback."""
    proxy_url = proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("ALL_PROXY")
    if not proxy_url:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.2)
                if s.connect_ex(("127.0.0.1", 7890)) == 0:
                    proxy_url = "http://127.0.0.1:7890"
        except Exception:
            pass
    return proxy_url


def harvest_threads_url(
    url: str,
    output_dir: str = "data/threads_staged",
    filter_non_kazakh: bool = True,
    proxy: Optional[str] = None,
    timeout_sec: float = 15.0,
) -> ThreadsHarvestResult:
    """
    Public stateless scraper fetching a Threads post or profile without authentication.

    Args:
        url: Public Threads URL (e.g. https://www.threads.net/@user/post/C_abc123).
        output_dir: Staging directory for batch JSON files.
        filter_non_kazakh: If True, auto-discards non-Kazakh comments.
        proxy: Optional HTTP/HTTPS proxy URL (e.g. http://127.0.0.1:7890).
        timeout_sec: HTTP request timeout in seconds.

    Returns:
        ThreadsHarvestResult with status, batch path, and comment count.
    """
    proxy_url = _resolve_proxy(proxy)
    # Normalize threads.com mobile share links to threads.net
    target_url = url
    if "threads.com" in target_url:
        target_url = target_url.replace("threads.com", "threads.net")

    try:
        client_kwargs: Dict[str, Any] = {
            "headers": DEFAULT_HEADERS,
            "follow_redirects": True,
            "timeout": timeout_sec,
            "verify": False,
        }
        if proxy_url:
            client_kwargs["proxy"] = proxy_url

        with httpx.Client(**client_kwargs) as client:
            resp = client.get(target_url)

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
            filter_non_kazakh=filter_non_kazakh,
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


def search_threads_posts(
    query: str,
    limit: int = 15,
    proxy: Optional[str] = None,
    timeout_sec: float = 15.0,
) -> List[str]:
    """
    Searches public Threads for a topic/keyword and returns distinct post URLs.

    Args:
        query: Search term (e.g. 'қазақша', 'алматы', 'каспи').
        limit: Maximum number of post URLs to retrieve.
        proxy: Optional proxy URL.
        timeout_sec: Request timeout in seconds.

    Returns:
        List of canonical post URLs (https://www.threads.net/t/<post_id>).
    """
    proxy_url = _resolve_proxy(proxy)
    encoded_query = urllib.parse.quote(query.strip())
    search_url = f"https://www.threads.net/search?q={encoded_query}&serp_type=default"

    client_kwargs: Dict[str, Any] = {
        "headers": DEFAULT_HEADERS,
        "follow_redirects": True,
        "timeout": timeout_sec,
        "verify": False,
    }
    if proxy_url:
        client_kwargs["proxy"] = proxy_url

    try:
        with httpx.Client(**client_kwargs) as client:
            resp = client.get(search_url)

        if resp.status_code != 200:
            return []

        post_ids = re.findall(r'/(?:@[a-zA-Z0-9_\.]+/)?(?:post|t)/([A-Za-z0-9_\-]+)', resp.text)
        seen_ids = set()
        urls: List[str] = []
        for pid in post_ids:
            if pid not in seen_ids and len(pid) >= 5:
                seen_ids.add(pid)
                urls.append(f"https://www.threads.net/t/{pid}")
                if len(urls) >= limit:
                    break
        return urls
    except Exception:
        return []


def harvest_threads_user_posts(
    username: str,
    limit: int = 10,
    proxy: Optional[str] = None,
    timeout_sec: float = 15.0,
) -> List[str]:
    """
    Discovers recent public post URLs for a given Threads creator or channel.

    Args:
        username: Handle with or without leading '@' (e.g. 'tengrinewskz').
        limit: Maximum number of post URLs to retrieve.
        proxy: Optional proxy URL.
        timeout_sec: Request timeout in seconds.

    Returns:
        List of canonical post URLs.
    """
    clean_user = username.strip().lstrip("@")
    profile_url = f"https://www.threads.net/@{clean_user}"
    proxy_url = _resolve_proxy(proxy)

    client_kwargs: Dict[str, Any] = {
        "headers": DEFAULT_HEADERS,
        "follow_redirects": True,
        "timeout": timeout_sec,
        "verify": False,
    }
    if proxy_url:
        client_kwargs["proxy"] = proxy_url

    try:
        with httpx.Client(**client_kwargs) as client:
            resp = client.get(profile_url)

        if resp.status_code != 200:
            return []

        post_ids = re.findall(r'/(?:@[a-zA-Z0-9_\.]+/)?(?:post|t)/([A-Za-z0-9_\-]+)', resp.text)
        seen_ids = set()
        urls: List[str] = []
        for pid in post_ids:
            if pid not in seen_ids and len(pid) >= 5:
                seen_ids.add(pid)
                urls.append(f"https://www.threads.net/t/{pid}")
                if len(urls) >= limit:
                    break
        return urls
    except Exception:
        return []


def harvest_threads_bulk(
    urls: List[str],
    output_dir: str = "data/threads_staged",
    filter_non_kazakh: bool = True,
    proxy: Optional[str] = None,
    delay_sec: float = 0.5,
    timeout_sec: float = 15.0,
) -> ThreadsHarvestResult:
    """
    Fetches comments from multiple Threads posts and aggregates them into a single batch.

    Args:
        urls: List of post URLs to scrape.
        output_dir: Destination staging folder.
        filter_non_kazakh: If True, automatically marks non-Kazakh text as discarded.
        proxy: Optional proxy URL.
        delay_sec: Delay between post requests to respect polite scraping rates.
        timeout_sec: Per-request timeout in seconds.

    Returns:
        Combined ThreadsHarvestResult.
    """
    if not urls:
        return ThreadsHarvestResult(
            success=False,
            error_message="No URLs provided for bulk harvesting",
        )

    proxy_url = _resolve_proxy(proxy)
    client_kwargs: Dict[str, Any] = {
        "headers": DEFAULT_HEADERS,
        "follow_redirects": True,
        "timeout": timeout_sec,
        "verify": False,
    }
    if proxy_url:
        client_kwargs["proxy"] = proxy_url

    all_comments: List[Dict[str, Any]] = []
    seen_texts = set()

    with httpx.Client(**client_kwargs) as client:
        for idx, post_url in enumerate(urls):
            target_url = post_url.replace("threads.com", "threads.net")
            post_id_match = re.search(r'/(?:post|t)/([A-Za-z0-9_\-]+)', target_url)
            default_pid = post_id_match.group(1) if post_id_match else f"p_{idx+1}"

            try:
                resp = client.get(target_url)
                if resp.status_code == 200:
                    comments = extract_comments_from_html(resp.text, default_post_id=default_pid)
                    for c in comments:
                        txt = c.get("text_anonymized", "")
                        if txt and txt not in seen_texts:
                            seen_texts.add(txt)
                            all_comments.append(c)
            except Exception:
                pass

            if delay_sec > 0 and idx < len(urls) - 1:
                time.sleep(delay_sec)

    if not all_comments:
        return ThreadsHarvestResult(
            success=False,
            error_message="Failed to parse comments from any of the provided URLs",
        )

    batch_path, batch_data = create_staging_batch(
        source_query=f"bulk://{len(urls)}_posts",
        raw_items=all_comments,
        output_dir=output_dir,
        filter_non_kazakh=filter_non_kazakh,
    )

    return ThreadsHarvestResult(
        success=True,
        batch_file_path=batch_path,
        batch_id=batch_data["batch_id"],
        comments_count=len(all_comments),
    )


def harvest_threads_search(
    query: str,
    limit: int = 15,
    output_dir: str = "data/threads_staged",
    filter_non_kazakh: bool = True,
    proxy: Optional[str] = None,
) -> ThreadsHarvestResult:
    """
    End-to-end convenience workflow: searches Threads by keyword, crawls discovered
    posts, filters out non-Kazakh text, and creates a combined staging batch.
    """
    discovered_urls = search_threads_posts(query=query, limit=limit, proxy=proxy)
    if not discovered_urls:
        return ThreadsHarvestResult(
            success=False,
            error_message=f"No posts found for query '{query}'",
        )
    return harvest_threads_bulk(
        urls=discovered_urls,
        output_dir=output_dir,
        filter_non_kazakh=filter_non_kazakh,
        proxy=proxy,
    )
