# -*- coding: utf-8 -*-
"""
models/fst_sentinel.py: Auto-FST Sentinel and continuous morphological lexicon induction engine.
Scans verified human social comments, extracts novel roots via affix peeling and morphophonological checks,
and promotes validated roots to DynamicFSTOverlay while rejecting AI spam.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.auto_fst.overlay import DynamicFSTOverlay
from src.auto_fst.inducer import AutoFSTInducer, InductionResult
from src.auto_fst.phonology import MorphophonologicalRestorer
from src.auto_fst.guard import RegressionGuard, RESERVED_NATIVE_ROOTS
from models.document_detector import ONNXDualStreamDetector


VOICELESS_CODAS = set("пфкқтсшщхцч")
VOICED_ONSET_CONFLICTS = set("дгғблм")


def passes_coda_assimilation(stem: str, suffix: str) -> bool:
    """
    Verifies that stem coda and affix onset satisfy Kazakh assimilation rules.

    Args:
        stem: Candidate root stem string.
        suffix: Extracted suffix string (e.g. '-дан', 'ден', '-тық').

    Returns:
        True if the combination satisfies phonotactic assimilation, False otherwise.
    """
    if not stem or not suffix:
        return True

    last_char = stem[-1].lower()
    suffix_clean = suffix.lstrip("-")
    if not suffix_clean:
        return True
    first_char = suffix_clean[0].lower()

    # If stem ends in voiceless consonant:
    if last_char in VOICELESS_CODAS:
        # Suffix cannot begin with voiced consonant when voiceless variant exists
        if first_char in VOICED_ONSET_CONFLICTS:
            return False
    else:
        # If stem ends in vowel or voiced consonant:
        # Suffix starting with 'т' (e.g. -тан/-тен, -тық/-тік, -тау/-теу) is invalid
        if first_char == "т" and suffix_clean in {"тан", "тен", "тық", "тік", "тау", "теу"}:
            return False

    return True


@dataclass
class SentinelRunStats:
    """Audit and runtime metrics container for an AutoFSTSentinel run."""
    run_timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    batches_processed: int = 0
    comments_scored: int = 0
    ai_spam_rejected: int = 0
    human_accepted: int = 0
    promoted_stems: List[str] = field(default_factory=list)
    pending_buffer_size: int = 0
    guard_rejections: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_timestamp": self.run_timestamp,
            "batches_processed": self.batches_processed,
            "comments_scored": self.comments_scored,
            "ai_spam_rejected": self.ai_spam_rejected,
            "human_accepted": self.human_accepted,
            "promoted_stems": self.promoted_stems,
            "pending_buffer_size": self.pending_buffer_size,
            "guard_rejections": self.guard_rejections,
        }


class AutoFSTSentinel:
    """
    Continuous active-learning sentinel:
    1. Filters verified social comments via ONNXDualStreamDetector (rejecting AI spam).
    2. Identifies unknown words and peels legitimate Kazakh suffixes.
    3. Maintains cross-post memory buffer for candidate root recurrence.
    4. Validates and promotes candidates to DynamicFSTOverlay with RegressionGuard.
    """

    def __init__(
        self,
        detector: Optional[Any] = None,
        overlay: Optional[DynamicFSTOverlay] = None,
        inducer: Optional[AutoFSTInducer] = None,
        restorer: Optional[MorphophonologicalRestorer] = None,
        guard: Optional[RegressionGuard] = None,
        min_distinct_posts: int = 3,
        min_distinct_affixes: int = 2,
        lexicon_path: str = "data/discovered_lexicon.json",
        audit_path: str = "data/sentinel_audit.json",
        dry_run: bool = False,
    ) -> None:
        self.detector = detector or ONNXDualStreamDetector()
        self.overlay = overlay or DynamicFSTOverlay()
        self.inducer = inducer or AutoFSTInducer()
        self.restorer = restorer or MorphophonologicalRestorer()
        self.guard = guard or RegressionGuard()

        self.min_distinct_posts = int(min_distinct_posts)
        self.min_distinct_affixes = int(min_distinct_affixes)
        self.lexicon_path = lexicon_path
        self.audit_path = audit_path
        self.dry_run = bool(dry_run)

        # Candidate buffer: Map[canon_stem -> dict]
        self.buffer: Dict[str, Dict[str, Any]] = {}

        # Load existing discovered lexicon if present
        if os.path.isfile(self.lexicon_path):
            try:
                self.overlay.load_from_json(self.lexicon_path)
            except Exception:
                pass

    def process_batch(self, batch_data: Dict[str, Any]) -> SentinelRunStats:
        """
        Processes a staged Threads batch adhering to the Sentinel specification.

        Args:
            batch_data: Loaded JSON dictionary of a staged batch.

        Returns:
            SentinelRunStats containing processing counts and promoted stems.
        """
        stats = SentinelRunStats(batches_processed=1)
        items = batch_data.get("items", [])

        for item in items:
            # Only process verified and non-discarded comments
            if not item.get("verified") or item.get("discarded"):
                continue

            stats.comments_scored += 1
            text = str(item.get("text_anonymized") or item.get("text") or "").strip()
            if not text:
                continue

            # 1. AI Contamination Gate
            det_res = self.detector.predict_document(text)
            is_ai = bool(
                det_res.document_ai_probability >= self.detector.calibrated_threshold
                or det_res.verdict == "Machine-Generated"
            )
            if is_ai:
                stats.ai_spam_rejected += 1
                continue

            stats.human_accepted += 1
            post_id = str(item.get("post_id") or item.get("comment_id") or "p_default")

            # 2. Tokenize and inspect candidate words
            words = re.findall(r'[а-яёәіңғүұқөһА-ЯЁӘІҢҒҮҰҚӨҺa-zA-Z0-9\-]+', text)
            for token in words:
                token_clean = token.strip().lower()
                if len(token_clean) < 4:
                    continue

                # Skip if already known in standard FST or custom lexicon
                if token_clean in self.overlay.custom_lexicon:
                    continue
                if hasattr(self.overlay, "loanword_roots") and token_clean in self.overlay.loanword_roots:
                    continue
                if token_clean in RESERVED_NATIVE_ROOTS:
                    continue

                # 3. Induce stem via affix peeling
                cand = self.inducer.induce_stem(token_clean)
                if cand is None or not cand.suffixes:
                    continue

                # 4. Check Coda Assimilation
                first_sfx = cand.suffixes[0]
                if not passes_coda_assimilation(cand.stem, first_sfx):
                    continue

                # 5. Restore Canonical Stem
                known_custom = set(self.overlay.custom_lexicon.keys())
                canon_stem = self.restorer.restore_stem(cand.stem.lower(), first_sfx, known_custom)
                if len(canon_stem) < 3 or canon_stem in RESERVED_NATIVE_ROOTS:
                    continue
                if canon_stem in self.overlay.custom_lexicon:
                    continue

                # 6. Accumulate into Memory Buffer
                entry = self.buffer.setdefault(
                    canon_stem,
                    {
                        "posts": set(),
                        "affixes": set(),
                        "count": 0,
                        "harmony": cand.harmony,
                        "coda": cand.coda,
                    },
                )
                entry["posts"].add(post_id)
                entry["affixes"].add(first_sfx)
                entry["count"] += 1
                entry["harmony"] = cand.harmony
                entry["coda"] = cand.coda

                # 7. Check Promotion Threshold
                if (
                    len(entry["posts"]) >= self.min_distinct_posts
                    and len(entry["affixes"]) >= self.min_distinct_affixes
                ):
                    ok, reason = self.guard.validate_candidate(canon_stem)
                    if ok:
                        metadata = {
                            "harmony": entry["harmony"],
                            "coda": entry["coda"],
                            "origin": "threads_sentinel",
                            "occurrences": entry["count"],
                            "distinct_posts": len(entry["posts"]),
                            "suffix_count": len(entry["affixes"]),
                            "observed_affixes": sorted(list(entry["affixes"])),
                            "induced_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        }
                        self.overlay.register_root(canon_stem, metadata)
                        stats.promoted_stems.append(canon_stem)
                        del self.buffer[canon_stem]
                    else:
                        stats.guard_rejections.append((canon_stem, reason))

        stats.pending_buffer_size = len(self.buffer)
        self.save_discovered_lexicon()
        self.save_audit(stats)
        return stats

    def save_discovered_lexicon(self) -> None:
        """Saves current custom roots to discovered lexicon JSON."""
        if self.dry_run:
            return
        self.overlay.save_to_json(self.lexicon_path)

    def save_audit(self, stats: SentinelRunStats) -> None:
        """Saves run statistics to sentinel audit JSON log with historical log."""
        if self.dry_run:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.audit_path)), exist_ok=True)
        history: List[Dict[str, Any]] = []
        if os.path.isfile(self.audit_path):
            try:
                with open(self.audit_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if isinstance(existing, dict):
                        if "history" in existing and isinstance(existing["history"], list):
                            history = existing["history"]
                        else:
                            clean_copy = {k: v for k, v in existing.items() if k != "history"}
                            history = [clean_copy]
                    elif isinstance(existing, list):
                        history = existing
            except Exception:
                history = []

        current_run = stats.to_dict()
        history.append(current_run)

        audit_payload = dict(current_run)
        audit_payload["history"] = history

        with open(self.audit_path, "w", encoding="utf-8") as f:
            json.dump(audit_payload, f, indent=2, ensure_ascii=False)
