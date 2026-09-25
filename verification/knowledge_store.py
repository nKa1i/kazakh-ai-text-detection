# -*- coding: utf-8 -*-
"""
verification/knowledge_store.py: Knowledge Corpus Store & FST-Stemmed Inverted Index.
"""

import os
import re
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

from fst_analyzer import AdvancedKazakhFSTAnalyzer
from verification.evidence import EvidencePassage


class KnowledgeStore:
    """
    Manages verified Kazakh encyclopedic reference passages and builds
    an in-memory inverted index keyed on FST root stems to bridge agglutinative variation.
    """

    def __init__(self, fst_analyzer: Optional[AdvancedKazakhFSTAnalyzer] = None):
        self.fst = fst_analyzer or AdvancedKazakhFSTAnalyzer()
        self.passages: Dict[str, EvidencePassage] = {}
        # stem -> list of (passage_id, term_frequency)
        self.inverted_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0

    def stem_word(self, word: str) -> List[str]:
        """
        Extracts morphological stems and lemmas for a Kazakh word:
        returns [surface_form, case_base, fst_root] to maximize retrieval recall.
        """
        w = (word or "").strip().lower()
        # Clean leading/trailing non-alphanumeric (except internal hyphens)
        w = re.sub(r'^[^\w\d]+|[^\w\d]+$', '', w, flags=re.UNICODE)
        if not w:
            return []
        if len(w) <= 3 or w.isdigit():
            return [w]

        stems = [w]

        # 1. Check Case Suffix Base (e.g., алматының -> алматы)
        if w.endswith("стан") and len(w) >= 5:
            pass  # -стан is a toponymic root suffix ending in 'н', not ablative -тан
        else:
            match = self.fst.case_re.search(w)
            if match and len(w[:match.start()]) >= 3:
                case_base = w[:match.start()]
                if case_base not in stems:
                    stems.append(case_base)

        # 2. Check FST Deep Root (e.g., алматының -> алмат)
        seg = self.fst.analyze_and_segment(w)
        if " -" in seg:
            root = seg.split(" -")[0].strip()
            if len(root) >= 2 and root not in stems:
                stems.append(root)

        return stems

    def stem_text(self, text: str) -> List[str]:
        """Extracts list of clean, lower-case root stems and lemmas from input text."""
        if not text:
            return []
        raw_tokens = re.findall(r'[\w\d]+(?:-[\w\d]+)*', text, flags=re.UNICODE)
        all_stems = []
        for t in raw_tokens:
            word_stems = self.stem_word(t)
            all_stems.extend(word_stems)
        return all_stems

    def add_passage(self, passage: EvidencePassage) -> None:
        """Indexes an EvidencePassage into the knowledge store."""
        pid = passage.passage_id
        self.passages[pid] = passage

        # Extract stems for title and text
        full_content = f"{passage.title} {passage.text}"
        stems = self.stem_text(full_content)
        passage.stemmed_tokens = stems

        doc_len = len(stems)
        self.doc_lengths[pid] = doc_len

        # Count frequencies
        freqs: Dict[str, int] = defaultdict(int)
        for s in stems:
            freqs[s] += 1

        for stem, count in freqs.items():
            self.inverted_index[stem].append((pid, count))

        # Update average doc length
        if self.doc_lengths:
            self.avg_doc_len = sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_passage(self, passage_id: str) -> Optional[EvidencePassage]:
        """Retrieves an EvidencePassage by its ID."""
        return self.passages.get(passage_id)

    def get_inverted_index(self) -> Dict[str, List[Tuple[str, int]]]:
        """Returns the internal inverted index."""
        return dict(self.inverted_index)

    def __len__(self) -> int:
        return len(self.passages)

    @classmethod
    def from_jsonl(cls, file_path: str) -> "KnowledgeStore":
        """Loads and indexes passages from a JSONL corpus file."""
        store = cls()
        if not os.path.exists(file_path):
            return store

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    passage = EvidencePassage(
                        passage_id=data.get("passage_id", f"p_{len(store)}"),
                        title=data.get("title", ""),
                        text=data.get("text", ""),
                        source_url=data.get("source_url", "")
                    )
                    store.add_passage(passage)
                except json.JSONDecodeError:
                    continue
        return store
