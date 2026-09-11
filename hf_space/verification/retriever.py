# -*- coding: utf-8 -*-
"""
verification/retriever.py: FST-Stemmed Hybrid BM25 & Semantic Evidence Retriever.
"""

import math
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

from verification.evidence import EvidencePassage
from verification.knowledge_store import KnowledgeStore


KAZAKH_STOPWORDS = {
    "және", "мен", "бен", "пен", "де", "да", "та", "те", "бұл", "ол", "оның", "олар",
    "үшін", "туралы", "бойынша", "арқылы", "сияқты", "тура", "бойы", "дейін",
    "болып", "бола", "болған", "болды", "екен", "етіп", "еткен", "етті",
    "бір", "барлық", "бар", "жоқ", "ресми", "түрде", "деген", "деп", "осы", "сол"
}


class HybridEvidenceRetriever:
    """
    Combines FST morphological root-stemmed BM25 sparse search with
    dense semantic overlap reranking to retrieve verified encyclopedic evidence.
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        k1: float = 1.5,
        b: float = 0.75,
        dense_weight: float = 0.30
    ):
        self.store = knowledge_store
        self.k1 = k1
        self.b = b
        self.dense_weight = max(0.0, min(1.0, dense_weight))

    def _compute_idf(self, stem: str) -> float:
        """Computes Robertson-Spärck Jones IDF for a morphological stem."""
        N = len(self.store)
        if N == 0:
            return 0.0
        postings = self.store.inverted_index.get(stem, [])
        n = len(postings)
        if n == 0:
            return 0.0
        # Standard Okapi BM25 IDF with +0.5 smoothing
        return math.log(1.0 + (N - n + 0.5) / (n + 0.5))

    def retrieve(self, query: str, top_k: int = 5) -> List[EvidencePassage]:
        """
        Retrieves the top-k most relevant evidence passages for a query claim.
        """
        if not query or not query.strip():
            return []

        all_q_stems = self.store.stem_text(query)
        # Filter stopwords
        query_stems = [s for s in all_q_stems if s not in KAZAKH_STOPWORDS]
        if not query_stems:
            query_stems = all_q_stems  # fallback if all were stopwords
        if not query_stems:
            return []

        # Unique query stems
        unique_q_stems = set(query_stems)
        scores: Dict[str, float] = defaultdict(float)
        matched_stems_map: Dict[str, Set[str]] = defaultdict(set)

        avgdl = self.store.avg_doc_len if self.store.avg_doc_len > 0 else 1.0

        for q_stem in unique_q_stems:
            idf = self._compute_idf(q_stem)
            if idf <= 0.0:
                continue

            postings = self.store.inverted_index.get(q_stem, [])
            for pid, tf in postings:
                doc_len = self.store.doc_lengths.get(pid, 1)
                # BM25 term saturation
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl))
                term_score = idf * ((tf * (self.k1 + 1.0)) / denom)
                scores[pid] += term_score
                matched_stems_map[pid].add(q_stem)

        if not scores:
            return []

        results: List[EvidencePassage] = []

        for pid, bm25_val in scores.items():
            orig_passage = self.store.get_passage(pid)
            if not orig_passage:
                continue

            # Asymptotic saturation curve for BM25 score (score / (score + 5.0))
            # Prevents single isolated stop-term from dominating
            norm_bm25 = bm25_val / (bm25_val + 5.0)

            # Dense / semantic overlap component
            matched = matched_stems_map[pid]
            doc_unique_stems = set(orig_passage.stemmed_tokens) - KAZAKH_STOPWORDS
            overlap_count = len(matched)
            semantic_overlap = (
                overlap_count / math.sqrt(len(unique_q_stems) * max(1, len(doc_unique_stems)))
                if unique_q_stems else 0.0
            )

            # Combined hybrid score
            final_score = (1.0 - self.dense_weight) * norm_bm25 + self.dense_weight * semantic_overlap

            # Return detached copy with query-specific score and matched stems
            results.append(
                EvidencePassage(
                    passage_id=orig_passage.passage_id,
                    title=orig_passage.title,
                    text=orig_passage.text,
                    source_url=orig_passage.source_url,
                    similarity_score=round(float(final_score), 4),
                    matched_stems=sorted(list(matched))
                )
            )

        # Sort descending by score
        results.sort(key=lambda p: p.similarity_score, reverse=True)
        return results[:top_k]
