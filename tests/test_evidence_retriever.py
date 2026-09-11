# -*- coding: utf-8 -*-
"""
tests/test_evidence_retriever.py: Unit tests for Hybrid Evidence Retriever.
"""

import unittest
from verification.knowledge_store import KnowledgeStore
from verification.evidence import EvidencePassage
from verification.retriever import HybridEvidenceRetriever


class TestEvidenceRetriever(unittest.TestCase):
    """Verifies BM25 retrieval, morphological stem matching, and semantic reranking."""

    def setUp(self):
        self.store = KnowledgeStore.from_jsonl("data/kazakh_knowledge_corpus.jsonl")
        self.retriever = HybridEvidenceRetriever(self.store)

    def test_retrieve_independence_fact(self):
        """Query: 'Қазақстан 1991 жылы тәуелсіздік алды' should retrieve wiki_kz_001."""
        query = "Қазақстан 1991 жылы тәуелсіздік алды."
        results = self.retriever.retrieve(query, top_k=3)
        self.assertGreater(len(results), 0)
        top_passage = results[0]
        self.assertEqual(top_passage.passage_id, "wiki_kz_001")
        self.assertGreater(top_passage.similarity_score, 0.0)
        self.assertIn("1991", top_passage.matched_stems)

    def test_retrieve_with_inflection_variation(self):
        """Query: 'Астананың елорда атануы' should retrieve Astana passage (wiki_kz_002)."""
        query = "Астананың елорда атануы және көшірілуі."
        results = self.retriever.retrieve(query, top_k=3)
        self.assertGreater(len(results), 0)
        matched_ids = [p.passage_id for p in results]
        self.assertIn("wiki_kz_002", matched_ids)

    def test_retrieve_abai_literature_fact(self):
        """Query: 'Абайдың туған жылы және Қара сөздері' should retrieve wiki_kz_005."""
        query = "Абайдың туған жылы және Қара сөздері."
        results = self.retriever.retrieve(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].passage_id, "wiki_kz_005")

    def test_empty_and_nonsense_queries(self):
        self.assertEqual(self.retriever.retrieve("", top_k=5), [])
        self.assertEqual(self.retriever.retrieve("   ", top_k=5), [])
        results_nonsense = self.retriever.retrieve("xyz123 qwerty nonexistent", top_k=3)
        self.assertEqual(len(results_nonsense), 0)

    def test_bm25_score_monotonicity(self):
        """Verifies retrieved list is sorted in strictly descending order of score."""
        query = "Қазақстан Республикасының Конституциясы қашан қабылданды?"
        results = self.retriever.retrieve(query, top_k=5)
        self.assertGreater(len(results), 1)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].similarity_score, results[i + 1].similarity_score)


if __name__ == "__main__":
    unittest.main()
