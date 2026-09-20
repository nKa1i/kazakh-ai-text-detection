# -*- coding: utf-8 -*-
"""
tests/test_hybrid_retriever.py: Unit tests for Hybrid Evidence Retriever (BM25 + Dense + RRF).
"""

import os
import unittest
from src.retrieval.hybrid_retriever import HybridEvidenceRetriever, compute_rrf_score


class TestHybridEvidenceRetriever(unittest.TestCase):
    """Verifies HybridEvidenceRetriever with FST BM25, Dense embeddings, and RRF fusion."""

    def setUp(self):
        self.sample_docs = [
            {
                "id": "doc_abai",
                "title": "Абай Құнанбайұлы",
                "text": "Абай Құнанбайұлы 1845 жылы туған ұлы қазақ ақыны және ойшылы. Қара сөздер жазған."
            },
            {
                "id": "doc_astana",
                "title": "Астана қаласы",
                "text": "Астана — Қазақстанның елордасы, заманауи сәулетімен танымал бас қала."
            },
            {
                "id": "doc_space",
                "title": "Байқоңыр айлағы",
                "text": "Байқоңыр — әлемдегі ең үлкен ғарыш айлағы, 1955 жылы салынған."
            }
        ]
        self.retriever = HybridEvidenceRetriever()
        self.retriever.index_corpus(self.sample_docs)

    def test_rrf_calculation(self):
        """Asserts exact numerical match for RRF(1, 1, 60) = 1/61 + 1/61 approx 0.032787."""
        score = compute_rrf_score(sparse_rank=1, dense_rank=1, k_const=60)
        expected = (1.0 / 61.0) + (1.0 / 61.0)
        self.assertAlmostEqual(score, expected, places=6)
        self.assertAlmostEqual(score, 0.032787, places=5)

        # Test retriever instance method as well
        instance_score = self.retriever.compute_rrf_score(1, 1, 60)
        self.assertEqual(score, instance_score)

    def test_sparse_only_and_dense_only_fallbacks(self):
        """Asserts documents present in only one stream still get valid RRF scores."""
        sparse_only_score = compute_rrf_score(sparse_rank=1, dense_rank=None, k_const=60)
        self.assertAlmostEqual(sparse_only_score, 1.0 / 61.0, places=6)

        dense_only_score = compute_rrf_score(sparse_rank=None, dense_rank=2, k_const=60)
        self.assertAlmostEqual(dense_only_score, 1.0 / 62.0, places=6)

        both_none_score = compute_rrf_score(sparse_rank=None, dense_rank=None, k_const=60)
        self.assertEqual(both_none_score, 0.0)

    def test_retrieval_returns_relevant_top_k(self):
        """Asserts queries return correct documents with top-K constraint."""
        results = self.retriever.retrieve("ғарыш кемесі және Байқоңыр", top_k=2)
        self.assertLessEqual(len(results), 2)
        self.assertGreater(len(results), 0)

        top_doc = results[0]
        self.assertEqual(top_doc["id"], "doc_space")
        self.assertIn("rrf_score", top_doc)
        self.assertIn("sparse_rank", top_doc)
        self.assertIn("dense_rank", top_doc)
        self.assertGreater(top_doc["rrf_score"], 0.0)

    def test_empty_or_single_doc_corpus(self):
        """Asserts graceful handling of empty or 1-element corpora."""
        empty_retriever = HybridEvidenceRetriever()
        empty_retriever.index_corpus([])
        self.assertEqual(empty_retriever.retrieve("кез келген сұрақ", top_k=3), [])

        # Empty query handling
        self.assertEqual(self.retriever.retrieve("", top_k=3), [])
        self.assertEqual(self.retriever.retrieve("   ", top_k=3), [])

        # Single document corpus
        single_doc = [{"id": "single_01", "text": "Алматы — әсем қала."}]
        single_retriever = HybridEvidenceRetriever()
        single_retriever.index_corpus(single_doc)
        results = single_retriever.retrieve("Алматы", top_k=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "single_01")

    def test_morphological_stemming_effect(self):
        """Asserts stemmed query (e.g. inflected forms Абайдың, Абайға) matches documents containing Абай."""
        results_genitive = self.retriever.retrieve("Абайдың туындылары", top_k=1)
        self.assertGreater(len(results_genitive), 0)
        self.assertEqual(results_genitive[0]["id"], "doc_abai")

        results_dative = self.retriever.retrieve("Абайға арналған өлең", top_k=1)
        self.assertGreater(len(results_dative), 0)
        self.assertEqual(results_dative[0]["id"], "doc_abai")

    def test_index_jsonl_and_retrieval(self):
        """Asserts indexing from JSONL file works properly on real knowledge corpus."""
        corpus_path = os.path.join("data", "kazakh_knowledge_corpus.jsonl")
        if os.path.exists(corpus_path):
            jsonl_retriever = HybridEvidenceRetriever()
            jsonl_retriever.index_jsonl(corpus_path)

            results = jsonl_retriever.retrieve("Қазақстан 1991 жылы тәуелсіздік алды", top_k=3)
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]["id"], "wiki_kz_001")


if __name__ == "__main__":
    unittest.main()
