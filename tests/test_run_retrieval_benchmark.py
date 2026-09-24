# -*- coding: utf-8 -*-
"""
tests/test_run_retrieval_benchmark.py: Unit tests for Hybrid Retrieval Benchmark Evaluation Engine.
"""

import unittest
from scripts.run_retrieval_benchmark import (
    evaluate_retrieval_metrics,
    run_retrieval_ablation_experiment,
    format_retrieval_latex_table
)


class TestRunRetrievalBenchmark(unittest.TestCase):
    """Verifies retrieval benchmark metrics, ablation experiment, and LaTeX table generator."""

    def test_evaluate_retrieval_metrics_exact(self):
        retrieved = [
            ["doc_1", "doc_2", "doc_3"],
            ["doc_4", "doc_5", "doc_6"]
        ]
        gold = [
            ["doc_1"],
            ["doc_6"]
        ]
        metrics = evaluate_retrieval_metrics(retrieved, gold, k_values=[1, 3])
        self.assertEqual(metrics["recall_at_1"], 0.5)
        self.assertEqual(metrics["recall_at_3"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], (1.0 + 1 / 3) / 2)

    def test_evaluate_retrieval_metrics_default_k(self):
        retrieved = [
            ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"],
            ["doc_10", "doc_20", "doc_30", "doc_40", "doc_2"]
        ]
        gold = [
            ["doc_1"],
            ["doc_2"]
        ]
        metrics = evaluate_retrieval_metrics(retrieved, gold)
        self.assertIn("recall_at_1", metrics)
        self.assertIn("recall_at_3", metrics)
        self.assertIn("recall_at_5", metrics)
        self.assertIn("mrr", metrics)
        self.assertEqual(metrics["recall_at_1"], 0.5)
        self.assertEqual(metrics["recall_at_3"], 0.5)
        self.assertEqual(metrics["recall_at_5"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], (1.0 + 1 / 5) / 2)

    def test_evaluate_retrieval_metrics_empty_and_mismatched(self):
        empty_metrics = evaluate_retrieval_metrics([], [], k_values=[1, 3, 5])
        self.assertEqual(empty_metrics["recall_at_1"], 0.0)
        self.assertEqual(empty_metrics["recall_at_3"], 0.0)
        self.assertEqual(empty_metrics["recall_at_5"], 0.0)
        self.assertEqual(empty_metrics["mrr"], 0.0)

        mismatch_metrics = evaluate_retrieval_metrics([["doc_1"]], [], k_values=[1, 3, 5])
        self.assertEqual(mismatch_metrics["recall_at_1"], 0.0)
        self.assertEqual(mismatch_metrics["mrr"], 0.0)

    def test_evaluate_retrieval_metrics_no_match(self):
        retrieved = [["doc_1", "doc_2"]]
        gold = [["doc_99"]]
        metrics = evaluate_retrieval_metrics(retrieved, gold, k_values=[1, 3])
        self.assertEqual(metrics["recall_at_1"], 0.0)
        self.assertEqual(metrics["recall_at_3"], 0.0)
        self.assertEqual(metrics["mrr"], 0.0)

    def test_run_retrieval_ablation_experiment(self):
        corpus = [
            {"passage_id": "p1", "text": "Абай Құнанбайұлы Семей өңірінде дүниеге келген."},
            {"passage_id": "p2", "text": "Байқоңыр ғарыш айлағы Қызылорда облысында орналасқан."}
        ]
        queries = [
            {"claim": "Абай Құнанбайұлы қай өңірде дүниеге келді?", "gold_passage_ids": ["p1"]}
        ]
        results = run_retrieval_ablation_experiment(corpus, queries)
        self.assertIn("bm25_standard", results)
        self.assertIn("bm25_fst_stemmed", results)
        self.assertIn("dense_semantic", results)
        self.assertIn("hybrid_rrf", results)

        for mode in ["bm25_standard", "bm25_fst_stemmed", "dense_semantic", "hybrid_rrf"]:
            self.assertIn("recall_at_1", results[mode])
            self.assertIn("recall_at_3", results[mode])
            self.assertIn("recall_at_5", results[mode])
            self.assertIn("mrr", results[mode])

    def test_run_retrieval_ablation_experiment_empty(self):
        results = run_retrieval_ablation_experiment([], [])
        self.assertIn("bm25_standard", results)
        self.assertEqual(results["bm25_standard"]["recall_at_1"], 0.0)

    def test_format_retrieval_latex_table(self):
        results = {
            "bm25_standard": {"recall_at_1": 0.462, "recall_at_3": 0.621, "recall_at_5": 0.704, "mrr": 0.542},
            "bm25_fst_stemmed": {"recall_at_1": 0.528, "recall_at_3": 0.694, "recall_at_5": 0.768, "mrr": 0.611},
            "dense_semantic": {"recall_at_1": 0.541, "recall_at_3": 0.712, "recall_at_5": 0.785, "mrr": 0.627},
            "hybrid_rrf": {"recall_at_1": 0.635, "recall_at_3": 0.792, "recall_at_5": 0.846, "mrr": 0.718}
        }
        latex = format_retrieval_latex_table(results)
        self.assertIn("\\begin{table}", latex)
        self.assertIn("Recall@5", latex)
        self.assertIn("Hybrid (BM25 + mContriever)", latex)
        self.assertIn("\\textbf{63.5}", latex)
        self.assertIn("\\textbf{0.718}", latex)

    def test_format_retrieval_latex_table_partial(self):
        results = {
            "bm25_standard": {"recall_at_1": 0.462, "recall_at_3": 0.621, "recall_at_5": 0.704, "mrr": 0.542}
        }
        latex = format_retrieval_latex_table(results)
        self.assertIn("\\begin{table}", latex)
        self.assertIn("BM25 (Standard)", latex)
        self.assertIn("\\end{table}", latex)


if __name__ == "__main__":
    unittest.main()
