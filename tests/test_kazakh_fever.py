# -*- coding: utf-8 -*-
"""
tests/test_kazakh_fever.py: Unit tests for Kazakh-FEVER Benchmark Generator and Evaluator.
"""

import os
import json
import tempfile
import unittest

from verification.knowledge_store import KnowledgeStore
from verification.fever_generator import KazakhFEVERGenerator
from verification.evaluator import FEVEREvaluator
from verification.verifier import TrustworthyDocumentVerifier


class TestKazakhFEVERBenchmark(unittest.TestCase):
    """Verifies synthetic FEVER dataset generation, mutation logic, and evaluation metrics."""

    def setUp(self):
        self.store = KnowledgeStore.from_jsonl("data/kazakh_knowledge_corpus.jsonl")
        self.generator = KazakhFEVERGenerator(self.store)
        self.evaluator = FEVEREvaluator()

    def test_generate_benchmark_distribution(self):
        """Verifies generator creates balanced SUPPORTED, REFUTED, and NOT ENOUGH INFO claims."""
        benchmark = self.generator.generate_benchmark(samples_per_class=4)
        self.assertEqual(len(benchmark), 12)

        labels = [item["label"] for item in benchmark]
        self.assertEqual(labels.count("SUPPORTED"), 4)
        self.assertEqual(labels.count("REFUTED"), 4)
        self.assertEqual(labels.count("NOT ENOUGH INFO"), 4)

        for item in benchmark:
            self.assertIn("claim", item)
            self.assertIn("evidence_id", item)
            self.assertIn("label", item)
            self.assertIn("mutation_type", item)

    def test_temporal_mutation_logic(self):
        """Verifies temporal mutations reliably alter calendar years."""
        text = "Қазақстан 1991 жылы тәуелсіздік алды."
        mutated, m_type = self.generator.mutate_claim(text)
        self.assertEqual(m_type, "temporal_mutation")
        self.assertNotIn("1991", mutated)
        self.assertTrue(any(yr in mutated for yr in ["1995", "1998", "2001", "1986"]))

    def test_evaluator_metrics_calculation(self):
        """Verifies precision, recall, macro-F1, and FEVER score calculation."""
        y_true = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]
        y_pred = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]
        retrieved_ids = [["p1", "p2"], ["p3"], ["p4"]]
        ground_truth_ids = ["p1", "p3", "p5"]

        metrics = self.evaluator.compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            retrieved_ids=retrieved_ids,
            ground_truth_ids=ground_truth_ids
        )

        self.assertAlmostEqual(metrics["nli_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["nli_macro_f1"], 1.0)
        # Recall@5: p1 in ['p1', 'p2'] (yes), p3 in ['p3'] (yes), p5 in ['p4'] (no) -> 2/3
        self.assertAlmostEqual(metrics["retrieval_recall_at_5"], 2.0 / 3.0, places=3)
        # FEVER score: both correct label and correct retrieval -> 2/3
        self.assertAlmostEqual(metrics["fever_score"], 2.0 / 3.0, places=3)

    def test_end_to_end_system_evaluation(self):
        """Evaluates the TrustworthyDocumentVerifier against a mini benchmark."""
        verifier = TrustworthyDocumentVerifier(knowledge_store=self.store)
        mini_benchmark = [
            {
                "claim": "Қазақстан 1991 жылы өз тәуелсіздігін жариялады.",
                "evidence_id": "wiki_kz_001",
                "label": "SUPPORTED"
            },
            {
                "claim": "Қазақстан 1998 жылы өз тәуелсіздігін жариялады.",
                "evidence_id": "wiki_kz_001",
                "label": "REFUTED"
            }
        ]

        report = self.evaluator.evaluate_verifier(verifier, mini_benchmark)
        self.assertIn("nli_accuracy", report)
        self.assertIn("fever_score", report)
        self.assertGreaterEqual(report["nli_accuracy"], 0.50)


if __name__ == "__main__":
    unittest.main()
