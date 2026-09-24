# -*- coding: utf-8 -*-
"""
tests/test_run_verification_benchmark.py: Unit tests for Fact Verification Scoring
and 2D Trust Matrix Benchmark Engine.
"""

import math
import unittest
from scripts.run_verification_benchmark import (
    compute_fever_score,
    compute_trust_matrix_quadrants,
    compute_macro_f1,
    compute_hard_nei_f1,
    format_verification_latex_table
)


class TestRunVerificationBenchmark(unittest.TestCase):
    """Verifies strict FEVER scoring, 2D Trust Matrix quadrants, and LaTeX table formatting."""

    def test_compute_fever_score_strict(self):
        gold = [
            {"id": "c1", "label": "SUPPORTS", "evidence_sentences": ["ev1"]},
            {"id": "c2", "label": "REFUTES", "evidence_sentences": ["ev2"]},
            {"id": "c3", "label": "NOT_ENOUGH_INFO", "evidence_sentences": []}
        ]
        preds = [
            {"id": "c1", "label": "SUPPORTS"},
            {"id": "c2", "label": "REFUTES"},
            {"id": "c3", "label": "NOT_ENOUGH_INFO"}
        ]
        # c1 has ev1 retrieved; c2 missed ev2; c3 NEI needs no evidence
        retrieved = [
            ["ev1", "ev_other"],
            ["ev_wrong"],
            []
        ]
        res = compute_fever_score(preds, gold, retrieved)
        self.assertEqual(res["label_accuracy"], 1.0)
        self.assertAlmostEqual(res["strict_fever_score"], 2 / 3)

    def test_compute_fever_score_empty_and_mismatched(self):
        empty_res = compute_fever_score([], [], [])
        self.assertEqual(empty_res["label_accuracy"], 0.0)
        self.assertEqual(empty_res["strict_fever_score"], 0.0)

        mismatched_res = compute_fever_score(
            [{"id": "c1", "label": "SUPPORTS"}],
            [],
            [["ev1"]]
        )
        self.assertEqual(mismatched_res["label_accuracy"], 0.0)
        self.assertEqual(mismatched_res["strict_fever_score"], 0.0)

        mismatched_ev_res = compute_fever_score(
            [{"id": "c1", "label": "SUPPORTS"}],
            [{"id": "c1", "label": "SUPPORTS", "evidence_sentences": ["ev1"]}],
            []
        )
        self.assertEqual(mismatched_ev_res["label_accuracy"], 0.0)
        self.assertEqual(mismatched_ev_res["strict_fever_score"], 0.0)

    def test_compute_fever_score_wrong_labels(self):
        gold = [
            {"id": "c1", "label": "SUPPORTS", "evidence_sentences": ["ev1"]},
            {"id": "c2", "label": "REFUTES", "evidence_sentences": ["ev2"]},
            {"id": "c3", "label": "NOT_ENOUGH_INFO", "evidence_sentences": []}
        ]
        preds = [
            {"id": "c1", "label": "REFUTES"},
            {"id": "c2", "label": "SUPPORTS"},
            {"id": "c3", "label": "REFUTES"}
        ]
        retrieved = [
            ["ev1"],
            ["ev2"],
            []
        ]
        res = compute_fever_score(preds, gold, retrieved)
        self.assertEqual(res["label_accuracy"], 0.0)
        self.assertEqual(res["strict_fever_score"], 0.0)

    def test_compute_fever_score_case_insensitivity_and_multiple_evidence(self):
        gold = [
            {"id": "c1", "label": "supports", "evidence_sentences": ["ev_a", "ev_b"]},
            {"id": "c2", "label": "Refutes", "evidence_sentences": ["ev_c"]}
        ]
        preds = [
            {"id": "c1", "label": "SUPPORTS "},
            {"id": "c2", "label": " refutes"}
        ]
        retrieved = [
            ["ev_other", "ev_b"],  # ev_b matches one of gold
            ["ev_other"]           # ev_c not retrieved
        ]
        res = compute_fever_score(preds, gold, retrieved)
        self.assertEqual(res["label_accuracy"], 1.0)
        self.assertEqual(res["strict_fever_score"], 0.5)

    def test_compute_trust_matrix_quadrants(self):
        items = [
            {"id": "1", "p_support": 0.9, "p_refute": 0.05, "p_ai": 0.1},   # Q1: High fact, Low AI
            {"id": "2", "p_support": 0.1, "p_refute": 0.8, "p_ai": 0.1},    # Q2: Low fact, Low AI
            {"id": "3", "p_support": 0.8, "p_refute": 0.1, "p_ai": 0.9},    # Q3: High fact, High AI
            {"id": "4", "p_support": 0.05, "p_refute": 0.9, "p_ai": 0.95}   # Q4: Low fact, High AI
        ]
        matrix = compute_trust_matrix_quadrants(items)
        self.assertEqual(matrix["total_items"], 4)
        self.assertEqual(matrix["quadrants"]["Q1"], 1)
        self.assertEqual(matrix["quadrants"]["Q2"], 1)
        self.assertEqual(matrix["quadrants"]["Q3"], 1)
        self.assertEqual(matrix["quadrants"]["Q4"], 1)
        self.assertIn("mean_trust_score", matrix)
        self.assertIsInstance(matrix["mean_trust_score"], float)

    def test_compute_trust_matrix_quadrants_custom_thresholds(self):
        items = [
            {"id": "1", "p_support": 0.3, "p_refute": 0.1, "p_ai": 0.4}  # T_fact = 0.2, p_ai = 0.4
        ]
        # At default threshold_fact = 0.0, threshold_ai = 0.5 -> Q1
        m1 = compute_trust_matrix_quadrants(items, threshold_fact=0.0, threshold_ai=0.5)
        self.assertEqual(m1["quadrants"]["Q1"], 1)

        # With threshold_fact = 0.3 (0.2 < 0.3) -> Q2
        m2 = compute_trust_matrix_quadrants(items, threshold_fact=0.3, threshold_ai=0.5)
        self.assertEqual(m2["quadrants"]["Q2"], 1)

        # With threshold_ai = 0.3 (0.4 >= 0.3) -> Q3
        m3 = compute_trust_matrix_quadrants(items, threshold_fact=0.0, threshold_ai=0.3)
        self.assertEqual(m3["quadrants"]["Q3"], 1)

        # With both threshold_fact = 0.3 and threshold_ai = 0.3 -> Q4
        m4 = compute_trust_matrix_quadrants(items, threshold_fact=0.3, threshold_ai=0.3)
        self.assertEqual(m4["quadrants"]["Q4"], 1)

    def test_compute_trust_matrix_empty(self):
        empty_matrix = compute_trust_matrix_quadrants([])
        self.assertEqual(empty_matrix["total_items"], 0)
        self.assertEqual(empty_matrix["quadrants"]["Q1"], 0)
        self.assertEqual(empty_matrix["quadrants"]["Q2"], 0)
        self.assertEqual(empty_matrix["quadrants"]["Q3"], 0)
        self.assertEqual(empty_matrix["quadrants"]["Q4"], 0)
        self.assertEqual(empty_matrix["mean_trust_score"], 0.0)

    def test_trust_score_formula_exact(self):
        # Case 1: T_fact = 1.0 (p_sup=1.0, p_ref=0.0), T_gen = 1.0 (p_ai=0.0)
        # T(x) = sqrt(0.5 * (1^2 + 1^2)) = 1.0
        item_perfect = [{"id": "p", "p_support": 1.0, "p_refute": 0.0, "p_ai": 0.0}]
        res_perfect = compute_trust_matrix_quadrants(item_perfect)
        self.assertAlmostEqual(res_perfect["mean_trust_score"], 1.0)

        # Case 2: T_fact = -1.0 (p_sup=0.0, p_ref=1.0), T_gen = 0.0 (p_ai=1.0)
        # max(0, -1.0) = 0.0, T_gen = 0.0 -> T(x) = 0.0
        item_worst = [{"id": "w", "p_support": 0.0, "p_refute": 1.0, "p_ai": 1.0}]
        res_worst = compute_trust_matrix_quadrants(item_worst)
        self.assertAlmostEqual(res_worst["mean_trust_score"], 0.0)

        # Case 3: T_fact = -0.5, T_gen = 0.6 -> max(0, -0.5) = 0.0, score = sqrt(0.5 * 0.36) = sqrt(0.18)
        item_negative_fact = [{"id": "n", "p_support": 0.1, "p_refute": 0.6, "p_ai": 0.4}]
        res_neg = compute_trust_matrix_quadrants(item_negative_fact)
        expected = math.sqrt(0.5 * (0.6 ** 2))
        self.assertAlmostEqual(res_neg["mean_trust_score"], expected)

    def test_format_verification_latex_table(self):
        results = {
            "mBERT-base": {"accuracy": 64.2, "macro_f1": 63.8, "fever_score": 51.4, "hard_nei_f1": 48.2},
            "XLM-RoBERTa-base": {"accuracy": 71.8, "macro_f1": 71.2, "fever_score": 58.6, "hard_nei_f1": 55.7},
            "Ours (Hybrid + Morpho)": {"accuracy": 82.6, "macro_f1": 82.1, "fever_score": 71.4, "hard_nei_f1": 72.8}
        }
        latex = format_verification_latex_table(results)
        self.assertIn("\\begin{table}", latex)
        self.assertIn("FEVER Score", latex)
        self.assertIn("Ours (Hybrid + Morpho)", latex)
        self.assertIn("\\toprule", latex)
        self.assertIn("\\bottomrule", latex)
        self.assertIn("\\textbf{82.6}", latex)
        self.assertIn("\\textbf{71.4}", latex)

    def test_compute_macro_f1_and_hard_nei_f1(self):
        gold = [
            {"id": "c1", "label": "SUPPORTS"},
            {"id": "c2", "label": "REFUTES"},
            {"id": "c3", "label": "NOT_ENOUGH_INFO"},
            {"id": "c4", "label": "NOT_ENOUGH_INFO"}
        ]
        # Perfect predictions
        preds_perfect = [
            {"id": "c1", "label": "SUPPORTS"},
            {"id": "c2", "label": "REFUTES"},
            {"id": "c3", "label": "NOT_ENOUGH_INFO"},
            {"id": "c4", "label": "NOT_ENOUGH_INFO"}
        ]
        self.assertAlmostEqual(compute_macro_f1(preds_perfect, gold), 1.0)
        self.assertAlmostEqual(compute_hard_nei_f1(preds_perfect, gold), 1.0)

        # Imperfect predictions: c3 predicted as SUPPORTS
        preds_imperfect = [
            {"id": "c1", "label": "SUPPORTS"},
            {"id": "c2", "label": "REFUTES"},
            {"id": "c3", "label": "SUPPORTS"},
            {"id": "c4", "label": "NOT_ENOUGH_INFO"}
        ]
        # NOT_ENOUGH_INFO: tp=1, fp=0, fn=1 -> prec=1.0, rec=0.5 -> f1 = 2 * 0.5 / 1.5 = 2/3
        self.assertAlmostEqual(compute_hard_nei_f1(preds_imperfect, gold), 2 / 3)
        self.assertLess(compute_macro_f1(preds_imperfect, gold), 1.0)

        # Empty/mismatch tests
        self.assertEqual(compute_macro_f1([], []), 0.0)
        self.assertEqual(compute_hard_nei_f1([], []), 0.0)

    def test_format_verification_latex_table_empty(self):
        latex = format_verification_latex_table({})
        self.assertIn("\\begin{table}", latex)
        self.assertIn("\\begin{tabular}", latex)
        self.assertIn("\\bottomrule", latex)
        self.assertIn("\\end{table}", latex)


if __name__ == "__main__":
    unittest.main()
