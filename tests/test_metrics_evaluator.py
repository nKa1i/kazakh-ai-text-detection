import unittest
import math
import time

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

class TestMetricsEvaluator(unittest.TestCase):
    def test_metrics_and_roc_auc_calibration(self):
        """Plan-specified integration test for compute_comprehensive_metrics."""
        from scripts.metrics_evaluator import compute_comprehensive_metrics
        if HAS_NUMPY:
            y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
            y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 0.95])
            lengths = np.array([40, 50, 70, 90, 45, 55, 75, 100])
        else:
            y_true = [0, 0, 0, 0, 1, 1, 1, 1]
            y_prob = [0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 0.95]
            lengths = [40, 50, 70, 90, 45, 55, 75, 100]

        res = compute_comprehensive_metrics(y_true, y_prob, lengths)
        self.assertIn("roc_auc", res)
        self.assertIn("optimal_threshold", res)
        self.assertIn("eer", res)
        self.assertEqual(res["roc_auc"], 1.0)
        self.assertIn("short", res["length_stratification"])
        self.assertEqual(res["length_stratification"]["short"]["fp"], 0)

    def test_compute_comprehensive_metrics_perfect_ranking(self):
        from scripts.metrics_evaluator import compute_comprehensive_metrics
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_prob = [0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 0.95]
        lengths = [40, 50, 70, 90, 45, 55, 75, 100]

        res = compute_comprehensive_metrics(y_true, y_prob, lengths=lengths, threshold=0.5)

        # Threshold-independent metrics
        self.assertIn("roc_auc", res)
        self.assertIn("optimal_threshold", res)
        self.assertIn("eer", res)
        self.assertEqual(res["roc_auc"], 1.0)
        self.assertAlmostEqual(res["optimal_threshold"], 0.7, places=2)
        self.assertEqual(res["eer"], 0.0)

        # Fixed threshold metrics (threshold=0.5)
        self.assertEqual(res["accuracy"], 100.0)
        self.assertEqual(res["f1"], 100.0)
        self.assertEqual(res["precision"], 100.0)
        self.assertEqual(res["recall"], 100.0)
        self.assertEqual(res["tn"], 4)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 0)
        self.assertEqual(res["tp"], 4)
        self.assertEqual(res["fpr"], 0.0)
        self.assertEqual(res["fnr"], 0.0)

        # Length stratification:
        # short: <= 60 -> indices 0 (40), 1 (50), 4 (45), 5 (55) -> 4 samples
        # medium: 61..85 -> indices 2 (70), 6 (75) -> 2 samples
        # long: > 85 -> indices 3 (90), 7 (100) -> 2 samples
        self.assertIn("length_stratification", res)
        strat = res["length_stratification"]
        self.assertIn("short", strat)
        self.assertIn("medium", strat)
        self.assertIn("long", strat)

        self.assertEqual(strat["short"]["total"], 4)
        self.assertEqual(strat["short"]["accuracy"], 100.0)
        self.assertEqual(strat["short"]["fp"], 0)
        self.assertEqual(strat["short"]["fn"], 0)
        self.assertEqual(strat["short"]["fpr"], 0.0)
        self.assertEqual(strat["short"]["fnr"], 0.0)

        self.assertEqual(strat["medium"]["total"], 2)
        self.assertEqual(strat["medium"]["accuracy"], 100.0)
        self.assertEqual(strat["long"]["total"], 2)
        self.assertEqual(strat["long"]["accuracy"], 100.0)

    def test_compute_comprehensive_metrics_imperfect(self):
        from scripts.metrics_evaluator import compute_comprehensive_metrics
        # 4 negatives, 4 positives
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        # index 1 is negative with prob 0.6 (FP at 0.5)
        # index 4 is positive with prob 0.3 (FN at 0.5)
        y_prob = [0.2, 0.6, 0.3, 0.4, 0.3, 0.8, 0.9, 0.7]
        lengths = [30, 40, 70, 90, 50, 60, 80, 120]

        res = compute_comprehensive_metrics(y_true, y_prob, lengths=lengths, threshold=0.5)

        # y_pred at 0.5: [0, 1, 0, 0, 0, 1, 1, 1]
        # TN=3, FP=1, FN=1, TP=3
        self.assertEqual(res["tn"], 3)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 1)
        self.assertEqual(res["tp"], 3)
        self.assertEqual(res["accuracy"], 75.0)
        self.assertEqual(res["precision"], 75.0)
        self.assertEqual(res["recall"], 75.0)
        self.assertEqual(res["f1"], 75.0)
        self.assertEqual(res["fpr"], 25.0)
        self.assertEqual(res["fnr"], 25.0)

        # Length stratification check for short (lengths <= 60):
        # indices 0 (30, y=0, p=0.2 -> TN)
        # index 1 (40, y=0, p=0.6 -> FP)
        # index 4 (50, y=1, p=0.3 -> FN)
        # index 5 (60, y=1, p=0.8 -> TP)
        short_res = res["length_stratification"]["short"]
        self.assertEqual(short_res["total"], 4)
        self.assertEqual(short_res["fp"], 1)
        self.assertEqual(short_res["fn"], 1)
        self.assertEqual(short_res["accuracy"], 50.0)
        self.assertEqual(short_res["fpr"], 50.0)
        self.assertEqual(short_res["fnr"], 50.0)

    def test_single_class_or_empty_edge_cases(self):
        from scripts.metrics_evaluator import compute_comprehensive_metrics
        # All negatives
        res = compute_comprehensive_metrics([0, 0, 0], [0.1, 0.2, 0.3])
        self.assertEqual(res["roc_auc"], 0.5)
        self.assertEqual(res["optimal_threshold"], 0.5)
        self.assertEqual(res["eer"], 0.5)
        self.assertEqual(res["accuracy"], 100.0)

        # All positives
        res_pos = compute_comprehensive_metrics([1, 1, 1], [0.8, 0.9, 0.7])
        self.assertEqual(res_pos["roc_auc"], 0.5)
        self.assertEqual(res_pos["accuracy"], 100.0)

    def test_duck_array_support(self):
        from scripts.metrics_evaluator import compute_comprehensive_metrics
        # Simulates numpy array or torch tensor having .tolist()
        class DuckArray:
            def __init__(self, data):
                self.data = data
            def tolist(self):
                return self.data

        y_true = DuckArray([0, 0, 1, 1])
        y_prob = DuckArray([0.1, 0.2, 0.8, 0.9])
        lengths = DuckArray([50, 55, 65, 95])

        res = compute_comprehensive_metrics(y_true, y_prob, lengths=lengths)
        self.assertEqual(res["roc_auc"], 1.0)
        self.assertEqual(res["accuracy"], 100.0)
        self.assertIn("short", res["length_stratification"])

    def test_compute_mcnemar_test(self):
        from scripts.metrics_evaluator import compute_mcnemar_test
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        p_base = [0, 1, 0, 0, 1, 0, 1, 1]  # 6/8 correct: errors at idx 1 (FP), idx 5 (FN)
        p_prop = [0, 0, 0, 0, 1, 1, 1, 1]  # 8/8 correct

        # Evaluate on all samples
        res = compute_mcnemar_test(y_true, p_base, p_prop)
        self.assertIn("contingency_matrix", res)
        self.assertIn("chi2_statistic", res)
        self.assertIn("p_value", res)
        self.assertIn("is_significant", res)

        # Contingency table:
        # n00 (both correct) = 6
        # n01 (base correct, prop wrong) = 0
        # n10 (base wrong, prop correct) = 2
        # n11 (both wrong) = 0
        self.assertEqual(res["contingency_matrix"], [[6, 0], [2, 0]])
        # chi2 = (|0 - 2| - 1)^2 / 2 = 1 / 2 = 0.5
        self.assertAlmostEqual(res["chi2_statistic"], 0.5, places=3)
        self.assertFalse(res["is_significant"])

        # Test with short_mask
        short_mask = [True, True, False, False, True, True, False, False]
        # In short subset:
        # y_true: [0, 0, 1, 1]
        # p_base: [0, 1, 1, 0] (2 correct: idx 0, idx 4; 2 wrong: idx 1, idx 5)
        # p_prop: [0, 0, 1, 1] (4 correct)
        res_short = compute_mcnemar_test(y_true, p_base, p_prop, short_mask=short_mask)
        self.assertEqual(res_short["contingency_matrix"], [[2, 0], [2, 0]])
        self.assertAlmostEqual(res_short["chi2_statistic"], 0.5, places=3)

        # Test highly significant difference
        # 30 cases where baseline is wrong and proposed is correct
        y_true_sig = [1] * 30
        p_base_sig = [0] * 30
        p_prop_sig = [1] * 30
        res_sig = compute_mcnemar_test(y_true_sig, p_base_sig, p_prop_sig)
        self.assertEqual(res_sig["contingency_matrix"], [[0, 0], [30, 0]])
        self.assertTrue(res_sig["is_significant"])
        self.assertLess(res_sig["p_value"], 0.001)

    def test_measure_inference_efficiency(self):
        from scripts.metrics_evaluator import measure_inference_efficiency

        def dummy_predictor(batch):
            time.sleep(0.001)
            return [0.5] * len(batch)

        texts = ["Бұл жақсы өнім", "Сапасы төмен", "Керемет дүкен!"]
        res = measure_inference_efficiency(dummy_predictor, texts, num_runs=5)

        self.assertIn("latency_ms_per_sample", res)
        self.assertIn("throughput_samples_per_sec", res)
        self.assertIn("total_time_sec", res)
        self.assertEqual(res["num_samples"], 3)
        self.assertEqual(res["num_runs"], 5)
        self.assertGreater(res["latency_ms_per_sample"], 0.0)
        self.assertGreater(res["throughput_samples_per_sec"], 0.0)

        # Single string predictor support
        def single_predictor(text):
            time.sleep(0.0005)
            return {"score": len(text)}

        res_single = measure_inference_efficiency(single_predictor, texts, num_runs=3)
        self.assertEqual(res_single["num_samples"], 3)
        self.assertEqual(res_single["num_runs"], 3)
        self.assertGreater(res_single["latency_ms_per_sample"], 0.0)

if __name__ == "__main__":
    unittest.main()
