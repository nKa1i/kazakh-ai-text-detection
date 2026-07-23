import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import unittest
import numpy as np
from evaluate_statistical_significance import compute_mcnemar_test, compute_bootstrap_ci

class TestStatisticalEval(unittest.TestCase):

    def test_compute_mcnemar_test_identical(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        preds_a = np.array([0, 1, 0, 1, 0, 1])
        preds_b = np.array([0, 1, 0, 1, 0, 1])
        res = compute_mcnemar_test(y_true, preds_a, preds_b)
        self.assertEqual(res["chi2_statistic"], 0.0)
        self.assertEqual(res["p_value"], 1.0)
        self.assertFalse(res["is_statistically_significant"])

    def test_compute_mcnemar_test_disagreement(self):
        y_true = np.array([0]*50 + [1]*50)
        preds_a = np.array([0]*45 + [1]*5 + [1]*45 + [0]*5)
        # B makes errors where A was correct
        preds_b = np.array([0]*30 + [1]*20 + [1]*45 + [0]*5)
        res = compute_mcnemar_test(y_true, preds_a, preds_b)
        self.assertGreater(res["chi2_statistic"], 0.0)
        self.assertIn("contingency_matrix", res)

    def test_compute_bootstrap_ci_structure(self):
        np.random.seed(42)
        y_true = np.array([0]*100 + [1]*100)
        preds_pure = y_true.copy()
        preds_fst = y_true.copy()
        preds_pure[0:10] = 1  # 10 FPs
        preds_fst[0:5] = 1    # 5 FPs
        is_short = np.array([True]*200)

        res = compute_bootstrap_ci(y_true, preds_pure, preds_fst, is_short, n_bootstraps=50, seed=42)
        self.assertIn("pure", res)
        self.assertIn("fst", res)
        self.assertIn("fp_reduction_percent", res)
        self.assertIn("ci_str", res["pure"]["f1"])
        self.assertLessEqual(res["fst"]["fp_short"]["mean"], res["pure"]["fp_short"]["mean"])

if __name__ == "__main__":
    unittest.main()
