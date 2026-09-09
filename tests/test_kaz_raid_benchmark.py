import unittest


class TestKazRaidBenchmark(unittest.TestCase):
    def test_condition_generation_dry_run(self):
        from scripts.generate_kaz_raid_dataset import generate_benchmark_splits
        sample_records = [
            {"id": "0", "label": 0, "text": "Каспиден тауар алдым, өте жақсы сапа."},
            {"id": "1", "label": 1, "text": "Бұл өнім жоғары сапалы стандарттарға сәйкес келеді."}
        ]
        benchmark = generate_benchmark_splits(
            sample_records,
            rates=[0.10],
            selected_attacks=["homoglyph_swap"],
            seed=42
        )
        self.assertIn("clean", benchmark)
        self.assertIn("homoglyph_swap_rate_0.10", benchmark)
        self.assertEqual(len(benchmark["clean"]), 2)
        self.assertEqual(len(benchmark["homoglyph_swap_rate_0.10"]), 2)
        self.assertEqual(benchmark["homoglyph_swap_rate_0.10"][0]["condition"], "homoglyph_swap_rate_0.10")
        self.assertEqual(benchmark["homoglyph_swap_rate_0.10"][0]["perturbation_rate"], 0.10)
        self.assertEqual(benchmark["homoglyph_swap_rate_0.10"][0]["tier"], "tier1_orthographic")

    def test_all_28_conditions_generation(self):
        from scripts.generate_kaz_raid_dataset import generate_benchmark_splits
        sample_records = [
            {"id": "0", "label": 0, "text": "Каспиден тауар алдым, өте жақсы сапа."},
            {"id": "1", "label": 1, "text": "Бұл өнім жоғары сапалы стандарттарға сәйкес келеді."}
        ]
        # 1 clean + 9 attacks * 3 rates = 28 conditions
        benchmark = generate_benchmark_splits(
            sample_records,
            rates=[0.05, 0.10, 0.20],
            seed=42
        )
        self.assertEqual(len(benchmark), 28)
        self.assertIn("clean", benchmark)
        self.assertIn("homoglyph_swap_rate_0.05", benchmark)
        self.assertIn("suffix_tamperer_rate_0.20", benchmark)
        self.assertIn("loanword_swap_rate_0.10", benchmark)

    def test_evaluate_benchmark_and_degradation(self):
        from scripts.evaluate_kaz_raid import evaluate_benchmark_results, compute_bootstrap_ci
        # Mock benchmark results for clean and perturbed
        y_true = [0, 0, 1, 1]
        lengths = [30, 40, 70, 90]
        # Perfect clean predictions
        clean_probs = [0.05, 0.10, 0.90, 0.95]
        # Degraded perturbed predictions (ranking inverted: negative has higher prob than positive)
        perturbed_probs = [0.10, 0.45, 0.25, 0.85]


        condition_predictions = {
            "clean": clean_probs,
            "homoglyph_swap_rate_0.20": perturbed_probs
        }

        results = evaluate_benchmark_results(
            y_true=y_true,
            condition_predictions=condition_predictions,
            lengths=lengths,
            n_bootstraps=100
        )

        self.assertIn("clean", results["conditions"])
        self.assertIn("homoglyph_swap_rate_0.20", results["conditions"])
        self.assertEqual(results["conditions"]["clean"]["roc_auc"], 1.0)
        self.assertLess(results["conditions"]["homoglyph_swap_rate_0.20"]["roc_auc"], 1.0)
        self.assertIn("asr", results["conditions"]["homoglyph_swap_rate_0.20"])
        self.assertIn("degradation", results)

    def test_compute_bootstrap_ci(self):
        from scripts.evaluate_kaz_raid import compute_bootstrap_ci
        y_true = [0] * 50 + [1] * 50
        y_prob = [0.1] * 45 + [0.8] * 5 + [0.2] * 5 + [0.9] * 45
        ci = compute_bootstrap_ci(y_true, y_prob, n_bootstraps=100, seed=42)
        self.assertIn("auc_mean", ci)
        self.assertIn("auc_ci_lower", ci)
        self.assertIn("auc_ci_upper", ci)
        self.assertLessEqual(ci["auc_ci_lower"], ci["auc_mean"])
        self.assertGreaterEqual(ci["auc_ci_upper"], ci["auc_mean"])


if __name__ == "__main__":
    unittest.main()
