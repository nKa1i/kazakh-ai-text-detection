import json
import os
import tempfile
import unittest
from scripts.evaluate_kaz_mage import (
    compute_mage_metrics,
    calculate_domain_degradation,
    evaluate_quadrant_matrix,
    evaluate_kaz_mage_matrix,
    generate_mage_markdown_report,
    _compute_roc_and_youden_pure,
    main as cli_main
)

class TestEvaluateKazMage(unittest.TestCase):
    def test_compute_mage_metrics(self):
        y_true = [0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9]
        metrics = compute_mage_metrics(y_true, y_prob)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertIn("optimal_threshold", metrics)
        self.assertIn("f1", metrics)

    def test_calculate_domain_degradation(self):
        quadrant_results = {
            "Q1": {"roc_auc": 0.9900},
            "Q2": {"roc_auc": 0.9850},
            "Q3": {"roc_auc": 0.9500},
            "Q4": {"roc_auc": 0.9300}
        }
        deg = calculate_domain_degradation(quadrant_results)
        self.assertAlmostEqual(deg["delta_auc_domain"], 0.0400, places=4)
        self.assertAlmostEqual(deg["delta_auc_wild"], 0.0600, places=4)
        self.assertAlmostEqual(deg["delta_auc_generator"], 0.0050, places=4)

    def test_evaluate_quadrant_matrix(self):
        records = [
            {"quadrant": "Q1", "domain": "consumer_reviews", "y_true": 1, "y_prob": 0.9, "gate": 0.45},
            {"quadrant": "Q1", "domain": "consumer_reviews", "y_true": 0, "y_prob": 0.1, "gate": 0.46},
            {"quadrant": "Q2", "domain": "consumer_reviews", "y_true": 1, "y_prob": 0.85, "gate": 0.44},
            {"quadrant": "Q2", "domain": "consumer_reviews", "y_true": 0, "y_prob": 0.15, "gate": 0.47},
            {"quadrant": "Q3", "domain": "news", "y_true": 1, "y_prob": 0.88, "gate": 0.65},
            {"quadrant": "Q3", "domain": "news", "y_true": 0, "y_prob": 0.2, "gate": 0.68},
            {"quadrant": "Q4", "domain": "wikipedia", "y_true": 1, "y_prob": 0.82, "gate": 0.70},
            {"quadrant": "Q4", "domain": "wikipedia", "y_true": 0, "y_prob": 0.25, "gate": 0.72}
        ]
        results = evaluate_quadrant_matrix(records)
        self.assertIn("quadrants", results)
        self.assertIn("degradation", results)
        self.assertIn("gate_dynamics", results)
        self.assertIn("mean_gate_by_domain", results["gate_dynamics"])
        self.assertAlmostEqual(results["gate_dynamics"]["mean_gate_by_domain"]["consumer_reviews"], 0.455, places=3)
        self.assertAlmostEqual(results["gate_dynamics"]["mean_gate_by_domain"]["news"], 0.665, places=3)
        self.assertAlmostEqual(results["gate_dynamics"]["mean_gate_by_domain"]["wikipedia"], 0.710, places=3)

        # Verify gate shift calculations
        self.assertAlmostEqual(results["gate_dynamics"]["delta_gate_news"], 0.210, places=3)
        self.assertAlmostEqual(results["gate_dynamics"]["delta_gate_wiki"], 0.255, places=3)

    def test_bootstrap_ci_and_thresholds(self):
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_prob = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        metrics = compute_mage_metrics(y_true, y_prob, bootstrap_ci=True, n_bootstrap=200)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertLessEqual(metrics["ci_lower"], metrics["roc_auc"])
        self.assertGreaterEqual(metrics["ci_upper"], metrics["roc_auc"])
        self.assertIn("accuracy_optimal", metrics)
        self.assertIn("accuracy_05", metrics)
        self.assertIn("macro_f1", metrics)

    def test_edge_cases(self):
        # Empty input
        empty_metrics = compute_mage_metrics([], [])
        self.assertEqual(empty_metrics["roc_auc"], 0.5)
        self.assertEqual(empty_metrics["n_samples"], 0)

        # Single class (all positives)
        single_class = compute_mage_metrics([1, 1, 1], [0.8, 0.9, 0.7])
        self.assertEqual(single_class["roc_auc"], 0.5)

        # Tied probabilities
        tied = compute_mage_metrics([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(tied["roc_auc"], 0.5)

    def test_markdown_report_generation(self):
        quadrant_results = {
            "Q1": {"roc_auc": 0.9900, "ci_lower": 0.9850, "ci_upper": 0.9950, "accuracy": 0.98, "f1": 0.98, "optimal_threshold": 0.52},
            "Q2": {"roc_auc": 0.9850, "ci_lower": 0.9780, "ci_upper": 0.9910, "accuracy": 0.96, "f1": 0.96, "optimal_threshold": 0.51},
            "Q3": {"roc_auc": 0.9500, "ci_lower": 0.9380, "ci_upper": 0.9620, "accuracy": 0.92, "f1": 0.92, "optimal_threshold": 0.48},
            "Q4": {"roc_auc": 0.9300, "ci_lower": 0.9150, "ci_upper": 0.9450, "accuracy": 0.89, "f1": 0.89, "optimal_threshold": 0.45}
        }
        deg = calculate_domain_degradation(quadrant_results)
        results = {
            "quadrants": quadrant_results,
            "degradation": deg,
            "gate_dynamics": {
                "mean_gate_by_domain": {"consumer_reviews": 0.455, "news": 0.665, "wikipedia": 0.710},
                "delta_gate_news": 0.210,
                "delta_gate_wiki": 0.255
            }
        }
        report = generate_mage_markdown_report(results, model_name="Morpho-Contrastive-v3")
        self.assertIn("# Kaz-MAGE Benchmark Evaluation Report", report)
        self.assertIn("Morpho-Contrastive-v3", report)
        self.assertIn("Q1", report)
        self.assertIn("Q4", report)
        self.assertIn("Generator Degradation", report)
        self.assertIn("Dynamic Gate Routing Dynamics", report)

    def test_human_pairing_in_quadrants(self):
        # AI samples in Q1 (label 1) and separate human controls in consumer_reviews (label 0)
        records = [
            {"quadrant": "Q1", "domain": "consumer_reviews", "label": 1, "prob": 0.95, "gate": 0.45},
            {"quadrant": "Q1", "domain": "consumer_reviews", "label": 1, "prob": 0.85, "gate": 0.46},
            {"quadrant": "human_consumer_reviews", "domain": "consumer_reviews", "label": 0, "prob": 0.10, "gate": 0.44},
            {"quadrant": "human_consumer_reviews", "domain": "consumer_reviews", "label": 0, "prob": 0.20, "gate": 0.47},
        ]
        results = evaluate_quadrant_matrix(records, bootstrap_ci=False)
        self.assertEqual(results["quadrants"]["Q1"]["roc_auc"], 1.0)
        self.assertEqual(results["quadrants"]["Q1"]["accuracy"], 1.0)

    def test_cli_execution(self):
        records = [
            {"quadrant": "Q1", "domain": "consumer_reviews", "y_true": 1, "y_prob": 0.9, "gate": 0.45},
            {"quadrant": "Q1", "domain": "consumer_reviews", "y_true": 0, "y_prob": 0.1, "gate": 0.46},
            {"quadrant": "Q2", "domain": "consumer_reviews", "y_true": 1, "y_prob": 0.85, "gate": 0.44},
            {"quadrant": "Q2", "domain": "consumer_reviews", "y_true": 0, "y_prob": 0.15, "gate": 0.47},
            {"quadrant": "Q3", "domain": "news", "y_true": 1, "y_prob": 0.88, "gate": 0.65},
            {"quadrant": "Q3", "domain": "news", "y_true": 0, "y_prob": 0.2, "gate": 0.68},
            {"quadrant": "Q4", "domain": "wikipedia", "y_true": 1, "y_prob": 0.82, "gate": 0.70},
            {"quadrant": "Q4", "domain": "wikipedia", "y_true": 0, "y_prob": 0.25, "gate": 0.72}
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.json")
            output_json = os.path.join(tmpdir, "output.json")
            output_md = os.path.join(tmpdir, "output.md")

            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(records, f)

            # Test JSON output
            import sys
            orig_argv = sys.argv
            try:
                sys.argv = ["evaluate_kaz_mage.py", "--input", input_path, "--output", output_json, "--format", "json", "--n-bootstrap", "50"]
                cli_main()
                self.assertTrue(os.path.exists(output_json))
                with open(output_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.assertIn("quadrants", data)

                # Test Markdown output
                sys.argv = ["evaluate_kaz_mage.py", "--input", input_path, "--output", output_md, "--format", "markdown", "--n-bootstrap", "50"]
                cli_main()
                self.assertTrue(os.path.exists(output_md))
                with open(output_md, "r", encoding="utf-8") as f:
                    md_text = f.read()
                self.assertIn("Kaz-MAGE Benchmark Evaluation", md_text)
            finally:
                sys.argv = orig_argv

    def test_evaluate_kaz_mage_matrix_mock_model(self):
        try:
            import torch
            def make_logits():
                return torch.tensor([[0.2, 0.8]])
            def make_gate():
                return torch.tensor([0.65])
        except ImportError:
            class MockTensor:
                def __init__(self, data):
                    self.data = data
                    if isinstance(data, list):
                        if data and isinstance(data[0], list):
                            self.shape = (len(data), len(data[0]))
                            self.ndim = 2
                        else:
                            self.shape = (len(data),)
                            self.ndim = 1
                    else:
                        self.shape = ()
                        self.ndim = 0

                def mean(self):
                    return self

                def cpu(self):
                    return self

                def item(self):
                    if isinstance(self.data, list):
                        if isinstance(self.data[0], list):
                            return self.data[0][-1]
                        return self.data[-1]
                    return self.data

                def tolist(self):
                    return self.data

                def __float__(self):
                    return float(self.item())

            def make_logits():
                return MockTensor([[0.2, 0.8]])
            def make_gate():
                return MockTensor([0.65])

        class MockDetector:
            def eval(self):
                pass
            def to(self, device):
                return self
            def __call__(self, *args, **kwargs):
                return {
                    "logits": make_logits(),
                    "gate": make_gate()
                }

        mock_model = MockDetector()
        dataset = [
            {"text": "Сәлем әлем", "label": 1, "quadrant": "Q1", "domain": "consumer_reviews"},
            {"text": "Жақсы тауар", "label": 0, "quadrant": "Q1", "domain": "consumer_reviews"},
        ]

        results = evaluate_kaz_mage_matrix(
            model=mock_model,
            dataset=dataset,
            bootstrap_ci=False
        )

        self.assertIn("records", results)
        records = results["records"]
        self.assertEqual(len(records), 2)

        # Expected softmax probability for [0.2, 0.8]
        import math
        expected_prob = math.exp(0.8) / (math.exp(0.2) + math.exp(0.8))

        rec0 = records[0]
        self.assertEqual(rec0["quadrant"], "Q1")
        self.assertEqual(rec0["y_true"], 1)
        self.assertAlmostEqual(rec0["y_prob"], expected_prob, places=3)
        self.assertAlmostEqual(rec0["gate"], 0.65, places=3)

        rec1 = records[1]
        self.assertEqual(rec1["quadrant"], "Q1")
        self.assertEqual(rec1["y_true"], 0)
        self.assertAlmostEqual(rec1["y_prob"], expected_prob, places=3)
        self.assertAlmostEqual(rec1["gate"], 0.65, places=3)

        # Gate dynamics verification
        self.assertIn("gate_dynamics", results)
        self.assertAlmostEqual(results["gate_dynamics"]["mean_gate_by_quadrant"]["Q1"], 0.65, places=3)
        self.assertAlmostEqual(results["gate_dynamics"]["mean_gate_by_domain"]["consumer_reviews"], 0.65, places=3)

    def test_evaluate_kaz_mage_matrix_predict_conventions(self):
        class ModelWithPredict:
            def predict(self, text):
                return {
                    "probability": 0.75,
                    "gate_semantic_weight": 0.60
                }

        dataset = [
            {"text": "Сынақ мәтіні", "label": 1, "quadrant": "Q1", "domain": "consumer_reviews"},
            {"text": "Адам мәтіні", "label": 0, "quadrant": "Q1", "domain": "consumer_reviews"}
        ]
        res1 = evaluate_kaz_mage_matrix(model=ModelWithPredict(), dataset=dataset, bootstrap_ci=False)
        self.assertAlmostEqual(res1["records"][0]["y_prob"], 0.75, places=3)
        self.assertAlmostEqual(res1["records"][0]["gate"], 0.60, places=3)

        class ModelWithPredictText:
            def predict_text(self, text):
                return {
                    "ai_probability": 0.85,
                    "gate_value": 0.70
                }

        res2 = evaluate_kaz_mage_matrix(model=ModelWithPredictText(), dataset=dataset, bootstrap_ci=False)
        self.assertAlmostEqual(res2["records"][0]["y_prob"], 0.85, places=3)
        self.assertAlmostEqual(res2["records"][0]["gate"], 0.70, places=3)

    def test_youden_threshold_capped_at_one(self):
        # Even if optimal index selects threshold > 1.0, it must be capped at 1.0
        auc, thresh, eer = _compute_roc_and_youden_pure([0, 1], [0.5, 0.9])
        self.assertLessEqual(thresh, 1.0)

if __name__ == "__main__":
    unittest.main()
