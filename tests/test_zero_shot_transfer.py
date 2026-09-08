import unittest
from scripts.evaluate_zero_shot_transfer import compute_transfer_metrics

class TestZeroShotTransfer(unittest.TestCase):
    def test_compute_transfer_metrics(self):
        sample_eval_data = [
            # True Human (label=0)
            {"generator": "human", "label": 0, "prediction": 0, "length_bracket": "short"},
            {"generator": "human", "label": 0, "prediction": 1, "length_bracket": "short"}, # False positive
            {"generator": "human", "label": 0, "prediction": 0, "length_bracket": "medium"},
            # Unseen AI (label=1)
            {"generator": "qwen_2.5_7b", "label": 1, "prediction": 1, "length_bracket": "short"},
            {"generator": "qwen_2.5_7b", "label": 1, "prediction": 0, "length_bracket": "short"}, # False negative
            {"generator": "qwen_2.5_7b", "label": 1, "prediction": 1, "length_bracket": "medium"},
        ]
        results = compute_transfer_metrics(sample_eval_data)
        self.assertIn("overall_accuracy", results)
        self.assertIn("generators", results)
        self.assertIn("length_stratification", results)
        self.assertEqual(results["generators"]["human"]["false_positives"], 1)
        self.assertEqual(results["generators"]["qwen_2.5_7b"]["false_negatives"], 1)

if __name__ == "__main__":
    unittest.main()
