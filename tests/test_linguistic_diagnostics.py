import unittest
from scripts.compute_linguistic_diagnostics import (
    calculate_token_inflation,
    calculate_lexical_diversity,
    detect_code_switched_loanwords,
    compute_dataset_profile
)

class TestLinguisticDiagnostics(unittest.TestCase):
    def test_token_inflation(self):
        text = "Бұл өте жақсы және ыңғайлы қосымша екен"
        tw = calculate_token_inflation(text)
        self.assertGreaterEqual(tw, 1.0)
        self.assertIsInstance(tw, float)

    def test_lexical_diversity(self):
        texts = ["жақсы тауар", "өте жақсы тауар сапасы"]
        res = calculate_lexical_diversity(texts)
        self.assertIn("ttr", res)
        self.assertIn("distinct_1", res)
        self.assertIn("distinct_2", res)
        self.assertGreater(res["ttr"], 0.0)

    def test_loanword_detection(self):
        # доставка and каспи are common commercial loanwords
        text = "Доставкасы өте тез болды, каспиге рақмет"
        loanwords = detect_code_switched_loanwords(text)
        self.assertGreaterEqual(len(loanwords), 1)

    def test_dataset_profile(self):
        sample_data = [
            {"text": "Керемет тауар!", "generator": "human", "length_bracket": "short"},
            {"text": "Сапасы жақсы, жеткізу тез болды.", "generator": "human", "length_bracket": "medium"},
            {"text": "Өте керемет сатып алу болды.", "generator": "qwen_2.5_7b", "length_bracket": "short"}
        ]
        summary = compute_dataset_profile(sample_data)
        self.assertIn("human", summary)
        self.assertIn("qwen_2.5_7b", summary)
        self.assertIn("overall_ttr", summary["human"])
        self.assertIn("avg_tw", summary["human"])

if __name__ == "__main__":
    unittest.main()
