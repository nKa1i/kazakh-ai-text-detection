import unittest
import os
import json
import tempfile
from scripts.generate_kaz_mage_dataset import extract_prefix, build_synthetic_pair, assemble_kaz_mage_dataset

class TestGenerateKazMage(unittest.TestCase):
    def test_extract_prefix(self):
        text = "Бүгін Астана қаласында халықаралық экономикалық форум өз жұмысын бастады. Жиынға әлемнің отыз елінен өкілдер қатысуда."
        prefix = extract_prefix(text, num_words=5)
        self.assertEqual(prefix, "Бүгін Астана қаласында халықаралық экономикалық")

    def test_build_synthetic_pair(self):
        human_sample = {
            "id": "news_test_001",
            "domain": "news",
            "generator": "human",
            "text": "Бүгін Астана қаласында халықаралық экономикалық форум өз жұмысын бастады. Жиынға әлемнің отыз елінен өкілдер қатысуда. Талқылау барысында жаңа жобалар қаралды.",
            "label": 0
        }
        sherkala_ai = build_synthetic_pair(human_sample, generator="Sherkala-7B", seed=42)
        self.assertEqual(sherkala_ai["generator"], "Sherkala-7B")
        self.assertEqual(sherkala_ai["label"], 1)
        self.assertEqual(sherkala_ai["quadrant"], "Q3")
        self.assertTrue(sherkala_ai["is_unseen_domain"])
        self.assertFalse(sherkala_ai["is_unseen_generator"])
        self.assertTrue(sherkala_ai["text"].startswith(sherkala_ai["prefix"]))

        qwen_ai = build_synthetic_pair(human_sample, generator="Qwen-2.5-7B-Instruct", seed=42)
        self.assertEqual(qwen_ai["generator"], "Qwen-2.5-7B-Instruct")
        self.assertEqual(qwen_ai["label"], 1)
        self.assertEqual(qwen_ai["quadrant"], "Q4")
        self.assertTrue(qwen_ai["is_unseen_domain"])
        self.assertTrue(qwen_ai["is_unseen_generator"])
        self.assertTrue(qwen_ai["text"].startswith(qwen_ai["prefix"]))

    def test_assemble_kaz_mage_dataset_mini(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "mini_mage.json")
            assemble_kaz_mage_dataset(output_path=out_file, num_samples_per_domain=10)
            self.assertTrue(os.path.exists(out_file))
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertGreaterEqual(len(data), 30)
            domains = set(d["domain"] for d in data)
            self.assertIn("news", domains)
            self.assertIn("wikipedia", domains)
            self.assertIn("consumer_reviews", domains)
            # Verify quadrants exist
            quadrants = set(d["quadrant"] for d in data if d["label"] == 1)
            self.assertTrue({"Q1", "Q2", "Q3", "Q4"}.issubset(quadrants))

if __name__ == "__main__":
    unittest.main()
