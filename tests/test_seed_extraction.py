import os
import json
import unittest
import tempfile
from scripts.extract_seed_human_data import extract_stratified_human_reviews

class TestSeedExtraction(unittest.TestCase):
    def test_extract_stratified_human_reviews(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "test_seed.json")
            data = extract_stratified_human_reviews(
                source_csv="dataset_package/data/train.csv",
                output_json=output_path,
                total_samples=100
            )
            self.assertTrue(os.path.exists(output_path))
            self.assertEqual(len(data), 100)
            short_count = sum(1 for d in data if d["length_bracket"] == "short")
            med_count = sum(1 for d in data if d["length_bracket"] == "medium")
            long_count = sum(1 for d in data if d["length_bracket"] == "long")
            self.assertEqual(short_count, 30)
            self.assertEqual(med_count, 45)
            self.assertEqual(long_count, 25)
            self.assertEqual(data[0]["domain"], "consumer_reviews")
            self.assertIn("char_length", data[0])

if __name__ == "__main__":
    unittest.main()
