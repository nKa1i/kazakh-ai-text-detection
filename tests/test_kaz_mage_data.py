import unittest
import os
import json
import tempfile
from kaz_mage.data import KazMageSample, load_mage_dataset, filter_quadrant, get_quadrant_slices

class TestKazMageData(unittest.TestCase):
    def setUp(self):
        self.sample_records = [
            {
                "id": "s1",
                "domain": "consumer_reviews",
                "generator": "Sherkala-7B",
                "is_unseen_domain": False,
                "is_unseen_generator": False,
                "quadrant": "Q1",
                "prefix": "Өте жақсы",
                "text": "Өте жақсы тауар, бәрі ұнады.",
                "label": 1,
                "char_length": 28,
                "word_count": 5
            },
            {
                "id": "s2",
                "domain": "consumer_reviews",
                "generator": "Qwen-2.5-7B-Instruct",
                "is_unseen_domain": False,
                "is_unseen_generator": True,
                "quadrant": "Q2",
                "prefix": "Каспиден алдым",
                "text": "Каспиден алдым, өте тез жеткізді.",
                "label": 1,
                "char_length": 33,
                "word_count": 4
            },
            {
                "id": "s3",
                "domain": "news",
                "generator": "Sherkala-7B",
                "is_unseen_domain": True,
                "is_unseen_generator": False,
                "quadrant": "Q3",
                "prefix": "Қазақстанда жаңа заң",
                "text": "Қазақстанда жаңа заң қабылданды. Үкімет отырысында осы мәселе қаралды.",
                "label": 1,
                "char_length": 70,
                "word_count": 8
            },
            {
                "id": "s4",
                "domain": "wikipedia",
                "generator": "Qwen-2.5-7B-Instruct",
                "is_unseen_domain": True,
                "is_unseen_generator": True,
                "quadrant": "Q4",
                "prefix": "Алматы қаласы",
                "text": "Алматы қаласы – Қазақстанның ірі мәдени орталығы.",
                "label": 1,
                "char_length": 49,
                "word_count": 6
            },
            {
                "id": "s0",
                "domain": "news",
                "generator": "human",
                "is_unseen_domain": True,
                "is_unseen_generator": False,
                "quadrant": "human_news",
                "prefix": "",
                "text": "Үкімет жаңа қаулы қабылдады.",
                "label": 0,
                "char_length": 28,
                "word_count": 4
            }
        ]
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(self.sample_records, self.temp_file, ensure_ascii=False)
        self.temp_file.close()

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_load_mage_dataset(self):
        dataset = load_mage_dataset(self.temp_file.name)
        self.assertEqual(len(dataset), 5)
        self.assertIsInstance(dataset[0], KazMageSample)
        self.assertEqual(dataset[0].quadrant, "Q1")

    def test_filter_quadrant(self):
        dataset = load_mage_dataset(self.temp_file.name)
        q1 = filter_quadrant(dataset, "Q1")
        self.assertEqual(len(q1), 1)
        self.assertEqual(q1[0].id, "s1")

        q4 = filter_quadrant(dataset, "Q4")
        self.assertEqual(len(q4), 1)
        self.assertEqual(q4[0].id, "s4")

    def test_get_quadrant_slices(self):
        dataset = load_mage_dataset(self.temp_file.name)
        slices = get_quadrant_slices(dataset)
        self.assertIn("Q1", slices)
        self.assertIn("Q2", slices)
        self.assertIn("Q3", slices)
        self.assertIn("Q4", slices)
        self.assertEqual(len(slices["Q1"]), 1)
        self.assertEqual(len(slices["Q4"]), 1)

if __name__ == "__main__":
    unittest.main()