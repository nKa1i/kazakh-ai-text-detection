import unittest
import os
import json
import tempfile
from scripts.generate_multi_domain_train_dataset import (
    generate_multi_domain_train_dataset,
    audit_dataset_isolation
)

class TestMultiDomainData(unittest.TestCase):
    def test_mini_train_dataset_generation_and_balance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_out = os.path.join(tmpdir, 'test_train_mini.json')
            dummy_eval = os.path.join(tmpdir, 'dummy_eval.json')
            with open(dummy_eval, 'w', encoding='utf-8') as f:
                json.dump([{'text': 'Бұл тестілік бөлек мәтін.', 'id': 'eval_1'}], f)

            generate_multi_domain_train_dataset(
                output_path=train_out,
                eval_benchmark_path=dummy_eval,
                samples_per_domain=30,
                seed=42
            )
            self.assertTrue(os.path.exists(train_out))

            with open(train_out, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.assertEqual(len(data), 90)
            domain_counts = {}
            for item in data:
                domain_counts[item['domain']] = domain_counts.get(item['domain'], 0) + 1
                self.assertNotEqual(item.get('generator'), 'Qwen-2.5-7B-Instruct')
                self.assertEqual(item.get('generator'), 'Sherkala-7B' if item['label'] == 1 else 'human')

            self.assertEqual(domain_counts.get('consumer_reviews'), 30)
            self.assertEqual(domain_counts.get('news'), 30)
            self.assertEqual(domain_counts.get('wikipedia'), 30)

            audit = audit_dataset_isolation(train_out, dummy_eval)
            self.assertTrue(audit['is_clean'])
            self.assertEqual(audit['overlap_count'], 0)

if __name__ == '__main__':
    unittest.main()
