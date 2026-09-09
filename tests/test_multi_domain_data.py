import json
import os
import tempfile
import unittest

from scripts.generate_multi_domain_train_dataset import (
    audit_dataset_isolation,
    generate_multi_domain_train_dataset
)


class TestMultiDomainData(unittest.TestCase):
    def test_mini_train_dataset_generation_and_balance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_out = os.path.join(tmpdir, "test_train_mini.json")
            dummy_eval = os.path.join(tmpdir, "dummy_eval.json")
            with open(dummy_eval, "w", encoding="utf-8") as f:
                json.dump([{"text": "Бұл тестілік бөлек мәтін.", "id": "eval_1"}], f)

            generate_multi_domain_train_dataset(
                output_path=train_out,
                eval_benchmark_path=dummy_eval,
                samples_per_domain=30,
                seed=42
            )
            self.assertTrue(os.path.exists(train_out))

            with open(train_out, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(len(data), 90)

            # Assert exact 50% human (y=0) and 50% AI (y=1) label balance
            human_samples = [item for item in data if item["label"] == 0]
            ai_samples = [item for item in data if item["label"] == 1]
            self.assertEqual(len(human_samples), 45)
            self.assertEqual(len(ai_samples), 45)

            domain_counts = {}
            for item in data:
                dom = item["domain"]
                domain_counts[dom] = domain_counts.get(dom, 0) + 1
                self.assertNotEqual(item.get("generator"), "Qwen-2.5-7B-Instruct")
                self.assertEqual(
                    item.get("generator"),
                    "Sherkala-7B" if item["label"] == 1 else "human"
                )

                # Assert that no human Wikipedia sample has word count < 30 (verifying zero fallback stubs)
                if item["domain"] == "wikipedia" and item["label"] == 0:
                    self.assertGreaterEqual(
                        item.get("word_count", 0),
                        30,
                        f"Human Wikipedia sample {item['id']} is a stub with word count < 30!"
                    )

            self.assertEqual(domain_counts.get("consumer_reviews"), 30)
            self.assertEqual(domain_counts.get("news"), 30)
            self.assertEqual(domain_counts.get("wikipedia"), 30)

            # Verify audit isolation return dictionary structure and clean status
            audit = audit_dataset_isolation(train_out, dummy_eval)
            self.assertIn("train_total_records", audit)
            self.assertIn("train_unique_hashes", audit)
            self.assertIn("eval_unique_hashes", audit)
            self.assertIn("overlap_count", audit)
            self.assertIn("is_clean", audit)
            self.assertIn("overlapping_hashes", audit)

            self.assertEqual(audit["train_total_records"], 90)
            self.assertTrue(audit["is_clean"])
            self.assertEqual(audit["overlap_count"], 0)
            self.assertEqual(len(audit["overlapping_hashes"]), 0)

    def test_negative_collision_detection(self):
        """Test negative collision: intentional overlap must cause is_clean=False and overlap_count > 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train_collision.json")
            eval_path = os.path.join(tmpdir, "eval_collision.json")

            train_records = [
                {"id": "tr_1", "text": "Бұл ортақ ұқсас сөйлем сынағы болып табылады."},
                {"id": "tr_2", "text": "Каспи дүкеннен тауар алдым, бәрі ұнады."}
            ]
            eval_records = [
                {"id": "ev_1", "text": "Басқа бөлек мәтін мазмұны."},
                {"id": "ev_2", "text": "  Бұл ортақ ұқсас сөйлем сынағы болып табылады.  "}
            ]

            with open(train_path, "w", encoding="utf-8") as f:
                json.dump(train_records, f, ensure_ascii=False)
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(eval_records, f, ensure_ascii=False)

            audit = audit_dataset_isolation(train_path, eval_path)
            self.assertFalse(audit["is_clean"])
            self.assertGreater(audit["overlap_count"], 0)
            self.assertEqual(audit["overlap_count"], 1)
            self.assertEqual(len(audit["overlapping_hashes"]), 1)

    def test_missing_eval_benchmark_file_raises_filenotfound(self):
        """Test that missing evaluation benchmark explicitly raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_out = os.path.join(tmpdir, "test_train_error.json")
            nonexistent_eval = os.path.join(tmpdir, "nonexistent_eval_file.json")

            with self.assertRaises(FileNotFoundError) as cm:
                generate_multi_domain_train_dataset(
                    output_path=train_out,
                    eval_benchmark_path=nonexistent_eval,
                    samples_per_domain=10,
                    seed=42
                )
            self.assertIn("Evaluation benchmark file not found at:", str(cm.exception))

            # Also check audit_dataset_isolation directly with dummy train file
            with open(train_out, "w", encoding="utf-8") as f:
                json.dump([{"text": "Тест"}], f)

            with self.assertRaises(FileNotFoundError) as cm2:
                audit_dataset_isolation(train_out, nonexistent_eval)
            self.assertIn("Evaluation benchmark file not found at:", str(cm2.exception))

    def test_full_dataset_verification_if_present(self):
        """If the canonical 6k train dataset exists, verify 50/50 balance and zero stubs."""
        train_path = "data/kaz_multi_domain_train_6k.json"
        eval_path = "data/kaz_mage_eval_6k.json"

        if not os.path.exists(train_path):
            self.skipTest(f"{train_path} not found; skipping full dataset test.")

        with open(train_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data), 6000)

        # Exact 50% human / 50% AI balance
        human_samples = [d for d in data if d["label"] == 0]
        ai_samples = [d for d in data if d["label"] == 1]
        self.assertEqual(len(human_samples), 3000)
        self.assertEqual(len(ai_samples), 3000)

        # Domain breakdown and zero stubs verification
        for domain in ["consumer_reviews", "news", "wikipedia"]:
            dom_samples = [d for d in data if d["domain"] == domain]
            self.assertEqual(len(dom_samples), 2000)
            dom_human = [d for d in dom_samples if d["label"] == 0]
            dom_ai = [d for d in dom_samples if d["label"] == 1]
            self.assertEqual(len(dom_human), 1000)
            self.assertEqual(len(dom_ai), 1000)

        # Zero fallback stubs for human Wikipedia
        wiki_human = [d for d in data if d["domain"] == "wikipedia" and d["label"] == 0]
        for item in wiki_human:
            self.assertGreaterEqual(
                item["word_count"],
                30,
                f"Human Wikipedia sample {item['id']} is a stub with word count < 30!"
            )

        # Isolation audit against eval benchmark if present
        if os.path.exists(eval_path):
            audit = audit_dataset_isolation(train_path, eval_path)
            self.assertTrue(audit["is_clean"])
            self.assertEqual(audit["overlap_count"], 0)
            self.assertEqual(audit["train_total_records"], 6000)


if __name__ == "__main__":
    unittest.main()
