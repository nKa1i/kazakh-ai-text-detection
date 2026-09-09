import json
import os
import tempfile
import unittest

from scripts.train_multi_domain import (
    MultiDomainCollator,
    MultiDomainDataset,
    build_arg_parser,
    evaluate_validation,
    run_multi_domain_training,
    train_epoch,
    train_multi_domain,
)


class TestTrainMultiDomain(unittest.TestCase):
    def test_dry_run_training_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_file = os.path.join(tmpdir, "train.json")
            eval_file = os.path.join(tmpdir, "eval.json")
            out_dir = os.path.join(tmpdir, "model_out")

            dummy_samples = [
                {"id": f"s{i}", "domain": d, "text": f"Мәтін үлгісі {i}", "label": i % 2, "generator": "Sherkala-7B"}
                for i, d in enumerate(["consumer_reviews", "news", "wikipedia"] * 4)
            ]
            with open(train_file, "w", encoding="utf-8") as f:
                json.dump(dummy_samples, f)
            with open(eval_file, "w", encoding="utf-8") as f:
                json.dump(dummy_samples[:6], f)

            results = run_multi_domain_training(
                train_path=train_file,
                eval_path=eval_file,
                output_dir=out_dir,
                epochs=1,
                batch_size=6,
                dry_run=True,
            )
            self.assertIn("train_loss", results)
            self.assertIn("eval_auc", results)
            self.assertIn("epochs_completed", results)
            self.assertTrue(os.path.exists(out_dir))

            # Verify saved training_results.json
            saved_results_file = os.path.join(out_dir, "training_results.json")
            self.assertTrue(os.path.exists(saved_results_file))
            with open(saved_results_file, "r", encoding="utf-8") as f:
                loaded_results = json.load(f)
            self.assertEqual(loaded_results["epochs_completed"], 1)
            self.assertIn("train_loss", loaded_results)
            self.assertIn("eval_auc", loaded_results)

    def test_multidomain_dataset(self):
        records = [
            {"id": "s1", "domain": "news", "text": "Бұл жаңалық мәтіні.", "label": 0, "generator": "human"},
            {"id": "s2", "domain": "wikipedia", "text": "Бұл энциклопедиялық мақала.", "label": 1, "generator": "Sherkala-7B"},
        ]
        dataset = MultiDomainDataset(records, max_length=128, max_morph_length=32)
        self.assertEqual(len(dataset), 2)

        sample0 = dataset[0]
        self.assertEqual(sample0["id"], "s1")
        self.assertEqual(sample0["domain"], "news")
        self.assertEqual(sample0["label"], 0)
        self.assertIn("morpheme_ids", sample0)
        self.assertIsInstance(sample0["morpheme_ids"], list)
        self.assertLessEqual(len(sample0["morpheme_ids"]), 32)

    def test_multidomain_collator(self):
        records = [
            {"id": "s1", "domain": "news", "text": "Мәтін 1", "label": 0, "morpheme_ids": [1, 2, 3]},
            {"id": "s2", "domain": "wikipedia", "text": "Мәтін 2", "label": 1, "morpheme_ids": [4, 5]},
        ]
        collator = MultiDomainCollator(pad_morph_id=0, max_morph_length=5)
        batch = collator(records)
        self.assertEqual(len(batch["texts"]), 2)
        self.assertEqual(len(batch["domains"]), 2)
        self.assertEqual(len(batch["morpheme_ids"]), 2)
        self.assertEqual(len(batch["morpheme_ids"][0]), 3)  # padded to max in batch (3)
        self.assertEqual(batch["morpheme_ids"][1], [4, 5, 0])

    def test_train_epoch_fallback(self):
        res = train_epoch(model=None, dataloader=None)
        self.assertIn("loss", res)
        self.assertIn("ce_loss", res)
        self.assertIn("supcon_loss", res)
        self.assertIsInstance(res["loss"], float)

    def test_evaluate_validation_records(self):
        val_records = [
            {"id": f"v{i}", "domain": "news", "text": f"Мәтін {i}", "label": i % 2}
            for i in range(10)
        ]
        res = evaluate_validation(model=None, val_data=val_records)
        self.assertIn("roc_auc", res)
        self.assertIn("accuracy", res)
        self.assertGreaterEqual(res["roc_auc"], 0.5)

    def test_cli_argument_parser(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--train_data", "data/test_train.json",
            "--eval_data", "data/test_eval.json",
            "--output_dir", "output/test_out",
            "--epochs", "5",
            "--batch_size", "24",
            "--lr_backbone", "3e-5",
            "--lr_morph", "2e-4",
            "--lambda_supcon", "0.8",
            "--dry_run",
        ])
        self.assertEqual(args.train_data, "data/test_train.json")
        self.assertEqual(args.eval_data, "data/test_eval.json")
        self.assertEqual(args.output_dir, "output/test_out")
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.batch_size, 24)
        self.assertAlmostEqual(args.lr_backbone, 3e-5)
        self.assertAlmostEqual(args.lr_morph, 2e-4)
        self.assertAlmostEqual(args.lambda_supcon, 0.8)
        self.assertTrue(args.dry_run)

    def test_train_multi_domain_interface(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_file = os.path.join(tmpdir, "train.json")
            out_dir = os.path.join(tmpdir, "out")
            dummy_samples = [
                {"id": f"s{i}", "domain": "news", "text": f"Үлгі {i}", "label": i % 2, "generator": "Sherkala-7B"}
                for i in range(6)
            ]
            with open(train_file, "w", encoding="utf-8") as f:
                json.dump(dummy_samples, f)

            results = train_multi_domain(
                train_path=train_file,
                output_dir=out_dir,
                epochs=1,
                batch_size=6,
                dry_run=True,
            )
            self.assertIn("train_loss", results)
            self.assertIn("eval_auc", results)
            self.assertTrue(os.path.exists(out_dir))


if __name__ == "__main__":
    unittest.main()
