"""
Unit and Integration Tests for Training & LOGO Benchmark Pipeline (Task 5).

Tests:
1. DualViewDataset construction and dual-view sample generation (raw vs FST segmented + morphemes).
2. DualViewCollator batch packing for contrastive pairing.
3. Command-line argument parsing and default configurations.
4. Leave-One-Generator-Out (LOGO) evaluation suite across generators.
5. Dry-run end-to-end execution pipeline (defensive local execution).
"""

import os
import sys
import shutil
import tempfile
import unittest

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.train_morpho_contrastive import (
    DualViewDataset,
    DualViewCollator,
    build_arg_parser,
    train_epoch,
    evaluate_model,
    evaluate_logo_benchmark,
    run_training_pipeline
)


class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_records = [
            {"text": "Бұл керемет өнім, қатты ұнады!", "label": 0, "generator": "human", "char_length": 31},
            {"text": "Сапасы өте нашар, сатып алуға кеңес бермеймін.", "label": 0, "generator": "human", "char_length": 45},
            {"text": "Жеткізу қызметінің жылдамдығы таңғалдырды, барлығы ұқыпты оралған.", "label": 1, "generator": "qwen_2.5_7b", "char_length": 65},
            {"text": "Қызмет көрсету сапасы жоғары деңгейде жүзеге асырылды.", "label": 1, "generator": "qwen_2.5_7b", "char_length": 53},
        ]

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dual_view_dataset_construction(self):
        dataset = DualViewDataset(self.sample_records, max_morph_length=16)
        self.assertEqual(len(dataset), 4)

        sample = dataset[0]
        self.assertIn("text_v1", sample)
        self.assertIn("text_v2", sample)
        self.assertIn("morpheme_ids_v1", sample)
        self.assertIn("morpheme_ids_v2", sample)
        self.assertIn("label", sample)

        # View 1 must equal raw text
        self.assertEqual(sample["text_v1"], self.sample_records[0]["text"])
        self.assertEqual(sample["label"], 0)

        # Morpheme sequence must have length <= max_morph_length
        self.assertLessEqual(len(sample["morpheme_ids_v1"]), 16)
        self.assertLessEqual(len(sample["morpheme_ids_v2"]), 16)

    def test_dual_view_collator(self):
        dataset = DualViewDataset(self.sample_records, max_morph_length=16)
        collator = DualViewCollator(max_morph_length=16)
        batch = [dataset[i] for i in range(len(dataset))]
        collated = collator(batch)

        self.assertIn("labels", collated)
        self.assertIn("morpheme_ids", collated)
        self.assertIn("texts_v1", collated)
        self.assertIn("texts_v2", collated)

        # Labels must be present for all samples
        self.assertEqual(len(collated["labels"]), len(self.sample_records))

        # Check morpheme tensor/array dimensions
        morph_ids = collated["morpheme_ids"]
        if hasattr(morph_ids, "shape"):
            self.assertEqual(morph_ids.shape[0], len(self.sample_records))
            self.assertEqual(morph_ids.shape[1], 16)
        else:
            self.assertEqual(len(morph_ids), len(self.sample_records))
            self.assertEqual(len(morph_ids[0]), 16)

    def test_argument_parser_defaults(self):
        parser = build_arg_parser()
        args = parser.parse_args([])

        self.assertTrue(hasattr(args, "train_data"))
        self.assertTrue(hasattr(args, "test_data"))
        self.assertEqual(args.epochs, 4)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.lr_backbone, 2e-5)
        self.assertEqual(args.lr_morph, 1e-4)
        self.assertEqual(args.lambda_supcon, 0.5)
        self.assertEqual(args.output_dir, "output/morpho_contrastive")
        self.assertFalse(args.dry_run)

    def test_evaluate_logo_benchmark(self):
        from models.morpho_contrastive_detector import MorphoContrastiveDetector
        model = MorphoContrastiveDetector(
            morpheme_vocab_size=100,
            embed_dim=768,
            proj_dim=128
        )
        report = evaluate_logo_benchmark(
            model=model,
            test_data=self.sample_records,
            batch_size=2
        )
        self.assertIn("overall", report)
        self.assertIn("generators", report)
        self.assertIn("roc_auc", report["overall"])
        self.assertIn("accuracy", report["overall"])
        self.assertIn("qwen_2.5_7b", report["generators"])

    def test_dry_run_pipeline_execution(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--dry_run",
            "--output_dir", self.temp_dir,
            "--epochs", "1",
            "--batch_size", "2"
        ])

        summary = run_training_pipeline(args)
        self.assertIsInstance(summary, dict)
        self.assertIn("train_epochs", summary)
        self.assertIn("evaluation", summary)
        self.assertIn("output_dir", summary)

        # Check output directory artifacts
        results_file = os.path.join(self.temp_dir, "training_results.json")
        self.assertTrue(os.path.exists(results_file), "training_results.json must be saved in output_dir")


if __name__ == "__main__":
    unittest.main()
