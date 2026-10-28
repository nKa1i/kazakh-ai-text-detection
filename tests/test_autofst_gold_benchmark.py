# -*- coding: utf-8 -*-
"""Tests for the Gold Morphological & Auto-FST Segmentation Benchmark."""

import os
import json
import tempfile
import unittest

from scripts.evaluate_autofst_benchmark import (
    build_gold_autofst_benchmark,
    evaluate_autofst_on_benchmark,
    VALID_MORPH_CATEGORIES
)


class TestAutoFSTGoldBenchmark(unittest.TestCase):
    def test_build_gold_benchmark_structure_and_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "gold.jsonl")
            items = build_gold_autofst_benchmark(out_path)
            self.assertGreaterEqual(len(items), 120)
            categories = {item["category"] for item in items}
            self.assertEqual(categories, set(VALID_MORPH_CATEGORIES))
            for item in items:
                self.assertIn("surface", item)
                self.assertIn("normalized", item)
                self.assertIn("expected_stem", item)
                self.assertIn("expected_suffixes", item)
                self.assertIsInstance(item["expected_suffixes"], list)

    def test_evaluate_autofst_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_path = os.path.join(tmpdir, "gold.jsonl")
            res_path = os.path.join(tmpdir, "results.json")
            build_gold_autofst_benchmark(bench_path)
            metrics = evaluate_autofst_on_benchmark(bench_path, res_path)
            self.assertIn("overall", metrics)
            self.assertIn("by_category", metrics)
            self.assertGreaterEqual(metrics["overall"]["stem_accuracy"], 0.95)
            self.assertGreaterEqual(metrics["overall"]["exact_segmentation_accuracy"], 0.94)
            self.assertGreaterEqual(metrics["overall"]["boundary_f1"], 0.95)
            self.assertGreaterEqual(metrics["overall"]["normalization_accuracy"], 0.98)


if __name__ == "__main__":
    unittest.main()
