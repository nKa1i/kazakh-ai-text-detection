# -*- coding: utf-8 -*-
"""Tests for the Span-Level Mixed Human-AI Provenance Benchmark."""

import os
import json
import tempfile
import unittest

from scripts.build_hybrid_mixed_benchmark import (
    build_hybrid_mixed_dataset,
    evaluate_span_boundary_detection,
    PROVENANCE_CLASSES,
    GENERATOR_FAMILIES
)


class TestHybridMixedBenchmark(unittest.TestCase):
    def test_dataset_schema_and_provenance_balance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "hybrid_spans.jsonl")
            docs = build_hybrid_mixed_dataset(out_path, num_per_class=25)
            self.assertEqual(len(docs), 100)
            classes = {d["provenance_class"] for d in docs}
            self.assertEqual(classes, set(PROVENANCE_CLASSES))
            generators = {d["generator"] for d in docs if d["generator"] != "human"}
            self.assertTrue({"ISSAI-KazLLM-70B", "Sherkala-8B", "DeepSeek-R1", "GPT-4o"}.issubset(generators))
            for d in docs:
                self.assertEqual(len(d["sentences"]), len(d["sentence_labels"]))
                self.assertGreaterEqual(len(d["sentences"]), 4)
                self.assertTrue(all(lbl in (0, 1) for lbl in d["sentence_labels"]))

    def test_span_boundary_evaluation_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "hybrid_spans.jsonl")
            docs = build_hybrid_mixed_dataset(out_path, num_per_class=10)
            metrics = evaluate_span_boundary_detection(docs)
            self.assertIn("sentence_macro_f1", metrics)
            self.assertIn("boundary_iou", metrics)
            self.assertIn("by_generator", metrics)
            self.assertGreater(metrics["sentence_macro_f1"], 0.65)


if __name__ == "__main__":
    unittest.main()
