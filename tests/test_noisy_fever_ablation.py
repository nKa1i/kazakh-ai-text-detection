# -*- coding: utf-8 -*-
"""Tests for the Orthographically Noisy & Morphophonological FEVER Ablation Benchmark."""

import os
import tempfile
import unittest

from scripts.evaluate_noisy_fever_ablation import (
    build_noisy_fever_stress_set,
    run_noisy_fever_ablation,
)


class TestNoisyFeverAblation(unittest.TestCase):
    def test_build_and_evaluate_noisy_fever_ablation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stress_path = os.path.join(tmpdir, "noisy_fever.jsonl")
            out_json = os.path.join(tmpdir, "ablation.json")
            items = build_noisy_fever_stress_set(stress_path)
            self.assertGreaterEqual(len(items), 30)

            report = run_noisy_fever_ablation(
                stress_path=stress_path,
                corpus_path="data/kazakh_knowledge_corpus.jsonl",
                output_path=out_json
            )
            self.assertIn("legacy_fst_v1", report)
            self.assertIn("autofst_v2", report)
            self.assertIn("improvement", report)
            # Auto-FST v2 must strictly outperform Legacy FST v1 on noisy/alternated claims
            self.assertGreater(
                report["autofst_v2"]["recall_at_1"],
                report["legacy_fst_v1"]["recall_at_1"]
            )
            self.assertGreaterEqual(report["autofst_v2"]["recall_at_5"], 0.85)


if __name__ == "__main__":
    unittest.main()
