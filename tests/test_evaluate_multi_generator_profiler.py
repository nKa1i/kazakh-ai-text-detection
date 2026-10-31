# -*- coding: utf-8 -*-
"""
Unit tests for Phase 2 Multi-Model Generator Profiling & 5-Fold LOGO + 6th Wild Unseen Benchmark.
"""

import json
import os
import tempfile
import unittest

from scripts.evaluate_multi_generator_profiler import (
    UNSEEN_CALQUE_SENTENCES,
    build_multi_generator_eval_corpus,
    evaluate_logo_unseen_detection,
    evaluate_profiler_attribution_and_ood,
    run_phase2_study,
)


class TestEvaluateMultiGeneratorProfiler(unittest.TestCase):
    def test_corpus_structure_covers_5_known_1_wild_and_human_controls(self):
        self.assertEqual(len(UNSEEN_CALQUE_SENTENCES), 12)
        corpus = build_multi_generator_eval_corpus(docs_per_family=6, random_seed=42)
        # 5 known families * 6 + 1 wild family * 6 + human controls (12) = 48 docs
        self.assertEqual(len(corpus), 48)
        families = {doc["generator"] for doc in corpus}
        for expected in (
            "ISSAI-KazLLM-70B",
            "Sherkala-8B",
            "DeepSeek-R1",
            "GPT-4o",
            "Qwen-2.5-72B",
            "Llama-3.3-70B-Calque",
            "human",
        ):
            self.assertIn(expected, families)

    def test_logo_unseen_binary_detector_beats_closed_set_lexical_baseline(self):
        corpus = build_multi_generator_eval_corpus(docs_per_family=10, random_seed=42)
        logo = evaluate_logo_unseen_detection(corpus)

        self.assertLessEqual(logo["human_fpr_pct"]["proposed_morpho_invariant"], 5.0)
        self.assertGreaterEqual(logo["mean_unseen_recall_pct"]["proposed_morpho_invariant"], 92.0)
        self.assertGreaterEqual(
            logo["mean_unseen_recall_pct"]["proposed_morpho_invariant"]
            - logo["mean_unseen_recall_pct"]["closed_set_lexical_baseline"],
            25.0,
        )
        self.assertGreaterEqual(
            logo["by_held_out_family"]["Llama-3.3-70B-Calque"]["proposed_recall_pct"],
            90.0,
        )

    def test_profiler_closed_set_attribution_and_open_set_ood_rejection(self):
        corpus = build_multi_generator_eval_corpus(docs_per_family=10, random_seed=42)
        prof = evaluate_profiler_attribution_and_ood(corpus)

        self.assertGreaterEqual(prof["closed_set_5family_accuracy_pct"], 85.0)
        self.assertGreaterEqual(prof["wild_calque_ood_detection_rate_pct"], 90.0)
        self.assertIn("family_fingerprint_matrix", prof)
        self.assertEqual(len(prof["family_fingerprint_matrix"]), 7)

    def test_end_to_end_study_generates_json_and_300dpi_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "multi_gen.json")
            out_png = os.path.join(tmpdir, "multi_gen.png")
            res = run_phase2_study(output_json=out_json, output_png=out_png, docs_per_family=8)
            self.assertTrue(os.path.isfile(out_json))
            self.assertTrue(os.path.isfile(out_png))
            self.assertGreater(os.path.getsize(out_png), 15_000)

            with open(out_json, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIn("logo_unseen_detection", loaded)
            self.assertIn("profiler_attribution_and_ood", loaded)
            self.assertEqual(res["total_documents"], loaded["total_documents"])


if __name__ == "__main__":
    unittest.main()
