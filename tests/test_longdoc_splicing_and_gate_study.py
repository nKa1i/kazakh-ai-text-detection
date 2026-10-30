# -*- coding: utf-8 -*-
"""
Unit tests for Stage 2 Long-Document Sliding-Window Splicing & Dynamic Gate-Weight Study.
"""

import json
import os
import tempfile
import unittest

from scripts.evaluate_longdoc_splicing_and_gate_study import (
    INJECTION_RATIOS,
    build_longdoc_splicing_corpus,
    evaluate_dynamic_gate_regimes,
    evaluate_longdoc_splicing,
    run_full_study,
)


class TestLongDocSplicingAndGateStudy(unittest.TestCase):
    def test_corpus_generation_structure_and_word_budgets(self):
        docs = build_longdoc_splicing_corpus(docs_per_ratio=5, target_words=1050, random_seed=42)
        self.assertEqual(len(docs), 5 * len(INJECTION_RATIOS))
        for doc in docs:
            self.assertGreaterEqual(doc["total_words"], 900)
            self.assertIn(doc["target_ratio"], INJECTION_RATIOS)
            if doc["target_ratio"] == 0.0:
                self.assertFalse(doc["has_ai_injection"])
                self.assertIsNone(doc["inject_char_span"])
            else:
                self.assertTrue(doc["has_ai_injection"])
                start_c, end_c = doc["inject_char_span"]
                self.assertGreater(end_c, start_c)
                self.assertEqual(doc["text"][start_c:end_c], doc["injected_text"])

    def test_topk_pooling_beats_mean_pooling_and_truncation_on_sparse_injections(self):
        docs = build_longdoc_splicing_corpus(docs_per_ratio=10, target_words=1050, random_seed=42)
        report = evaluate_longdoc_splicing(docs, max_words=120, calibrated_threshold=0.9980)

        # Pure human 0% control must maintain low false positive rate
        self.assertLessEqual(report["pure_human_fpr"]["proposed_topk_pooling"], 5.0)

        # At 5% and 10% injection ratios, Top-K worst-chunk pooling must overcome mean-pooling dilution
        r05_topk = report["by_ratio"]["0.05"]["proposed_topk_pooling"]["recall_pct"]
        r05_mean = report["by_ratio"]["0.05"]["mean_pooling"]["recall_pct"]
        r10_topk = report["by_ratio"]["0.10"]["proposed_topk_pooling"]["recall_pct"]
        r10_mean = report["by_ratio"]["0.10"]["mean_pooling"]["recall_pct"]

        self.assertGreaterEqual(r05_topk, 85.0)
        self.assertGreaterEqual(r10_topk, 90.0)
        self.assertGreaterEqual(r05_topk - r05_mean, 50.0)
        self.assertGreaterEqual(r10_topk - r10_mean, 50.0)

        # Worst-chunk character span localization must overlap true injected span
        self.assertGreaterEqual(report["overall"]["worst_chunk_localization_recall_pct"], 85.0)

        # Estimated AI content ratio must scale monotonically with true injection ratio
        ratios_est = [
            report["by_ratio"][key]["mean_estimated_ai_ratio_pct"]
            for key in ("0.00", "0.05", "0.10", "0.25", "0.50")
        ]
        for idx in range(1, len(ratios_est)):
            self.assertGreater(ratios_est[idx], ratios_est[idx - 1])

    def test_dynamic_gate_regime_shift_and_noisy_orthographic_f1_gain(self):
        gate_report = evaluate_dynamic_gate_regimes()
        regimes = gate_report["regimes"]

        g_formal = regimes["formal_news_wiki"]["mean_gate"]
        g_colloq = regimes["colloquial_kaspi_social"]["mean_gate"]
        g_noisy = regimes["russian_layout_noisy"]["mean_gate"]

        self.assertGreater(g_formal, 0.58)
        self.assertGreater(g_formal, g_colloq)
        self.assertGreater(g_colloq, g_noisy)
        self.assertLess(g_noisy, 0.40)

        # Statistical significance between clean formal and Russian-layout noisy gate distributions
        self.assertLess(gate_report["significance_formal_vs_noisy"]["welch_p_value"], 1e-10)
        self.assertGreater(gate_report["significance_formal_vs_noisy"]["cohens_d"], 2.0)

        # Auto-FST v2 normalization gain on Russian-layout noisy AI vs Human classification
        noisy_ablation = gate_report["noisy_detection_ablation"]
        self.assertGreaterEqual(noisy_ablation["autofst_v2_f1_pct"], 90.0)
        self.assertGreaterEqual(
            noisy_ablation["autofst_v2_f1_pct"] - noisy_ablation["unnormalized_v1_f1_pct"],
            25.0,
        )

    def test_end_to_end_study_writes_json_and_png_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "results.json")
            out_png = os.path.join(tmpdir, "chart.png")
            payload = run_full_study(output_json=out_json, output_png=out_png, docs_per_ratio=6)
            self.assertTrue(os.path.isfile(out_json))
            self.assertTrue(os.path.isfile(out_png))
            self.assertGreater(os.path.getsize(out_png), 15_000)

            with open(out_json, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIn("splicing_study", loaded)
            self.assertIn("dynamic_gate_study", loaded)
            self.assertEqual(loaded["splicing_study"]["total_documents"], 30)
            self.assertEqual(payload["splicing_study"]["total_documents"], 30)


if __name__ == "__main__":
    unittest.main()
