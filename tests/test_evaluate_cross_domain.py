# -*- coding: utf-8 -*-
"""
Unit tests for Phase 4 Cross-Domain Evaluation Runner (scripts/evaluate_cross_domain_robustness.py):
- CLI argument parsing (--benchmark-path, --output-json, --output-report, --dry-run)
- Fast --dry-run execution (2 records per domain = 10 total records)
- Full 500-record acceptance criteria verification:
  * ROC-AUC >= 0.92 across all 5 genres
  * FPR@95%TPR <= 0.08 on Academic and Official/Legal domains
  * Zero human false positives at global calibrated threshold tau = 0.9980
- Markdown/LaTeX report and JSON artifact generation with zero decorative emojis
"""

import json
import os
import re
import tempfile
import unittest

from scripts.evaluate_cross_domain_robustness import (
    parse_args,
    render_cross_domain_markdown_report,
    run_cross_domain_evaluation,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "data", "cross_domain_benchmark.json")
RESULTS_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "cross_domain_evaluation_results.json")
REPORT_MD_PATH = os.path.join(
    PROJECT_ROOT, "docs", "reports", "2026-10-31-cross-domain-benchmark-report.md"
)
EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\ufe0f]")


class TestEvaluateCrossDomainRobustness(unittest.TestCase):
    def test_parse_args_defaults_and_flags(self):
        args_default = parse_args([])
        self.assertFalse(args_default.dry_run)
        self.assertTrue(args_default.benchmark_path.endswith("cross_domain_benchmark.json"))
        self.assertTrue(args_default.output_json.endswith("cross_domain_evaluation_results.json"))
        self.assertTrue(args_default.output_report.endswith("2026-10-31-cross-domain-benchmark-report.md"))

        args_custom = parse_args([
            "--benchmark-path", "custom_bench.json",
            "--output-json", "custom_out.json",
            "--output-report", "custom_rep.md",
            "--dry-run",
        ])
        self.assertTrue(args_custom.dry_run)
        self.assertEqual(args_custom.benchmark_path, "custom_bench.json")
        self.assertEqual(args_custom.output_json, "custom_out.json")
        self.assertEqual(args_custom.output_report, "custom_rep.md")

    def test_dry_run_execution_creates_json_and_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "dry_results.json")
            out_md = os.path.join(tmpdir, "dry_report.md")
            res = run_cross_domain_evaluation(
                benchmark_path=BENCHMARK_PATH,
                output_json=out_json,
                output_report=out_md,
                dry_run=True,
            )
            self.assertTrue(os.path.isfile(out_json))
            self.assertTrue(os.path.isfile(out_md))
            self.assertTrue(res["dry_run"])
            self.assertEqual(res["total_evaluated"], 10)
            self.assertEqual(len(res["per_domain"]), 5)

            with open(out_md, "r", encoding="utf-8") as f:
                md_text = f.read()
            self.assertIn("Cross-Domain Multi-Genre Robustness Benchmark", md_text)
            self.assertIn("\\begin{table}", md_text)
            self.assertIsNone(EMOJI_RE.search(md_text))

    def test_full_benchmark_meets_phase4_acceptance_criteria(self):
        if not os.path.isfile(RESULTS_JSON_PATH) or not os.path.isfile(REPORT_MD_PATH):
            run_cross_domain_evaluation(
                benchmark_path=BENCHMARK_PATH,
                output_json=RESULTS_JSON_PATH,
                output_report=REPORT_MD_PATH,
                dry_run=False,
            )

        with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)

        self.assertEqual(results["total_evaluated"], 500)
        per_dom = results["per_domain"]
        for dom_key in ("news", "academic", "official_legal", "fiction_literature", "social_casual"):
            self.assertIn(dom_key, per_dom)
            m = per_dom[dom_key]
            self.assertGreaterEqual(
                m["roc_auc"],
                0.92,
                msg=f"Domain {dom_key} ROC-AUC {m['roc_auc']} must be >= 0.92",
            )
            self.assertEqual(
                m["fpr_at_calibrated_tau"],
                0.0,
                msg=f"Domain {dom_key} must have zero human false positives at tau = 0.9980",
            )
            self.assertIn("transfer_gap_auc", m)
            self.assertIn("attribution_accuracy", m)
            self.assertIn("unseen_ood_recall", m)
            self.assertIn("threshold_stability_curve", m)

        self.assertLessEqual(per_dom["academic"]["fpr_at_95_tpr"], 0.08)
        self.assertLessEqual(per_dom["official_legal"]["fpr_at_95_tpr"], 0.08)

        md_report = render_cross_domain_markdown_report(results)
        self.assertIsNone(EMOJI_RE.search(md_report))


if __name__ == "__main__":
    unittest.main()
