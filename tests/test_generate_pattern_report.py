# -*- coding: utf-8 -*-
"""
Unit tests for Phase 3 Comparative Pattern Report Suite:
- Executes scripts/generate_generator_pattern_report.py with --dry-run
- Verifies JSON statistical profile schema (mean, std, 95% CI, Welch t-tests)
- Verifies academic Markdown report sections, LaTeX summary table, and zero emojis
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

from models.pattern_prober import PROBE_METRIC_NAMES
from scripts.generate_generator_pattern_report import (
    evaluate_generator_linguistic_profiles,
    generate_pattern_report,
    main,
    render_comparative_markdown_report,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\ufe0f]")


class TestGenerateGeneratorPatternReport(unittest.TestCase):
    def test_evaluate_profiles_covers_all_families_and_statistical_tests(self):
        report = evaluate_generator_linguistic_profiles(docs_per_family=12, random_seed=42)
        self.assertIn("profiles_by_generator", report)
        self.assertIn("hypothesis_tests_human_vs_ai", report)
        self.assertIn("family_weaknesses", report)

        expected_groups = (
            "ISSAI-KazLLM-70B",
            "Sherkala-8B",
            "DeepSeek-R1",
            "GPT-4o",
            "Qwen-2.5-72B",
            "Llama-3.3-70B-Calque",
            "human",
        )
        for grp in expected_groups:
            self.assertIn(grp, report["profiles_by_generator"])
            grp_metrics = report["profiles_by_generator"][grp]["metrics"]
            for m_name in PROBE_METRIC_NAMES:
                self.assertIn(m_name, grp_metrics)
                stat = grp_metrics[m_name]
                for field in ("mean", "std", "ci95_low", "ci95_high", "n"):
                    self.assertIn(field, stat)
                self.assertLessEqual(stat["ci95_low"], stat["mean"] + 1e-9)
                self.assertGreaterEqual(stat["ci95_high"], stat["mean"] - 1e-9)

        tests = report["hypothesis_tests_human_vs_ai"]
        self.assertLess(tests["chain_len_entropy"]["p_value"], 0.001)
        self.assertGreater(tests["chain_len_entropy"]["mean_diff"], 0.0)
        self.assertLess(tests["copula_ratio"]["p_value"], 0.001)
        self.assertLess(tests["copula_ratio"]["mean_diff"], 0.0)

    def test_dry_run_cli_and_artifact_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "profiles.json")
            out_md = os.path.join(tmpdir, "report.md")

            exit_code = main([
                "--dry-run",
                "--output-json", out_json,
                "--output-md", out_md,
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.isfile(out_json))
            self.assertTrue(os.path.isfile(out_md))

            with open(out_json, "r", encoding="utf-8") as f_json:
                payload = json.load(f_json)
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["docs_per_family"], 4)

            with open(out_md, "r", encoding="utf-8") as f_md:
                md_text = f_md.read()

            self.assertIn("Cross-Model Morphological & Syntactic Profile Table", md_text)
            self.assertIn("Primary Linguistic Weaknesses Identified per LLM Family", md_text)
            self.assertIn("Unseen Detector Generalization Rationale", md_text)
            self.assertIn("\\begin{tabular}", md_text)
            self.assertIsNone(EMOJI_RE.search(md_text))

    def test_subprocess_dry_run_execution(self):
        script_path = os.path.join(PROJECT_ROOT, "scripts", "generate_generator_pattern_report.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "sub_profiles.json")
            out_md = os.path.join(tmpdir, "sub_report.md")
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            proc = subprocess.run(
                [sys.executable, script_path, "--dry-run", "--output-json", out_json, "--output-md", out_md],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(os.path.isfile(out_json))
            self.assertTrue(os.path.isfile(out_md))


if __name__ == "__main__":
    unittest.main()
