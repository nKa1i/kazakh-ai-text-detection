# -*- coding: utf-8 -*-
"""
tests/test_benchmark_onnx_inference.py: Unit tests for ONNX CPU benchmarking suite.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest


class TestBenchmarkONNXInference(unittest.TestCase):
    def test_benchmark_script_runs_dry_run_and_generates_valid_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "bench_results.json")
            cmd = [
                sys.executable,
                os.path.join("scripts", "benchmark_onnx_inference.py"),
                "--dry-run",
                "--output-json",
                out_json,
            ]
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"

            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            )
            self.assertEqual(
                res.returncode,
                0,
                f"Benchmark script failed with return code {res.returncode}.\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}",
            )
            self.assertTrue(os.path.isfile(out_json))

            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertIn("int8_latency_ms", data)
            int8_stats = data["int8_latency_ms"]
            self.assertIn("p50_ms", int8_stats)
            self.assertIn("p90_ms", int8_stats)
            self.assertIn("p99_ms", int8_stats)
            self.assertIn("mean_ms", int8_stats)

            # Sub-50ms CPU latency constraint
            self.assertLessEqual(int8_stats["p50_ms"], 50.0)

            # Check parity metric
            self.assertIn("mae_probability_vs_pytorch", data)
            self.assertLessEqual(data["mae_probability_vs_pytorch"], 0.05)


if __name__ == "__main__":
    unittest.main()
