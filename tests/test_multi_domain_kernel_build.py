import unittest
import os
import py_compile

class TestMultiDomainKernelBuild(unittest.TestCase):
    def test_kernel_build_and_syntax(self):
        from scripts.prepare_multi_domain_kernel import build_multi_domain_kernel
        kernel_path = os.path.join("kaggle_runner", "test_md_kernel.py")
        build_multi_domain_kernel(output_path=kernel_path, dry_run=True)
        self.assertTrue(os.path.exists(kernel_path))
        py_compile.compile(kernel_path, doraise=True)

        with open(kernel_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Verify all required components are embedded in the generated kernel
        required_symbols = [
            "AdvancedKazakhFSTAnalyzer",
            "MorphemeTokenizer",
            "MorphemeEncoder",
            "MorphoContrastiveDetector",
            "DomainStratifiedBatchSampler",
            "MultiDomainCollator",
            "SupConLoss",
            "InvarianceLoss",
            "compute_mage_metrics",
            "calculate_domain_degradation",
            "evaluate_quadrant_matrix",
            "generate_mage_markdown_report",
            "kaz_multi_domain_benchmark_results.json",
            "kaz_multi_domain_paper_report.md",
        ]
        for sym in required_symbols:
            self.assertIn(sym, code, f"Symbol '{sym}' missing from generated kernel.")

        if os.path.exists(kernel_path):
            os.remove(kernel_path)

if __name__ == "__main__":
    unittest.main()

