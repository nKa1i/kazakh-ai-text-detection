# tests/test_presentation_figures.py
import os
import unittest
from PIL import Image

EXPECTED_FIGURES = [
    "fig01_subword_vs_fst.png",
    "fig02_domain_collapse.png",
    "fig03_end_to_end_pipeline.png",
    "fig03_tripartite_framework.png",
    "fig04_morpho_gate_arch.png",
    "fig05_chunking_topk_flow.png",
    "fig06_trust_matrix_pipeline.png",
    "fig07_dataset_distribution.png",
    "fig08_roc_curves_kaz_mage.png",
    "fig09_long_doc_injection.png",
    "fig10_kaz_fever_confusion_matrix.png",
    "fig11_four_quadrant_trust_matrix.png",
    "fig12_ablation_study_barchart.png",
    "fig13_gradio_dashboard_panels.png",
    "fig14_cloud_deployment_pipeline.png",
    "fig15_methodological_innovations_framework.png",
]


class TestPresentationFigures(unittest.TestCase):
    def setUp(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.fig_dir = os.path.join(project_root, "presentation_figures")

    def test_all_figures_exist_and_valid(self):
        self.assertTrue(os.path.exists(self.fig_dir), f"Directory {self.fig_dir} must exist")
        for fig_name in EXPECTED_FIGURES:
            path = os.path.join(self.fig_dir, fig_name)
            self.assertTrue(os.path.exists(path), f"Figure {fig_name} must exist")
            size = os.path.getsize(path)
            self.assertGreater(size, 5000, f"Figure {fig_name} must be larger than 5KB")
            with open(path, "rb") as f:
                header = f.read(8)
                self.assertEqual(header[:4], b"\x89PNG", f"Figure {fig_name} must have valid PNG header")

    def test_squarish_aspect_ratios_fig03_04_13_14(self):
        target_figs = [
            "fig03_end_to_end_pipeline.png",
            "fig04_morpho_gate_arch.png",
            "fig13_gradio_dashboard_panels.png",
            "fig14_cloud_deployment_pipeline.png",
        ]
        for fig_name in target_figs:
            path = os.path.join(self.fig_dir, fig_name)
            self.assertTrue(os.path.isfile(path), f"Missing {fig_name}")
            with Image.open(path) as img:
                w, h = img.size
                ratio = w / h
                self.assertGreaterEqual(ratio, 1.10, f"{fig_name} ratio {ratio:.2f} too tall (< 1.10)")
                self.assertLessEqual(ratio, 1.48, f"{fig_name} ratio {ratio:.2f} too wide (> 1.48); expected squarish ~4:3")


if __name__ == "__main__":
    unittest.main()

