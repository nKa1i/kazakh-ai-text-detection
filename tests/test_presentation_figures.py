# tests/test_presentation_figures.py
import os
import unittest

EXPECTED_FIGURES = [
    "fig01_subword_vs_fst.png",
    "fig02_domain_collapse.png",
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
    def test_all_figures_exist_and_valid(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fig_dir = os.path.join(project_root, "presentation_figures")
        self.assertTrue(os.path.exists(fig_dir), f"Directory {fig_dir} must exist")
        for fig_name in EXPECTED_FIGURES:
            path = os.path.join(fig_dir, fig_name)
            self.assertTrue(os.path.exists(path), f"Figure {fig_name} must exist")
            size = os.path.getsize(path)
            self.assertGreater(size, 5000, f"Figure {fig_name} must be larger than 5KB")
            with open(path, "rb") as f:
                header = f.read(8)
                self.assertEqual(header[:4], b"\x89PNG", f"Figure {fig_name} must have valid PNG header")


if __name__ == "__main__":
    unittest.main()
