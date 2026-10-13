"""Validation tests for proposal visual figures extracted from research presentation."""

import os
import unittest
from PIL import Image


class TestExtractedProposalFigures(unittest.TestCase):
    """Test suite ensuring all required PPT figures are extracted and valid."""

    def setUp(self):
        self.fig_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "proposal", "figures")
        )
        self.expected_figures = [
            "fig1_fst_fragmentation.png",
            "fig2_overall_framework.png",
            "fig3_dual_stream_fusion.png",
            "fig4_chunk_pooling.png",
            "fig5_fact_checking_trust_matrix.png",
            "fig6_trust_matrix_validation.png",
            "fig7_gradio_prototype.png",
        ]

    def test_all_figures_exist_and_valid(self):
        """Verify each figure exists, is non-empty, and has width >= 1500px and height >= 1000px."""
        for fig_name in self.expected_figures:
            fig_path = os.path.join(self.fig_dir, fig_name)
            self.assertTrue(os.path.exists(fig_path), f"Figure missing: {fig_name}")
            with Image.open(fig_path) as img:
                self.assertGreaterEqual(
                    img.width, 1500, f"Image {fig_name} width too small: {img.width}"
                )
                self.assertGreaterEqual(
                    img.height, 1000, f"Image {fig_name} height too small: {img.height}"
                )


if __name__ == "__main__":
    unittest.main()
