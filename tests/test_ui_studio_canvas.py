# -*- coding: utf-8 -*-
"""Unit tests verifying Notion Light canvas enforcement and CSS de-boxification."""

import unittest
from ui.app import APP_CSS
from hf_space.ui.app import APP_CSS as HF_APP_CSS

class TestUIStudioCanvas(unittest.TestCase):
    def test_light_canvas_enforced(self):
        self.assertIn("color-scheme: light", APP_CSS)
        self.assertIn("#fbfbfa", APP_CSS)

    def test_dark_mode_override_present(self):
        self.assertIn(".dark", APP_CSS)
        self.assertIn(".dark .gradio-container", APP_CSS)

    def test_live_stats_and_diff_css_classes(self):
        self.assertIn(".live-stats-ribbon", APP_CSS)
        self.assertIn(".diff-norm", APP_CSS)
        self.assertIn(".diff-induce", APP_CSS)

    def test_hf_parity(self):
        self.assertEqual(APP_CSS, HF_APP_CSS)

if __name__ == '__main__':
    unittest.main()
