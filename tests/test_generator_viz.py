# -*- coding: utf-8 -*-
"""
Unit tests for Phase 3 GeneratorFingerprintVisualizer:
- Valid XML SVG output verified via xml.etree.ElementTree.fromstring
- Overlaid 8-axis radar polygon coordinate generation
- Horizontal diverging z-score deviation bar rendering
- NaN/out-of-range resilience, strict XSS escaping, and zero-emoji compliance
"""

import re
import unittest
import xml.etree.ElementTree as ET

from models.generator_viz import GeneratorFingerprintVisualizer
from models.pattern_prober import RADAR_AXIS_KEYS, RADAR_AXIS_LABELS, GeneratorPatternProber


EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\ufe0f]")


class TestGeneratorFingerprintVisualizer(unittest.TestCase):
    def setUp(self):
        self.viz = GeneratorFingerprintVisualizer()
        self.prober = GeneratorPatternProber()

    def test_radar_chart_svg_is_valid_xml_with_overlaid_polygons(self):
        target_text = (
            "Табиғи ресурстарды тиімді пайдалану мәселелері заманауи экологиялық саясаттың басым бағыттарының бірі болып табылады. "
            "Осыған байланысты, экономикалық өсімді қамтамасыз етудің маңызды құралы болып саналады."
        )
        probes = {
            "Target Document": self.prober.normalize_for_radar(self.prober.probe(target_text)),
            "Human Baseline": self.prober.get_human_radar_baseline(),
            "Sherkala-8B Centroid": self.prober.get_family_radar_profile("Sherkala-8B") or {},
        }
        labels = [RADAR_AXIS_LABELS[k]["en"] for k in RADAR_AXIS_KEYS]

        svg_str = self.viz.render_radar_chart_svg(
            probes=probes,
            labels=labels,
            width=500,
            height=400,
            axis_keys=RADAR_AXIS_KEYS,
        )

        self.assertTrue(svg_str.strip().startswith("<svg"))
        root = ET.fromstring(svg_str)
        self.assertTrue(root.tag.endswith("svg"))
        self.assertIn("0 0 500 400", root.attrib.get("viewBox", ""))

        polygons = [elem for elem in root.iter() if elem.tag.endswith("polygon")]
        # 4 concentric grid rings + 3 series polygons = 7 polygons
        self.assertGreaterEqual(len(polygons), 7)
        for poly in polygons:
            pts = poly.attrib.get("points", "").strip()
            self.assertGreaterEqual(len(pts.split()), 8)

        self.assertIsNone(EMOJI_RE.search(svg_str))

    def test_deviation_bars_svg_is_valid_xml_with_diverging_rects(self):
        deviations = {
            "Copula Crutch": 8.42,
            "Chain Entropy": -1.75,
            "Discourse Rigidity": 2.90,
            "Synthetic Verbs": -3.10,
            "SOV Compliance": -0.40,
            "Loan Harmony": 0.15,
            "Lexical TTR": 0.10,
            "Sent. Burstiness": -1.25,
        }
        svg_str = self.viz.render_deviation_bars_svg(deviations, width=500, height=240)
        root = ET.fromstring(svg_str)
        self.assertTrue(root.tag.endswith("svg"))

        rects = [elem for elem in root.iter() if elem.tag.endswith("rect")]
        # Background rect + 8 horizontal bar rects >= 9
        self.assertGreaterEqual(len(rects), 9)
        self.assertIn("+8.42z", svg_str)
        self.assertIn("-3.10z", svg_str)
        self.assertIsNone(EMOJI_RE.search(svg_str))

    def test_xss_escaping_and_edge_case_resilience(self):
        malicious_probes = {
            "<script>alert(1)</script>": {
                "Axis <img onerror=1>": float("nan"),
                "Axis & 2": 4.5,
                "Axis \"3\"": -2.0,
            }
        }
        malicious_labels = ["Axis <img onerror=1>", "Axis & 2", 'Axis "3"']
        radar_svg = self.viz.render_radar_chart_svg(malicious_probes, malicious_labels, width=420, height=340)
        self.assertNotIn("<script>", radar_svg)
        self.assertNotIn("<img", radar_svg)
        ET.fromstring(radar_svg)

        bars_svg = self.viz.render_deviation_bars_svg(
            {"<svg/onload=alert(1)>": float("inf"), "Valid & Metric": -0.5},
            width=420,
            height=180,
        )
        self.assertNotIn("<svg/onload", bars_svg)
        ET.fromstring(bars_svg)

        # Empty input safety
        empty_radar = self.viz.render_radar_chart_svg({}, [], width=400, height=300)
        empty_bars = self.viz.render_deviation_bars_svg({}, width=400, height=180)
        ET.fromstring(empty_radar)
        ET.fromstring(empty_bars)


if __name__ == "__main__":
    unittest.main()
