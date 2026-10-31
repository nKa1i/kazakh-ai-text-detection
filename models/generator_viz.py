# -*- coding: utf-8 -*-
"""
models/generator_viz.py: Zero-dependency SVG Visualizer for Kazakh LLM Linguistic Fingerprints.

Renders publication-grade, theme-adaptive, XSS-safe SVG vector graphics using only
the Python standard library (no matplotlib, seaborn, or browser JS dependencies):
  1. render_radar_chart_svg   - 8-axis overlaid polygon radar chart comparing target prose
                                against the Authentic Human Baseline and model centroids.
  2. render_deviation_bars_svg - Horizontal diverging z-score deviation bar chart highlighting
                                anomalous morphosyntactic dimensions.
"""

from __future__ import annotations

import html
import math
from typing import Dict, List, Optional, Sequence, Tuple

from models.pattern_prober import RADAR_AXIS_KEYS


SERIES_COLORS: Tuple[Tuple[str, float], ...] = (
    ("#38bdf8", 0.24),  # Sky blue: Target Document
    ("#34d399", 0.16),  # Emerald green: Authentic Human Baseline
    ("#f472b6", 0.16),  # Rose pink: Attributed Model Centroid
    ("#fbbf24", 0.16),  # Amber: Secondary Comparison
)


class GeneratorFingerprintVisualizer:
    """Generates self-contained, XML-valid SVG radar and z-score deviation charts."""

    @staticmethod
    def _clamp01(value: float) -> float:
        try:
            fval = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(fval):
            return 0.0
        return max(0.0, min(1.0, fval))

    @staticmethod
    def _safe_float(value: float, fallback: float = 0.0) -> float:
        try:
            fval = float(value)
        except (TypeError, ValueError):
            return fallback
        if not math.isfinite(fval):
            return fallback
        return fval

    def render_radar_chart_svg(
        self,
        probes: Dict[str, Dict[str, float]],
        labels: List[str],
        width: int = 500,
        height: int = 400,
        axis_keys: Optional[Sequence[str]] = None,
    ) -> str:
        """
        Renders an overlaid multi-series SVG radar chart with concentric grid polygons,
        radial spokes, escaped axis titles, vertex markers, and a series legend.
        """
        w = max(260, int(width))
        h = max(220, int(height))

        if not probes or not labels:
            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
                f'width="100%" height="{h}" role="img" aria-label="Empty Radar Chart">'
                f'<rect x="0" y="0" width="{w}" height="{h}" rx="8" fill="#181a20" stroke="#2d313d" stroke-width="1"/>'
                f'<text x="{w / 2:.1f}" y="{h / 2:.1f}" fill="#9ca3af" font-family="Inter, sans-serif" '
                f'font-size="12" text-anchor="middle">No linguistic profile data available</text>'
                f"</svg>"
            )

        num_axes = len(labels)
        first_series = next(iter(probes.values()), {})
        if axis_keys is not None and len(axis_keys) == num_axes:
            resolved_keys = list(axis_keys)
        elif all(lbl in first_series for lbl in labels):
            resolved_keys = list(labels)
        elif num_axes == len(RADAR_AXIS_KEYS) and any(k in first_series for k in RADAR_AXIS_KEYS):
            resolved_keys = list(RADAR_AXIS_KEYS)
        else:
            s_keys = list(first_series.keys())
            resolved_keys = [s_keys[i] if i < len(s_keys) else labels[i] for i in range(num_axes)]

        legend_height = 30
        cx = w / 2.0
        cy = (h + legend_height) / 2.0
        radius = max(55.0, min(w * 0.30, (h - legend_height) * 0.34))

        elements: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="100%" height="{h}" role="img" aria-label="8-Axis Linguistic Fingerprint Radar Chart">',
            f'<rect x="0" y="0" width="{w}" height="{h}" rx="8" fill="#181a20" stroke="#2d313d" stroke-width="1"/>',
        ]

        # 1. Concentric grid polygons (25%, 50%, 75%, 100%)
        for level in (0.25, 0.50, 0.75, 1.00):
            ring_pts: List[str] = []
            for idx in range(num_axes):
                angle = -math.pi / 2.0 + (2.0 * math.pi * idx) / float(num_axes)
                px = cx + radius * level * math.cos(angle)
                py = cy + radius * level * math.sin(angle)
                ring_pts.append(f"{px:.2f},{py:.2f}")
            dash = ' stroke-dasharray="3,3"' if level < 1.0 else ""
            elements.append(
                f'<polygon points="{" ".join(ring_pts)}" fill="none" '
                f'stroke="#2d313d" stroke-width="1"{dash}/>'
            )

        # 2. Radial spokes and escaped outer axis labels
        for idx, raw_label in enumerate(labels):
            angle = -math.pi / 2.0 + (2.0 * math.pi * idx) / float(num_axes)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            x_end = cx + radius * cos_a
            y_end = cy + radius * sin_a
            elements.append(
                f'<line x1="{cx:.2f}" y1="{cy:.2f}" x2="{x_end:.2f}" y2="{y_end:.2f}" '
                f'stroke="#334155" stroke-width="1"/>'
            )

            lbl_r = radius + 18.0
            lx = cx + lbl_r * cos_a
            ly = cy + lbl_r * sin_a + 4.0
            if abs(cos_a) < 0.22:
                anchor = "middle"
            elif cos_a > 0:
                anchor = "start"
            else:
                anchor = "end"

            safe_label = html.escape(str(raw_label), quote=True)
            elements.append(
                f'<text x="{lx:.2f}" y="{ly:.2f}" fill="#cbd5e1" '
                f'font-family="Inter, -apple-system, sans-serif" font-size="10.5" '
                f'font-weight="500" text-anchor="{anchor}">{safe_label}</text>'
            )

        # 3. Overlaid series polygons and vertex markers
        series_items = list(probes.items())
        for s_idx, (series_name, metric_map) in enumerate(series_items):
            stroke_hex, fill_alpha = SERIES_COLORS[s_idx % len(SERIES_COLORS)]
            poly_pts: List[str] = []
            vertices: List[Tuple[float, float]] = []

            for idx, m_key in enumerate(resolved_keys):
                norm_val = self._clamp01(metric_map.get(m_key, 0.0))
                angle = -math.pi / 2.0 + (2.0 * math.pi * idx) / float(num_axes)
                vx = cx + radius * norm_val * math.cos(angle)
                vy = cy + radius * norm_val * math.sin(angle)
                poly_pts.append(f"{vx:.2f},{vy:.2f}")
                vertices.append((vx, vy))

            stroke_dash = ' stroke-dasharray="5,3"' if s_idx == 1 else ""
            elements.append(
                f'<polygon points="{" ".join(poly_pts)}" fill="{stroke_hex}" '
                f'fill-opacity="{fill_alpha:.2f}" stroke="{stroke_hex}" '
                f'stroke-width="2"{stroke_dash}/>'
            )
            for vx, vy in vertices:
                elements.append(
                    f'<circle cx="{vx:.2f}" cy="{vy:.2f}" r="3" '
                    f'fill="{stroke_hex}" stroke="#0f172a" stroke-width="1"/>'
                )

        # 4. Top horizontal series legend
        n_series = len(series_items)
        slot_w = (w - 24.0) / float(max(1, n_series))
        for s_idx, (series_name, _) in enumerate(series_items):
            stroke_hex, _ = SERIES_COLORS[s_idx % len(SERIES_COLORS)]
            lx = 16.0 + s_idx * slot_w
            ly = 18.0
            safe_name = html.escape(str(series_name), quote=True)
            elements.append(
                f'<rect x="{lx:.1f}" y="{ly - 8.0:.1f}" width="10" height="10" rx="2" fill="{stroke_hex}"/>'
            )
            elements.append(
                f'<text x="{lx + 15.0:.1f}" y="{ly:.1f}" fill="#e2e8f0" '
                f'font-family="Inter, -apple-system, sans-serif" font-size="10.5" '
                f'font-weight="600">{safe_name}</text>'
            )

        elements.append("</svg>")
        return "\n".join(elements)

    def render_deviation_bars_svg(
        self,
        deviations: Dict[str, float],
        width: int = 500,
        height: int = 240,
    ) -> str:
        """
        Renders a horizontal diverging z-score bar chart relative to the Human Baseline (z = 0.0),
        highlighting anomalous dimensions (|z| >= 2.0 rose, |z| >= 1.2 amber, normal emerald).
        """
        w = max(260, int(width))
        items = list(deviations.items())
        n_rows = len(items)
        h = max(int(height), 48 + n_rows * 24) if n_rows > 0 else max(140, int(height))

        if not items:
            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
                f'width="100%" height="{h}" role="img" aria-label="Empty Deviation Chart">'
                f'<rect x="0" y="0" width="{w}" height="{h}" rx="8" fill="#181a20" stroke="#2d313d" stroke-width="1"/>'
                f'<text x="{w / 2:.1f}" y="{h / 2:.1f}" fill="#9ca3af" font-family="Inter, sans-serif" '
                f'font-size="12" text-anchor="middle">No z-score deviations available</text>'
                f"</svg>"
            )

        left_pad = 138.0
        right_pad = 64.0
        top_pad = 30.0
        plot_w = max(80.0, w - left_pad - right_pad)
        zero_x = left_pad + plot_w / 2.0
        max_abs_z = 4.0
        row_h = (h - top_pad - 14.0) / float(n_rows)

        elements: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="100%" height="{h}" role="img" aria-label="Z-Score Deviation Bar Chart">',
            f'<rect x="0" y="0" width="{w}" height="{h}" rx="8" fill="#181a20" stroke="#2d313d" stroke-width="1"/>',
            f'<text x="{left_pad:.1f}" y="18" fill="#94a3b8" font-family="Inter, sans-serif" '
            f'font-size="10" font-weight="600">Z-Score Deviation vs. Authentic Human Baseline (z = 0)</text>',
        ]

        # Threshold guide lines (-2z, 0z, +2z)
        for z_tick, dash in ((-2.0, "3,3"), (0.0, ""), (2.0, "3,3")):
            tx = zero_x + (z_tick / max_abs_z) * (plot_w / 2.0)
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            stroke_col = "#475569" if z_tick == 0.0 else "#2d313d"
            elements.append(
                f'<line x1="{tx:.2f}" y1="{top_pad - 4.0:.2f}" x2="{tx:.2f}" y2="{h - 10.0:.2f}" '
                f'stroke="{stroke_col}" stroke-width="1"{dash_attr}/>'
            )

        for idx, (raw_dim, raw_z) in enumerate(items):
            z_val = self._safe_float(raw_z, fallback=0.0)
            clamped_z = max(-max_abs_z, min(max_abs_z, z_val))
            cy = top_pad + idx * row_h + row_h / 2.0
            bar_h = min(13.0, max(8.0, row_h * 0.54))

            abs_z = abs(z_val)
            if abs_z >= 2.0:
                bar_color = "#fb7185"  # Rose anomaly
            elif abs_z >= 1.2:
                bar_color = "#fbbf24"  # Amber caution
            else:
                bar_color = "#34d399"  # Emerald within human range

            bar_len = abs(clamped_z / max_abs_z) * (plot_w / 2.0)
            bar_len = max(2.0, bar_len)
            bar_x = zero_x if clamped_z >= 0.0 else (zero_x - bar_len)

            safe_dim = html.escape(str(raw_dim), quote=True)
            elements.append(
                f'<text x="{left_pad - 8.0:.1f}" y="{cy + 3.5:.2f}" fill="#cbd5e1" '
                f'font-family="Inter, -apple-system, sans-serif" font-size="10.5" '
                f'text-anchor="end">{safe_dim}</text>'
            )
            elements.append(
                f'<rect x="{bar_x:.2f}" y="{cy - bar_h / 2.0:.2f}" width="{bar_len:.2f}" '
                f'height="{bar_h:.2f}" rx="2.5" fill="{bar_color}"/>'
            )
            elements.append(
                f'<text x="{w - 10.0:.1f}" y="{cy + 3.5:.2f}" fill="{bar_color}" '
                f'font-family="SFMono-Regular, Consolas, monospace" font-size="10" '
                f'font-weight="600" text-anchor="end">{z_val:+.2f}z</text>'
            )

        elements.append("</svg>")
        return "\n".join(elements)
