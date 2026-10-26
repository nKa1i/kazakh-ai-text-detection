# -*- coding: utf-8 -*-
"""
Generate Publication-Grade Timeline Figure for Master's Proposal Defense.
Matching Screenshot 2 from Professor Guo's laboratory standard presentation:
- Period: 2026年10月-2027年6月 (October 2026 -- June 2027)
- 5 alternating red/blue milestone nodes along a directional axis
- High-resolution 300 DPI rendering via matplotlib
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# Configure Chinese Font
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimSun", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "presentation_figures")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig15_timeline_2026_2027.png")

# Palette tokens
COLOR_NAVY = "#1E3A8A"
COLOR_AXIS = "#9BB4DB"
COLOR_RED_NODE = "#C00000"
COLOR_BLUE_NODE = "#2E75B6"
COLOR_RED_LINE = "#C00000"
COLOR_DATE = "#111827"
COLOR_DESC = "#1F2937"
COLOR_BG = "#FFFFFF"


def generate_timeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.333, 6.2), dpi=300, facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 1. Top-Left Period Header: Dark Navy Vertical Bar + "2026年10月-2027年6月"
    rect_bar = patches.Rectangle((0.04, 0.90), 0.008, 0.055, facecolor=COLOR_NAVY, edgecolor="none")
    ax.add_patch(rect_bar)
    ax.text(
        0.056, 0.927, "2026年10月-2027年6月",
        fontsize=17, fontweight="bold", color=COLOR_NAVY, va="center", ha="left"
    )

    # 2. Central Horizontal Axis with Right-Pointing Arrow
    axis_y = 0.50
    axis_start_x = 0.03
    axis_end_x = 0.97

    # Main line
    line = Line2D([axis_start_x, axis_end_x - 0.02], [axis_y, axis_y], color=COLOR_AXIS, linewidth=6.5, zorder=2)
    ax.add_line(line)

    # Arrow head
    arrow = patches.Polygon(
        [[axis_end_x - 0.025, axis_y + 0.022], [axis_end_x, axis_y], [axis_end_x - 0.025, axis_y - 0.022]],
        closed=True, facecolor=COLOR_AXIS, edgecolor="none", zorder=2
    )
    ax.add_patch(arrow)

    # 3. Define 5 Milestones
    # Positions x: 0.13, 0.31, 0.49, 0.67, 0.85
    milestones = [
        {
            "id": 1,
            "color": COLOR_RED_NODE,
            "x": 0.13,
            "pos": "below",
            "date": "2026.10-2026.11",
            "desc": "开题报告答辩与修改完善\n确立论文技术路线及任务分工"
        },
        {
            "id": 2,
            "color": COLOR_BLUE_NODE,
            "x": 0.31,
            "pos": "above",
            "date": "2026.11-2027.01",
            "desc": "长文档滑动窗口检测与跨领域优化\n完成4-Tab Gradio云端原型平台部署"
        },
        {
            "id": 3,
            "color": COLOR_RED_NODE,
            "x": 0.49,
            "pos": "below",
            "date": "2027.02-2027.03",
            "desc": "录用论文修改与补充消融实验\n扩展Kazakh-FEVER事实核验基准评测"
        },
        {
            "id": 4,
            "color": COLOR_BLUE_NODE,
            "x": 0.67,
            "pos": "above",
            "date": "2027.04-2027.05",
            "desc": "撰写硕士学位论文各章节初稿\n参加学院组织的学位论文预答辩"
        },
        {
            "id": 5,
            "color": COLOR_RED_NODE,
            "x": 0.85,
            "pos": "below",
            "date": "2027.05-2027.06",
            "desc": "论文双盲评审、盲审意见修改\n硕士学位论文正式答辩与毕业"
        }
    ]

    circle_radius = 0.038
    line_span = 0.085

    for m in milestones:
        mx = m["x"]
        # Draw Circle on Axis
        circle = patches.Circle((mx, axis_y), circle_radius, facecolor=m["color"], edgecolor="none", zorder=4)
        ax.add_patch(circle)
        ax.text(mx, axis_y, str(m["id"]), fontsize=15, fontweight="bold", color="white", ha="center", va="center", zorder=5)

        if m["pos"] == "below":
            # Date below circle
            date_y = axis_y - circle_radius - 0.05
            ax.text(mx, date_y, m["date"], fontsize=12.5, fontweight="bold", color=COLOR_DATE, ha="center", va="center")

            # Red line below date
            line_y = date_y - 0.03
            ax.add_line(Line2D([mx - line_span, mx + line_span], [line_y, line_y], color=COLOR_RED_LINE, linewidth=1.8))

            # Description below line
            desc_y = line_y - 0.035
            ax.text(mx, desc_y, m["desc"], fontsize=11, color=COLOR_DESC, ha="center", va="top", linespacing=1.4)

        else: # "above"
            # Date on top
            date_y = axis_y + circle_radius + 0.25
            ax.text(mx, date_y, m["date"], fontsize=12.5, fontweight="bold", color=COLOR_DATE, ha="center", va="center")

            # Red line below date
            line_y = date_y - 0.03
            ax.add_line(Line2D([mx - line_span, mx + line_span], [line_y, line_y], color=COLOR_RED_LINE, linewidth=1.8))

            # Description below line (above circle)
            desc_y = line_y - 0.035
            ax.text(mx, desc_y, m["desc"], fontsize=11, color=COLOR_DESC, ha="center", va="top", linespacing=1.4)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, facecolor=COLOR_BG, edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Timeline figure generated successfully at: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_timeline()
