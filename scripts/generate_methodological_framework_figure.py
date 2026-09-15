# -*- coding: utf-8 -*-
"""
Generate publication-grade 3-column Methodological Innovation Framework diagram.
Replicates the architecture provided by Prof. Guo's assistant (media_1789459331351.png)
mapped 1-to-1 to Daulet's Master's thesis methodology:
- Left Column: Client & Ingestion Subsystem
- Center Column: Core Detection & Verification Engine
- Right Column: Portals & Verification
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Institutional Palette Constants
NAVY = "#1E3A8A"        # Deep Academic Navy
BLUE_ACCENT = "#2563EB" # Royal Blue Accent
TEAL = "#0D9488"        # FST Morphology / Secondary
AMBER = "#D97706"       # Gating / Warning
GREEN = "#16A34A"       # Verified / Normal
SLATE = "#0F172A"       # Dark Slate / Text
CORAL = "#DC2626"       # Critical / Alert
MUTED_GRAY = "#64748B"  # Secondary Muted
LIGHT_BG = "#F8FAFC"    # Card Fill
BORDER_GRAY = "#CBD5E1" # Card Border
PILL_TEAL = "#0D9488"   # Output tensor pill
PILL_GRAY = "#64748B"   # Excluded pill


def draw_pill(ax, x, y, width, height, text, bg_color=PILL_TEAL, text_color="white", fontsize=7.2, bold=True):
    """Draws a rounded status pill with centered text."""
    pill = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        edgecolor=bg_color, facecolor=bg_color, linewidth=0.8, zorder=6
    )
    ax.add_patch(pill)
    ax.text(
        x + width / 2.0, y + height / 2.0, text,
        ha="center", va="center",
        fontsize=fontsize, fontweight="bold" if bold else "normal",
        color=text_color, zorder=7
    )


def draw_badge(ax, x, y, radius, number_str, bg_color="#1E3A8A", text_color="white"):
    """Draws a numbered circular badge."""
    circle = patches.Circle((x, y), radius, facecolor=bg_color, edgecolor="white", linewidth=1.5, zorder=8)
    ax.add_patch(circle)
    ax.text(x, y, number_str, ha="center", va="center", fontsize=9.5, fontweight="bold", color=text_color, zorder=9)


def draw_icon_circle(ax, x, y, radius, bg_color="#F97316", icon_type="camera"):
    """Draws a circular icon container with crisp vector glyphs."""
    circle = patches.Circle((x, y), radius, facecolor=bg_color, edgecolor="none", zorder=5)
    ax.add_patch(circle)
    
    if icon_type == "camera":
        # Video / camera glyph
        rect = patches.Rectangle((x - radius*0.48, y - radius*0.3), radius*0.62, radius*0.6, facecolor="white", zorder=6)
        ax.add_patch(rect)
        wedge = patches.Polygon([
            [x + radius*0.18, y - radius*0.1],
            [x + radius*0.52, y - radius*0.32],
            [x + radius*0.52, y + radius*0.32],
            [x + radius*0.18, y + radius*0.1]
        ], facecolor="white", zorder=6)
        ax.add_patch(wedge)
    elif icon_type == "document":
        # Document / page glyph
        doc = patches.Rectangle((x - radius*0.36, y - radius*0.48), radius*0.72, radius*0.96, facecolor="white", zorder=6)
        ax.add_patch(doc)
        ax.plot([x + radius*0.06, x + radius*0.36], [y + radius*0.18, y + radius*0.48], color=bg_color, lw=1.2, zorder=7)
    elif icon_type == "clock":
        # Clock / window glyph
        inner = patches.Circle((x, y), radius*0.58, facecolor="none", edgecolor="white", lw=1.5, zorder=6)
        ax.add_patch(inner)
        ax.plot([x, x], [y, y + radius*0.35], color="white", lw=1.5, zorder=7)
        ax.plot([x, x + radius*0.28], [y, y], color="white", lw=1.5, zorder=7)
    elif icon_type == "robot":
        # Robot head glyph
        head = patches.FancyBboxPatch(
            (x - radius*0.52, y - radius*0.4), radius*1.04, radius*0.75,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor="white", edgecolor="none", zorder=6
        )
        ax.add_patch(head)
        ax.add_patch(patches.Circle((x - radius*0.22, y - radius*0.02), radius*0.12, facecolor=bg_color, zorder=7))
        ax.add_patch(patches.Circle((x + radius*0.22, y - radius*0.02), radius*0.12, facecolor=bg_color, zorder=7))
        ax.plot([x, x], [y + radius*0.35, y + radius*0.55], color="white", lw=1.5, zorder=6)
        ax.add_patch(patches.Circle((x, y + radius*0.55), radius*0.1, facecolor="white", zorder=7))
    elif icon_type == "database":
        # Cylinder / database stack glyph
        for dy in [-0.25, 0.05, 0.35]:
            ellipse = patches.Ellipse((x, y + radius*dy), radius*0.75, radius*0.3, facecolor="white", edgecolor=bg_color, lw=0.8, zorder=6)
            ax.add_patch(ellipse)
    elif icon_type == "telegram":
        # Paper airplane / portal glyph
        plane = patches.Polygon([
            [x - radius*0.5, y - radius*0.35],
            [x + radius*0.55, y],
            [x - radius*0.5, y + radius*0.4],
            [x - radius*0.15, y]
        ], facecolor="white", zorder=6)
        ax.add_patch(plane)


def generate_methodological_framework_diagram(output_path="presentation_figures/fig03_methodological_innovation_framework.png"):
    """
    Renders the 3-column Methodological Innovation Framework diagram at 300 DPI.
    Canvas: 14.5 x 6.0 inches.
    """
    fig, ax = plt.subplots(figsize=(14.5, 6.0), facecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # =========================================================================
    # COLUMN 1: CLIENT & INGESTION SUBSYSTEM (Left: X = 2.0 to 28.5)
    # =========================================================================
    ax.text(15.25, 96.2, "Edge & Ingestion Subsystem", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=SLATE)

    # Outer dashed container
    c1_outer = patches.FancyBboxPatch(
        (2.0, 3.5), 26.5, 90.0,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        edgecolor=SLATE, facecolor="#FAFAFA", linewidth=1.4, linestyle="--"
    )
    ax.add_patch(c1_outer)

    # -------------------------------------------------------------------------
    # Column 1 - Top Box: Input Text & Chunking (SentencePreservingChunker Node)
    # -------------------------------------------------------------------------
    c1_top = patches.FancyBboxPatch(
        (3.2, 39.0), 24.1, 52.5,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        edgecolor="#94A3B8", facecolor="white", linewidth=1.2
    )
    ax.add_patch(c1_top)

    ax.text(15.25, 87.5, "Input Text & Segmentation\n(SentencePreservingChunker Node)",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color=SLATE)

    # Sub-box 1.1: Input Text (Top-Left)
    box_doc = patches.FancyBboxPatch(
        (4.2, 65.5), 10.6, 17.5,
        boxstyle="round,pad=0.2,rounding_size=0.5",
        edgecolor="#CBD5E1", facecolor="#F8FAFC", linewidth=0.9
    )
    ax.add_patch(box_doc)
    draw_icon_circle(ax, 9.5, 78.8, 1.7, bg_color="#8B5CF6", icon_type="document")
    ax.text(9.5, 74.5, "Input Text", ha="center", va="center", fontsize=8.0, fontweight="bold", color=SLATE)
    ax.text(9.5, 70.2, "Multi-genre corpus\n& raw documents", ha="center", va="center", fontsize=6.5, color=MUTED_GRAY)
    ax.text(9.5, 66.5, ".txt · .docx · .pdf", ha="center", va="center", fontsize=6.8, fontweight="bold", color=NAVY)

    # Sub-box 1.2: Sentence Chunking (Bottom-Left)
    box_chunk = patches.FancyBboxPatch(
        (4.2, 40.8), 10.6, 23.2,
        boxstyle="round,pad=0.2,rounding_size=0.5",
        edgecolor="#CBD5E1", facecolor="#F8FAFC", linewidth=0.9
    )
    ax.add_patch(box_chunk)
    draw_icon_circle(ax, 9.5, 60.2, 1.7, bg_color="#3B82F6", icon_type="clock")
    ax.text(9.5, 55.4, "Sentence\nChunking", ha="center", va="center", fontsize=7.6, fontweight="bold", color=SLATE)
    ax.text(9.5, 49.0, "10 Abbrev Guards\n(т.б., ж.б., ғ.)\nL=256 tokens\n1-sent overlap",
            ha="center", va="center", fontsize=6.3, color=SLATE)
    ax.text(9.5, 42.6, "output -> (N, 256)", ha="center", va="center", fontsize=6.4, fontstyle="italic", color=MUTED_GRAY)

    # Sub-box 1.3: FST Morpheme Tiling Node (Right side of top box)
    box_fst = patches.FancyBboxPatch(
        (15.6, 40.8), 10.7, 42.2,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        edgecolor="#99F6E4", facecolor="#F0FDFA", linewidth=1.0
    )
    ax.add_patch(box_fst)
    draw_icon_circle(ax, 20.95, 77.8, 2.0, bg_color="#F97316", icon_type="camera")
    ax.text(20.95, 72.8, "FST Parsing", ha="center", va="center", fontsize=8.2, fontweight="bold", color=SLATE)
    ax.text(20.95, 63.8, "83-Rule FST\nTransducer Engine", ha="center", va="center", fontsize=7.2, fontweight="bold", color=TEAL)
    ax.text(20.95, 55.5, "Stem-affix rule\ndecomposition\n\n|V_morph| = 128",
            ha="center", va="center", fontsize=6.6, color=SLATE)
    draw_pill(ax, 16.3, 43.5, 9.3, 4.0, "-> morpheme_seq", bg_color=TEAL, fontsize=6.5)

    # Internal arrow between chunking and FST parsing
    ax.annotate("", xy=(15.6, 52.5), xytext=(14.8, 52.5),
                arrowprops=dict(arrowstyle="->", color=MUTED_GRAY, lw=1.2))

    # -------------------------------------------------------------------------
    # Column 1 - Bottom Box: Client Devices & Serving Endpoints
    # -------------------------------------------------------------------------
    # Vertical label along left border
    ax.text(3.4, 21.0, "Client Ingestion (AI/API)", rotation=90, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=MUTED_GRAY)

    # Inner blue dashed box
    c1_bot_blue = patches.FancyBboxPatch(
        (5.2, 12.0), 22.1, 23.5,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        edgecolor="#3B82F6", facecolor="#EFF6FF", linewidth=1.3, linestyle="--"
    )
    ax.add_patch(c1_bot_blue)

    # Client icons inside blue box
    draw_icon_circle(ax, 11.5, 29.5, 1.8, bg_color="#F97316", icon_type="camera")
    ax.text(11.5, 26.0, "Web Client", ha="center", va="center", fontsize=6.8, color=SLATE)

    draw_icon_circle(ax, 20.5, 29.5, 1.8, bg_color="#F97316", icon_type="camera")
    ax.text(20.5, 26.0, "API Client", ha="center", va="center", fontsize=6.8, color=SLATE)

    ax.text(16.0, 20.5, "Client Ingestion & Memory Guards", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=NAVY)
    ax.text(16.0, 16.8, "Max 10MB file · < 1.4 GB VRAM", ha="center", va="center",
            fontsize=6.5, color=MUTED_GRAY)

    # Numbered Badge 1 on lower-right of blue box
    draw_badge(ax, 25.8, 13.8, 1.6, "1", bg_color=NAVY)

    # Edge / Server Serving Node at very bottom
    box_server = patches.FancyBboxPatch(
        (8.5, 4.8), 15.5, 5.5,
        boxstyle="round,pad=0.2,rounding_size=0.4",
        edgecolor="#CBD5E1", facecolor="white", linewidth=0.9
    )
    ax.add_patch(box_server)
    ax.text(16.25, 7.55, "Edge Device (AI/Inference)\nCPU / Nvidia RTX Node", ha="center", va="center",
            fontsize=6.5, fontweight="bold", color=SLATE)

    # =========================================================================
    # CONNECTIONS: Column 1 to Column 2
    # =========================================================================
    # Top bidirectional arrow: HTTP / REST
    ax.annotate("", xy=(32.0, 75.0), xytext=(28.5, 75.0),
                arrowprops=dict(arrowstyle="<->", color=SLATE, lw=1.5))
    ax.text(30.25, 77.2, "HTTP/REST", ha="center", va="center", fontsize=7.0, fontweight="bold", color=SLATE)

    # Solid arrow from Badge 1 into Column 2 Feature Streams
    ax.annotate("", xy=(32.0, 22.0), xytext=(27.4, 15.0),
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5,
                                connectionstyle="arc3,rad=-0.15"))

    # =========================================================================
    # COLUMN 2: CORE DETECTION & VERIFICATION ENGINE (Center: X = 32.0 to 80.0)
    # =========================================================================
    ax.text(56.0, 96.2, "Core Detection & Verification Engine", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=SLATE)

    # Outer solid container
    c2_outer = patches.FancyBboxPatch(
        (32.0, 3.5), 48.0, 90.0,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        edgecolor=SLATE, facecolor="white", linewidth=1.5
    )
    ax.add_patch(c2_outer)

    # -------------------------------------------------------------------------
    # Column 2 - Left Sub-Column: Processing Modules (X = 33.5 to 54.0)
    # -------------------------------------------------------------------------
    c2_sub1 = patches.FancyBboxPatch(
        (33.5, 5.5), 20.5, 85.5,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        edgecolor=SLATE, facecolor="white", linewidth=1.2
    )
    ax.add_patch(c2_sub1)
    ax.text(43.75, 87.5, "Processing\nModules", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=SLATE)

    # Module 1: Contextual Semantic Stream (Top)
    m1_box = patches.FancyBboxPatch(
        (34.7, 62.5), 18.1, 20.5,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        edgecolor="#3B82F6", facecolor="#F8FAFC", linewidth=1.1
    )
    ax.add_patch(m1_box)
    ax.text(43.75, 79.5, "Contextual Semantic", ha="center", va="center", fontsize=8.5, fontweight="bold", color=NAVY)
    ax.text(43.75, 73.8, "KazRoBERTa Backbone\n12-layer · 768-dim\nMean-pooling",
            ha="center", va="center", fontsize=6.8, color=SLATE)
    draw_pill(ax, 36.2, 64.5, 15.1, 4.0, r"-> h_sem $\in \mathbb{R}^{768}$", bg_color=TEAL, fontsize=6.6)

    # Double-headed vertical arrow between Module 1 and Module 2
    ax.annotate("", xy=(43.75, 62.5), xytext=(43.75, 58.5),
                arrowprops=dict(arrowstyle="<->", color=MUTED_GRAY, lw=1.2))

    # Module 2: Morphological Inductive Stream (Middle)
    m2_box = patches.FancyBboxPatch(
        (34.7, 36.8), 18.1, 21.7,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        edgecolor="#0D9488", facecolor="#F0FDFA", linewidth=1.1
    )
    ax.add_patch(m2_box)
    ax.text(43.75, 55.0, "Morphological Inductive", ha="center", va="center", fontsize=8.5, fontweight="bold", color=TEAL)
    ax.text(43.75, 48.2, "83-Rule FST Transducer\nBiLSTM Morpheme Embed\nd_m=128 · W_proj proj",
            ha="center", va="center", fontsize=6.8, color=SLATE)
    draw_pill(ax, 35.5, 38.6, 16.5, 4.0, r"-> $W_{proj} \cdot h_{morph} \in \mathbb{R}^{768}$", bg_color=TEAL, fontsize=6.6)

    # Double-headed vertical arrow between Module 2 and Module 3
    ax.annotate("", xy=(43.75, 36.8), xytext=(43.75, 32.8),
                arrowprops=dict(arrowstyle="<->", color=MUTED_GRAY, lw=1.2))

    # Module 3: Ablated Baselines (Bottom - Gray dashed with flag & excluded pill)
    m3_box = patches.FancyBboxPatch(
        (34.7, 8.5), 18.1, 24.3,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        edgecolor="#94A3B8", facecolor="#F1F5F9", linewidth=1.1, linestyle="--"
    )
    ax.add_patch(m3_box)
    # Flag icon
    ax.plot([43.75, 43.75], [26.8, 30.8], color=MUTED_GRAY, lw=1.5)
    flag = patches.Polygon([[43.75, 30.8], [46.8, 29.3], [43.75, 27.8]], facecolor=MUTED_GRAY)
    ax.add_patch(flag)
    ax.text(43.75, 24.8, "Ablated Baselines", ha="center", va="center", fontsize=8.2, fontweight="bold", color=MUTED_GRAY)
    ax.text(43.75, 18.5, "Perplexity & BPE-only\nSevere root slicing\nOOD collapse (-42.18%)",
            ha="center", va="center", fontsize=6.4, color=SLATE)
    draw_pill(ax, 38.0, 10.5, 11.5, 4.0, "[ X excluded ]", bg_color="#64748B", fontsize=6.8)

    # -------------------------------------------------------------------------
    # Column 2 - Right Sub-Column: Core Processing (Transformer) (X = 55.5 to 78.5)
    # -------------------------------------------------------------------------
    c2_sub2 = patches.FancyBboxPatch(
        (55.5, 5.5), 23.0, 85.5,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        edgecolor=SLATE, facecolor="white", linewidth=1.2
    )
    ax.add_patch(c2_sub2)
    ax.text(67.0, 87.5, "Core Processing\n(Transformer & Fusion)", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=SLATE)

    # Sub-box 2.1: Dynamic Feature Fusion (Top)
    top_fuse = patches.FancyBboxPatch(
        (56.8, 52.0), 20.4, 31.0,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        edgecolor="#CBD5E1", facecolor="#F8FAFC", linewidth=1.0
    )
    ax.add_patch(top_fuse)

    # Glyphs side by side at top
    draw_icon_circle(ax, 64.0, 78.2, 1.4, bg_color="#8B5CF6", icon_type="camera")
    draw_icon_circle(ax, 70.0, 78.2, 1.4, bg_color="#3B82F6", icon_type="database")

    ax.text(67.0, 73.8, "Feature Fusion · Dynamic Gating", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=SLATE)
    ax.text(67.0, 69.8, r"$g = \sigma(W_g [h_{sem}; W_{proj} h_{morph}] + b_g)$",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color=NAVY)
    ax.text(67.0, 66.0, r"$h_{fused} = g \odot h_{sem} + (1-g) \odot (W_{proj} h_{morph})$",
            ha="center", va="center", fontsize=6.8, color=SLATE)

    # 3 Teal Status Pills
    draw_pill(ax, 60.5, 60.8, 13.0, 3.4, "h_sem  (Semantic)", bg_color=TEAL, fontsize=6.5)
    draw_pill(ax, 60.5, 56.8, 13.0, 3.4, "W_proj • h_morph", bg_color=TEAL, fontsize=6.5)
    draw_pill(ax, 60.5, 52.8, 13.0, 3.4, "h_fused  (Gated)", bg_color=NAVY, fontsize=6.5)

    # Vertical double-headed arrow between Fusion and Transformer Core
    ax.annotate("", xy=(67.0, 52.0), xytext=(67.0, 48.0),
                arrowprops=dict(arrowstyle="<->", color=SLATE, lw=1.3))

    # Horizontal arrows between Processing Streams and Core Fusion
    ax.annotate("", xy=(56.8, 72.5), xytext=(52.8, 72.5),
                arrowprops=dict(arrowstyle="->", color=MUTED_GRAY, lw=1.2))
    ax.annotate("", xy=(56.8, 47.5), xytext=(52.8, 47.5),
                arrowprops=dict(arrowstyle="<->", color=MUTED_GRAY, lw=1.2))

    # Sub-box 2.2: Transformer Encoder & Top-K Engine (Bottom)
    bot_trans = patches.FancyBboxPatch(
        (56.8, 8.5), 20.4, 39.5,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        edgecolor="#3B82F6", facecolor="#F8FAFC", linewidth=1.1
    )
    ax.add_patch(bot_trans)

    # Robot / AI icon at top of transformer box
    draw_icon_circle(ax, 67.0, 44.5, 1.8, bg_color="#EC4899", icon_type="robot")

    ax.text(67.0, 40.2, "Transformer Encoder & Top-K", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=NAVY)
    ax.text(67.0, 34.0, r"4 layers · 4 heads · $d_{model}=64$" + "\n" +
                        r"Document Top-K: $K = \max(1, \lfloor 0.25 \cdot N \rfloor)$" + "\n" +
                        r"Dual-loss: $\mathcal{L}_{BCE} + \lambda \mathcal{L}_{SupCon}$ ($\mathbb{R}^{128}$)" + "\n" +
                        r"global avg pool $\rightarrow \hat{y} \in [0, 1]$",
            ha="center", va="center", fontsize=6.5, color=SLATE)

    # Empirical Metrics Bullet Points
    ax.text(67.0, 20.5, "• Dropout = 0.20 · Temp = 0.07\n"
                        "• ROC-AUC = 0.9980 (Macro)\n"
                        "• OOD Kaspi Gain = +42.18%\n"
                        "• Tamper Precision = 100.0%\n"
                        "• Latency < 85ms",
            ha="center", va="center", fontsize=6.6, fontweight="bold", color=NAVY)

    # Numbered Badge 2 on bottom-right of transformer box
    draw_badge(ax, 75.6, 10.5, 1.6, "2", bg_color=NAVY)

    # =========================================================================
    # CONNECTIONS: Column 2 to Column 3
    # =========================================================================
    # Top bidirectional arrow: API / JSON
    ax.annotate("", xy=(83.5, 75.0), xytext=(80.0, 75.0),
                arrowprops=dict(arrowstyle="<->", color=SLATE, lw=1.5))
    ax.text(81.75, 77.2, "API", ha="center", va="center", fontsize=7.0, fontweight="bold", color=SLATE)

    # Gray horizontal arrow from y_hat to Column 3 Threshold Alert Level
    ax.annotate("", xy=(83.5, 28.0), xytext=(80.0, 28.0),
                arrowprops=dict(arrowstyle="->", color=MUTED_GRAY, lw=1.5))

    # =========================================================================
    # COLUMN 3: PORTALS & VERIFICATION (Right: X = 83.5 to 98.0)
    # =========================================================================
    ax.text(90.75, 96.2, "Portals", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=SLATE)

    # Outer dashed container
    c3_outer = patches.FancyBboxPatch(
        (83.5, 3.5), 14.5, 90.0,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        edgecolor=SLATE, facecolor="#FAFAFA", linewidth=1.4, linestyle="--"
    )
    ax.add_patch(c3_outer)

    # -------------------------------------------------------------------------
    # Column 3 - Top Box: Response Delivery
    # -------------------------------------------------------------------------
    c3_top = patches.FancyBboxPatch(
        (84.5, 57.0), 12.5, 34.0,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        edgecolor="#CBD5E1", facecolor="white", linewidth=1.0
    )
    ax.add_patch(c3_top)

    ax.text(90.75, 87.5, "Response\nDelivery", ha="center", va="center",
            fontsize=9.2, fontweight="bold", color=SLATE)

    draw_icon_circle(ax, 90.75, 78.5, 2.0, bg_color="#60A5FA", icon_type="telegram")
    ax.text(90.75, 72.5, "Gradio / Telegram", ha="center", va="center",
            fontsize=7.6, fontweight="bold", color=NAVY)
    ax.text(90.75, 64.5, "4-Tab Dashboard\n• Sentence Heatmap\n• 36-Art Kazakh-FEVER\n• No specialized HW\nKazakhstan context",
            ha="center", va="center", fontsize=6.2, color=SLATE)

    # -------------------------------------------------------------------------
    # Column 3 - Bottom Box: Threshold Alert Level
    # -------------------------------------------------------------------------
    c3_bot = patches.FancyBboxPatch(
        (84.5, 5.5), 12.5, 50.0,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        edgecolor="#CBD5E1", facecolor="white", linewidth=1.0
    )
    ax.add_patch(c3_bot)

    ax.text(90.75, 52.5, "Threshold\nAlert Level", ha="center", va="center",
            fontsize=9.2, fontweight="bold", color=SLATE)

    # 3 Stacked Color Cards
    # 1. CRITICAL (Red)
    card_crit = patches.FancyBboxPatch(
        (85.3, 36.2), 10.9, 13.0,
        boxstyle="round,pad=0.2,rounding_size=0.5",
        edgecolor="#991B1B", facecolor="#B91C1C", linewidth=1.0
    )
    ax.add_patch(card_crit)
    ax.text(90.75, 45.6, "CRITICAL", ha="center", va="center", fontsize=8.2, fontweight="bold", color="white")
    ax.text(90.75, 42.0, r"$\hat{y} \geq 0.70$", ha="center", va="center", fontsize=7.8, fontweight="bold", color="white")
    ax.text(90.75, 38.4, "Synthetic / Malicious\nFlag immediate review", ha="center", va="center", fontsize=5.8, color="#FEE2E2")

    # 2. WARNING (Amber / Orange)
    card_warn = patches.FancyBboxPatch(
        (85.3, 21.0), 10.9, 13.2,
        boxstyle="round,pad=0.2,rounding_size=0.5",
        edgecolor="#B45309", facecolor="#D97706", linewidth=1.0
    )
    ax.add_patch(card_warn)
    ax.text(90.75, 30.6, "!  WARNING", ha="center", va="center", fontsize=8.2, fontweight="bold", color="white")
    ax.text(90.75, 27.0, r"$0.40 \leq \hat{y} < 0.70$", ha="center", va="center", fontsize=7.5, fontweight="bold", color="white")
    ax.text(90.75, 23.4, "Ambiguous / Hybrid\nManual review required", ha="center", va="center", fontsize=5.8, color="#FEF3C7")

    # 3. NORMAL (Green)
    card_norm = patches.FancyBboxPatch(
        (85.3, 6.0), 10.9, 13.0,
        boxstyle="round,pad=0.2,rounding_size=0.5",
        edgecolor="#15803D", facecolor="#16A34A", linewidth=1.0
    )
    ax.add_patch(card_norm)
    ax.text(90.75, 15.6, "✓  NORMAL", ha="center", va="center", fontsize=8.2, fontweight="bold", color="white")
    ax.text(90.75, 12.0, r"$\hat{y} < 0.40$", ha="center", va="center", fontsize=7.8, fontweight="bold", color="white")
    ax.text(90.75, 8.4, "Natural human text\nRoutine publishing safe", ha="center", va="center", fontsize=5.8, color="#DCFCE7")

    # Save at 300 DPI
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Generated publication-grade architecture diagram: {output_path}")


if __name__ == "__main__":
    generate_methodological_framework_diagram("presentation_figures/fig03_methodological_innovation_framework.png")
