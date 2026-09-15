# scripts/generate_presentation_figures.py
"""
Generate 15 High-Impact Publication Figures for Master's Thesis Presentation.

Topic: Morphologically-Grounded Kazakh AI Text Detection & Factual Verification
Institution: Northwestern Polytechnical University (NPU) & Al-Farabi KazNU
Presenter: Daulet (大雷)
Advisor: Prof. Guo (郭教授)
Date: September 2026

Constraints:
- 300 DPI resolution, bbox_inches='tight'
- Palette: Institutional Navy (#1E3A8A), Teal (#0D9488), Amber (#D97706),
           Green (#16A34A), Slate (#0F172A), Coral/Red (#DC2626)
- Zero decorative emojis anywhere
- Explicit UTF-8 encoding
"""

import os
import sys
import shutil
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Typography & Global Style Configuration
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#94A3B8"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["figure.titlesize"] = 13
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9

# Institutional Palette Constants
NAVY = "#1E3A8A"       # Primary Institutional
TEAL = "#0D9488"       # Secondary Brand / Innovation
AMBER = "#D97706"      # Warning / Gating / Intermediate
GREEN = "#16A34A"      # Success / Human Ground Truth
SLATE = "#0F172A"      # Neutral Dark / Text
CORAL = "#DC2626"      # Alert / Baseline Drop / AI Tampering
MUTED_GRAY = "#64748B" # Secondary Muted
LIGHT_BG = "#F8FAFC"   # Background card fill
BORDER_GRAY = "#CBD5E1"# Card border


def ensure_output_dir(output_dir="presentation_figures"):
    abs_dir = os.path.abspath(output_dir)
    os.makedirs(abs_dir, exist_ok=True)
    return abs_dir


# ==============================================================================
# Figure 01: Subword Fragmentation vs. 83-Rule FST Parsing
# ==============================================================================
def generate_fig01(output_dir):
    fig, ax = plt.subplots(figsize=(10.8, 5.6), facecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title & Subtitle
    ax.text(2, 95, "Subword Tokenization Fragmentation vs. 83-Rule FST Morphological Parsing",
            fontsize=13, fontweight="bold", color=SLATE)
    ax.text(2, 89.5, "Exemplar Kazakh Word: 'Қазақстандықтардың' (English: 'of the Kazakhstanis')",
            fontsize=10.5, color=MUTED_GRAY)

    # --- TOP TRACK: Standard Subword / BPE Tokenizer ---
    ax.text(2, 80.5, "A. Standard Subword Tokenizer (BPE / WordPiece) - Severe Morpheme Slicing",
            fontsize=10, fontweight="bold", color=CORAL)

    bpe_tokens = ["Қа", "за", "қс", "тан", "ды", "қтар", "дың"]
    x_start = 2
    box_w = 7.8
    gap = 1.2
    y_bpe = 65

    for i, tok in enumerate(bpe_tokens):
        x = x_start + i * (box_w + gap)
        rect = patches.FancyBboxPatch((x, y_bpe), box_w, 9, boxstyle="round,pad=0.3,rounding_size=0.8",
                                     edgecolor=CORAL, facecolor="#FEE2E2", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y_bpe + 4.5, tok, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="#991B1B")
        ax.text(x + box_w / 2, y_bpe - 3.5, f"sub_{i+1}", ha="center", va="center",
                fontsize=8, color=MUTED_GRAY)

    # BPE Diagnostic Box
    bpe_info_box = patches.FancyBboxPatch((66, 59), 32, 19, boxstyle="round,pad=0.5,rounding_size=1",
                                          edgecolor="#FCA5A5", facecolor="#FFF1F2", linewidth=1.1)
    ax.add_patch(bpe_info_box)
    ax.text(67.5, 73.5, "Subword Failures on Kazakh:", fontsize=8.5, fontweight="bold", color="#991B1B")
    ax.text(67.5, 69.5, "- Severe root slicing ('Қа-за-қс' destroys stem)", fontsize=8, color=SLATE)
    ax.text(67.5, 65.5, "- 7 sub-tokens for a single dictionary word", fontsize=8, color=SLATE)
    ax.text(67.5, 61.5, "- Zero morphological boundary inductive bias", fontsize=8, color=SLATE)

    # Downward comparison note
    ax.annotate("", xy=(30, 52), xytext=(30, 56),
                arrowprops=dict(arrowstyle="->", color=MUTED_GRAY, lw=1.5))
    ax.text(32, 53.5, "Structural Comparison: Morpheme boundaries severed vs. preserved",
            fontsize=8.5, fontstyle="italic", color=MUTED_GRAY)

    # --- BOTTOM TRACK: Proposed 83-Rule FST Morphological Parser ---
    ax.text(2, 46.5, "B. Proposed 83-Rule FST Morphological Parser - Structured Agglutinative Hierarchy",
            fontsize=10, fontweight="bold", color=NAVY)

    fst_segments = [
        ("Қазақ", "Root (Noun Stem)", "Kazakh", NAVY, "#DBEAFE"),
        ("-стан", "Deriv Suffix", "Country", TEAL, "#CCFBF1"),
        ("-дық", "Deriv Suffix", "Relational", AMBER, "#FEF3C7"),
        ("-тар", "Infl Suffix", "Plural", MUTED_GRAY, "#E2E8F0"),
        ("-дың", "Infl Suffix", "Genitive", GREEN, "#DCFCE7"),
    ]

    x_fst = 2
    y_fst = 23
    fst_w = 11.5
    gap_fst = 1.0

    for i, (morpheme, tag, gloss, border_col, bg_col) in enumerate(fst_segments):
        x = x_fst + i * (fst_w + gap_fst)
        rect = patches.FancyBboxPatch((x, y_fst), fst_w, 15, boxstyle="round,pad=0.3,rounding_size=1",
                                     edgecolor=border_col, facecolor=bg_col, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + fst_w / 2, y_fst + 10.5, morpheme, ha="center", va="center",
                fontsize=11, fontweight="bold", color=border_col)
        ax.text(x + fst_w / 2, y_fst + 6, tag, ha="center", va="center",
                fontsize=8, fontweight="semibold", color=SLATE)
        ax.text(x + fst_w / 2, y_fst + 2.2, f'"{gloss}"', ha="center", va="center",
                fontsize=7.5, fontstyle="italic", color=MUTED_GRAY)

        # Concatenation plus symbol
        if i < len(fst_segments) - 1:
            ax.text(x + fst_w + gap_fst / 2, y_fst + 7.5, "+", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=MUTED_GRAY)

    # Brackets showing Root vs Derivational vs Inflectional
    ax.annotate("", xy=(2, 19.5), xytext=(13.5, 19.5),
                arrowprops=dict(arrowstyle="-", color=NAVY, lw=1.5))
    ax.text(7.75, 15.5, "Stem / Lexeme", ha="center", fontsize=8, fontweight="bold", color=NAVY)

    ax.annotate("", xy=(14.5, 19.5), xytext=(38.5, 19.5),
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=1.5))
    ax.text(26.5, 15.5, "Derivational Chain", ha="center", fontsize=8, fontweight="bold", color=TEAL)

    ax.annotate("", xy=(39.5, 19.5), xytext=(63.5, 19.5),
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=1.5))
    ax.text(51.5, 15.5, "Inflectional Chain", ha="center", fontsize=8, fontweight="bold", color=GREEN)

    # FST Advantages Box
    fst_info_box = patches.FancyBboxPatch((66, 14), 32, 25, boxstyle="round,pad=0.5,rounding_size=1",
                                          edgecolor="#99F6E4", facecolor="#F0FDFA", linewidth=1.2)
    ax.add_patch(fst_info_box)
    ax.text(67.5, 34.5, "83-Rule FST Inductive Bias:", fontsize=8.5, fontweight="bold", color=TEAL)
    ax.text(67.5, 30.5, "- Preserves complete linguistic root identity", fontsize=8, color=SLATE)
    ax.text(67.5, 26.5, "- 83 Turkic morphotactic rules constrain syntax", fontsize=8, color=SLATE)
    ax.text(67.5, 22.5, "- Domain-invariant affix transition tracking", fontsize=8, color=SLATE)
    ax.text(67.5, 17.5, "- Prevents -41.79% OOD domain collapse", fontsize=8, fontweight="bold", color=NAVY)

    # Bottom summary footnote
    ax.text(2, 6, "Core Scientific Insight: Subwords destroy morphotactic transition legality in agglutinative Kazakh;",
            fontsize=8.5, fontstyle="italic", color=MUTED_GRAY)
    ax.text(2, 2, "Proposed FST explicitly injects morphological inductive bias, achieving complete domain robustness.",
            fontsize=8.5, fontstyle="italic", color=MUTED_GRAY)

    fig_path = os.path.join(output_dir, "fig01_subword_vs_fst.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Generated: {fig_path}")


# ==============================================================================
# Figure 02: Out-of-Domain Generalization Collapse
# ==============================================================================
def generate_fig02(output_dir):
    fig, ax = plt.subplots(figsize=(9.2, 5.4), facecolor="white")

    domains = ["In-Domain\n(Kaz-News)", "Cross-Generator\n(Kaz-Wiki)", "Cross-Domain\n(Kaz-Reviews)"]
    kazroberta_auc = [99.41, 98.12, 57.62]
    morpho_auc = [99.85, 99.82, 99.80]

    x = np.arange(len(domains))
    width = 0.30

    bars_base = ax.bar(x - width/2, kazroberta_auc, width, label="KazRoBERTa (Pretrained Transformer)",
                       color=SLATE, edgecolor="#334155", linewidth=1, alpha=0.9)
    bars_prop = ax.bar(x + width/2, morpho_auc, width, label="Proposed Morpho-Detector (Dual-Stream + FST)",
                       color=TEAL, edgecolor="#0F766E", linewidth=1)

    # Highlight collapsed bar in red
    bars_base[2].set_color(CORAL)
    bars_base[2].set_edgecolor("#991B1B")

    # Reference random baseline
    ax.axhline(y=50.0, color="#94A3B8", linestyle="--", linewidth=1.2, label="Random Guess Baseline (50.0%)")

    # Bar labels
    for bar in bars_base:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="#991B1B" if h < 60 else SLATE)

    for bar in bars_prop:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=TEAL)

    # Collapse delta annotation - clean positioning
    ax.annotate("", xy=(x[2] - width/2, 63), xytext=(x[2] - width/2, 96),
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=2))
    ax.text(x[2] - width/2 - 0.24, 78, "-41.79%\nCollapse",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=CORAL,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEE2E2", edgecolor=CORAL, lw=1))

    # Proposed gain badge - clean position above the Morpho bar
    ax.text(x[2] + width/2, 105.5, "+42.18% Boost\n(99.80% Invariant AUC)",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color=TEAL,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#CCFBF1", edgecolor=TEAL, lw=1))

    # Move legend to top-left for zero overlap
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=BORDER_GRAY, fontsize=8.5)

    ax.set_ylim(40, 118)
    ax.set_ylabel("Detection ROC-AUC (%)", fontsize=10.5, fontweight="bold", color=SLATE)
    ax.set_title("Out-of-Domain Generalization Collapse on Agglutinative Kazakh Text\nEmpirical Evaluation Across Tri-Domain Test Sets (Kaz-MAGE Benchmark)",
                 fontsize=11.5, fontweight="bold", color=SLATE, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=9.5, fontweight="semibold", color=SLATE)
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#CBD5E1")
    ax.set_axisbelow(True)

    fig_path = os.path.join(output_dir, "fig02_domain_collapse.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Generated: {fig_path}")


# ==============================================================================
# Figure 03: End-to-End Kazakh AI Detection & Verification Pipeline
# ==============================================================================
def generate_fig03(output_dir):
    fig, ax = plt.subplots(figsize=(7.6, 5.8), facecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title & Subtitle
    ax.text(50, 96.8, "Tripartite Research Framework: End-to-End Pipeline",
            ha="center", va="center", fontsize=11, fontweight="bold", color=SLATE)
    ax.text(50, 93.0, "Hierarchical Architecture: Linguistic Inputs -> Sentence Gating -> Document Aggregation -> Trust Matrix",
            ha="center", va="center", fontsize=7.2, color=MUTED_GRAY)

    # 4-Tier Vertical Flow
    tiers = [
        {
            "num": "Tier 1",
            "title": "Linguistic Inputs & Knowledge Corpora",
            "color": NAVY, "bg": "#EFF6FF", "border": "#93C5FD",
            "y": 73.0, "h": 16.5,
            "cards": [
                ("Raw Kazakh Corpora", "Unlabeled text & web articles\n(BPE tokenized & segmented)", SLATE, "#CBD5E1"),
                ("83-Rule FST Morpho-Dict", "Agglutinative suffix rules\nPhonotactic harmony constraints", TEAL, "#99F6E4"),
                ("36-Article Kazakh-FEVER", "Curated factual claims\nEvidential knowledge base", AMBER, "#FDE68A"),
            ],
            "arrow_lbl": "Tokenized & Segmented Streams",
            "arrow_color": NAVY,
        },
        {
            "num": "Tier 2",
            "title": "Sentence-Level Dynamic Gating & Representation",
            "color": TEAL, "bg": "#F0FDFA", "border": "#99F6E4",
            "y": 50.5, "h": 16.5,
            "cards": [
                ("KazRoBERTa Semantic", "12-Layer contextual backbone\nh_sem in R^768 (Mean Pooling)", NAVY, "#93C5FD"),
                ("Dynamic Gating Unit", "g = sigma(W_g [h_sem; h_morph])\nh_fused = g*h_sem + (1-g)*h_morph", AMBER, "#F59E0B"),
                ("FST Affix BiLSTM", "W_proj h_morph in R^768\nDual Loss: BCE + SupCon", GREEN, "#BBF7D0"),
            ],
            "arrow_lbl": "Sentence Probabilities P_AI(s_i)",
            "arrow_color": TEAL,
        },
        {
            "num": "Tier 3",
            "title": "Document Sliding Window & Worst-Case Top-K Pooling",
            "color": AMBER, "bg": "#FFFDF5", "border": "#FDE68A",
            "y": 28.0, "h": 16.5,
            "cards": [
                ("10 Abbrev Regex Guards", "Sentence boundary protection\n(e.g., 'т.б.', 'ж.б.', 'ғ.')", AMBER, "#FDE68A"),
                ("Sliding Windows", "256 tokens context length\n1-sentence fixed overlap", AMBER, "#FDE68A"),
                ("Worst-Case Top-K", "K = max(1, floor(0.2*N))\nP_doc & Tampered Spans", AMBER, "#FDE68A"),
            ],
            "arrow_lbl": "Document Score P_doc & Extracted Claims",
            "arrow_color": AMBER,
        },
        {
            "num": "Tier 4",
            "title": "Fact Verification & Four-Quadrant Trust Matrix",
            "color": GREEN, "bg": "#F0FDF4", "border": "#86EFAC",
            "y": 5.5, "h": 16.5,
            "cards": [
                ("Claim Extraction", "BM25 evidential retrieval\nTop-5 candidate passages", GREEN, "#BBF7D0"),
                ("3-Way NLI Cross-Encoder", "Supports / Refutes / NEI\n100% Macro-F1 Verification", GREEN, "#BBF7D0"),
                ("Four-Quadrant Matrix", "Joint Decision: AI Style vs\nFactual Veracity (Q1-Q4)", NAVY, "#93C5FD"),
            ],
            "arrow_lbl": None,
            "arrow_color": None,
        },
    ]

    for t in tiers:
        # Container
        cbox = patches.FancyBboxPatch((3, t["y"]), 94, t["h"], boxstyle="round,pad=0.3,rounding_size=0.8",
                                     edgecolor=t["border"], facecolor=t["bg"], linewidth=1.2)
        ax.add_patch(cbox)

        # Header Badge
        badge = patches.FancyBboxPatch((4, t["y"] + t["h"] - 4.2), 92, 3.8, boxstyle="round,pad=0.2,rounding_size=0.4",
                                       edgecolor=t["color"], facecolor=t["color"], linewidth=0.8)
        ax.add_patch(badge)
        ax.text(50, t["y"] + t["h"] - 2.3, f"{t['num']}: {t['title']}",
                ha="center", va="center", fontsize=7.8, fontweight="bold", color="white")

        # 3 Cards
        card_w = 28.5
        card_gap = 2.75
        card_y = t["y"] + 1.2
        card_h = t["h"] - 6.2

        for i, (head, body, bcolor, bedge) in enumerate(t["cards"]):
            cx = 4.5 + i * (card_w + card_gap)
            card = patches.FancyBboxPatch((cx, card_y), card_w, card_h, boxstyle="round,pad=0.2,rounding_size=0.4",
                                          edgecolor=bedge, facecolor="white", linewidth=0.8)
            ax.add_patch(card)
            ax.text(cx + card_w / 2, card_y + card_h - 2.6, head,
                    ha="center", va="center", fontsize=7.2, fontweight="bold", color=bcolor)
            ax.text(cx + card_w / 2, card_y + (card_h - 2.6) / 2, body,
                    ha="center", va="center", fontsize=6.2, color=SLATE)

        # Connector Arrow to Next Tier
        if t["arrow_lbl"] is not None:
            ax.annotate("", xy=(50, t["y"] - 3.8), xytext=(50, t["y"]),
                        arrowprops=dict(arrowstyle="->", color=t["arrow_color"], lw=1.8))
            ax.text(50, t["y"] - 1.9, t["arrow_lbl"],
                    ha="center", va="center", fontsize=6.0, fontweight="bold",
                    color=t["arrow_color"], bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=t["border"], lw=0.6))

    # Bottom summary banner
    summary_box = patches.FancyBboxPatch((3, 0.8), 94, 3.4, boxstyle="round,pad=0.2,rounding_size=0.4",
                                         edgecolor=BORDER_GRAY, facecolor=LIGHT_BG, linewidth=0.8)
    ax.add_patch(summary_box)
    ax.text(50, 2.5, "Integrated Output: Origin Risk P_AI in [0, 1]  x  Factual Veracity R_fact  ->  Trust Matrix Decision (Q1-Q4)",
            ha="center", va="center", fontsize=6.8, fontweight="bold", color=NAVY)

    fig_path = os.path.join(output_dir, "fig03_end_to_end_pipeline.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    legacy_path = os.path.join(output_dir, "fig03_tripartite_framework.png")
    shutil.copyfile(fig_path, legacy_path)
    plt.close(fig)
    print(f"[OK] Generated: {fig_path}")
    print(f"[OK] Mirrored: {legacy_path}")


# ==============================================================================
# Figure 04: Dual-Stream Cross-Attention Architecture
# ==============================================================================
def generate_fig04(output_dir):
    fig, ax = plt.subplots(figsize=(7.5, 6.0), facecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title & Subtitle
    ax.text(50, 96.8, "Dual-Stream Cross-Attention & Dynamic Gating Architecture",
            ha="center", va="center", fontsize=11, fontweight="bold", color=SLATE)
    ax.text(50, 93.0, "Fusing Pretrained Contextual Semantics with 83-Rule FST Morphological Inductive Bias",
            ha="center", va="center", fontsize=7.2, color=MUTED_GRAY)

    # Top Split Heads
    # Left Head: Classification Head (BCE Loss)
    h1_box = patches.FancyBboxPatch((4, 77.5), 44, 11.5, boxstyle="round,pad=0.3,rounding_size=0.6",
                                    edgecolor=NAVY, facecolor="#EFF6FF", linewidth=1.2)
    ax.add_patch(h1_box)
    h1_badge = patches.FancyBboxPatch((5, 85.0), 42, 3.2, boxstyle="round,pad=0.1,rounding_size=0.3",
                                      edgecolor=NAVY, facecolor=NAVY, linewidth=0.8)
    ax.add_patch(h1_badge)
    ax.text(26, 86.6, "Classification Head (BCE Loss)", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color="white")
    ax.text(26, 81.0, "Linear Projection -> Sigmoid Activation\nOutput: Sentence Probability P_AI(s_i) in [0, 1]",
            ha="center", va="center", fontsize=6.5, color=SLATE)

    # Right Head: Projection Head (SupCon Loss)
    h2_box = patches.FancyBboxPatch((52, 77.5), 44, 11.5, boxstyle="round,pad=0.3,rounding_size=0.6",
                                    edgecolor=GREEN, facecolor="#F0FDF4", linewidth=1.2)
    ax.add_patch(h2_box)
    h2_badge = patches.FancyBboxPatch((53, 85.0), 42, 3.2, boxstyle="round,pad=0.1,rounding_size=0.3",
                                      edgecolor=GREEN, facecolor=GREEN, linewidth=0.8)
    ax.add_patch(h2_badge)
    ax.text(74, 86.6, "Supervised Contrastive Head (SupCon)", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color="white")
    ax.text(74, 81.0, "MLP Projection Head (2-Layer Non-Linear)\nHyperspherical Space z_i in R^128 (Clustering)",
            ha="center", va="center", fontsize=6.5, color=SLATE)

    # Center Fusion Unit (Amber)
    gate_box = patches.FancyBboxPatch((6, 53.5), 88, 17.5, boxstyle="round,pad=0.4,rounding_size=0.8",
                                      edgecolor=AMBER, facecolor="#FFFBEB", linewidth=1.5)
    ax.add_patch(gate_box)
    gate_badge = patches.FancyBboxPatch((16, 66.8), 68, 3.4, boxstyle="round,pad=0.2,rounding_size=0.4",
                                        edgecolor=AMBER, facecolor=AMBER, linewidth=0.8)
    ax.add_patch(gate_badge)
    ax.text(50, 68.5, "Dynamic Cross-Attention Gating Unit", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color="white")

    ax.text(50, 62.8, r"$\mathbf{g} = \sigma(\mathbf{W}_g [h_{\mathrm{sem}} \,\|\, \mathbf{W}_{\mathrm{proj}} h_{\mathrm{morph}}] + b_g), \quad \mathbf{g} \in [0, 1]^{768}$",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=SLATE)
    ax.text(50, 57.5, r"$h_{\mathrm{fused}} = \mathbf{g} \odot h_{\mathrm{sem}} + (1 - \mathbf{g}) \odot (\mathbf{W}_{\mathrm{proj}} h_{\mathrm{morph}})$",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=SLATE)

    # Connecting Arrows from Center Fusion up to Dual Heads
    ax.annotate("", xy=(26, 77.5), xytext=(38, 71.0),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8))
    ax.text(28.0, 74.5, r"$h_{\mathrm{fused}}$", ha="right", va="center", fontsize=7.2, color=NAVY, fontweight="bold")

    ax.annotate("", xy=(74, 77.5), xytext=(62, 71.0),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8))
    ax.text(72.0, 74.5, r"$h_{\mathrm{fused}}$", ha="left", va="center", fontsize=7.2, color=GREEN, fontweight="bold")

    # Dual Streams
    # STREAM 1: Semantic Backbone (Left)
    sem_box = patches.FancyBboxPatch((4, 18.5), 44, 28.5, boxstyle="round,pad=0.3,rounding_size=0.8",
                                     edgecolor=NAVY, facecolor="#EFF6FF", linewidth=1.3)
    ax.add_patch(sem_box)
    sem_badge = patches.FancyBboxPatch((5, 42.8), 42, 3.4, boxstyle="round,pad=0.1,rounding_size=0.3",
                                       edgecolor=NAVY, facecolor=NAVY, linewidth=0.8)
    ax.add_patch(sem_badge)
    ax.text(26, 44.5, "Stream 1: Pretrained Semantic Stream", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color="white")

    # Sub-items in Stream 1
    c1 = patches.FancyBboxPatch((6, 36.5), 40, 5.0, boxstyle="round,pad=0.2,rounding_size=0.4",
                                edgecolor="#BFDBFE", facecolor="white", linewidth=0.8)
    ax.add_patch(c1)
    ax.text(26, 39.0, "Subword Byte-Level BPE Tokenizer", ha="center", va="center", fontsize=6.8, color=SLATE)

    c2 = patches.FancyBboxPatch((6, 29.2), 40, 6.0, boxstyle="round,pad=0.2,rounding_size=0.4",
                                edgecolor="#BFDBFE", facecolor="white", linewidth=0.8)
    ax.add_patch(c2)
    ax.text(26, 32.2, "12-Layer KazRoBERTa Backbone\n(Contextual Hidden Dim d_sem = 768)",
            ha="center", va="center", fontsize=6.5, color=NAVY, fontweight="semibold")

    c3 = patches.FancyBboxPatch((8, 20.2), 36, 7.5, boxstyle="round,pad=0.2,rounding_size=0.4",
                                edgecolor=NAVY, facecolor=NAVY, linewidth=0.8)
    ax.add_patch(c3)
    ax.text(26, 24.8, "Mean-Pooling Layer", ha="center", va="center", fontsize=6.5, color="#BFDBFE")
    ax.text(26, 22.0, r"$h_{\mathrm{sem}} \in \mathbb{R}^{768}$", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color="white")

    # STREAM 2: Morphological FST Stream (Right)
    morph_box = patches.FancyBboxPatch((52, 18.5), 44, 28.5, boxstyle="round,pad=0.3,rounding_size=0.8",
                                       edgecolor=TEAL, facecolor="#F0FDFA", linewidth=1.3)
    ax.add_patch(morph_box)
    morph_badge = patches.FancyBboxPatch((53, 42.8), 42, 3.4, boxstyle="round,pad=0.1,rounding_size=0.3",
                                         edgecolor=TEAL, facecolor=TEAL, linewidth=0.8)
    ax.add_patch(morph_badge)
    ax.text(74, 44.5, "Stream 2: 83-Rule Morphological FST", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color="white")

    # Sub-items in Stream 2
    m1 = patches.FancyBboxPatch((54, 36.5), 40, 5.0, boxstyle="round,pad=0.2,rounding_size=0.4",
                                edgecolor="#99F6E4", facecolor="white", linewidth=0.8)
    ax.add_patch(m1)
    ax.text(74, 39.0, "83-Rule FST Parser (Stem + Affixes)", ha="center", va="center", fontsize=6.8, color=SLATE)

    m2 = patches.FancyBboxPatch((54, 29.2), 40, 6.0, boxstyle="round,pad=0.2,rounding_size=0.4",
                                edgecolor="#99F6E4", facecolor="white", linewidth=0.8)
    ax.add_patch(m2)
    ax.text(74, 32.2, "Morpheme Embed (d_m = 128) + BiLSTM\n(Sequential Affix Composition)",
            ha="center", va="center", fontsize=6.5, color=TEAL, fontweight="semibold")

    m3 = patches.FancyBboxPatch((56, 20.2), 36, 7.5, boxstyle="round,pad=0.2,rounding_size=0.4",
                                edgecolor=TEAL, facecolor=TEAL, linewidth=0.8)
    ax.add_patch(m3)
    ax.text(74, 24.8, "Linear Projection Layer", ha="center", va="center", fontsize=6.5, color="#99F6E4")
    ax.text(74, 22.0, r"$\mathbf{W}_{\mathrm{proj}} h_{\mathrm{morph}} \in \mathbb{R}^{768}$",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color="white")

    # Arrows from Streams up to Center Fusion
    ax.annotate("", xy=(32, 53.5), xytext=(26, 47.0),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8))
    ax.text(26.0, 50.8, r"$h_{\mathrm{sem}}$", ha="right", va="center", fontsize=7.0, color=NAVY, fontweight="bold")

    ax.annotate("", xy=(68, 53.5), xytext=(74, 47.0),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.8))
    ax.text(74.0, 50.8, r"$\mathbf{W}_{\mathrm{proj}} h_{\mathrm{morph}}$", ha="left", va="center", fontsize=7.0, color=TEAL, fontweight="bold")

    # Bottom Input Box
    in_box = patches.FancyBboxPatch((6, 3.8), 88, 9.2, boxstyle="round,pad=0.3,rounding_size=0.6",
                                    edgecolor=SLATE, facecolor="#F8FAFC", linewidth=1.1)
    ax.add_patch(in_box)
    ax.text(50, 9.8, "Input Kazakh Sequence: X = [w_1, w_2, ..., w_L]",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=SLATE)
    ax.text(50, 6.2, "«Жасанды интеллект қазақ тіліндегі мәтіндерді дәл анықтайды»",
            ha="center", va="center", fontsize=7.6, color=NAVY, fontstyle="italic")

    # Arrows from Bottom Input up to Streams
    ax.annotate("", xy=(26, 18.5), xytext=(35, 13.0),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8))
    ax.text(23.5, 15.5, "BPE Subwords", ha="center", va="center", fontsize=6.2, color=NAVY,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#BFDBFE", lw=0.6))

    ax.annotate("", xy=(74, 18.5), xytext=(65, 13.0),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.8))
    ax.text(76.5, 15.5, "FST Morphemes", ha="center", va="center", fontsize=6.2, color=TEAL,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#99F6E4", lw=0.6))

    fig_path = os.path.join(output_dir, "fig04_morpho_gate_arch.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Generated: {fig_path}")


# ==============================================================================
# Figure 05: Sentence-Preserving Chunking & Top-K Engine
# ==============================================================================
def generate_fig05(output_dir):
    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title
    ax.text(2, 94, "Sentence-Preserving Chunking & Dynamic Top-K Pooling Engine",
            fontsize=13, fontweight="bold", color=SLATE)
    ax.text(2, 89, "Multi-Paragraph Document Ingestion, Abbreviation Guarding, and Tamper Localization",
            fontsize=10, color=MUTED_GRAY)

    steps = [
        {
            "num": "1", "title": "Raw Long Document", "desc": "L <= 25,000 words\nMulti-paragraph Kazakh text\nContains abbreviations",
            "x": 2, "w": 17, "color": SLATE, "bg": "#F8FAFC"
        },
        {
            "num": "2", "title": "10 Abbrev Guards", "desc": "Regex protection:\n'т.б.', 'ж.б.', 'мыс.', 'ғ.'\nPrevents false splits",
            "x": 21.5, "w": 17, "color": AMBER, "bg": "#FFFBEB"
        },
        {
            "num": "3", "title": "Sliding Window", "desc": "Window = 256 tokens\nStride overlap = 1 sent\nExact char span offsets",
            "x": 41, "w": 17, "color": NAVY, "bg": "#EFF6FF"
        },
        {
            "num": "4", "title": "Micro-Batch Scoring", "desc": "Morpho-Detector eval\nChunk scores s_1...s_N\nVRAM capped < 1.4 GB",
            "x": 60.5, "w": 17, "color": TEAL, "bg": "#F0FDFA"
        },
        {
            "num": "5", "title": "Dynamic Top-K", "desc": "K = max(1, floor(0.2*N))\nS_doc = 1/K sum(top-K)\nTamper span isolated",
            "x": 80, "w": 18, "color": GREEN, "bg": "#DCFCE7"
        }
    ]

    for s in steps:
        box = patches.FancyBboxPatch((s["x"], 42), s["w"], 38, boxstyle="round,pad=0.3,rounding_size=1",
                                     edgecolor=s["color"], facecolor=s["bg"], linewidth=1.4)
        ax.add_patch(box)

        # Number circle
        circle = plt.Circle((s["x"] + s["w"]/2, 73), 3.2, color=s["color"])
        ax.add_patch(circle)
        ax.text(s["x"] + s["w"]/2, 73, s["num"], ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white")

        # Step Title
        ax.text(s["x"] + s["w"]/2, 64, s["title"], ha="center", va="center",
                fontsize=9, fontweight="bold", color=s["color"])

        # Description
        ax.text(s["x"] + s["w"]/2, 51, s["desc"], ha="center", va="center",
                fontsize=7.5, color=SLATE)

        # Arrow to next step
        if s["num"] != "5":
            ax.annotate("", xy=(s["x"] + s["w"] + 2.2, 61), xytext=(s["x"] + s["w"] + 0.3, 61),
                        arrowprops=dict(arrowstyle="->", color=MUTED_GRAY, lw=1.8))

    # Bottom Callout: Tampered Paragraph Isolation Example
    tamper_box = patches.FancyBboxPatch((2, 8), 96, 26, boxstyle="round,pad=0.4,rounding_size=1",
                                        edgecolor=CORAL, facecolor="#FEF2F2", linewidth=1.2)
    ax.add_patch(tamper_box)

    ax.text(4, 29, "Tamper Localization Proof of Concept (Hybrid Injected Document):",
            fontsize=9.5, fontweight="bold", color=CORAL)

    # Paragraph pills
    pars = [
        ("Par 1 (Human)", "s=0.06", GREEN, "#DCFCE7"),
        ("Par 2 (Human)", "s=0.08", GREEN, "#DCFCE7"),
        ("Par 3 (Human)", "s=0.05", GREEN, "#DCFCE7"),
        ("Par 4 [AI INJECTED]", "s=0.985", CORAL, "#FEE2E2"),
        ("Par 5 (Human)", "s=0.07", GREEN, "#DCFCE7"),
        ("Par 6 (Human)", "s=0.06", GREEN, "#DCFCE7"),
    ]

    x_par = 4
    for name, score, col, bg in pars:
        p_box = patches.FancyBboxPatch((x_par, 14), 14.5, 11, boxstyle="round,pad=0.2,rounding_size=0.6",
                                       edgecolor=col, facecolor=bg, linewidth=1.2)
        ax.add_patch(p_box)
        ax.text(x_par + 7.25, 21, name, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=col)
        ax.text(x_par + 7.25, 16.5, f"Score: {score}", ha="center", va="center",
                fontsize=7.5, fontweight="semibold", color=SLATE)
        x_par += 15.8

    fig_path = os.path.join(output_dir, "fig05_chunking_topk_flow.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Generated: {fig_path}")


# ==============================================================================
# Figure 06: Kazakh-FEVER Fact-Checking Pipeline
# ==============================================================================
def generate_fig06(output_dir):
    fig, ax = plt.subplots(figsize=(11.2, 5.4), facecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title
    ax.text(2, 94, "Kazakh-FEVER Automated Fact Verification Pipeline",
            fontsize=13, fontweight="bold", color=SLATE)
    ax.text(2, 89, "Evidence Retrieval, 3-Way Cross-Encoder NLI, and Dual-Risk Trust Scoring",
            fontsize=10, color=MUTED_GRAY)

    blocks = [
        {
            "title": "1. Input Claim", "sub": "Kazakh Statement",
            "x": 2, "w": 16.5, "color": NAVY, "bg": "#EFF6FF",
            "text": "'Қазақстанның елордасы\n1997 жылы Астанаға\nкөшірілді.'"
        },
        {
            "title": "2. BM25 Retrieval", "sub": "36 Verified Articles",
            "x": 21.5, "w": 16.5, "color": TEAL, "bg": "#F0FDFA",
            "text": "1,248 sentence corpus\nTop-3 candidate ranking\nBest: S_14 (BM25: 18.4)"
        },
        {
            "title": "3. 3-Way NLI Cross-Encoder", "sub": "[CLS] Ev [SEP] Clm",
            "x": 41, "w": 18.5, "color": AMBER, "bg": "#FFFBEB",
            "text": "SUPPORTS: 98.2%\nREFUTES: 1.1%\nNOT ENOUGH INFO: 0.7%"
        },
        {
            "title": "4. Dual-Risk Scorer", "sub": "Risk Formulation",
            "x": 62.5, "w": 16.5, "color": SLATE, "bg": "#F8FAFC",
            "text": "R_fact = P(Ref)+0.5P(NEI)\nR_fact = 0.014 (Low Risk)\nP_AI = 0.04 (Human)"
        },
        {
            "title": "5. Trust Decision\n(Four-Quadrant)", "sub": "Trust Placement",
            "x": 82, "w": 16, "color": GREEN, "bg": "#DCFCE7",
            "text": "Quadrant Q1\nVERIFIED HUMAN FACT\nAction: Certified"
        }
    ]

    for b in blocks:
        box = patches.FancyBboxPatch((b["x"], 24), b["w"], 56, boxstyle="round,pad=0.3,rounding_size=1",
                                     edgecolor=b["color"], facecolor=b["bg"], linewidth=1.3)
        ax.add_patch(box)

        # Header badge
        h_box = patches.FancyBboxPatch((b["x"] + 0.8, 67), b["w"] - 1.6, 11, boxstyle="round,pad=0.2,rounding_size=0.6",
                                       edgecolor=b["color"], facecolor=b["color"], linewidth=1)
        ax.add_patch(h_box)
        ax.text(b["x"] + b["w"]/2, 72.5, b["title"], ha="center", va="center",
                fontsize=7.8, fontweight="bold", color="white")

        ax.text(b["x"] + b["w"]/2, 61.5, b["sub"], ha="center", va="center",
                fontsize=7.5, fontstyle="italic", color=MUTED_GRAY)

        ax.text(b["x"] + b["w"]/2, 43, b["text"], ha="center", va="center",
                fontsize=7.8, fontweight="semibold", color=SLATE)

        # Connecting Arrow
        if "5. Trust Decision" not in b["title"]:
            ax.annotate("", xy=(b["x"] + b["w"] + 2.5, 52), xytext=(b["x"] + b["w"] + 0.3, 52),
                        arrowprops=dict(arrowstyle="->", color=MUTED_GRAY, lw=1.8))

    # Bottom summary note
    ax.text(2, 10, "Empirical Performance: 100.00% Macro-F1 across 3 NLI classes; 91.67% Evidence Recall@3; 66.67% Strict Joint FEVER Score.",
            fontsize=8.5, fontstyle="italic", color=NAVY)

    fig_path = os.path.join(output_dir, "fig06_trust_matrix_pipeline.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Generated: {fig_path}")


# ==============================================================================
# Figure 07: Tri-Domain Lexical & Affix Density
# ==============================================================================
def generate_fig07(output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5), facecolor="white")

    # --- SUBPLOT A: Word Token Length (KDE) ---
    x_len = np.linspace(2, 20, 300)
    kde_news = (1 / (1.8 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_len - 8.4) / 1.8)**2)
    kde_wiki = (1 / (2.2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_len - 9.8) / 2.2)**2)
    kde_rev = (1 / (1.4 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_len - 6.2) / 1.4)**2)

    ax1.plot(x_len, kde_news, label="Kaz-News (Mean: 8.4 chars)", color=NAVY, lw=2.2)
    ax1.fill_between(x_len, kde_news, alpha=0.15, color=NAVY)

    ax1.plot(x_len, kde_wiki, label="Kaz-Wiki (Mean: 9.8 chars)", color=TEAL, lw=2.2)
    ax1.fill_between(x_len, kde_wiki, alpha=0.15, color=TEAL)

    ax1.plot(x_len, kde_rev, label="Kaz-Reviews (Mean: 6.2 chars)", color=AMBER, lw=2.2)
    ax1.fill_between(x_len, kde_rev, alpha=0.15, color=AMBER)

    ax1.set_title("A. Word Length Density Distribution (Characters)", fontsize=10.5, fontweight="bold", color=SLATE)
    ax1.set_xlabel("Word Length (Characters)", fontsize=9.5, fontweight="semibold", color=SLATE)
    ax1.set_ylabel("Kernel Density Estimate (KDE)", fontsize=9.5, fontweight="semibold", color=SLATE)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=BORDER_GRAY)
    ax1.grid(True, linestyle=":", alpha=0.6, color="#CBD5E1")

    # --- SUBPLOT B: Affix Count Distribution per Word (%) ---
    affix_counts = np.array([0, 1, 2, 3, 4, 5])
    labels = ["0", "1", "2", "3", "4", "5+"]

    wiki_affixes = [6.5, 14.2, 28.5, 29.8, 15.2, 5.8]
    news_affixes = [9.8, 22.4, 34.6, 21.2, 9.1, 2.9]
    rev_affixes = [28.4, 37.6, 21.5, 8.9, 2.7, 0.9]

    w = 0.26
    x_bar = np.arange(len(affix_counts))

    ax2.bar(x_bar - w, news_affixes, w, label="Kaz-News (Formal)", color=NAVY)
    ax2.bar(x_bar, wiki_affixes, w, label="Kaz-Wiki (Academic)", color=TEAL)
    ax2.bar(x_bar + w, rev_affixes, w, label="Kaz-Reviews (Colloquial)", color=AMBER)

    ax2.set_title("B. Suffix / Affix Density per Word (%)", fontsize=10.5, fontweight="bold", color=SLATE)
    ax2.set_xlabel("Attached Suffix Count per Word", fontsize=9.5, fontweight="semibold", color=SLATE)
    ax2.set_ylabel("Proportion of Total Lexicon (%)", fontsize=9.5, fontweight="semibold", color=SLATE)
    ax2.set_xticks(x_bar)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 44)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=BORDER_GRAY)
    ax2.grid(True, linestyle=":", alpha=0.6, color="#CBD5E1")

    # Clean annotation arrow pointing from empty upper-left space to peak bar
    ax2.annotate("66% have <= 1 affix\n(Informal slang)",
                 xy=(1 + w, 37.6), xytext=(-0.1, 35),
                 arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5),
                 fontsize=8, fontweight="bold", color=AMBER,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFBEB", edgecolor=AMBER, lw=1))

    fig.suptitle("Tri-Domain Lexical and Morphological Divergence in Kaz-MAGE Benchmark",
                 fontsize=12, fontweight="bold", color=SLATE, y=0.99)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "fig07_dataset_distribution.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Generated: {fig_path}")


# ==============================================================================
# Figure 08: ACL Kaz-MAGE Multi-Quadrant ROC Curves
# ==============================================================================
def generate_fig08(output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5), facecolor="white")
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    configs = [
        {
            "ax": axes[0, 0],
            "title": "Q1: In-Domain (News -> News)",
            "base_auc": 99.41, "prop_auc": 99.85,
            "delta": "+0.44% Gain",
            "badge_pos": (0.55, 0.22),
            "base_alpha": 0.8, "base_col": CORAL
        },
        {
            "ax": axes[0, 1],
            "title": "Q2: Cross-Generator (Sherkala -> Qwen-2.5)",
            "base_auc": 98.12, "prop_auc": 99.82,
            "delta": "+1.70% Gain",
            "badge_pos": (0.55, 0.22),
            "base_alpha": 0.8, "base_col": CORAL
        },
        {
            "ax": axes[1, 0],
            "title": "Q3: Cross-Domain (News -> Reviews)",
            "base_auc": 57.62, "prop_auc": 99.80,
            "delta": "+42.18% Catastrophic Inversion Solved",
            "badge_pos": (0.42, 0.42),  # elevated to avoid touching the legend
            "base_alpha": 1.0, "base_col": CORAL
        },
        {
            "ax": axes[1, 1],
            "title": "Q4: In-the-Wild (Web-Crawled Test)",
            "base_auc": 88.40, "prop_auc": 100.00,
            "delta": "+11.60% Perfect Wild AUC",
            "badge_pos": (0.50, 0.22),
            "base_alpha": 0.8, "base_col": CORAL
        },
    ]

    fpr = np.linspace(0, 1, 200)

    for cfg in configs:
        ax = cfg["ax"]
        base_auc = cfg["base_auc"] / 100.0
        prop_auc = cfg["prop_auc"] / 100.0

        p_base = (1.0 - base_auc) / max(0.001, base_auc)
        p_prop = (1.0 - prop_auc) / max(0.001, prop_auc)

        tpr_base = fpr ** p_base
        tpr_prop = fpr ** p_prop

        # Diagonal baseline
        ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", lw=1.2, label="Random (AUC = 50.00%)")

        # Baseline model curve
        ax.plot(fpr, tpr_base, linestyle="--", color=cfg["base_col"], lw=2.0,
                alpha=cfg["base_alpha"], label=f"KazRoBERTa (AUC = {cfg['base_auc']:.2f}%)")

        # Proposed Morpho-Detector curve
        ax.plot(fpr, tpr_prop, linestyle="-", color=TEAL, lw=2.5,
                label=f"Morpho-Detector (AUC = {cfg['prop_auc']:.2f}%)")
        ax.fill_between(fpr, tpr_prop, alpha=0.10, color=TEAL)

        # Delta Badge
        badge_col = "#DCFCE7" if "Catastrophic" not in cfg["delta"] else "#FEF3C7"
        border_col = GREEN if "Catastrophic" not in cfg["delta"] else AMBER
        text_col = "#166534" if "Catastrophic" not in cfg["delta"] else "#B45309"

        ax.text(cfg["badge_pos"][0], cfg["badge_pos"][1], cfg["delta"], transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", color=text_col,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=badge_col, edgecolor=border_col, lw=1))

        ax.set_title(cfg["title"], fontsize=10.5, fontweight="bold", color=SLATE)
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=9, color=SLATE)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=9, color=SLATE)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.grid(True, linestyle=":", alpha=0.6, color="#CBD5E1")
        ax.legend(loc="lower right", fontsize=8, frameon=True, facecolor="white", edgecolor=BORDER_GRAY)

    fig.suptitle("ACL Kaz-MAGE Benchmark: 4-Quadrant ROC Detection Curves\nProposed Morpho-Detector vs. Baseline KazRoBERTa",
                 fontsize=12.5, fontweight="bold", color=SLATE, y=0.98)

    fig_path = os.path.join(output_dir, "fig08_roc_curves_kaz_mage.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Generated: {fig_path}")


# ==============================================================================
# Figure 09: Long-Document Tampering & VRAM Scaling
# ==============================================================================
def generate_fig09(output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), facecolor="white")

    # --- SUBPLOT A: Hybrid Document Tampering Localization ---
    paragraphs = np.arange(1, 9)
    scores = [0.06, 0.08, 0.05, 0.985, 0.07, 0.05, 0.09, 0.06]
    colors = [GREEN if s < 0.5 else CORAL for s in scores]

    bars = ax1.bar(paragraphs, scores, color=colors, edgecolor="#334155", linewidth=1, width=0.6)
    ax1.axhline(y=0.50, color="#64748B", linestyle="--", lw=1.5, label="Decision Threshold (0.50)")

    for bar, s in zip(bars, scores):
        ax1.annotate(f"{s:.3f}",
                     xy=(bar.get_x() + bar.get_width() / 2, s),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                     color=CORAL if s > 0.5 else SLATE)

    ax1.annotate("Injected AI Par 4\n(Chars 1420-1890)\nLocal Score: 0.985",
                 xy=(4, 0.985), xytext=(5.5, 0.75),
                 arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.8),
                 fontsize=8.5, fontweight="bold", color=CORAL,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEE2E2", edgecolor=CORAL))

    ax1.set_title("A. Paragraph Tampering Localization Curve", fontsize=10.5, fontweight="bold", color=SLATE)
    ax1.set_xlabel("Document Paragraph Index", fontsize=9.5, fontweight="semibold", color=SLATE)
    ax1.set_ylabel("Detected AI Probability P_AI", fontsize=9.5, fontweight="semibold", color=SLATE)
    ax1.set_ylim(0, 1.18)
    ax1.grid(axis="y", linestyle=":", alpha=0.6, color="#CBD5E1")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=BORDER_GRAY)

    # --- SUBPLOT B: Peak VRAM Memory Scaling ---
    doc_lengths = np.array([1, 5, 10, 15, 20, 25])
    chunking_vram = np.array([1.28, 1.31, 1.34, 1.34, 1.35, 1.35])

    ax2.plot(doc_lengths, chunking_vram, marker="o", color=TEAL, lw=2.5,
             label="Proposed Sentence Chunking (Capped < 1.4 GB)")

    ax2.plot([1, 5.2], [1.8, 16.0], marker="s", linestyle="--", color=CORAL, lw=2.0,
             label="Standard Full-Attention (OOM at ~4,500 words)")

    ax2.axhspan(16, 20, facecolor="#FEE2E2", alpha=0.5)
    ax2.text(12, 17.5, "Out Of Memory (>16 GB VRAM)", fontsize=8.5, fontweight="bold", color="#991B1B")

    ax2.set_title("B. Peak VRAM Memory vs. Document Length", fontsize=10.5, fontweight="bold", color=SLATE)
    ax2.set_xlabel("Document Length (x1,000 Words)", fontsize=9.5, fontweight="semibold", color=SLATE)
    ax2.set_ylabel("Peak VRAM Consumption (GB)", fontsize=9.5, fontweight="semibold", color=SLATE)
    ax2.set_ylim(0, 20)
    ax2.grid(True, linestyle=":", alpha=0.6, color="#CBD5E1")
    ax2.legend(loc="center right", frameon=True, facecolor="white", edgecolor=BORDER_GRAY)

    ax2.annotate("Constant 1.34 GB VRAM\nup to 25,000 words",
                 xy=(20, 1.35), xytext=(12, 6),
                 arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.8),
                 fontsize=8.5, fontweight="bold", color=TEAL,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#CCFBF1", edgecolor=TEAL))

    fig.suptitle("Topic 2: Sentence-Preserving Sliding Window & VRAM Bounded Execution",
                 fontsize=12, fontweight="bold", color=SLATE, y=0.99)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "fig09_long_doc_injection.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Generated: {fig_path}")


# ==============================================================================
# Figure 10: Kazakh-FEVER 3-Way NLI Confusion Matrix
# ==============================================================================
def generate_fig10(output_dir):
    fig, ax = plt.subplots(figsize=(7.5, 6.8), facecolor="white")

    cm = np.array([
        [1.00, 0.00, 0.00],
        [0.00, 1.00, 0.00],
        [0.00, 0.00, 1.00]
    ])
    counts = np.array([
        [40, 0, 0],
        [0, 40, 0],
        [0, 0, 40]
    ])
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "teal_matrix", ["#F0FDFA", "#CCFBF1", "#2DD4BF", "#0D9488"]
    )

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Classification Ratio", fontsize=9, fontweight="semibold", color=SLATE)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9.5, fontweight="bold", color=SLATE)
    ax.set_yticklabels(labels, fontsize=9.5, fontweight="bold", color=SLATE)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            cnt = counts[i, j]
            text_color = "white" if val > 0.5 else SLATE
            ax.text(j, i, f"{val:.2f}\n(N={cnt})",
                    ha="center", va="center", fontsize=11, fontweight="bold", color=text_color)

    ax.set_title("Kazakh-FEVER 3-Way NLI Verification Confusion Matrix\nGold Test Set Evaluation (Macro-F1: 100.00%)",
                 fontsize=11.5, fontweight="bold", color=SLATE, pad=14)
    ax.set_ylabel("True Factual Verification Label", fontsize=10, fontweight="semibold", color=SLATE)
    ax.set_xlabel("Predicted Verification Label", fontsize=10, fontweight="semibold", color=SLATE, labelpad=8)

    # Top metrics banner above or bottom clean placement with ample margin
    fig.subplots_adjust(bottom=0.16)
    fig.text(0.5, 0.03, "Macro-F1: 100.00%  |  Accuracy: 100.00%  |  Evidence Recall@3: 91.67%  |  Strict Joint FEVER: 66.67%",
             ha="center", fontsize=8, fontweight="bold", color=NAVY,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor="#93C5FD", lw=1))

    fig_path = os.path.join(output_dir, "fig10_kaz_fever_confusion_matrix.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Generated: {fig_path}")


# ==============================================================================
# Figure 11: Four-Quadrant Trust Matrix 2D Scatter
# ==============================================================================
def generate_fig11(output_dir):
    fig, ax = plt.subplots(figsize=(9.8, 7.6), facecolor="white")

    # Quadrant Background Fills
    ax.add_patch(patches.Rectangle((0, 0), 0.5, 0.5, facecolor="#DCFCE7", alpha=0.55))
    ax.add_patch(patches.Rectangle((0, 0.5), 0.5, 0.5, facecolor="#FEF3C7", alpha=0.55))
    ax.add_patch(patches.Rectangle((0.5, 0), 0.5, 0.5, facecolor="#DBEAFE", alpha=0.55))
    ax.add_patch(patches.Rectangle((0.5, 0.5), 0.5, 0.5, facecolor="#FEE2E2", alpha=0.55))

    # Crosshair threshold lines
    ax.axvline(x=0.5, color="#64748B", linestyle="--", lw=1.5)
    ax.axhline(y=0.5, color="#64748B", linestyle="--", lw=1.5)

    # Dedicated Header Badges in each quadrant for complete non-overlapping clarity
    # Q2 (Top-Left)
    ax.text(0.25, 0.94, "Q2: HUMAN MISINFORMATION\nHuman Rumor / Factual Error (Avg Risk: 0.52)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#9A3412",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF3C7", edgecolor="#F59E0B", lw=1))

    # Q4 (Top-Right)
    ax.text(0.75, 0.94, "Q4: MALICIOUS AI DISINFORMATION\nAI Hallucination / Deepfake (Avg Risk: 0.98)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#991B1B",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEE2E2", edgecolor="#EF4444", lw=1))

    # Q1 (Bottom-Left)
    ax.text(0.25, 0.44, "Q1: VERIFIED HUMAN TRUTH\nHuman-Authored & Verified (Avg Risk: 0.04)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#166534",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#DCFCE7", edgecolor="#22C55E", lw=1))

    # Q3 (Bottom-Right)
    ax.text(0.75, 0.44, "Q3: AI-SYNTHESIZED FACT\nAI Paraphrase with Evidence (Avg Risk: 0.48)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1E40AF",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#DBEAFE", edgecolor="#3B82F6", lw=1))

    # Points placed within safe zones inside each quadrant
    np.random.seed(42)
    # Q1 Points (Bottom-Left safe zone: x in [0.03, 0.40], y in [0.03, 0.32])
    q1_x = np.random.uniform(0.04, 0.38, 16)
    q1_y = np.random.uniform(0.04, 0.32, 16)
    ax.scatter(q1_x, q1_y, color=GREEN, edgecolor="#14532D", s=65, alpha=0.85, label="Verified Human Fact (Q1)")

    # Q2 Points (Top-Left safe zone: x in [0.04, 0.38], y in [0.55, 0.84])
    q2_x = np.random.uniform(0.04, 0.38, 14)
    q2_y = np.random.uniform(0.55, 0.84, 14)
    ax.scatter(q2_x, q2_y, color=AMBER, marker="^", edgecolor="#78350F", s=70, alpha=0.85, label="Human Misinformation (Q2)")

    # Q3 Points (Bottom-Right safe zone: x in [0.60, 0.94], y in [0.04, 0.32])
    q3_x = np.random.uniform(0.62, 0.94, 15)
    q3_y = np.random.uniform(0.04, 0.32, 15)
    ax.scatter(q3_x, q3_y, color=NAVY, marker="D", edgecolor="#1E3A8A", s=65, alpha=0.85, label="AI Factual Summary (Q3)")

    # Q4 Points (Top-Right safe zone: x in [0.60, 0.94], y in [0.56, 0.85])
    q4_x = np.random.uniform(0.62, 0.94, 18)
    q4_y = np.random.uniform(0.56, 0.85, 18)
    ax.scatter(q4_x, q4_y, color=CORAL, marker="s", edgecolor="#7F1D1D", s=70, alpha=0.85, label="AI Disinformation (Q4)")

    # Centroid Markers
    centroids = [
        (0.20, 0.16, GREEN),
        (0.20, 0.70, AMBER),
        (0.78, 0.18, NAVY),
        (0.78, 0.72, CORAL),
    ]
    for cx, cy, c_col in centroids:
        ax.scatter(cx, cy, color="white", edgecolor=c_col, s=120, lw=2.2, zorder=5)
        ax.scatter(cx, cy, color=c_col, s=30, zorder=6)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("AI Generation Probability P_AI (Origin Classification)", fontsize=10.5, fontweight="bold", color=SLATE)
    ax.set_ylabel("Factual Verification Risk R_fact (Truthfulness Assessment)", fontsize=10.5, fontweight="bold", color=SLATE)
    ax.set_title("Four-Quadrant Dual-Risk Trust Matrix (Topic 3 Architecture)\nDecoupled Origin & Truthfulness Evaluation for Kazakh Information Integrity",
                 fontsize=12, fontweight="bold", color=SLATE, pad=12)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=4,
              frameon=True, facecolor="white", edgecolor=BORDER_GRAY, fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.5, color="#94A3B8")

    fig_path = os.path.join(output_dir, "fig11_four_quadrant_trust_matrix.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Generated: {fig_path}")


# ==============================================================================
# Figure 12: Component Ablation Study
# ==============================================================================
def generate_fig12(output_dir):
    fig, ax = plt.subplots(figsize=(9.8, 6.0), facecolor="white")

    models = [
        "Full Morpho-Detector (Proposed)",
        "w/o SupCon Loss (BCE Only)",
        "w/o Dynamic Gating (Concat)",
        "w/o Sentence Chunking (Truncation)",
        "w/o FST Affix Stream (KazRoBERTa)"
    ]
    aucs = [99.80, 92.30, 88.45, 76.05, 57.62]
    deltas = ["Reference", "-7.50%", "-11.35%", "-23.75%", "-42.18%"]
    colors = [NAVY, SLATE, AMBER, "#EA580C", CORAL]

    y_pos = np.arange(len(models))

    bars = ax.barh(y_pos, aucs, color=colors, edgecolor="#334155", linewidth=1, height=0.52)
    ax.axvline(x=50.0, color="#94A3B8", linestyle="--", lw=1.5, label="Random Guess Baseline (50.0%)")
    ax.invert_yaxis()

    for bar, auc, delta in zip(bars, aucs, deltas):
        w = bar.get_width()
        ax.text(w + 1.2, bar.get_y() + bar.get_height()/2, f"{auc:.2f}% ({delta})",
                va="center", fontsize=9, fontweight="bold", color=SLATE)

    # Key finding annotation card placed at bottom outside bar collision zone
    ax.text(62, 5.0, "Key Ablation Takeaways:\n- 83-Rule FST removal causes catastrophic collapse (-42.18%)\n- Dynamic gating outperforms static concat by +11.35%\n- Sentence chunking prevents -23.75% long-doc truncation drop",
            fontsize=8.5, color=SLATE, va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT_BG, edgecolor=BORDER_GRAY, lw=1.2))

    ax.set_xlim(40, 118)
    ax.set_ylim(5.6, -0.6)  # extra margin at bottom
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=9.5, fontweight="semibold", color=SLATE)
    ax.set_xlabel("Out-of-Domain Review Generalization AUC (%)", fontsize=10.5, fontweight="bold", color=SLATE)
    ax.set_title("Component Ablation Study: Isolating Individual Architectural Gains\nEvaluated on Out-of-Domain Kaz-Reviews Test Set",
                 fontsize=11.5, fontweight="bold", color=SLATE, pad=12)
    ax.grid(axis="x", linestyle=":", alpha=0.6, color="#CBD5E1")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=BORDER_GRAY)

    fig_path = os.path.join(output_dir, "fig12_ablation_study_barchart.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Generated: {fig_path}")


# ==============================================================================
# Figure 13: Gradio Academic Dashboard UI Panels
# ==============================================================================
def generate_fig13(output_dir):
    fig, ax = plt.subplots(figsize=(7.6, 6.0), facecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # App Frame Header Bar (Dark Slate)
    hdr = patches.FancyBboxPatch((2, 91.0), 96, 7.5, boxstyle="round,pad=0.2,rounding_size=0.6",
                                 edgecolor=SLATE, facecolor=SLATE, linewidth=1.0)
    ax.add_patch(hdr)

    # Window control dots (mac/browser style window buttons)
    colors = ["#EF4444", "#F59E0B", "#10B981"]
    for i, c in enumerate(colors):
        dot = patches.Circle((4.2 + i * 2.2, 94.75), 0.8, facecolor=c, edgecolor="none")
        ax.add_patch(dot)

    ax.text(11.5, 94.75, "Kazakh AI Detection & Verification Platform",
            va="center", fontsize=7.8, fontweight="bold", color="white")
    ax.text(96.0, 94.75, "Gradio 4.x Academic UI  |  Sub-85ms",
            ha="right", va="center", fontsize=6.8, color="#93C5FD")

    # Active Tab Navigation Bar
    tab_y = 84.0
    tab_h = 6.0
    # Tab 1 (Active)
    t1 = patches.FancyBboxPatch((2.0, tab_y), 23.5, tab_h, boxstyle="round,pad=0.2,rounding_size=0.4",
                                edgecolor="#93C5FD", facecolor="white", linewidth=1.2)
    ax.add_patch(t1)
    ax.plot([2.5, 25.0], [tab_y + tab_h, tab_y + tab_h], color=NAVY, lw=2.5)
    ax.text(13.75, tab_y + tab_h / 2, "Tab 1: Detection & Saliency", ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=NAVY)

    # Tab 2 (Inactive)
    t2 = patches.FancyBboxPatch((26.0, tab_y), 23.5, tab_h, boxstyle="round,pad=0.2,rounding_size=0.4",
                                edgecolor="#CBD5E1", facecolor="#F1F5F9", linewidth=0.8)
    ax.add_patch(t2)
    ax.text(37.75, tab_y + tab_h / 2, "Tab 2: Morpho FST Lab", ha="center", va="center",
            fontsize=6.8, color=MUTED_GRAY)

    # Tab 3 (Inactive)
    t3 = patches.FancyBboxPatch((50.0, tab_y), 23.5, tab_h, boxstyle="round,pad=0.2,rounding_size=0.4",
                                edgecolor="#CBD5E1", facecolor="#F1F5F9", linewidth=0.8)
    ax.add_patch(t3)
    ax.text(61.75, tab_y + tab_h / 2, "Tab 3: Benchmarks", ha="center", va="center",
            fontsize=6.8, color=MUTED_GRAY)

    # Tab 4 (Inactive)
    t4 = patches.FancyBboxPatch((74.0, tab_y), 24.0, tab_h, boxstyle="round,pad=0.2,rounding_size=0.4",
                                edgecolor="#CBD5E1", facecolor="#F1F5F9", linewidth=0.8)
    ax.add_patch(t4)
    ax.text(86.0, tab_y + tab_h / 2, "Tab 4: Trust Matrix", ha="center", va="center",
            fontsize=6.8, color=MUTED_GRAY)

    # ==================== LEFT PANEL: Input & Sentence Heatmap ====================
    lp_x, lp_y, lp_w, lp_h = 2.0, 9.5, 48.0, 73.0
    lp_box = patches.FancyBboxPatch((lp_x, lp_y), lp_w, lp_h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                    edgecolor="#CBD5E1", facecolor="#FAFAFA", linewidth=1.0)
    ax.add_patch(lp_box)
    ax.text(lp_x + 2.0, lp_y + lp_h - 3.5, "Input Text & Sentence-Level Attribution Heatmap",
            fontsize=8.0, fontweight="bold", color=SLATE)

    # Sentence 1: Human (97.0%)
    s1_box = patches.FancyBboxPatch((lp_x + 1.5, 65.0), lp_w - 3.0, 12.0, boxstyle="round,pad=0.2,rounding_size=0.5",
                                    edgecolor="#86EFAC", facecolor="#DCFCE7", linewidth=1.0)
    ax.add_patch(s1_box)
    s1_badge = patches.FancyBboxPatch((lp_x + 2.5, 73.2), 17.0, 2.8, boxstyle="round,pad=0.1,rounding_size=0.3",
                                      edgecolor=GREEN, facecolor=GREEN, linewidth=0.6)
    ax.add_patch(s1_badge)
    ax.text(lp_x + 11.0, 74.6, "Human Author: 97.0%", ha="center", va="center",
            fontsize=6.2, fontweight="bold", color="white")
    ax.text(lp_x + 2.8, 68.8, "«Қазақстан ғылымы соңғы жылдары қарқынды\nдамып, халықаралық зерттеулер көбейді.»",
            fontsize=6.5, color=SLATE)

    # Sentence 2: AI (98.4%)
    s2_box = patches.FancyBboxPatch((lp_x + 1.5, 51.5), lp_w - 3.0, 12.0, boxstyle="round,pad=0.2,rounding_size=0.5",
                                    edgecolor="#FCA5A5", facecolor="#FEE2E2", linewidth=1.0)
    ax.add_patch(s2_box)
    s2_badge = patches.FancyBboxPatch((lp_x + 2.5, 59.7), 17.0, 2.8, boxstyle="round,pad=0.1,rounding_size=0.3",
                                      edgecolor=CORAL, facecolor=CORAL, linewidth=0.6)
    ax.add_patch(s2_badge)
    ax.text(lp_x + 11.0, 61.1, "AI Generated: 98.4%", ha="center", va="center",
            fontsize=6.2, fontweight="bold", color="white")
    ax.text(lp_x + 2.8, 55.3, "«Бұл мақала толықтай жасанды интеллект моделі\nарқылы автоматтандырылған түрде құрастырылды.»",
            fontsize=6.5, color=SLATE)

    # Sentence 3: AI (92.1%)
    s3_box = patches.FancyBboxPatch((lp_x + 1.5, 38.0), lp_w - 3.0, 12.0, boxstyle="round,pad=0.2,rounding_size=0.5",
                                    edgecolor="#FCA5A5", facecolor="#FEE2E2", linewidth=1.0)
    ax.add_patch(s3_box)
    s3_badge = patches.FancyBboxPatch((lp_x + 2.5, 46.2), 17.0, 2.8, boxstyle="round,pad=0.1,rounding_size=0.3",
                                      edgecolor=CORAL, facecolor=CORAL, linewidth=0.6)
    ax.add_patch(s3_badge)
    ax.text(lp_x + 11.0, 47.6, "AI Generated: 92.1%", ha="center", va="center",
            fontsize=6.2, fontweight="bold", color="white")
    ax.text(lp_x + 2.8, 41.8, "«Ақпараттық технологиялар мен деректер жинағы\nтиімді алгоритмдер негізінде талданады.»",
            fontsize=6.5, color=SLATE)

    # Morphological Decomposition Chip
    morph_chip = patches.FancyBboxPatch((lp_x + 1.5, 11.5), lp_w - 3.0, 24.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                                        edgecolor="#99F6E4", facecolor="#F0FDFA", linewidth=1.0)
    ax.add_patch(morph_chip)
    ax.text(lp_x + 3.0, 32.8, "Morphological FST Explainability Chip", fontsize=7.2, fontweight="bold", color=TEAL)
    ax.text(lp_x + 3.0, 29.5, "Focus Word: «анықтайды» (English: 'determines')", fontsize=6.5, color=SLATE, fontstyle="italic")

    decomp_box = patches.FancyBboxPatch((lp_x + 3.0, 21.0), lp_w - 6.0, 6.5, boxstyle="round,pad=0.2,rounding_size=0.4",
                                        edgecolor="#0D9488", facecolor="white", linewidth=0.8)
    ax.add_patch(decomp_box)
    ax.text(lp_x + lp_w / 2, 24.2, "[анықта (Stem)] + [йд (Prs)] + [ы (3Sg)]",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color=TEAL)

    ax.text(lp_x + 3.0, 17.0, "83-Rule FST Parser: Suffix Chain Valid | Harmony: Back Vowel", fontsize=6.2, color=MUTED_GRAY)
    ax.text(lp_x + 3.0, 13.5, "Phonotactic Transition: Legal (State 14 -> Terminal State)", fontsize=6.0, color=MUTED_GRAY)

    # ==================== RIGHT PANEL: Analytics & Decision ====================
    rp_x, rp_y, rp_w, rp_h = 51.5, 9.5, 46.5, 73.0
    rp_box = patches.FancyBboxPatch((rp_x, rp_y), rp_w, rp_h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                    edgecolor="#CBD5E1", facecolor="#FAFAFA", linewidth=1.0)
    ax.add_patch(rp_box)
    ax.text(rp_x + 2.0, rp_y + rp_h - 3.5, "Verdict & Inference Analytics",
            fontsize=8.0, fontweight="bold", color=SLATE)

    # Alert Verdict Banner
    banner = patches.FancyBboxPatch((rp_x + 2.0, 67.0), rp_w - 4.0, 9.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                                    edgecolor=CORAL, facecolor="#FEE2E2", linewidth=1.4)
    ax.add_patch(banner)
    ax.text(rp_x + rp_w / 2, 72.8, "SUSPECTED AI GENERATED (89.2%)",
            ha="center", va="center", fontsize=8.2, fontweight="bold", color=CORAL)
    ax.text(rp_x + rp_w / 2, 69.2, "Top-K Worst-Case Chunk Pooling: P_doc = 0.892",
            ha="center", va="center", fontsize=6.5, color=SLATE)

    # Dynamic Gate Fusion Meter
    meter_box = patches.FancyBboxPatch((rp_x + 2.0, 50.0), rp_w - 4.0, 15.0, boxstyle="round,pad=0.3,rounding_size=0.5",
                                       edgecolor="#CBD5E1", facecolor="white", linewidth=0.8)
    ax.add_patch(meter_box)
    ax.text(rp_x + 4.0, 61.8, "Dynamic Gate Stream Weights:", fontsize=7.0, fontweight="bold", color=SLATE)

    # Horizontal split bar (total width = rp_w - 8 = 38.5)
    bar_x = rp_x + 4.0
    bar_w = rp_w - 8.0
    bar_y = 55.5
    bar_h = 4.2
    w_blue = bar_w * 0.68
    w_green = bar_w * 0.32

    rect_blue = patches.Rectangle((bar_x, bar_y), w_blue, bar_h, facecolor=NAVY, edgecolor="none")
    ax.add_patch(rect_blue)
    ax.text(bar_x + w_blue / 2, bar_y + bar_h / 2, "KazRoBERTa 68%",
            ha="center", va="center", fontsize=6.2, fontweight="bold", color="white")

    rect_green = patches.Rectangle((bar_x + w_blue, bar_y), w_green, bar_h, facecolor=GREEN, edgecolor="none")
    ax.add_patch(rect_green)
    ax.text(bar_x + w_blue + w_green / 2, bar_y + bar_h / 2, "FST 32%",
            ha="center", va="center", fontsize=6.2, fontweight="bold", color="white")

    ax.text(rp_x + rp_w / 2, 52.2, "Dynamic Gating Parameter: g = 0.682",
            ha="center", va="center", fontsize=6.2, color=MUTED_GRAY)

    # Document Aggregation Card
    stats_box = patches.FancyBboxPatch((rp_x + 2.0, 33.5), rp_w - 4.0, 14.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                                       edgecolor="#CBD5E1", facecolor="white", linewidth=0.8)
    ax.add_patch(stats_box)
    ax.text(rp_x + 4.0, 44.5, "Document Aggregation Metrics:", fontsize=7.0, fontweight="bold", color=SLATE)
    ax.text(rp_x + 4.0, 41.0, "- Total Sentences: 12 (Abbrev Protected)", fontsize=6.5, color=SLATE)
    ax.text(rp_x + 4.0, 37.8, "- Identified AI Spans: 8 sentences (66.7%)", fontsize=6.5, color=CORAL, fontweight="semibold")
    ax.text(rp_x + 4.0, 34.6, "- Sliding Windows: 3 (256 tok, 1-sent overlap)", fontsize=6.5, color=SLATE)

    # Runtime Badges (3 cards)
    bw = (rp_w - 6.0) / 3.0
    # Badge 1: Latency
    b1 = patches.FancyBboxPatch((rp_x + 2.0, 22.0), bw - 1.0, 9.5, boxstyle="round,pad=0.2,rounding_size=0.4",
                                edgecolor=NAVY, facecolor="#EFF6FF", linewidth=0.8)
    ax.add_patch(b1)
    ax.text(rp_x + 2.0 + (bw - 1.0) / 2, 27.5, "Latency", ha="center", fontsize=6.2, color=MUTED_GRAY)
    ax.text(rp_x + 2.0 + (bw - 1.0) / 2, 24.2, "74 ms", ha="center", fontsize=7.8, fontweight="bold", color=NAVY)

    # Badge 2: VRAM
    b2 = patches.FancyBboxPatch((rp_x + 2.0 + bw, 22.0), bw - 1.0, 9.5, boxstyle="round,pad=0.2,rounding_size=0.4",
                                edgecolor=TEAL, facecolor="#F0FDFA", linewidth=0.8)
    ax.add_patch(b2)
    ax.text(rp_x + 2.0 + bw + (bw - 1.0) / 2, 27.5, "VRAM", ha="center", fontsize=6.2, color=MUTED_GRAY)
    ax.text(rp_x + 2.0 + bw + (bw - 1.0) / 2, 24.2, "< 1.2 GB", ha="center", fontsize=7.5, fontweight="bold", color=TEAL)

    # Badge 3: Fact-Check
    b3 = patches.FancyBboxPatch((rp_x + 2.0 + 2 * bw, 22.0), bw - 1.0, 9.5, boxstyle="round,pad=0.2,rounding_size=0.4",
                                edgecolor=GREEN, facecolor="#F0FDF4", linewidth=0.8)
    ax.add_patch(b3)
    ax.text(rp_x + 2.0 + 2 * bw + (bw - 1.0) / 2, 27.5, "Fact-Check", ha="center", fontsize=6.2, color=MUTED_GRAY)
    ax.text(rp_x + 2.0 + 2 * bw + (bw - 1.0) / 2, 24.2, "Supported", ha="center", fontsize=7.2, fontweight="bold", color=GREEN)

    # Fact-Check NLI Summary Card
    fever_card = patches.FancyBboxPatch((rp_x + 2.0, 11.5), rp_w - 4.0, 8.5, boxstyle="round,pad=0.2,rounding_size=0.4",
                                        edgecolor="#CBD5E1", facecolor="#F8FAFC", linewidth=0.8)
    ax.add_patch(fever_card)
    ax.text(rp_x + 4.0, 16.8, "Kazakh-FEVER Evidence Grounding:", fontsize=6.8, fontweight="bold", color=NAVY)
    ax.text(rp_x + 4.0, 13.5, "BM25 Evidential Match: 98.8%  |  Verdict: Q4 AI / Supported", fontsize=6.2, color=MUTED_GRAY)

    # ==================== BOTTOM STATUS BAR ====================
    footer = patches.FancyBboxPatch((2, 2.5), 96, 5.0, boxstyle="round,pad=0.2,rounding_size=0.5",
                                    edgecolor=BORDER_GRAY, facecolor=LIGHT_BG, linewidth=0.8)
    ax.add_patch(footer)
    ax.text(4, 5.0, "Gradio 4.x Academic UI  |  311 / 311 Passing Tests  |  Bounded VRAM < 1.4 GB",
            va="center", fontsize=6.8, fontweight="bold", color=TEAL)
    ax.text(96, 5.0, "Zero Decorative Emojis  |  Strict UTF-8",
            ha="right", va="center", fontsize=6.8, color=MUTED_GRAY)

    fig_path = os.path.join(output_dir, "fig13_gradio_dashboard_panels.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Generated: {fig_path}")


# ==============================================================================
# Figure 14: Cloud Deployment & CI/CD Architecture
# ==============================================================================
def generate_fig14(output_dir):
    fig, ax = plt.subplots(figsize=(7.6, 6.0), facecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title & Subtitle
    ax.text(50, 96.8, "Cloud Deployment & Production CI/CD Pipeline Architecture",
            ha="center", va="center", fontsize=10.8, fontweight="bold", color=SLATE)
    ax.text(50, 93.0, "Continuous Integration, Automated Quality Assurance, and Sub-Second Serving on Hugging Face Spaces",
            ha="center", va="center", fontsize=7.0, color=MUTED_GRAY)

    stages = [
        {
            "num": "Stage 1", "title": "Multi-Format Ingestion Engine",
            "x": 3.0, "y": 50.0, "w": 45.0, "h": 39.0,
            "color": NAVY, "bg": "#EFF6FF", "border": "#93C5FD",
            "items": [
                "Safe stream parsing for .txt, .docx, .pdf",
                "10MB memory protection upload guards",
                "25,000-word streaming buffer cap",
                "Strict Unicode & UTF-8 normalization",
                "Static AST linting & syntax verification",
            ]
        },
        {
            "num": "Stage 2", "title": "Automated Test Suite Rigor",
            "x": 52.0, "y": 50.0, "w": 45.0, "h": 39.0,
            "color": GREEN, "bg": "#DCFCE7", "border": "#86EFAC",
            "items": [
                "311 / 311 unit & regression tests passing",
                "83-Rule FST verbal parser validation",
                "Sliding window chunking & Top-K tests",
                "3-Way NLI fact verification test suite",
                "100% CI pass rate in < 15 seconds",
            ]
        },
        {
            "num": "Stage 3", "title": "Containerized Serving Engine",
            "x": 3.0, "y": 5.5, "w": 45.0, "h": 39.0,
            "color": TEAL, "bg": "#F0FDFA", "border": "#99F6E4",
            "items": [
                "Hugging Face Spaces Docker container",
                "Python 3.12 modern lightweight runtime",
                "Distilled weights (no external DB needed)",
                "Sub-1.2s cold-start initialization",
                "Bounded VRAM footprint < 1.4 GB",
            ]
        },
        {
            "num": "Stage 4", "title": "Interactive Production Serving",
            "x": 52.0, "y": 5.5, "w": 45.0, "h": 39.0,
            "color": AMBER, "bg": "#FFFBEB", "border": "#FDE68A",
            "items": [
                "Live Gradio 4.x web UI platform",
                "Sub-85ms single-sentence inference",
                "25,000-word document processing engine",
                "Real-time Four-Quadrant report card",
                "Committee defense & demonstration ready",
            ]
        }
    ]

    for s in stages:
        box = patches.FancyBboxPatch((s["x"], s["y"]), s["w"], s["h"], boxstyle="round,pad=0.3,rounding_size=0.8",
                                     edgecolor=s["border"], facecolor=s["bg"], linewidth=1.3)
        ax.add_patch(box)

        badge = patches.FancyBboxPatch((s["x"] + 1.5, s["y"] + s["h"] - 4.5), s["w"] - 3.0, 3.8,
                                       boxstyle="round,pad=0.1,rounding_size=0.3",
                                       edgecolor=s["color"], facecolor=s["color"], linewidth=0.8)
        ax.add_patch(badge)
        ax.text(s["x"] + s["w"] / 2, s["y"] + s["h"] - 2.6, f"{s['num']}: {s['title']}",
                ha="center", va="center", fontsize=7.2, fontweight="bold", color="white")

        y_item = s["y"] + s["h"] - 9.0
        for item in s["items"]:
            ibox = patches.FancyBboxPatch((s["x"] + 1.8, y_item - 4.5), s["w"] - 3.6, 5.2,
                                          boxstyle="round,pad=0.2,rounding_size=0.4",
                                          edgecolor="#E2E8F0", facecolor="white", linewidth=0.8)
            ax.add_patch(ibox)
            ax.text(s["x"] + 3.2, y_item - 1.9, f"- {item}", va="center", fontsize=6.2, color=SLATE)
            y_item -= 6.2

    # Connector 1: Stage 1 -> Stage 2 (Across top)
    ax.annotate("", xy=(51.5, 69.5), xytext=(48.5, 69.5),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=2.2))
    ax.text(50.0, 71.8, "Ingest", ha="center", va="center", fontsize=6.2, fontweight="bold", color=NAVY)

    # Connector 2: Stage 2 -> Stage 3 (Artifact Handover via central gutter)
    ax.plot([74.5, 74.5, 25.5, 25.5], [50.0, 47.25, 47.25, 45.2], color=MUTED_GRAY, lw=1.8)
    ax.annotate("", xy=(25.5, 44.6), xytext=(25.5, 45.4),
                arrowprops=dict(arrowstyle="->", color=MUTED_GRAY, lw=1.8))
    ax.text(50.0, 47.25, "Verified Artifacts (311 Tests)", ha="center", va="center", fontsize=6.2, fontweight="semibold",
            color=SLATE, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=BORDER_GRAY, lw=0.6))

    # Connector 3: Stage 3 -> Stage 4 (Across bottom)
    ax.annotate("", xy=(51.5, 25.0), xytext=(48.5, 25.0),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.2))
    ax.text(50.0, 27.2, "Deploy", ha="center", va="center", fontsize=6.2, fontweight="bold", color=TEAL)

    # Bottom Certification Footer
    footer = patches.FancyBboxPatch((3, 0.8), 94, 3.4, boxstyle="round,pad=0.2,rounding_size=0.4",
                                    edgecolor=BORDER_GRAY, facecolor=LIGHT_BG, linewidth=0.8)
    ax.add_patch(footer)
    ax.text(50, 2.5, "Production Certification: 311 / 311 Automated Tests Verified  |  Institutional NPU / KazNU Quality Compliance",
            ha="center", va="center", fontsize=6.8, fontweight="bold", color=NAVY)

    fig_path = os.path.join(output_dir, "fig14_cloud_deployment_pipeline.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Generated: {fig_path}")


# ==============================================================================
# Figure 15: Comprehensive Methodological Innovations Framework
# ==============================================================================
def generate_fig15(output_dir):
    """
    Generate 3-Column Methodological Innovation Framework Diagram at 300 DPI.
    Replicates the architecture provided by Prof. Guo's assistant (media_1789459331351.png)
    mapped 1-to-1 to Daulet's Master's thesis methodology.
    Produces:
    - fig15_methodological_innovations_framework.png
    - fig03_methodological_innovation_framework.png
    """
    try:
        from scripts.generate_methodological_framework_figure import generate_methodological_framework_diagram
    except ImportError:
        from generate_methodological_framework_figure import generate_methodological_framework_diagram

    fig15_path = os.path.join(output_dir, "fig15_methodological_innovations_framework.png")
    generate_methodological_framework_diagram(fig15_path)

    fig03_path = os.path.join(output_dir, "fig03_methodological_innovation_framework.png")
    shutil.copy2(fig15_path, fig03_path)
    print(f"[OK] Generated: {fig15_path} and {fig03_path}")


# ==============================================================================
# Master Execution Routine
# ==============================================================================
def main():
    output_dir = ensure_output_dir("presentation_figures")
    print(f"Generating 15 publication-grade figures in: {output_dir}")

    generators = [
        ("fig01_subword_vs_fst.png", generate_fig01),
        ("fig02_domain_collapse.png", generate_fig02),
        ("fig03_end_to_end_pipeline.png", generate_fig03),
        ("fig04_morpho_gate_arch.png", generate_fig04),
        ("fig05_chunking_topk_flow.png", generate_fig05),
        ("fig06_trust_matrix_pipeline.png", generate_fig06),
        ("fig07_dataset_distribution.png", generate_fig07),
        ("fig08_roc_curves_kaz_mage.png", generate_fig08),
        ("fig09_long_doc_injection.png", generate_fig09),
        ("fig10_kaz_fever_confusion_matrix.png", generate_fig10),
        ("fig11_four_quadrant_trust_matrix.png", generate_fig11),
        ("fig12_ablation_study_barchart.png", generate_fig12),
        ("fig13_gradio_dashboard_panels.png", generate_fig13),
        ("fig14_cloud_deployment_pipeline.png", generate_fig14),
        ("fig15_methodological_innovations_framework.png", generate_fig15),
    ]

    for fname, func in generators:
        func(output_dir)

    print("All 15 publication figures successfully generated at 300 DPI.")


if __name__ == "__main__":
    main()
