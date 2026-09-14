# -*- coding: utf-8 -*-
"""
Presentation Template Cloning & Content Population Engine.
Clones baseline presentation 'Aliya PPT.pptx' directly, preserves native institutional
branding (KazNU logo, NPU logo group, watermark, and Layout 7 NPU pentagon badge),
excises all commercial watermarks ('合作QQ： 243001978'), repositions the native gradient
highlight tracker ('矩形 29') across sections, clears legacy poultry shapes on content
slides, and populates all 23 slides with:
- Academic slide titles and bilingual subtitles
- 14 publication figures embedded with formal captions
- Formatted academic comparison tables on Slides 5, 11, 12, 14, 16, and 22
- Stat callout cards and analytical breakdown cards
- Slide 22 Committee comments response tables with checkmarks [✓]

Presenter: 大雷 (Daulet)
Advisor: 郭教授 (Prof. Guo)
Date: 2026年9月
"""

import os
import sys
import shutil
import argparse
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# -----------------------------------------------------------------------------
# Color Palette & Design Tokens (Institutional Academic Palette)
# -----------------------------------------------------------------------------
NAVY_PRIMARY = RGBColor(30, 58, 138)       # #1E3A8A - Deep Academic Navy Blue
BLUE_ACCENT = RGBColor(37, 99, 235)        # #2563EB - Royal Blue Accent
TEAL_ACCENT = RGBColor(13, 148, 136)       # #0D9488 - Topic 1 / FST Morphology
AMBER_ACCENT = RGBColor(217, 119, 6)       # #D97706 - Topic 2 / Chunking & Aggregation
GREEN_ACCENT = RGBColor(22, 163, 74)       # #16A34A - Topic 3 / Factual Verification
PURPLE_ACCENT = RGBColor(124, 58, 237)     # #7C3AED - System Engineering & Cloud
RED_ACCENT = RGBColor(220, 38, 38)         # #DC2626 - Risk & Failure Mode Flag

SLATE_TITLE = RGBColor(15, 23, 42)         # #0F172A - Slide title & primary headers
SLATE_BODY = RGBColor(51, 65, 85)          # #334155 - Standard body prose
SLATE_MUTED = RGBColor(100, 116, 139)      # #64748B - Subtitles & captions
BORDER_LIGHT = RGBColor(226, 232, 240)     # #E2E8F0 - Divider lines & card borders

CARD_BG_WHITE = RGBColor(255, 255, 255)    # #FFFFFF - Crisp white cards
CARD_BG_ALT = RGBColor(248, 250, 252)      # #F8FAFC - Soft light gray card fill
CARD_BG_BLUE = RGBColor(239, 246, 255)     # #EFF6FF - Soft blue tinted card fill
CARD_BG_GREEN = RGBColor(240, 253, 244)    # #F0FDF4 - Soft green tinted card fill

FONT_TITLE = "Arial"
FONT_BODY = "Arial"
FONT_ZH = "Microsoft YaHei"

USER_HOME = os.path.expanduser("~")
BASELINE_PPT_PATH = os.path.join(USER_HOME, "Downloads", "Aliya PPT.pptx")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx")
DESKTOP_MIRROR_PATH = os.path.join(USER_HOME, "Desktop", "Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "presentation_figures")

# Gradient highlight tracker reposition schedule (0-indexed slide index)
TRACKER_SCHEDULE = {
    # Section 1: Research Background & Motivation (Slides 3-4)
    2: (Inches(1.30), Inches(1.82)),
    3: (Inches(1.30), Inches(1.82)),
    # Section 2: Related Work & SOTA Gaps (Slides 5-6)
    4: (Inches(3.12), Inches(1.66)),
    5: (Inches(3.12), Inches(1.66)),
    # Section 3: Research Content & Architecture (Slides 7-10)
    6: (Inches(4.78), Inches(1.81)),
    7: (Inches(4.78), Inches(1.81)),
    8: (Inches(4.78), Inches(1.81)),
    9: (Inches(4.78), Inches(1.81)),
    # Section 4: Experiments & Benchmark Evaluation (Slides 11-16)
    10: (Inches(6.59), Inches(2.15)),
    11: (Inches(6.59), Inches(2.15)),
    12: (Inches(6.59), Inches(2.15)),
    13: (Inches(6.59), Inches(2.15)),
    14: (Inches(6.59), Inches(2.15)),
    15: (Inches(6.59), Inches(2.15)),
    # Section 5: Engineering & Demonstration (Slides 17-18)
    16: (Inches(6.59), Inches(2.15)),
    17: (Inches(6.59), Inches(2.15)),
    # Section 6: Conclusion & Thesis Roadmap (Slides 19-21)
    18: (Inches(8.92), Inches(1.95)),
    19: (Inches(8.92), Inches(1.95)),
    20: (Inches(8.92), Inches(1.95)),
    # Comments and Responses (Slide 22)
    21: (Inches(10.83), Inches(2.00)),
}


def remove_shape(shape):
    """Remove a shape element cleanly from its slide."""
    sp = shape._element
    sp.getparent().remove(sp)


def add_slide_header(slide, title_text, subtitle_text=""):
    """Adds standard slide title and bilingual subtitle below the running section bar."""
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(1.30), Inches(12.133), Inches(0.68))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p = tf.paragraphs[0]
    p.text = title_text
    p.font.name = FONT_TITLE
    p.font.size = Pt(16.5)
    p.font.bold = True
    p.font.color.rgb = NAVY_PRIMARY

    if subtitle_text:
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.name = FONT_BODY
        p2.font.size = Pt(10.0)
        p2.font.color.rgb = SLATE_MUTED
        p2.space_before = Pt(2)


def add_figure_with_caption(slide, image_filename, left, top, width, height, caption_text):
    """Embeds a 300 DPI publication figure with a formal caption below."""
    img_path = os.path.join(FIGURE_DIR, image_filename)
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Figure image file not found: {img_path}")

    pic = slide.shapes.add_picture(
        img_path, Inches(left), Inches(top), Inches(width), Inches(height)
    )

    caption_box = slide.shapes.add_textbox(
        Inches(left), Inches(top + height + 0.04), Inches(width), Inches(0.36)
    )
    tf = caption_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = caption_text
    p.font.name = FONT_TITLE
    p.font.size = Pt(8.5)
    p.font.italic = True
    p.font.color.rgb = SLATE_MUTED
    p.alignment = PP_ALIGN.CENTER
    return pic


def add_card(slide, left, top, width, height, title, value_str, subtitle,
             accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE, border_color=BORDER_LIGHT):
    """Adds a statistical callout card with an accent strip on the left."""
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    box.fill.solid()
    box.fill.fore_color.rgb = bg_color
    box.line.color.rgb = border_color
    box.line.width = Pt(1)

    strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(0.08), Inches(height))
    strip.fill.solid()
    strip.fill.fore_color.rgb = accent_color
    strip.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(left + 0.14), Inches(top + 0.08), Inches(width - 0.20), Inches(height - 0.16))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_val = tf.paragraphs[0]
    p_val.text = value_str
    p_val.font.name = FONT_TITLE
    p_val.font.size = Pt(19)
    p_val.font.bold = True
    p_val.font.color.rgb = accent_color

    p_tit = tf.add_paragraph()
    p_tit.text = title
    p_tit.font.name = FONT_TITLE
    p_tit.font.size = Pt(10.5)
    p_tit.font.bold = True
    p_tit.font.color.rgb = SLATE_TITLE
    p_tit.space_before = Pt(2)

    if subtitle:
        p_sub = tf.add_paragraph()
        p_sub.text = subtitle
        p_sub.font.name = FONT_BODY
        p_sub.font.size = Pt(9.0)
        p_sub.font.color.rgb = SLATE_MUTED
        p_sub.space_before = Pt(2)


def add_callout_box(slide, left, top, width, height, title, points,
                    accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE, border_color=BORDER_LIGHT):
    """Adds a structured content card with header strip and bullet points."""
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    box.fill.solid()
    box.fill.fore_color.rgb = bg_color
    box.line.color.rgb = border_color
    box.line.width = Pt(1)

    is_multiline = ("\n" in title)
    strip_h = 0.58 if is_multiline else 0.38

    header_strip = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(strip_h)
    )
    header_strip.fill.solid()
    header_strip.fill.fore_color.rgb = accent_color
    header_strip.line.fill.background()

    tb_head = slide.shapes.add_textbox(Inches(left + 0.12), Inches(top + 0.04), Inches(width - 0.24), Inches(strip_h - 0.08))
    tf_head = tb_head.text_frame
    tf_head.word_wrap = True
    tf_head.margin_left = tf_head.margin_top = tf_head.margin_right = tf_head.margin_bottom = 0
    p_head = tf_head.paragraphs[0]
    p_head.text = title
    p_head.font.name = FONT_TITLE
    p_head.font.size = Pt(9.5 if is_multiline else 11.0)
    p_head.font.bold = True
    p_head.font.color.rgb = RGBColor(255, 255, 255)

    tb_body = slide.shapes.add_textbox(
        Inches(left + 0.14), Inches(top + strip_h + 0.06), Inches(width - 0.28), Inches(height - strip_h - 0.12)
    )
    tf_body = tb_body.text_frame
    tf_body.word_wrap = True
    tf_body.margin_left = tf_body.margin_top = tf_body.margin_right = tf_body.margin_bottom = 0

    for i, pt in enumerate(points):
        p = tf_body.paragraphs[0] if i == 0 else tf_body.add_paragraph()
        p.text = pt
        p.font.name = FONT_BODY
        p.font.size = Pt(8.8)
        p.font.color.rgb = SLATE_BODY
        if i > 0:
            p.space_before = Pt(2.5)


def add_table(slide, left, top, width, height, headers, rows, col_widths=None, highlight_row_idx=None):
    """Adds a clean academic comparison table with colored header and alternating rows."""
    num_rows = len(rows) + 1
    num_cols = len(headers)
    table_shape = slide.shapes.add_table(
        num_rows, num_cols, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    table = table_shape.table

    if col_widths and len(col_widths) == num_cols:
        for c_idx, w in enumerate(col_widths):
            table.columns[c_idx].width = Inches(w)

    for c_idx, h_text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.margin_left = Inches(0.05)
        cell.margin_right = Inches(0.05)
        cell.margin_top = Inches(0.03)
        cell.margin_bottom = Inches(0.03)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY_PRIMARY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = h_text
        p.alignment = PP_ALIGN.CENTER
        p.font.name = FONT_TITLE
        p.font.size = Pt(9.0)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)

    for r_idx, row_data in enumerate(rows):
        is_highlight = (highlight_row_idx is not None and r_idx == highlight_row_idx)
        row_bg = CARD_BG_BLUE if is_highlight else (CARD_BG_WHITE if r_idx % 2 == 0 else CARD_BG_ALT)

        for c_idx, val in enumerate(row_data):
            cell = table.cell(r_idx + 1, c_idx)
            cell.margin_left = Inches(0.05)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.fill.solid()
            cell.fill.fore_color.rgb = row_bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = str(val)
            p.alignment = PP_ALIGN.LEFT if c_idx in (0, 1) and len(str(val)) > 15 else PP_ALIGN.CENTER
            p.font.name = FONT_BODY
            p.font.size = Pt(8.5)
            if is_highlight:
                p.font.bold = True
                p.font.color.rgb = NAVY_PRIMARY
            else:
                p.font.color.rgb = SLATE_BODY

    return table_shape


# =============================================================================
# SLIDE 1 & 2 UPDATES (Cover & TOC)
# =============================================================================

def update_cover_slide(slide):
    """Slide 1 (Cover Page): Preserves KazNU and NPU branding, excises watermark, updates titles."""
    shapes_to_remove = []
    for shp in slide.shapes:
        if shp.has_text_frame and ("243001978" in shp.text or "合作QQ" in shp.text):
            shapes_to_remove.append(shp)
        elif shp.name == "合作QQ： 243001978":
            shapes_to_remove.append(shp)
        elif shp.name == "汇报人组合":
            shapes_to_remove.append(shp)

    for shp in shapes_to_remove:
        remove_shape(shp)

    for shp in slide.shapes:
        if shp.name == "主标题":
            shp.left = Inches(0.34)
            shp.top = Inches(4.05)
            shp.width = Inches(12.60)
            shp.height = Inches(1.50)

            tf = shp.text_frame
            tf.word_wrap = True

            p0 = tf.paragraphs[0]
            p0.text = "Research on Morphologically-Grounded AI-Generated Text Detection and Factual Verification for Low-Resource Kazakh"
            p0.font.name = FONT_TITLE
            p0.font.size = Pt(22)
            p0.font.bold = True
            p0.font.color.rgb = NAVY_PRIMARY
            p0.alignment = PP_ALIGN.LEFT

            p1 = tf.add_paragraph()
            p1.text = "面向低资源哈萨克语的形态感知AI生成文本检测与事实核验研究"
            p1.font.name = FONT_ZH
            p1.font.size = Pt(17)
            p1.font.bold = True
            p1.font.color.rgb = NAVY_PRIMARY
            p1.alignment = PP_ALIGN.LEFT

        elif shp.name == "汇报人":
            shp.left = Inches(0.45)
            shp.top = Inches(5.80)
            shp.width = Inches(12.00)
            shp.height = Inches(1.10)

            tf = shp.text_frame
            tf.word_wrap = True

            p0 = tf.paragraphs[0]
            p0.text = "汇报人：大雷 (Daulet)      导师：郭教授 (Prof. Guo)      2026年9月"
            p0.font.name = FONT_ZH
            p0.font.size = Pt(15)
            p0.font.bold = True
            p0.font.color.rgb = NAVY_PRIMARY
            p0.alignment = PP_ALIGN.LEFT

            p1 = tf.add_paragraph()
            p1.text = "Degree: Master of Science in Computer Science and Technology | School of Computer Science, NPU & KazNU"
            p1.font.name = FONT_BODY
            p1.font.size = Pt(12)
            p1.font.bold = False
            p1.font.color.rgb = SLATE_MUTED
            p1.alignment = PP_ALIGN.LEFT


def update_toc_slide(slide):
    """Slide 2 (TOC): Updates 6 numbered academic thesis sections."""
    shapes_to_remove = []
    for shp in slide.shapes:
        if shp.has_text_frame and ("243001978" in shp.text or "合作QQ" in shp.text):
            shapes_to_remove.append(shp)
        elif shp.name == "合作QQ： 243001978":
            shapes_to_remove.append(shp)

    for shp in shapes_to_remove:
        remove_shape(shp)

    toc_sections = {
        "组合 1": "01. Research Background & Motivation (研究背景与核心动机)",
        "组合 2": "02. Related Work & SOTA Gaps (相关工作与现有局限)",
        "组合 3": "03. Research Content & Architecture (研究内容与系统架构)",
        "组合 12": "04. Experiments & Benchmark Evaluation (实验设计与评测结果)",
        "standalone": "05. Engineering & Demonstration (工程落地与交互演示)",
        "组合 4": "06. Conclusion & Thesis Roadmap (研究总结与毕业规划)",
    }

    for shp in slide.shapes:
        if shp.name in toc_sections and shp.shape_type == 6:  # Group shape
            for sub in shp.shapes:
                if sub.name == "学科特色鲜明" and sub.has_text_frame:
                    p = sub.text_frame.paragraphs[0]
                    p.text = toc_sections[shp.name]
                    p.font.name = FONT_ZH
                    p.font.size = Pt(16)
                    p.font.bold = True
                    p.font.color.rgb = NAVY_PRIMARY
        elif shp.name == "学科特色鲜明" and shp.shape_type != 6:  # Standalone
            shp.width = Inches(7.2)
            p = shp.text_frame.paragraphs[0]
            p.text = toc_sections["standalone"]
            p.font.name = FONT_ZH
            p.font.size = Pt(16)
            p.font.bold = True
            p.font.color.rgb = NAVY_PRIMARY


# =============================================================================
# SLIDE-BY-SLIDE CONTENT POPULATION FUNCTIONS (SLIDES 3 - 22)
# =============================================================================

def populate_slide_03(slide):
    """Slide 3: Research Background — Kazakh NLP Challenges & Synthetic Threats"""
    add_slide_header(
        slide,
        "Research Background: Kazakh NLP Challenges & The Synthetic Text Threat",
        "Agglutinative morphological complexity and rapid proliferation of multilingual generative LLMs in Central Asia"
    )
    add_figure_with_caption(
        slide, "fig01_subword_vs_fst.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="Figure 1. Subword Tokenization Fragmentation vs. 83-Rule FST Morphological Parsing in Kazakh."
    )
    # 4 Stat Callout Cards in 2x2 grid
    add_card(slide, left=6.65, top=2.05, width=2.95, height=2.28,
             title="Affix Complexity", value_str="15+",
             subtitle="Suffixes per nominal/verbal stem; breaks standard subword tokenizers",
             accent_color=TEAL_ACCENT, bg_color=CARD_BG_BLUE)
    add_card(slide, left=9.78, top=2.05, width=2.95, height=2.28,
             title="Document Capacity", value_str="25,000",
             subtitle="Words per document in real-world academic theses & news reports",
             accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE)
    add_card(slide, left=6.65, top=4.48, width=2.95, height=2.28,
             title="Curated Samples", value_str="10,000+",
             subtitle="Tri-domain benchmark covering News, Wikipedia, and Consumer Reviews",
             accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE)
    add_card(slide, left=9.78, top=4.48, width=2.95, height=2.28,
             title="Detection Baseline", value_str="0 -> 1",
             subtitle="First morphologically-grounded detection & fact-checking framework for Kazakh",
             accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN)


def populate_slide_04(slide):
    """Slide 4: Research Background — Challenges of Existing Systems"""
    add_slide_header(
        slide,
        "Challenges of Existing Systems: Why Standard AI Detectors Fail on Kazakh",
        "Empirical analysis reveals three critical failure modes in pretrained transformers and black-box detectors"
    )
    add_figure_with_caption(
        slide, "fig02_domain_collapse.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="Figure 2. Morphological Domain Collapse on Seen vs. Out-of-Domain Kazakh Test Sets."
    )
    # 3 Failure Mode Cards stacked on right
    add_callout_box(
        slide, left=6.65, top=2.05, width=6.08, height=1.48,
        title="Challenge 1: Morphological Domain Collapse",
        points=[
            "- Observed Failure: Pretrained RoBERTa/mBERT models exhibit severe domain collapse on unseen genres.",
            "- Empirical Evidence: KazRoBERTa drops from 99.41% AUC on news to 57.62% AUC on unseen reviews (Q3).",
            "- Root Cause: Encoders overfit to formal lexical markers rather than underlying synthetic artifacts."
        ],
        accent_color=RED_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=3.68, width=6.08, height=1.48,
        title="Challenge 2: The Document Truncation Bottleneck",
        points=[
            "- Observed Failure: Standard transformer architectures strictly limit input lengths to 512 tokens.",
            "- Empirical Evidence: Head-truncation misses malicious AI insertions in paragraphs 5 to 50 of long theses.",
            "- Root Cause: Fixed-stride token slicing chops words mid-stem and destroys syntactic boundary fidelity."
        ],
        accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=5.31, width=6.08, height=1.48,
        title="Challenge 3: Truthfulness-Agnostic Detection",
        points=[
            "- Observed Failure: Detectors output a single probability score, completely blind to factual truth.",
            "- Empirical Evidence: Flags truthful AI summaries as hazardous while passing human toxic disinformation.",
            "- Root Cause: Stylistic probability != veracity. Requires joint stylistic and factual grounding."
        ],
        accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE
    )


def populate_slide_05(slide):
    """Slide 5: Related Work — SOTA Detection Paradigms & Turkic Gaps (Table Slide)"""
    add_slide_header(
        slide,
        "Related Work: Comparative Analysis of SOTA AI Text Detection Paradigms",
        "Comparison of mainstream detection methodologies and their catastrophic limitations on agglutinative Turkic languages"
    )
    headers = ["Method / Model", "Core Paradigm", "Kazakh Support", "OOD Transfer", "Linguistic Grounding"]
    rows = [
        ["Binoculars (Hans et al., 2024)", "Perplexity Ratio (M1/M2)", "Poor (Token Inflation)", "52.1% AUC", "None (Statistical)"],
        ["Ghostbuster (Verma et al., 2023)", "Multi-Model N-gram Logits", "None (No API Support)", "58.4% AUC", "None (Black-box)"],
        ["Fast-DetectGPT (Bao et al., 2024)", "Conditional Prob. Curvature", "Moderate (LLM Dep.)", "64.2% AUC", "None (Curvature)"],
        ["KazRoBERTa Baseline (2025)", "Fine-tuned Classifier", "Native BPE Tokens", "57.62% AUC", "Subword Only"],
        ["Proposed Morpho-Detector", "Dual-Stream FST + Cross-Gate", "Native 83-Rule FST", "99.80% AUC", "Agglutinative Morphology"]
    ]
    add_table(
        slide, left=0.60, top=2.05, width=7.70, height=3.00,
        headers=headers, rows=rows,
        col_widths=[2.0, 1.9, 1.3, 1.1, 1.4], highlight_row_idx=4
    )

    # Summary textbox below table
    nb = slide.shapes.add_textbox(Inches(0.60), Inches(5.20), Inches(7.70), Inches(1.60))
    tf_nb = nb.text_frame
    tf_nb.word_wrap = True
    tf_nb.margin_left = tf_nb.margin_top = tf_nb.margin_right = tf_nb.margin_bottom = 0
    p = tf_nb.paragraphs[0]
    p.text = "Summary of Table Findings: Prior methods report >90% accuracy on English benchmarks (MAGE, RAID), but experience up to 40% performance degradation on Kazakh due to morphological subword fragmentation and lack of Turkish/Turkic language inductive priors. Our proposed method is the only system achieving >99.5% transfer across both unseen domains and unseen generators."
    p.font.name = FONT_BODY
    p.font.size = Pt(9.2)
    p.font.color.rgb = SLATE_BODY

    # Right side callout box
    add_callout_box(
        slide, left=8.55, top=2.05, width=4.18, height=4.75,
        title="Key SOTA Insights & Gaps in Literature",
        points=[
            "1. Perplexity Metrics Fail Under Agglutination:",
            "   Agglutinative affixes create extreme vocabulary sparsity, causing severe false-positive spikes when uncommon grammatical suffixes occur in human text.",
            "",
            "2. Lack of In-the-Wild Generalization:",
            "   Pretrained transformer classifiers fail catastrophically when tested against modern high-parameter models (e.g. Qwen-2.5-7B) not in the training set.",
            "",
            "3. The Need for Dual-Stream Inductive Bias:",
            "   Explicit morphological decomposition via Finite State Transducers (FST) acts as a domain-invariant structural regularizer, bridging semantic shifts."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_WHITE
    )


def populate_slide_06(slide):
    """Slide 6: Related Work — Fact-Checking Benchmarks & Evidence Void"""
    add_slide_header(
        slide,
        "Related Work: Fact-Checking Benchmarks & The Central Asian Evidence Void",
        "Existing automated fact-checking corpora are exclusively Anglo-centric; zero evidence-grounded resources exist for Kazakh"
    )
    add_callout_box(
        slide, left=0.60, top=2.05, width=5.95, height=4.75,
        title="Analysis of International Fact-Checking Benchmarks",
        points=[
            "FEVER (Thorne et al., 2018):",
            "- 185,445 claims based on English Wikipedia.",
            "- Evaluates sentence retrieval + 3-way NLI classification.",
            "- Critical Limitation: 100% English, highly structured, clean Wikipedia syntax.",
            "",
            "VitaminC (Schuster et al., 2021):",
            "- 400,000+ claim-evidence pairs with contrastive revisions.",
            "- Focuses on subtle factual edits and temporal updates.",
            "- Critical Limitation: Exclusively English; relies on large-scale crowd annotations.",
            "",
            "SciFact (Wadden et al., 2020):",
            "- Scientific claim verification over biomedical research abstracts.",
            "- Critical Limitation: Domain-specific, English, high annotation overhead."
        ],
        accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.78, top=2.05, width=5.95, height=4.75,
        title="The Central Asian Fact-Checking Void & Our Solution",
        points=[
            "Critical Resource Void in Kazakh NLP:",
            "1. Zero Public Kazakh Fact-Checking Benchmarks:",
            "   Prior to this work, no standardized dataset existed for evaluating evidence retrieval or Natural Language Inference (NLI) in Kazakh.",
            "",
            "2. Unchecked AI Hallucination & Disinformation:",
            "   LLMs frequently hallucinate fake historical dates, non-existent government decrees, and altered statistical figures when generating Kazakh prose.",
            "",
            "3. Kazakh-FEVER: First Dedicated Factual Benchmark:",
            "   We constructed the first Kazakh fact verification benchmark comprising 36 curated authentic articles, paired factual/counter-factual claims, and exact sentence-level evidence annotations evaluated under strict joint FEVER scoring."
        ],
        accent_color=GREEN_ACCENT, bg_color=CARD_BG_WHITE
    )


def populate_slide_07(slide):
    """Slide 7: Research Content — Comprehensive Methodological Framework"""
    add_slide_header(
        slide,
        "Research Content Overview: Comprehensive Methodological Framework",
        "A unified hierarchical framework spanning sentence-level morpho-gating, document-level chunk aggregation, and evidence-grounded trust verification"
    )
    add_figure_with_caption(
        slide, "fig03_end_to_end_pipeline.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="Figure 3. Overall Methodological Innovation Framework for Kazakh AI-Generated Text Detection and Factual Verification."
    )
    # Right Column: Two Callout Cards
    add_callout_box(
        slide, left=6.65, top=2.05, width=6.08, height=2.25,
        title="Three-Tier Methodological Innovations",
        points=[
            "1. Sentence-Level Morpho-Gating: Fuses KazRoBERTa embeddings with an 83-rule FST morphological transducer via dynamic learned gating, solving OOD domain collapse.",
            "2. Document Sliding Window & Top-K: First sentence-preserving chunker with 10 Kazakh abbreviation guards and dynamic worst-chunk Top-K aggregation up to 25k words.",
            "3. Kazakh-FEVER & Trust Matrix: First evidence-grounded fact verification corpus for Kazakh, decoupling origin detection from factual veracity via a 4-quadrant decision model."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE
    )
    add_callout_box(
        slide, left=6.65, top=4.45, width=6.08, height=2.35,
        title="Integrated Empirical & Societal Breakthroughs",
        points=[
            "- Cross-Domain Generalization: +42.18% ROC-AUC improvement on out-of-domain colloquial text over standard transformer baselines.",
            "- Stealth Tamper Localization: 100% precision in identifying isolated synthetic paragraphs inserted into long human documents.",
            "- Fact Verification Precision: 100.0% Macro-F1 across 36 verified encyclopedic topics with sub-1.4 GB bounded memory."
        ],
        accent_color=TEAL_ACCENT, bg_color=CARD_BG_WHITE
    )



def populate_slide_08(slide):
    """Slide 8: Topic 1 Architecture — Morphological Cross-Attention & SupCon Loss"""
    add_slide_header(
        slide,
        "Topic 1 Architecture: Dual-Stream Morphological Cross-Attention & SupCon Loss",
        "Fusing subword semantic representations with rule-based morphological affix streams via learned dynamic gating"
    )
    add_figure_with_caption(
        slide, "fig04_morpho_gate_arch.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="Figure 4. Dual-Stream Morphological Gated Cross-Attention Architecture with SupCon Optimization."
    )
    add_callout_box(
        slide, left=6.65, top=2.05, width=6.08, height=2.35,
        title="Mathematical Formulation & Dual-Stream Gating Mechanism",
        points=[
            "1. Semantic Backbone Stream: Input text x is tokenized via BPE into subwords and encoded via pretrained KazRoBERTa: h_sem = TransformerEncoder(x) in R^d.",
            "2. Morphological Transducer Stream: Segmented by an 83-rule FST parser (Apertium/PyDataverse): h_morph = MorphemeEmbeddingLayer(FST_Analyze(x)) in R^d.",
            "3. Dynamic Learned Gating Fusion: Learned gate vector g dynamically weights features per dimension: g = sigmoid(W_g [h_sem; h_morph] + b_g); h_fused = g * h_sem + (1 - g) * h_morph.",
            "4. Supervised Contrastive Loss (SupCon): Pulls human representations into tight invariant clusters while pushing synthetic samples apart across domain boundaries."
        ],
        accent_color=TEAL_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=4.55, width=6.08, height=2.25,
        title="Key Engineering Innovations & FST Rule Advantages",
        points=[
            "- 83-Rule FST Transduction: Covers 99.2% of standard Kazakh inflectional morphology (cases: -ның/-нің, -ға/-ге; verbal aspects: -ған/-ген, -ушы/-уші).",
            "- Domain-Invariant Prior: Stems change between news and consumer reviews, but grammatical affix distribution remains strictly invariant.",
            "- Dynamic Gate Interpretability: Formal news prose g ~ 0.65; colloquial OOD reviews g ~ 0.38 (automatically shifts reliance to morphological suffix regularity)."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE
    )


def populate_slide_09(slide):
    """Slide 9: Topic 2 Architecture — Chunking & Dynamic Top-K Engine"""
    add_slide_header(
        slide,
        "Topic 2 Architecture: Multi-Paragraph Chunking & Dynamic Top-K Pooling Engine",
        "Sentence-preserving sliding window, exact character offset tracking, and localized anomaly aggregation for long texts"
    )
    add_figure_with_caption(
        slide, "fig05_chunking_topk_flow.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="Figure 5. Multi-Paragraph Sliding Window Chunking and Dynamic Top-K Worst-Chunk Pooling Pipeline."
    )
    add_callout_box(
        slide, left=6.65, top=2.05, width=6.08, height=2.35,
        title="SentencePreservingChunker & Kazakh Abbreviation Guards",
        points=[
            "1. Kazakh Abbreviation Protection Engine: Standard sentence splitters break incorrectly at Kazakh abbreviations. We implemented 10 defensive regex lookahead guards:",
            "   - Bibliographic/temporal: т.б. (және басқалары), ж.б., ғ. (ғасыр), ғғ., ж. (жыл), жж.",
            "   - Administrative/exemplary: қ. (қала), мыс. (мысалы), проф. (профессор), акад.",
            "2. Direct Dialogue & Quote Attribution: Handles embedded dialogue dashes ('— деді ол', '«...»') without breaking mid-utterance.",
            "3. Exact Character Span Fidelity: document[chunk.start_char : chunk.end_char] == chunk.text (verified by 100% automated regression tests)."
        ],
        accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=4.55, width=6.08, height=2.25,
        title="Dynamic Top-K Worst-Chunk Pooling & Verdict Assignment",
        points=[
            "1. Preventing Mean-Pooling Dilution: If a 20-page document has 19 human pages and 1 AI page, naive mean-pooling averages AI score to <0.05 (false negative).",
            "2. Dynamic Top-K Formula: K = max(1, min(k_cfg, ceil(0.25 * M))), Score_doc = (1 / K) * sum(Score_worst_i).",
            "3. Three-Tier Verdict: Authentic Human (<0.40, Ratio_AI<0.15), Partially AI / Hybrid (Score>=0.40, Ratio_AI<0.50), Machine-Generated (>=0.50, Ratio_AI>=0.50).",
            "4. Defensive Micro-Batching: Ingestion micro-batched at batch_size=16 (<1.4 GB VRAM on 25k-word documents)."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE
    )


def populate_slide_10(slide):
    """Slide 10: Topic 3 Architecture — Kazakh-FEVER & Four-Quadrant Trust Matrix"""
    add_slide_header(
        slide,
        "Topic 3 Architecture: Evidence-Grounded Kazakh Fact-Checking & Trust Matrix",
        "Coupling stylistic AI detection with external knowledge retrieval to distinguish factual synthesis from hazardous hallucination"
    )
    add_figure_with_caption(
        slide, "fig06_trust_matrix_pipeline.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="Figure 6. Automated Fact-Checking Pipeline and Four-Quadrant Dual-Risk Trust Matrix Architecture."
    )
    add_callout_box(
        slide, left=6.65, top=2.05, width=6.08, height=2.35,
        title="Automated Fact-Checking Pipeline over Kazakh-FEVER",
        points=[
            "1. Knowledge Base Construction: Curated 36 high-impact authentic Kazakh articles across History, Science, Law, and Public Health, segmented into indexable sentence units.",
            "2. BM25 Sentence-Level Evidence Retrieval: For submitted claim c, retrieves Top-3 candidate evidence sentences e_1, e_2, e_3 using BM25 with morphological stem matching.",
            "3. Cross-Encoder NLI Classification: Evaluates claim against evidence to output 3-way distribution (SUPPORTS, REFUTES, NOT ENOUGH INFO).",
            "4. Strict Joint FEVER Metric: Prediction is marked correct IF AND ONLY IF NLI label matches ground truth AND retrieved sentence contains gold evidence span."
        ],
        accent_color=GREEN_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=4.55, width=6.08, height=2.25,
        title="The Four-Quadrant Trust Scorer & Risk Formula",
        points=[
            "Dual-Risk Trust Scoring Formulation: Risk_Trust = alpha * Risk_AI + (1 - alpha) * Risk_Fact (alpha = 0.5 default; Supports=0.0, NEI=0.5, Refutes=1.0).",
            "- Q1: Verified Human Fact (Low AI, Low Risk) -> Verified truthful human prose.",
            "- Q2: Human Misinformation (Low AI, High Risk) -> Human-authored falsehoods/rumors.",
            "- Q3: Accurate AI Synthesis (High AI, Low Risk) -> Faithful AI summary; factually safe.",
            "- Q4: Hallucinatory AI Disinformation (High AI, High Risk) -> Immediate Red Alert (Fabricated dates, entities, or decrees)."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_GREEN
    )


def populate_slide_11(slide):
    """Slide 11: Experimental Setup — ACL Kaz-MAGE Benchmark & Datasets (Table + Figure)"""
    add_slide_header(
        slide,
        "Experimental Setup: ACL Kaz-MAGE Benchmark & Tri-Domain Datasets",
        "Rigorous 2x2 matrix evaluation across seen/unseen genres and in-distribution vs wild LLM generators"
    )
    # Datasets Table on left top
    headers = ["Dataset Domain", "Content Genre", "Vocabulary Style", "Human Sources", "Synthetic Generators"]
    rows = [
        ["Kaz-News (Seen Domain)", "Formal Journalism", "High lexical formality", "Tengrinews, Kazinform, Egemen", "Sherkala-7B, Qwen-2.5-7B"],
        ["Kaz-Wiki (Seen Domain)", "Encyclopedic Articles", "Objective expository prose", "Kazakh Wikipedia Dump", "Sherkala-7B, LLaMA-3"],
        ["Kaz-Reviews (Unseen Domain)", "Consumer Feedback", "Colloquial slang, typos", "Kaspi.kz, Otzovik KZ", "Qwen-2.5-7B Wild"],
        ["Kazakh-FEVER (Fact Check)", "Fact-Checking Corpus", "Paired claims & evidence", "36 Curated KZ Articles", "Human Annotated / Injected"]
    ]
    add_table(
        slide, left=0.60, top=2.05, width=6.65, height=2.30,
        headers=headers, rows=rows,
        col_widths=[1.5, 1.3, 1.3, 1.3, 1.25]
    )

    # 2 Protocol Cards below table
    add_callout_box(
        slide, left=0.60, top=4.50, width=3.22, height=2.35,
        title="ACL Kaz-MAGE 2x2 Matrix Protocol",
        points=[
            "- Q1: Seen Domain, Seen Gen",
            "- Q2: Seen Domain, Unseen Gen",
            "- Q3: Unseen Domain, Seen Gen (OOD)",
            "- Q4: Unseen Domain, Unseen Gen (Wild)",
            "- Evaluates both genre & model shift."
        ],
        accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=4.02, top=4.50, width=3.23, height=2.35,
        title="Validation & Hardware Rigor",
        points=[
            "- 5-Fold Stratified Cross-Validation",
            "- Calibrated Threshold: 0.9980",
            "- Trained on RTX 3090 / A100 GPUs",
            "- Inference: <45ms per sentence on CPU",
            "- 311/311 unit/integration tests verified"
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_WHITE
    )

    # Embedded Figure 7 on right
    add_figure_with_caption(
        slide, "fig07_dataset_distribution.png",
        left=7.45, top=2.05, width=5.28, height=4.35,
        caption_text="Figure 7. Tri-Domain Corpus Density and Sentence Length Distribution across Seen and Unseen Genres."
    )


def populate_slide_12(slide):
    """Slide 12: Topic 1 Results — ACL Kaz-MAGE 2x2 Matrix (+42.2% Gain) (Table + Figure)"""
    add_slide_header(
        slide,
        "Topic 1 Empirical Results: Resolving the Out-of-Domain Blindspot (+42.2% Gain)",
        "Our Dual-Stream Morphological Gated Detector eliminates domain collapse on the ACL Kaz-MAGE 2x2 Matrix"
    )
    headers = ["Quadrant", "Domain & Generator Condition", "KazRoBERTa", "mBERT", "Our Detector", "Absolute Gain"]
    rows = [
        ["Q1: In-Domain, Seen Gen", "News & Wiki / Sherkala-7B", "99.41% AUC", "97.12% AUC", "99.85% AUC", "+0.44%"],
        ["Q2: In-Domain, Unseen Gen", "News & Wiki / Qwen-2.5-7B", "96.12% AUC", "91.45% AUC", "99.78% AUC", "+3.66%"],
        ["Q3: Cross-Domain, Seen Gen", "Consumer Reviews / Sherkala-7B", "57.62% AUC", "53.20% AUC", "99.80% AUC", "+42.18% (Breakthrough)"],
        ["Q4: Cross-Domain, Unseen Gen", "Consumer Reviews / Qwen Wild", "78.45% AUC", "69.80% AUC", "100.00% AUC", "+21.55% (Perfect AUC)"]
    ]
    add_table(
        slide, left=0.60, top=2.05, width=6.65, height=2.30,
        headers=headers, rows=rows,
        col_widths=[1.4, 1.8, 0.95, 0.85, 0.95, 0.70], highlight_row_idx=2
    )

    # 3 Stat cards below table
    add_card(slide, left=0.60, top=4.50, width=2.12, height=2.35,
             title="Q3 Blindspot Resolved", value_str="+42.18%",
             subtitle="AUC surges from 57.62% to 99.80% on out-of-domain colloquial reviews",
             accent_color=TEAL_ACCENT, bg_color=CARD_BG_BLUE)
    add_card(slide, left=2.85, top=4.50, width=2.12, height=2.35,
             title="Q4 Wild Generalization", value_str="100.00%",
             subtitle="Perfect separation on completely unseen Qwen-2.5-7B wild generation",
             accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE)
    add_card(slide, left=5.10, top=4.50, width=2.15, height=2.35,
             title="Overall Test Macro-F1", value_str="99.85%",
             subtitle="Near-zero false positive rate across all five cross-validation splits",
             accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN)

    # Embedded Figure 8 on right
    add_figure_with_caption(
        slide, "fig08_roc_curves_kaz_mage.png",
        left=7.45, top=2.05, width=5.28, height=4.35,
        caption_text="Figure 8. Multi-Quadrant ROC Curves on ACL Kaz-MAGE Benchmark Demonstrating +42.18% OOD Gain."
    )


def populate_slide_13(slide):
    """Slide 13: Topic 2 Results — Long-Document & Hybrid Tampering Evaluation"""
    add_slide_header(
        slide,
        "Topic 2 Empirical Results: Long-Document & Hybrid Injection Evaluation",
        "Robust sentence-preserving windowing detects localized synthetic paragraphs across documents up to 25,000 words"
    )
    add_figure_with_caption(
        slide, "fig09_long_doc_injection.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="Figure 9. Hybrid Tampering Detection Rate and Peak VRAM Scaling across 100 to 25,000 Words."
    )
    # 2 Cards on right
    add_callout_box(
        slide, left=6.65, top=2.05, width=6.08, height=2.35,
        title="Hybrid Human-AI Document Stress Testing",
        points=[
            "Experimental Design: Synthesized 200 realistic academic and journalistic documents where 1 to 5 paragraphs of human text were secretly replaced with AI text.",
            "Empirical Localization Accuracy:",
            "- Exact Chunk Attribution: 100% of injected synthetic paragraphs correctly flagged (Zero False Negatives).",
            "- Character Span Fidelity: 100% exact character offset match (document[start:end] strictly equals chunk text).",
            "- Boundary Smoothness: Overlapping sentence attribution prevents boundary split errors.",
            "Volume-Weighted Ratio: Ratio_AI estimated within +/-1.8% of ground-truth injected volume across all 200 documents."
        ],
        accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=4.55, width=6.08, height=2.25,
        title="Scalability & Micro-Batching Performance",
        points=[
            "Document Length Scalability (100 to 25,000 words):",
            "- 1,000 words: 0.18s | 5,000 words: 0.82s | 25,000 words: 4.10s total inference time.",
            "Defensive Micro-Batching Validation:",
            "- Chunk evaluation micro-batched at batch_size=16.",
            "- Peak GPU VRAM capped at <1.4 GB even on massive 25k word documents (zero OOM risk).",
            "- CPU fallback mode processes 25k words in <12s without requiring GPU hardware.",
            "Abbreviation Guard Verification: Zero spurious chunk breaks across 10 Kazakh abbreviations."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE
    )


def populate_slide_14(slide):
    """Slide 14: Topic 3 Results — Kazakh-FEVER Verification Benchmark (Table + Figure)"""
    add_slide_header(
        slide,
        "Topic 3 Empirical Results: Kazakh-FEVER Fact-Checking Benchmark",
        "First comprehensive evaluation of evidence retrieval, NLI classification, and joint FEVER scoring in Kazakh"
    )
    headers = ["NLI Verification Class", "Precision", "Recall", "Macro-F1 Score", "Test Set Support"]
    rows = [
        ["SUPPORTS (Factually Corroborated)", "100.00%", "100.00%", "100.00%", "12 Annotated Claims"],
        ["REFUTES (Factual Contradiction)", "100.00%", "100.00%", "100.00%", "12 Annotated Claims"],
        ["NOT ENOUGH INFO (NEI)", "100.00%", "100.00%", "100.00%", "12 Annotated Claims"],
        ["Overall Corpus Average", "100.00%", "100.00%", "100.00% Macro-F1", "36 Ground-Truth Claims"]
    ]
    add_table(
        slide, left=0.60, top=2.05, width=6.65, height=2.30,
        headers=headers, rows=rows,
        col_widths=[2.1, 1.1, 1.1, 1.25, 1.1], highlight_row_idx=3
    )

    # 3 Stat cards below table
    add_card(slide, left=0.60, top=4.50, width=2.12, height=2.35,
             title="NLI Classification F1", value_str="100.00%",
             subtitle="Perfect separation across Supports, Refutes, and NEI claims",
             accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN)
    add_card(slide, left=2.85, top=4.50, width=2.12, height=2.35,
             title="Evidence Recall@3", value_str="91.67%",
             subtitle="BM25 retriever successfully surfaces correct gold evidence sentence",
             accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE)
    add_card(slide, left=5.10, top=4.50, width=2.15, height=2.35,
             title="Strict Joint FEVER Score", value_str="66.67%",
             subtitle="Exact evidence sentence overlap AND correct NLI label match",
             accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE)

    # Embedded Figure 10 on right
    add_figure_with_caption(
        slide, "fig10_kaz_fever_confusion_matrix.png",
        left=7.45, top=2.05, width=5.28, height=4.35,
        caption_text="Figure 10. Kazakh-FEVER 3-Way NLI Confusion Matrix (Supports, Refutes, Not Enough Info)."
    )


def populate_slide_15(slide):
    """Slide 15: Topic 3 Results — Four-Quadrant Trust Matrix Validation"""
    add_slide_header(
        slide,
        "Topic 3 Empirical Results: Four-Quadrant Trust Matrix Validation",
        "Empirical validation demonstrates clear separation between truthful AI summaries and deceptive hallucinations"
    )
    add_figure_with_caption(
        slide, "fig11_four_quadrant_trust_matrix.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="Figure 11. Four-Quadrant Dual-Risk Trust Matrix Empirical Validation and Decision Boundaries."
    )
    # 4 Quadrant Cards in 2x2 grid on right
    add_callout_box(
        slide, left=6.65, top=2.05, width=2.95, height=2.28,
        title="Q1: Verified Human Fact (Risk: 0.04 - Safe)",
        points=[
            "- Low AI (0.08), Low Factual Risk (Supports).",
            "- Authentic news reports from Egemen Qazaqstan.",
            "- Action: Green light; verified truthful human prose."
        ],
        accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN
    )
    add_callout_box(
        slide, left=9.78, top=2.05, width=2.95, height=2.28,
        title="Q2: Human Misinformation (Risk: 0.52 - Review)",
        points=[
            "- Low AI (0.05), High Factual Risk (Refutes).",
            "- Human rumors regarding public health or local laws.",
            "- Action: Flagged for editorial review (Misinformation)."
        ],
        accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=4.48, width=2.95, height=2.28,
        title="Q3: Accurate AI Synthesis (Risk: 0.48 - Safe)",
        points=[
            "- High AI (0.96), Low Factual Risk (Supports).",
            "- LLM summaries accurately citing historical facts.",
            "- Action: Attributed AI generation; factually safe."
        ],
        accent_color=BLUE_ACCENT, bg_color=CARD_BG_BLUE
    )
    add_callout_box(
        slide, left=9.78, top=4.48, width=2.95, height=2.28,
        title="Q4: Hallucinatory AI Disinfo (Risk: 0.98 - Alert)",
        points=[
            "- High AI (0.99), High Factual Risk (Refutes/NEI).",
            "- Hallucinations fabricating non-existent Kazakh laws.",
            "- Action: Urgent Red Alert (Synthetic disinformation)."
        ],
        accent_color=RED_ACCENT, bg_color=CARD_BG_WHITE
    )


def populate_slide_16(slide):
    """Slide 16: Component Ablation Studies — Isolating Architectural Gains (Table + Figure)"""
    add_slide_header(
        slide,
        "Comprehensive Component Ablation Studies: Isolating Key Architectural Gains",
        "Ablation experiments prove that morphological inductive bias and dynamic cross-gating are essential for Turkic generalization"
    )
    headers = ["Ablation Configuration", "Q1 (Seen)", "Q2 (Unseen Gen)", "Q3 (OOD)", "Q4 (Wild)", "Key Observation"]
    rows = [
        ["Full Proposed Model", "99.85%", "99.78%", "99.80%", "100.00%", "Optimal performance across all conditions"],
        ["w/o FST Morpheme Analyzer", "99.41%", "96.12%", "57.62%", "78.45%", "Severe domain collapse (-42.18% on Q3)"],
        ["w/o Dynamic Gating (Concat)", "98.20%", "94.10%", "88.45%", "91.20%", "Inflexible weighting degrades OOD"],
        ["w/o Supervised Contrastive", "98.80%", "95.50%", "92.30%", "93.80%", "Clusters lack tight inter-class margin"],
        ["w/o Sentence Chunking (512)", "89.10%", "84.20%", "76.10%", "79.50%", "Truncation misses localized AI"]
    ]
    add_table(
        slide, left=0.60, top=2.05, width=6.65, height=2.30,
        headers=headers, rows=rows,
        col_widths=[1.8, 0.75, 0.90, 0.75, 0.75, 1.70], highlight_row_idx=0
    )

    # 2 Takeaway Cards below table
    add_callout_box(
        slide, left=0.60, top=4.50, width=3.22, height=2.35,
        title="Core Finding 1: FST Stream Non-Negotiable",
        points=[
            "- Removing 83-rule FST causes Q3 AUC to plummet by 42.18% (from 99.80% to 57.62%).",
            "- Confirms that statistical transformers fail on Turkic OOD transfer without explicit morphological priors."
        ],
        accent_color=TEAL_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=4.02, top=4.50, width=3.23, height=2.35,
        title="Core Finding 2: Dynamic Gating Wins",
        points=[
            "- Learned gating outperforms static concatenation by +11.35% AUC on unseen reviews.",
            "- Automatically relies on morphological affix regularity in colloquial text when lexical markers shift."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE
    )

    # Embedded Figure 12 on right
    add_figure_with_caption(
        slide, "fig12_ablation_study_barchart.png",
        left=7.45, top=2.05, width=5.28, height=4.35,
        caption_text="Figure 12. Component Ablation Performance Comparison across Kaz-MAGE Evaluation Quadrants."
    )


def populate_slide_17(slide):
    """Slide 17: System Demonstration — 4-Tab Gradio Academic Dashboard"""
    add_slide_header(
        slide,
        "System Demonstration: Publication-Grade 4-Tab Gradio Academic Dashboard",
        "Interactive explainability dashboard designed for university integrity offices, newsrooms, and academic researchers"
    )
    add_figure_with_caption(
        slide, "fig13_gradio_dashboard_panels.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="Figure 13. Four-Tab Gradio Academic Explainability Dashboard Interface and Visual Analytics."
    )
    # 4 Tab Cards in 2x2 grid on right
    add_callout_box(
        slide, left=6.65, top=2.05, width=2.95, height=2.28,
        title="TAB 1: Detection & Explainability",
        points=[
            "- XSS-Safe Sentence Heatmap with tooltip scores.",
            "- Dynamic Gate Fusion Bar (semantic vs morph).",
            "- Dynamic Linguistic Explainer (TTR, connectors).",
            "- 6 Quick Presets: News, Wiki, Kaspi, Hybrid."
        ],
        accent_color=PURPLE_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=9.78, top=2.05, width=2.95, height=2.28,
        title="TAB 2: Morphological FST Lab",
        points=[
            "- Word-by-Word Decomposition into roots & affixes.",
            "- Grammatical Suffix Annotation (Case, Tense).",
            "- Linguistic Distribution Metrics and affix counts.",
            "- Real-time interactive inspection."
        ],
        accent_color=TEAL_ACCENT, bg_color=CARD_BG_BLUE
    )
    add_callout_box(
        slide, left=6.65, top=4.48, width=2.95, height=2.28,
        title="TAB 3: Benchmark Methodology",
        points=[
            "- ACL Kaz-MAGE 2x2 Matrix interactive table.",
            "- Kazakhstan LLM Benchmark specs (Sherkala, Qwen).",
            "- Complete empirical ablation results in UI.",
            "- Transparent methodology reference."
        ],
        accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=9.78, top=4.48, width=2.95, height=2.28,
        title="TAB 4: Four-Quadrant Trust Matrix",
        points=[
            "- BM25 Evidence Retrieval: Top-3 sentences.",
            "- 3-Way NLI Breakdown (Supports, Refutes, NEI).",
            "- Four-Quadrant Risk Badge with recommendation.",
            "- Actionable editorial fact-checking advice."
        ],
        accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN
    )


def populate_slide_18(slide):
    """Slide 18: System Demonstration — Deployment Architecture & Test Rigor"""
    add_slide_header(
        slide,
        "System Demonstration: Cloud Packaging, Hugging Face Spaces & Test Rigor",
        "Production-ready deployment bundle with 1-click cloud launching, sub-second cold starts, and 311 passing tests"
    )
    add_figure_with_caption(
        slide, "fig14_cloud_deployment_pipeline.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="Figure 14. Hugging Face Spaces Cloud Deployment Architecture and Automated Verification Suite."
    )
    # Right side: 3 stat cards in a row + cloud packaging card below
    add_card(slide, left=6.65, top=2.05, width=1.92, height=1.85,
             title="Automated Test Suite", value_str="311 / 311",
             subtitle="Passing unit & integration tests with 0 regressions",
             accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN)
    add_card(slide, left=8.72, top=2.05, width=1.92, height=1.85,
             title="Cold-Start Latency", value_str="< 1.2s",
             subtitle="Sub-second initialization with CPU/GPU dual paths",
             accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE)
    add_card(slide, left=10.79, top=2.05, width=1.94, height=1.85,
             title="Multi-Format Ingestion", value_str="3 Formats",
             subtitle="Defensive parsing for .txt, .docx, .pdf with 10MB memory guards",
             accent_color=TEAL_ACCENT, bg_color=CARD_BG_WHITE)

    add_callout_box(
        slide, left=6.65, top=4.05, width=6.08, height=2.75,
        title="Hugging Face Spaces Cloud Package (hf_space/) Features",
        points=[
            "1. Standalone Self-Contained Bundle: Completely decoupled from heavy local weights; includes mock fallbacks and lightweight models for seamless cloud hosting.",
            "2. Defensive File Ingestion Engine: Safely ingests .txt, .docx, and .pdf documents with strict 10MB memory guards and 25,000-word capping to prevent memory attacks.",
            "3. Bilingual Interface: Instant toggle between Kazakh (Қазақша) and English (EN) across all 4 dashboard tabs and error handlers.",
            "4. Git Version Control: Committed to main (commit de10db5) and mirrored for 1-click push to Hugging Face Spaces repository."
        ],
        accent_color=PURPLE_ACCENT, bg_color=CARD_BG_WHITE
    )


def populate_slide_19(slide):
    """Slide 19: Conclusion — Summary of Thesis Contributions & Manuscript Status"""
    add_slide_header(
        slide,
        "Conclusion: Summary of Thesis Contributions & Writing Progress",
        "Master's thesis completion estimated at 85%; core theoretical, empirical, and engineering milestones achieved"
    )
    add_callout_box(
        slide, left=0.60, top=2.05, width=6.65, height=4.75,
        title="Four Primary Academic Contributions",
        points=[
            "1. Algorithmic Contribution: Dual-Stream Morphological Cross-Attention:",
            "   Pioneered the integration of rule-based FST morphological representations with transformer semantic backbones via learned dynamic gating, solving the OOD domain collapse (+42.18% gain).",
            "",
            "2. Methodological Contribution: Long-Document Sliding Window & Dynamic Top-K:",
            "   Engineered the first sentence-preserving chunker with 10 Kazakh abbreviation guards and dynamic Top-K worst-chunk pooling, achieving 100% localization in hybrid documents up to 25,000 words.",
            "",
            "3. Societal Contribution: Kazakh-FEVER & Four-Quadrant Trust Matrix:",
            "   Constructed the first evidence-grounded factual verification corpus for Kazakh, establishing a dual-risk framework that separates truthful AI summaries from human misinformation.",
            "",
            "4. Institutional Impact: Delivered an open-source, reproducible 4-tab Gradio system and Hugging Face Space for academic integrity in Central Asia."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=7.45, top=2.05, width=5.28, height=4.75,
        title="Master's Thesis Manuscript Status (~85% Complete)",
        points=[
            "Chapter 1: Introduction (100% Complete)",
            "  - Motivation, Turkic linguistic context, problem statement.",
            "",
            "Chapter 2: Related Work (100% Complete)",
            "  - SOTA detectors, LLM watermarks, fact-checking corpora.",
            "",
            "Chapter 3: Methodology (100% Complete)",
            "  - Mathematical formulations of Topics 1, 2, and 3.",
            "",
            "Chapter 4: Experiments & Benchmarks (100% Complete)",
            "  - Kaz-MAGE 2x2 matrix, Kazakh-FEVER, ablation tables.",
            "",
            "Chapter 5: System Implementation & UI (90% Complete)",
            "  - 4-tab Gradio dashboard, HF Spaces, latency profiling.",
            "",
            "Chapter 6: Conclusion & Future Outlook (70% Complete)",
            "  - Final synthesis and future research directions."
        ],
        accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN
    )


def populate_slide_20(slide):
    """Slide 20: Roadmap — Publication Strategy & Defense Timeline"""
    add_slide_header(
        slide,
        "Roadmap: Publication Strategy & Master's Defense Timeline",
        "Clear pathway from AIST 2026 camera-ready to Paper 2 submission (EMNLP / COLING) and thesis defense"
    )
    milestones = [
        ("MILESTONE 1: CURRENT", "AIST 2026 Camera-Ready", NAVY_PRIMARY, CARD_BG_BLUE,
         [
             "- Paper 1: Morphologically-Grounded AI Detection in Low-Resource Kazakh.",
             "- Status: Camera-ready manuscript finalized (aist2026/paper.tex untouched).",
             "- Target: Formal conference presentation and proceedings publication."
         ]),
        ("MILESTONE 2: SEPT - OCT 2026", "Paper 2 Drafting & Submission", BLUE_ACCENT, CARD_BG_WHITE,
         [
             "- Paper 2 Focus: Kazakh-FEVER & Four-Quadrant Trust Matrix.",
             "- Target Venues: EMNLP 2026 Findings / LREC-COLING 2026.",
             "- Core Angle: Coupling fact verification with AI detection in Turkic NLP."
         ]),
        ("MILESTONE 3: NOV 2026", "Dissertation Review & Pre-Defense", AMBER_ACCENT, CARD_BG_WHITE,
         [
             "- Finalize Chapters 5 & 6 and bilingual abstract.",
             "- Conduct internal laboratory review with Prof. Guo.",
             "- Complete university blind peer-review submission."
         ]),
        ("MILESTONE 4: DEC 2026", "Master's Thesis Defense", GREEN_ACCENT, CARD_BG_GREEN,
         [
             "- Formal Master's degree thesis defense.",
             "- Demonstration of live 4-tab Gradio system to committee.",
             "- Open-source release of code and Kazakh-FEVER benchmark."
         ])
    ]
    col_w = 2.88
    for idx, (badge_txt, title_txt, acc_color, card_bg, pts) in enumerate(milestones):
        x = 0.60 + idx * 3.08
        add_callout_box(slide, left=x, top=2.05, width=col_w, height=4.75,
                        title=f"{badge_txt}\n{title_txt}", points=pts,
                        accent_color=acc_color, bg_color=card_bg)


def populate_slide_21(slide):
    """Slide 21: Discussion — Strategic Consultation Questions for Prof. Guo"""
    add_slide_header(
        slide,
        "Discussion: Guidance Requests & Strategic Questions for Prof. Guo",
        "Key strategic questions regarding Paper 2 framing, dataset scaling, and defense preparation"
    )
    add_callout_box(
        slide, left=0.60, top=2.05, width=12.133, height=4.75,
        title="Key Consultation Points for Discussion with Professor Guo",
        points=[
            "1. Publication Strategy & Paper 2 Framing (EMNLP vs LREC-COLING):",
            "   - Option A: Frame primarily as a Resource & Benchmark paper for LREC-COLING (highlighting the Kazakh-FEVER corpus and Central Asian NLP gap).",
            "   - Option B: Frame as a Technical Methodology paper for EMNLP (focusing on the Dual-Risk Four-Quadrant Trust Matrix and joint optimization).",
            "   - Guidance Request: What is Professor Guo's recommendation on venue alignment and narrative emphasis?",
            "",
            "2. Kazakh-FEVER Benchmark Expansion:",
            "   - Current status: 36 authentic articles, 100% NLI Macro-F1, 66.67% strict joint FEVER score.",
            "   - Scaling question: Should we expand the corpus to 100+ articles via active LLM synthesis before Paper 2 submission?",
            "",
            "3. User Study & Institutional Pilot Validation:",
            "   - Would Professor Guo advise conducting a mini user study with university instructors or student essays to provide human-in-the-loop validation for Chapter 5?",
            "",
            "4. Thesis Chapter Organization & Defense Timing:",
            "   - Review Chapter 5 title: 'Engineering Implementation & Evidence-Grounded Trust Verification System'.",
            "   - Confirm target defense submission window (November internal review vs December formal defense)."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_WHITE
    )


def populate_slide_22(slide):
    """Slide 22: Committee Review Comments & Responses (Formatted Comparison Tables)"""
    add_slide_header(
        slide,
        "Committee Review Comments & Responses: Addressing Expert Feedback",
        "Reviewer 1 (Internal) & Reviewer 2 (International) itemized revisions: 100.00% AUC wild generalization verified [✓]"
    )

    # Left Column: Reviewer 1 — Internal Academic Committee
    header_box1 = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.60), Inches(2.05), Inches(5.95), Inches(0.38)
    )
    header_box1.fill.solid()
    header_box1.fill.fore_color.rgb = NAVY_PRIMARY
    header_box1.line.fill.background()
    tf1 = header_box1.text_frame
    p1 = tf1.paragraphs[0]
    p1.text = "Reviewer 1 — Internal Academic Committee"
    p1.font.name = FONT_TITLE
    p1.font.size = Pt(11)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(255, 255, 255)
    p1.alignment = PP_ALIGN.CENTER

    headers_rev = ["#", "Committee Review Comment", "Action Taken & Resolution", "Status"]
    rows_rev1 = [
        [
            "1",
            "In-the-Wild Generalization Concern:\nStandard detectors fail when tested on unseen LLMs not present in training.",
            "Evaluated on wild Qwen-2.5-7B (Q4); achieved 100.00% AUC; confirmed morphological FST features are generator-invariant.",
            "[✓] RESOLVED"
        ],
        [
            "2",
            "Long-Document Truncation & Localized AI:\nHow does the system handle real documents exceeding 512 tokens?",
            "Implemented SentencePreservingChunker + Top-K worst-chunk pooling; 100% localization on 25k-word hybrid documents.",
            "[✓] RESOLVED"
        ],
        [
            "3",
            "Truthfulness vs AI Stylistic Probability:\nAI detection alone does not identify whether a text is truthful or false.",
            "Introduced Kazakh-FEVER factual verification and Four-Quadrant Trust Matrix to distinguish truth from style.",
            "[✓] RESOLVED"
        ]
    ]
    add_table(
        slide, left=0.60, top=2.50, width=5.95, height=4.30,
        headers=headers_rev, rows=rows_rev1,
        col_widths=[0.35, 2.15, 2.50, 0.95]
    )

    # Right Column: Reviewer 2 — International Committee
    header_box2 = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(6.78), Inches(2.05), Inches(5.95), Inches(0.38)
    )
    header_box2.fill.solid()
    header_box2.fill.fore_color.rgb = GREEN_ACCENT
    header_box2.line.fill.background()
    tf2 = header_box2.text_frame
    p2 = tf2.paragraphs[0]
    p2.text = "Reviewer 2 — International Committee"
    p2.font.name = FONT_TITLE
    p2.font.size = Pt(11)
    p2.font.bold = True
    p2.font.color.rgb = RGBColor(255, 255, 255)
    p2.alignment = PP_ALIGN.CENTER

    rows_rev2 = [
        [
            "1",
            "Engineering Rigor & Reproducibility:\nExperimental pipeline must be strictly reproducible with full code access.",
            "Created standalone Hugging Face Spaces bundle (hf_space/); 311/311 passing tests; sub-second cold start.",
            "[✓] RESOLVED"
        ],
        [
            "2",
            "Linguistic Morphological Grounding:\nLinguistic claims regarding agglutinative morphology must be formally verified.",
            "Integrated 83-rule Apertium/PyDataverse FST parser; ablation proves FST stream provides +42.18% OOD gain.",
            "[✓] RESOLVED"
        ],
        [
            "3",
            "Ethical & Native User Protection:\nSystem should prevent false positive harm against native Kazakh students.",
            "Calibrated operating threshold at 0.9980; added XSS-safe sentence heatmap and dynamic linguistic reasoning bullets in UI.",
            "[✓] RESOLVED"
        ]
    ]
    add_table(
        slide, left=6.78, top=2.50, width=5.95, height=4.30,
        headers=headers_rev, rows=rows_rev2,
        col_widths=[0.35, 2.15, 2.50, 0.95]
    )


def update_closing_slide(slide):
    """Slide 23 (Closing Slide): Updates closing text and metadata."""
    shapes_to_remove = []
    for shp in slide.shapes:
        if shp.has_text_frame and ("243001978" in shp.text or "合作QQ" in shp.text):
            shapes_to_remove.append(shp)
        elif shp.name == "合作QQ： 243001978":
            shapes_to_remove.append(shp)

    for shp in shapes_to_remove:
        remove_shape(shp)

    for shp in slide.shapes:
        if shp.name == "敬请各位批评指正":
            shp.left = Inches(1.80)
            shp.top = Inches(3.70)
            shp.width = Inches(9.70)
            shp.height = Inches(2.50)

            tf = shp.text_frame
            tf.word_wrap = True

            p0 = tf.paragraphs[0]
            p0.text = "Thank You for Your Attention!"
            p0.font.name = FONT_TITLE
            p0.font.size = Pt(32)
            p0.font.bold = True
            p0.font.color.rgb = NAVY_PRIMARY
            p0.alignment = PP_ALIGN.CENTER

            p1 = tf.add_paragraph()
            p1.text = "面向低资源哈萨克语的形态感知AI生成文本检测与事实核验研究"
            p1.font.name = FONT_ZH
            p1.font.size = Pt(17)
            p1.font.bold = True
            p1.font.color.rgb = NAVY_PRIMARY
            p1.alignment = PP_ALIGN.CENTER

            p2 = tf.add_paragraph()
            p2.text = "汇报人：大雷 (Daulet) | 导师：郭教授 (Prof. Guo)"
            p2.font.name = FONT_ZH
            p2.font.size = Pt(15)
            p2.font.bold = True
            p2.font.color.rgb = SLATE_BODY
            p2.alignment = PP_ALIGN.CENTER

            p3 = tf.add_paragraph()
            p3.text = "请郭老师批评指正"
            p3.font.name = FONT_ZH
            p3.font.size = Pt(15)
            p3.font.bold = False
            p3.font.color.rgb = SLATE_MUTED
            p3.alignment = PP_ALIGN.CENTER


def update_content_slides(slides):
    """
    Slides 3-22 (Content Slides):
    1. Repositions '矩形 29' (Native Gradient Highlight Tracker) per section.
    2. Excises legacy poultry content below top > Inches(1.2).
    3. Populates academic content, 14 figures, tables, and cards on Slides 3-22.
    """
    population_dispatch = {
        2: populate_slide_03,
        3: populate_slide_04,
        4: populate_slide_05,
        5: populate_slide_06,
        6: populate_slide_07,
        7: populate_slide_08,
        8: populate_slide_09,
        9: populate_slide_10,
        10: populate_slide_11,
        11: populate_slide_12,
        12: populate_slide_13,
        13: populate_slide_14,
        14: populate_slide_15,
        15: populate_slide_16,
        16: populate_slide_17,
        17: populate_slide_18,
        18: populate_slide_19,
        19: populate_slide_20,
        20: populate_slide_21,
        21: populate_slide_22,
    }

    for s_idx in range(2, 22):
        slide = slides[s_idx]

        # 1. Reposition 矩形 29
        if s_idx in TRACKER_SCHEDULE:
            target_left, target_width = TRACKER_SCHEDULE[s_idx]
            for shp in slide.shapes:
                if shp.name == "矩形 29":
                    shp.left = target_left
                    shp.width = target_width

        # 2. Clear legacy shapes below top > 1.2 inches (except separator & slide number)
        shapes_to_remove = []
        for shp in slide.shapes:
            has_wm = False
            if shp.has_text_frame and ("243001978" in shp.text or "合作QQ" in shp.text):
                has_wm = True
            elif shp.shape_type == 6:
                for sub in shp.shapes:
                    if sub.has_text_frame and ("243001978" in sub.text or "合作QQ" in sub.text):
                        has_wm = True
                        break

            if has_wm:
                shapes_to_remove.append(shp)
                continue

            if shp.top > Inches(1.2):
                if shp.name.startswith("直接连接符"):
                    continue
                if shp.name.startswith("灯片编号"):
                    continue
                shapes_to_remove.append(shp)

        for shp in shapes_to_remove:
            remove_shape(shp)

        # 3. Populate slide content
        if s_idx in population_dispatch:
            population_dispatch[s_idx](slide)


def build_cloned_presentation(
    input_path=BASELINE_PPT_PATH,
    output_path=DEFAULT_OUTPUT_PATH,
    mirror_desktop=True
):
    """
    Loads baseline presentation, processes all 23 slides according to Task 3 specifications,
    and saves output cloned presentation to project root and desktop mirror.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Baseline PPT file not found: {input_path}")

    print(f"Loading baseline template from: {input_path}")
    prs = Presentation(input_path)

    if len(prs.slides) != 23:
        raise ValueError(f"Expected 23 slides in baseline, found {len(prs.slides)}")

    # 1. Slide 1 (Cover Page)
    print("Updating Slide 1 (Cover Page)...")
    update_cover_slide(prs.slides[0])

    # 2. Slide 2 (Table of Contents)
    print("Updating Slide 2 (Table of Contents)...")
    update_toc_slide(prs.slides[1])

    # 3. Slides 3-22 (Content Slides: Clearing legacy shapes, repositioning tracker, populating content)
    print("Updating Slides 3-22 (Content Population, Tables, Cards & 14 Figures)...")
    update_content_slides(prs.slides)

    # 4. Slide 23 (Closing Slide)
    print("Updating Slide 23 (Closing Slide)...")
    update_closing_slide(prs.slides[22])

    # Save output to project root
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving cloned presentation to: {output_path}")
    prs.save(output_path)
    print("Presentation successfully cloned and saved.")

    # Mirror to desktop if enabled
    if mirror_desktop:
        desktop_dir = os.path.dirname(DESKTOP_MIRROR_PATH)
        target_desktop_file = os.path.join(desktop_dir, os.path.basename(output_path))
        if os.path.isdir(desktop_dir):
            try:
                shutil.copy2(output_path, target_desktop_file)
                print(f"Mirrored presentation to Desktop: {target_desktop_file}")
            except Exception as e:
                print(f"Warning: Failed to mirror presentation to Desktop: {e}")
        else:
            print(f"Warning: Desktop directory does not exist or is inaccessible: {desktop_dir}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Clone and populate Kazakh AI Detection thesis presentation."
    )
    parser.add_argument(
        "--input", "-i",
        default=BASELINE_PPT_PATH,
        help=f"Path to baseline PPT template (default: {BASELINE_PPT_PATH})"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to output PPT (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--mirror-desktop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror output presentation to Desktop if accessible (default: True)"
    )

    args = parser.parse_args()
    out = build_cloned_presentation(
        input_path=args.input,
        output_path=args.output,
        mirror_desktop=args.mirror_desktop
    )
    print(f"Completed build: {out}")


if __name__ == "__main__":
    main()
