# -*- coding: utf-8 -*-
"""
Master's Thesis Proposal Defense Presentation Builder (10-Minute Compact English Deck).
Clones baseline presentation 'Aliya PPT.pptx' directly, preserves native institutional
branding (KazNU logo, NPU logo group, watermark, and Layout 7 NPU pentagon badge),
excises commercial watermarks ('合作QQ： 243001978'), installs a 5-element running
section tracker TITLED ONLY IN ENGLISH with enlarged prominent typography,
and populates a high-velocity 12-slide presentation for candidate Daulet Anekesh's
10-minute October 29 Proposal Defense.

Presenter: DAULET ANEKESH (大雷, Student No.: 2025291043)
Supervisor: Prof. Bin Guo (郭斌 教授)
Date of Defense: October 29, 2026 (2026年10月29日)
Degree: Master of Science in Computer Science and Technology
School: School of Computer Science, NPU & Al-Farabi KazNU
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
RED_ACCENT = RGBColor(192, 0, 0)           # #C00000 - Milestone Red / Failure Mode

SLATE_TITLE = RGBColor(15, 23, 42)         # #0F172A - Slide title & primary headers
SLATE_BODY = RGBColor(51, 65, 85)          # #334155 - Standard body prose
SLATE_MUTED = RGBColor(100, 116, 139)      # #64748B - Subtitles & captions
BORDER_LIGHT = RGBColor(226, 232, 240)     # #E2E8F0 - Divider lines & card borders

CARD_BG_WHITE = RGBColor(255, 255, 255)    # #FFFFFF - Crisp white cards
CARD_BG_ALT = RGBColor(248, 250, 252)      # #F8FAFC - Soft light gray card fill
CARD_BG_BLUE = RGBColor(239, 246, 255)     # #EFF6FF - Soft blue tinted card fill
CARD_BG_GREEN = RGBColor(240, 253, 244)    # #F0FDF4 - Soft green tinted card fill

FONT_TITLE = "Times New Roman"
FONT_BODY = "Times New Roman"
FONT_ZH = "Microsoft YaHei"

USER_HOME = os.path.expanduser("~")
BASELINE_PPT_PATH = os.path.join(USER_HOME, "Downloads", "Aliya PPT.pptx")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "Kazakh_AI_Detection_Thesis_Proposal_Defense_Oct29.pptx")
DESKTOP_MIRROR_PATH = os.path.join(USER_HOME, "Desktop", "Kazakh_AI_Detection_Thesis_Proposal_Defense_Oct29.pptx")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "presentation_figures")

# 5-Element Sections TITLED ONLY IN ENGLISH (Prominent Typography)
SECTION_NAMES_EN = [
    "Research Background",
    "Related Work",
    "Research Content",
    "Research Innovations",
    "Schedule"
]

# 5 Tracker slots spanning left = 1.15 to 11.90 inches
SLOT_COORDS = [
    (Inches(1.15), Inches(2.35)),  # Section 1: Research Background
    (Inches(3.55), Inches(1.95)),  # Section 2: Related Work
    (Inches(5.55), Inches(2.20)),  # Section 3: Research Content
    (Inches(7.80), Inches(2.40)),  # Section 4: Research Innovations
    (Inches(10.25), Inches(1.65)), # Section 5: Schedule
]

# Section index mapping for content slides (indices 2 to 10 in 12-slide deck):
# Slides 3-4 (indices 2, 3) -> Sec 0 (Research Background)
# Slide 5 (index 4)         -> Sec 1 (Related Work)
# Slides 6-9 (indices 5, 6, 7, 8) -> Sec 2 (Research Content)
# Slide 10 (index 9)        -> Sec 3 (Research Innovations)
# Slide 11 (index 10)       -> Sec 4 (Schedule)
SLIDE_SECTION_MAP = {
    2: 0, 3: 0,
    4: 1,
    5: 2, 6: 2, 7: 2, 8: 2,
    9: 3,
    10: 4
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
    p.font.size = Pt(17.0)
    p.font.bold = True
    p.font.color.rgb = NAVY_PRIMARY

    if subtitle_text:
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.name = FONT_TITLE
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
    p_tit.font.size = Pt(11.0)
    p_tit.font.bold = True
    p_tit.font.color.rgb = SLATE_TITLE
    p_tit.space_before = Pt(2)

    if subtitle:
        p_sub = tf.add_paragraph()
        p_sub.text = subtitle
        p_sub.font.name = FONT_TITLE
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
    p_head.font.size = Pt(10.0 if is_multiline else 11.5)
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
        p.font.name = FONT_TITLE
        p.font.size = Pt(9.0)
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
            p.font.name = FONT_TITLE
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
            shp.top = Inches(3.95)
            shp.width = Inches(12.60)
            shp.height = Inches(1.65)

            tf = shp.text_frame
            tf.word_wrap = True

            p0 = tf.paragraphs[0]
            p0.text = "Research on Morphologically-Grounded AI-Generated Text Detection and Factual Verification for Low-Resource Kazakh"
            p0.font.name = FONT_TITLE
            p0.font.size = Pt(21)
            p0.font.bold = True
            p0.font.color.rgb = NAVY_PRIMARY
            p0.alignment = PP_ALIGN.LEFT

            p1 = tf.add_paragraph()
            p1.text = "基于形态学感知的低资源黏着语AI生成文本检测与事实核查研究"
            p1.font.name = FONT_ZH
            p1.font.size = Pt(16)
            p1.font.bold = True
            p1.font.color.rgb = NAVY_PRIMARY
            p1.alignment = PP_ALIGN.LEFT

            p_sub = tf.add_paragraph()
            p_sub.text = "Master's Thesis Proposal Defense (硕士学位论文选题报告答辩)"
            p_sub.font.name = FONT_TITLE
            p_sub.font.size = Pt(13.5)
            p_sub.font.bold = True
            p_sub.font.color.rgb = BLUE_ACCENT
            p_sub.space_before = Pt(3)
            p_sub.alignment = PP_ALIGN.LEFT

        elif shp.name == "汇报人":
            shp.left = Inches(0.45)
            shp.top = Inches(5.85)
            shp.width = Inches(12.00)
            shp.height = Inches(1.10)

            tf = shp.text_frame
            tf.word_wrap = True

            p0 = tf.paragraphs[0]
            p0.text = "Candidate: DAULET ANEKESH (大雷, Student No.: 2025291043)    Supervisor: Prof. Bin Guo (郭斌 教授)    Date: October 29, 2026"
            p0.font.name = FONT_TITLE
            p0.font.size = Pt(14)
            p0.font.bold = True
            p0.font.color.rgb = NAVY_PRIMARY
            p0.alignment = PP_ALIGN.LEFT

            p1 = tf.add_paragraph()
            p1.text = "Discipline: Computer Science and Technology | School of Computer Science, NPU (西北工业大学) & KazNU"
            p1.font.name = FONT_TITLE
            p1.font.size = Pt(11.5)
            p1.font.bold = False
            p1.font.color.rgb = SLATE_MUTED
            p1.alignment = PP_ALIGN.LEFT


def update_toc_slide(slide):
    """Slide 2 (TOC): Updates 5 canonical sections matching Professor Guo's lab standard."""
    shapes_to_remove = []
    for shp in slide.shapes:
        if shp.has_text_frame and ("243001978" in shp.text or "合作QQ" in shp.text):
            shapes_to_remove.append(shp)
        elif shp.name == "合作QQ： 243001978":
            shapes_to_remove.append(shp)

    for shp in shapes_to_remove:
        remove_shape(shp)

    toc_sections = {
        "组合 1": "01. Research Background & Motivation (研究背景与意义)",
        "组合 2": "02. Related Work & SOTA Limitations (国内外研究现状)",
        "组合 3": "03. Research Content & System Architecture (研究内容与技术路线)",
        "组合 12": "04. Research Innovations & Key Challenges (研究创新与技术难点)",
        "standalone": "05. Dissertation Schedule & Timeline (进度安排与工作计划)",
    }

    for shp in slide.shapes:
        if shp.name in toc_sections and shp.shape_type == 6:  # Group shape
            for sub in shp.shapes:
                if sub.name == "学科特色鲜明" and sub.has_text_frame:
                    p = sub.text_frame.paragraphs[0]
                    p.text = toc_sections[shp.name]
                    p.font.name = FONT_TITLE
                    p.font.size = Pt(16)
                    p.font.bold = True
                    p.font.color.rgb = NAVY_PRIMARY
        elif shp.name == "学科特色鲜明" and shp.shape_type != 6:  # Standalone
            shp.width = Inches(7.5)
            p = shp.text_frame.paragraphs[0]
            p.text = toc_sections["standalone"]
            p.font.name = FONT_TITLE
            p.font.size = Pt(16)
            p.font.bold = True
            p.font.color.rgb = NAVY_PRIMARY
        elif shp.name == "组合 4":
            # Remove 6th group shape from TOC
            remove_shape(shp)


# =============================================================================
# SLIDE-BY-SLIDE CONTENT POPULATION FUNCTIONS (12 SLIDES TOTAL)
# =============================================================================

def populate_slide_03(slide):
    """Slide 3: 01 Research Background — Kazakh Linguistic Reality & Generative AI Threats"""
    add_slide_header(
        slide,
        "Research Background: Kazakh Linguistic Reality & Generative AI Threats",
        "Agglutinative morphology breaks standard tokenizers; rapid proliferation of LLMs creates critical risks in Central Asia"
    )
    add_figure_with_caption(
        slide, "fig01_subword_vs_fst.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="Figure 1. Subword Tokenization Fragmentation vs. 83-Rule FST Morphological Parsing in Kazakh."
    )
    add_card(slide, left=6.65, top=2.05, width=2.95, height=2.28,
             title="Affix Stacking Depth", value_str="15+",
             subtitle="Suffixes cascade after roots, causing statistical BPE to fragment into meaningless bytes",
             accent_color=TEAL_ACCENT, bg_color=CARD_BG_BLUE)
    add_card(slide, left=9.78, top=2.05, width=2.95, height=2.28,
             title="Long-Document Scale", value_str="25,000",
             subtitle="Words in real academic theses & news reports, requiring non-destructive auditing",
             accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE)
    add_card(slide, left=6.65, top=4.48, width=2.95, height=2.28,
             title="Tri-Domain Corpus", value_str="10,000+",
             subtitle="Balanced benchmark spanning News, Wikipedia, and Kaspi Consumer Reviews",
             accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE)
    add_card(slide, left=9.78, top=4.48, width=2.95, height=2.28,
             title="Native Baseline", value_str="0 -> 1",
             subtitle="First morphologically-grounded detection & factual verification framework for Kazakh",
             accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN)


def populate_slide_04(slide):
    """Slide 4: 01 Research Background — Three Fatal Failure Modes in Conventional Detectors"""
    add_slide_header(
        slide,
        "Research Background: Three Fatal Pathologies in Conventional Detectors",
        "Empirical evaluations reveal severe domain collapse, truncation blindness, and truthfulness-agnostic flaws"
    )
    add_figure_with_caption(
        slide, "fig02_domain_collapse.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="Figure 2. Baseline Kaz-RoBERTa Morphological Domain Collapse on Kaspi Reviews (-41.79% Drop)."
    )
    add_callout_box(
        slide, left=6.65, top=2.05, width=6.08, height=1.50,
        title="Pathology 1: Morphological Domain Collapse (-41.79% AUC Drop)",
        points=[
            "Standard Kaz-RoBERTa achieves 99.41% AUC on in-domain news, but collapses to 57.62% on Kaspi reviews.",
            "Classifiers overfit to formal editorial subwords rather than capturing generative synthesis invariants."
        ],
        accent_color=RED_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=3.65, width=6.08, height=1.35,
        title="Pathology 2: Document Truncation Bottleneck (512-Token Limit)",
        points=[
            "Academic manuscripts contain thousands of words; standard transformers truncate at 512 tokens.",
            "Detectors remain completely blind to localized synthetic paragraphs inserted in body sections."
        ],
        accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=5.10, width=6.08, height=1.35,
        title="Pathology 3: Truthfulness-Agnostic Paradigm",
        points=[
            "Classifiers merely output stylistic generation probabilities, conflating fluent summaries with false rumors.",
            "Decoupling AI generation probability from factual truthfulness is essential for trustworthy auditing."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE
    )


def populate_slide_05(slide):
    """Slide 5: 02 Related Work — Comparative Analysis of SOTA Paradigms & Fact-Checking Gaps"""
    add_slide_header(
        slide,
        "Related Work: Comparative Analysis of SOTA Paradigms & Fact-Checking Gaps",
        "Mainstream statistical curvature, watermarking, and classification baselines fail under agglutinative low-resource conditions"
    )
    headers = ["Detection Paradigm", "Representative Methods", "Morphological Prior", "Long-Doc Auditing", "Zero API Dependency", "Kaspi OOD AUC"]
    rows = [
        ["Statistical Curvature", "DetectGPT / Binoculars", "None (Surrogate models)", "Truncated (512 tokens)", "High API Cost", "54.20% (Random Guess)"],
        ["Generative Watermarking", "KGW / SynthID", "None (Soft token bias)", "Global sequence", "Impractical (Closed LLM)", "N/A (Key Dependent)"],
        ["Pretrained Transformers", "mBERT / XLM-R", "None (Byte-pair BPE)", "Truncated (512 tokens)", "Self-Contained", "61.35% (Severe Collapse)"],
        ["Native Monolingual", "Kaz-RoBERTa", "None (Subword fragmentation)", "Truncated (512 tokens)", "Self-Contained", "57.62% (-41.79% Drop)"],
        ["Proposed Framework", "Dual-Stream Gated Fusion", "83-Rule FST Engine", "Full-Length (25,000 words)", "Self-Contained", "99.80% (+42.18% Gain)"]
    ]
    add_table(
        slide, left=0.60, top=2.05, width=12.133, height=2.65,
        headers=headers, rows=rows,
        col_widths=[2.1, 2.2, 2.0, 1.8, 1.8, 2.233],
        highlight_row_idx=4
    )
    add_callout_box(
        slide, left=0.60, top=4.85, width=5.95, height=1.95,
        title="Literature Gap 1: Tokenization Fragmentation & OOD Collapse",
        points=[
            "• Curvature methods depend on surrogate LMs whose log-probability surfaces oscillate violently in low-resource settings.",
            "• Subword tokenizers fragment functional affixes into disjoint byte tokens, causing catastrophic generalization failure."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.78, top=4.85, width=5.95, height=1.95,
        title="Literature Gap 2: Fact Verification & Shortcut Learning",
        points=[
            "• In agglutinative languages, a 2-character negative suffix inverts truth value while retaining >80% lexical overlap.",
            "• Standard FEVER models succumb to lexical overlap shortcuts; an adversarial Hard NEI protocol is strictly required."
        ],
        accent_color=TEAL_ACCENT, bg_color=CARD_BG_WHITE
    )


def populate_slide_06(slide):
    """Slide 6: 03 Research Content — Comprehensive System Architecture & Technical Routes"""
    add_slide_header(
        slide,
        "Research Content: Tripartite System Architecture & Technical Routes",
        "A unified neural-symbolic framework coupling morphological fusion, long-document auditing, and factual verification"
    )
    add_figure_with_caption(
        slide, "fig03_tripartite_framework.png",
        left=0.60, top=2.05, width=6.65, height=4.35,
        caption_text="Figure 3. Tripartite System Architecture: Morphological Prior, Document Auditing, and Trust Verification."
    )
    add_callout_box(
        slide, left=7.45, top=2.05, width=5.28, height=1.35,
        title="Research Point 1: Morphology-Aware Gated Fusion",
        points=[
            "83-rule FST morphological analyzer + Kaz-RoBERTa semantic stream + BiLSTM morpheme stream + learned dynamic gate + SupCon."
        ],
        accent_color=TEAL_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=7.45, top=3.55, width=5.28, height=1.35,
        title="Research Point 2: Abbreviation-Guarded Auditing",
        points=[
            "10 regex lookahead guards + sliding window with 1-sentence overlap + dynamic Top-K worst-chunk pooling (100% localization on 25k words)."
        ],
        accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=7.45, top=5.05, width=5.28, height=1.35,
        title="Research Point 3: Kazakh-FEVER & Trust Matrix",
        points=[
            "Kazakh-FEVER 3K benchmark + hybrid BM25/M3 retrieval + 3-way NLI + (T_fact, T_gen) continuous Four-Quadrant Trust Matrix."
        ],
        accent_color=GREEN_ACCENT, bg_color=CARD_BG_WHITE
    )


def populate_slide_07(slide):
    """Slide 7: 03 Research Content — Research Point 1: Dual-Stream Morphology-Aware Gated Fusion"""
    add_slide_header(
        slide,
        "Research Content 1: Dual-Stream Morphology-Aware Gated Fusion",
        "Injecting deterministic finite-state linguistic priors into deep semantic representations to eliminate domain collapse"
    )
    add_figure_with_caption(
        slide, "fig04_morpho_gate_arch.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="Figure 4. Dual-Stream Cross-Attention Gated Fusion Architecture with 128D Hyperspherical Contrastive Learning."
    )
    add_callout_box(
        slide, left=6.65, top=2.05, width=6.08, height=2.30,
        title="Mathematical Formulation & Dynamic Gating Equations",
        points=[
            "1. Semantic Stream: h_sem = Kaz-RoBERTa(x_tokens) in R^d captures contextual discourse semantics.",
            "2. Morpheme Stream: 83-rule FST strips stems & affixes -> BiLSTM morpheme encoder produces h_morph in R^m.",
            "3. Dynamic Learned Gate: g = sigma(W_g [h_sem || h_morph] + b_g), dynamically balancing stream reliance.",
            "4. Fused Representation: h_fused = g * h_sem + (1 - g) * (W_m h_morph).",
            "5. Automatically amplifies the morphological stream whenever rare or out-of-domain agglutinative affixes appear."
        ],
        accent_color=TEAL_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=4.50, width=6.08, height=2.30,
        title="128D Hyperspherical Supervised Contrastive Loss (L_SupCon)",
        points=[
            "1. Joint Objective: L_total = L_CE + lambda * L_SupCon (lambda = 0.1).",
            "2. Latent Regularization: Tightly clusters identical classes on a 128D unit hypersphere while repelling human vs AI distributions.",
            "3. Leave-One-Generator-Out (LOGO) Rigor: Eliminates generator-specific stylistic memorization, achieving zero-shot transfer on unseen LLMs."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE
    )


def populate_slide_08(slide):
    """Slide 8: 03 Research Content — Research Point 2: Abbreviation-Guarded Long-Document Auditing"""
    add_slide_header(
        slide,
        "Research Content 2: Abbreviation-Guarded Long-Document Auditing",
        "Overcoming quadratic attention limits with linear O(L_chunk) memory and 100% paragraph-level tampering localization"
    )
    add_figure_with_caption(
        slide, "fig05_chunking_topk_flow.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="Figure 5. Linguistic Sentence Protection, Sliding Window Segmentation, and Dynamic Top-K Worst-Chunk Pooling."
    )
    add_callout_box(
        slide, left=6.65, top=2.05, width=6.08, height=2.30,
        title="Linguistic Sentence Boundary Protection (10 Regex Guards)",
        points=[
            "1. Boundary Breakdown: Standard sentence splitters break incorrectly at abbreviation periods (e.g., 't.b.', 'g.', 'zh.').",
            "2. Defensive Regex Lookaheads: Defends years (zh., zhzh.), centuries (g., gg.), cities (q.), professors (prof.), academic terms (akad.).",
            "3. Dialogue & Quote Attribution: Preserves embedded dialogue dashes and guillemets with zero span corruption.",
            "4. Overlapping Sliding Window: 1-sentence overlap smooths boundary discontinuities, preventing contextual truncation."
        ],
        accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=4.50, width=6.08, height=2.30,
        title="Dynamic Top-K Worst-Chunk Aggregation & Document Decision",
        points=[
            "1. Dilution Prevention: In a 20-page document with 1 forged page, naive mean-pooling dilutes the AI score to <0.05.",
            "2. Dynamic Formula: K = max(1, min(k_max, ceil(0.25 * M))), pooling only the worst suspicious chunks.",
            "3. Linear Hardware Footprint: Micro-batching maintains VRAM below 1.4 GB on documents up to 25,000 words.",
            "4. Empirical Verification: Delivers 100.0% precision in localizing synthetic paragraph insertions with <85ms latency."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE
    )


def populate_slide_09(slide):
    """Slide 9: 03 Research Content — Research Point 3: Evidence-Grounded Kazakh-FEVER & Trust Matrix"""
    add_slide_header(
        slide,
        "Research Content 3: Evidence-Grounded Kazakh-FEVER & Trust Matrix",
        "Decoupling stylistic AI probability from factual truthfulness through hybrid evidence retrieval and 2D Cartesian decision space"
    )
    add_figure_with_caption(
        slide, "fig06_trust_matrix_pipeline.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="Figure 6. Automated Fact-Checking Pipeline and Four-Quadrant Dual-Risk Trust Matrix Architecture."
    )
    add_callout_box(
        slide, left=6.65, top=2.05, width=6.08, height=2.30,
        title="Hybrid Evidence Retrieval & Cross-Encoder NLI",
        points=[
            "1. Authoritative Knowledge Base: Indexed 250,000 authoritative passages across History, Law, Science, and Public Health.",
            "2. Reciprocal Rank Fusion (RRF k0=60): Combines BM25 morphological stem retrieval with M3 dense embeddings.",
            "3. Cross-Encoder NLI: Jointly evaluates [CLS] Claim [SEP] Evidence [SEP] to output SUPPORTS / REFUTES / NEI.",
            "4. Strict Joint FEVER Metric: Requires both correct classification label and gold evidence span retrieval."
        ],
        accent_color=GREEN_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=4.50, width=6.08, height=2.30,
        title="Continuous Four-Quadrant Trust Decision Space (T_fact, T_gen)",
        points=[
            "Dual-Risk Trust Formulation: Risk_Trust = alpha * P_AI + (1 - alpha) * Risk_Fact (alpha = 0.5 default).",
            "• Quadrant I (Verified AI Assistance): High AI + High Fact -> Compliant AI summary; factually safe.",
            "• Quadrant II (Verified Human Prose): Low AI + High Fact -> Authentic human journalism; highly trusted.",
            "• Quadrant III (Human Misinformation): Low AI + Low Fact -> Human-authored falsehoods; malicious rumors.",
            "• Quadrant IV (Hallucinated AI Synthesis): High AI + Low Fact -> Fabricated synthetic assertions; immediate alert!"
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_GREEN
    )


def populate_slide_10(slide):
    """Slide 10: 04 Research Innovations — Scientific Contributions & Technical Challenges"""
    add_slide_header(
        slide,
        "Research Innovations: Scientific Contributions & Technical Challenges",
        "Core theoretical formulations and engineering solutions advancing low-resource NLP integrity"
    )
    add_callout_box(
        slide, left=0.60, top=2.05, width=5.95, height=4.75,
        title="Three Primary Scientific Innovations",
        points=[
            "1. Neural-Symbolic Inductive Prior (Innovation 1):",
            "   Pioneered the dynamic fusion of 83-rule deterministic finite-state transducers with transformer backbones, completely eliminating domain collapse (+42.18% OOD AUC gain on Kaspi).",
            "",
            "2. Boundary-Preserving Non-Destructive Auditing (Innovation 2):",
            "   Engineered 10 regex abbreviation guards and dynamic Top-K worst-chunk pooling, achieving 100% localization precision on 25,000-word hybrid documents with linear memory scaling.",
            "",
            "3. Dual-Risk Trust Space Decoupling (Innovation 3):",
            "   Constructed Central Asia's first Kazakh-FEVER 3K benchmark under adversarial Hard NEI and established the continuous 2D (T_fact, T_gen) trust matrix, separating stylistic form from factual truth."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.78, top=2.05, width=5.95, height=4.75,
        title="Key Technical Challenges & System Solutions",
        points=[
            "1. Challenge: Morphological Vocabulary Explosion & Affix Ambiguity",
            "   -> Solution: 83-rule FST state transition rules + BiLSTM morpheme encoder + dynamic learned gating mechanism.",
            "",
            "2. Challenge: Generator-Specific Stylistic Overfitting",
            "   -> Solution: Tri-Domain corpus across 4 LLM families + 128D hyperspherical supervised contrastive loss (L_SupCon) + LOGO cross-validation.",
            "",
            "3. Challenge: Retrieval Hallucination & Shortcut Learning",
            "   -> Solution: Adversarial 500 Hard NEI counterfactual protocol + hybrid BM25 and M3-Embedding Reciprocal Rank Fusion (RRF k0=60).",
            "",
            "4. Preliminary Milestones Achieved:",
            "   • Accepted oral paper at AIST 2026 (Springer LNCS).",
            "   • Submitted full research manuscript to COLING 2027 (ARR Track)."
        ],
        accent_color=TEAL_ACCENT, bg_color=CARD_BG_WHITE
    )


def populate_slide_11(slide):
    """Slide 11: 05 Schedule — Dissertation Schedule & Milestone Timeline (2026.10 – 2027.06)"""
    add_slide_header(
        slide,
        "Dissertation Schedule: Research Plan & Timeline (2026.10 – 2027.06)",
        "Structured five-stage research roadmap from proposal defense through final Master's degree defense in June 2027"
    )
    add_figure_with_caption(
        slide, "fig15_timeline_2026_2027.png",
        left=0.60, top=2.05, width=12.133, height=4.50,
        caption_text="Figure 7. Master's Dissertation Research Timeline and Milestone Execution Plan (October 2026 -- June 2027)."
    )


def update_closing_slide(slide):
    """Slide 12 (Closing Slide): Updates closing text and metadata matching laboratory standard."""
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
            shp.left = Inches(1.50)
            shp.top = Inches(3.60)
            shp.width = Inches(10.33)
            shp.height = Inches(2.80)

            tf = shp.text_frame
            tf.word_wrap = True

            p0 = tf.paragraphs[0]
            p0.text = "Thank You for Your Attention!"
            p0.font.name = FONT_TITLE
            p0.font.size = Pt(36)
            p0.font.bold = True
            p0.font.color.rgb = NAVY_PRIMARY
            p0.alignment = PP_ALIGN.CENTER

            p1 = tf.add_paragraph()
            p1.text = "敬请各位专家批评指正！"
            p1.font.name = FONT_ZH
            p1.font.size = Pt(22)
            p1.font.bold = True
            p1.font.color.rgb = NAVY_PRIMARY
            p1.alignment = PP_ALIGN.CENTER
            p1.space_before = Pt(4)

            p2 = tf.add_paragraph()
            p2.text = "Research on Morphologically-Grounded AI-Generated Text Detection and Factual Verification for Low-Resource Kazakh"
            p2.font.name = FONT_TITLE
            p2.font.size = Pt(13.5)
            p2.font.bold = True
            p2.font.color.rgb = SLATE_BODY
            p2.alignment = PP_ALIGN.CENTER
            p2.space_before = Pt(8)

            p3 = tf.add_paragraph()
            p3.text = "Candidate: DAULET ANEKESH (大雷, Student ID: 2025291043) | Supervisor: Prof. Bin Guo (郭斌 教授) | October 29, 2026"
            p3.font.name = FONT_TITLE
            p3.font.size = Pt(12)
            p3.font.bold = False
            p3.font.color.rgb = SLATE_MUTED
            p3.alignment = PP_ALIGN.CENTER
            p3.space_before = Pt(4)


# =============================================================================
# MAIN CONTENT SLIDE PROCESSOR & TEMPLATE CLONER
# =============================================================================

def update_content_slides(slides):
    """
    Slides 3-11 (indices 2 through 10):
    1. Sets up the 5-element running section tracker on top bar (TITLED ONLY IN ENGLISH, ENLARGED).
    2. Repositions '矩形 29' (Native Gradient Highlight Tracker) to active section.
    3. Excises legacy shapes below top > Inches(1.2).
    4. Populates academic content for Slides 3-11.
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
    }

    for s_idx in range(2, 11):
        slide = slides[s_idx]
        active_sec_idx = SLIDE_SECTION_MAP.get(s_idx, 0)
        target_left, target_width = SLOT_COORDS[active_sec_idx]

        # 1. Clear old header textboxes and reposition 矩形 29
        shapes_to_remove = []
        for shp in slide.shapes:
            # Check watermark
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

            # Remove old header textboxes on top bar
            if shp.top < Inches(1.1) and shp.has_text_frame and shp.name.startswith("TextBox"):
                shapes_to_remove.append(shp)
                continue

            # Clear legacy content shapes below top > 1.2 inches
            if shp.top > Inches(1.2):
                if shp.name.startswith("直接连接符") or shp.name.startswith("灯片编号"):
                    continue
                shapes_to_remove.append(shp)

        for shp in shapes_to_remove:
            remove_shape(shp)

        # Reposition 矩形 29 (the highlight tracker pill)
        for shp in slide.shapes:
            if shp.name == "矩形 29":
                shp.left = target_left
                shp.width = target_width

        # 2. Add 5 clean section textboxes on top bar: TITLED ONLY IN ENGLISH, ENLARGED
        for idx in range(5):
            s_left, s_width = SLOT_COORDS[idx]
            is_active = (idx == active_sec_idx)

            tb = slide.shapes.add_textbox(s_left, Inches(0.12), s_width, Inches(0.68))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

            p_en = tf.paragraphs[0]
            p_en.text = SECTION_NAMES_EN[idx]
            p_en.font.name = FONT_TITLE
            p_en.font.size = Pt(13.5)
            p_en.font.bold = True
            p_en.font.color.rgb = RGBColor(255, 255, 255) if is_active else NAVY_PRIMARY
            p_en.alignment = PP_ALIGN.CENTER

        # 3. Populate slide body content
        if s_idx in population_dispatch:
            population_dispatch[s_idx](slide)


def build_proposal_presentation(
    input_path=BASELINE_PPT_PATH,
    output_path=DEFAULT_OUTPUT_PATH,
    mirror_desktop=True
):
    """
    Loads baseline presentation, reduces to exactly 12 slides, applies 5-element English tracker,
    populates all slides, and saves to output_path and desktop mirror.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Baseline PPT file not found: {input_path}")

    print(f"Loading baseline template from: {input_path}")
    prs = Presentation(input_path)

    # In baseline (23 slides), drop index 21 down to 11 (11 slides), leaving 12 slides total.
    # Slide 22 (original closing slide) becomes index 11 (Slide 12).
    print(f"Initial slide count: {len(prs.slides)}")
    for idx in range(21, 10, -1):
        rId = prs.slides._sldIdLst[idx].rId
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[idx]

    print(f"Trimmed slide count: {len(prs.slides)} (Target: 12)")

    # 1. Update Slide 1 (Cover Page)
    print("Updating Slide 1: Cover Page...")
    update_cover_slide(prs.slides[0])

    # 2. Update Slide 2 (TOC)
    print("Updating Slide 2: 5-Element Table of Contents...")
    update_toc_slide(prs.slides[1])

    # 3. Update Slides 3 - 11 (Content Slides)
    print("Populating Slides 3 - 11 with academic English content, figures, and English tracker...")
    update_content_slides(prs.slides)

    # 4. Update Slide 12 (Closing Slide)
    print("Updating Slide 12: Closing & Guidance Slide...")
    update_closing_slide(prs.slides[11])

    # Save to project output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")

    # Mirror to Desktop
    if mirror_desktop:
        try:
            shutil.copy2(output_path, DESKTOP_MIRROR_PATH)
            print(f"Successfully mirrored presentation to Desktop: {DESKTOP_MIRROR_PATH}")
        except Exception as e:
            print(f"Warning: Failed to mirror to Desktop: {e}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Master's Proposal Defense Presentation (10-Minute Compact)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output PPTX path")
    args = parser.parse_args()

    build_proposal_presentation(output_path=args.output)
