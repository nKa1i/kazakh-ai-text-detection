# -*- coding: utf-8 -*-
"""
[LEGACY / SCRATCH PROTOTYPE - DO NOT OVERWRITE THE AUTHENTIC DECK]

IMPORTANT NOTICE:
This script creates a scratch presentation from a blank python-pptx template.
It does NOT clone or preserve native institutional KazNU / NPU branding elements
(KazNU logo, NPU logo group, NPU watermark, or Layout 7 pentagon badge from 'Aliya PPT.pptx').

The AUTHORITATIVE presentation builder is:
    scripts/build_cloned_presentation.py
which clones 'Aliya PPT.pptx' directly, preserves all authentic institutional branding,
excises commercial watermarks, embeds 14 publication-grade figures, formats academic
comparison tables, and outputs the official presentation:
    Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx

To prevent clobbering the authentic cloned deck, this script defaults its output to:
    Kazakh_AI_Detection_Thesis_Progress_Prof_Guo_scratch.pptx

Generate Academic Master's Thesis Progress Presentation for Professor Guo (Scratch Prototype).
Formatted in matching style with baseline 'Aliya PPT.pptx' (16:9 widescreen,
running section tracker, stat callout cards, comparison tables, and committee responses).

Presenter: 大雷 (Daulet)
Advisor: 郭教授 (Prof. Guo)
Date: September 2026 (2026年9月)
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# -----------------------------------------------------------------------------
# Color Palette & Design Tokens (Institutional Academic Palette)
# -----------------------------------------------------------------------------
NAVY_PRIMARY = RGBColor(30, 58, 138)     # #1E3A8A - Deep Academic Navy Blue
BLUE_ACCENT = RGBColor(37, 99, 235)      # #2563EB - Royal Blue Accent
TEAL_ACCENT = RGBColor(13, 148, 136)     # #0D9488 - Topic 1 / FST Morphology
AMBER_ACCENT = RGBColor(217, 119, 6)     # #D97706 - Topic 2 / Chunking & Aggregation
GREEN_ACCENT = RGBColor(22, 163, 74)     # #16A34A - Topic 3 / Factual Verification
PURPLE_ACCENT = RGBColor(124, 58, 237)   # #7C3AED - System Engineering & Cloud
RED_ACCENT = RGBColor(220, 38, 38)       # #DC2626 - Risk & Hallucination Flag

SLATE_TITLE = RGBColor(15, 23, 42)       # #0F172A - Slide title & primary headers
SLATE_BODY = RGBColor(51, 65, 85)        # #334155 - Standard body prose
SLATE_MUTED = RGBColor(100, 116, 139)    # #64748B - Subtitles & inactive items
BORDER_LIGHT = RGBColor(226, 232, 240)   # #E2E8F0 - Divider lines & card borders

BG_TRACKER = RGBColor(241, 245, 249)     # #F1F5F9 - Running tracker background
CARD_BG_WHITE = RGBColor(255, 255, 255)  # #FFFFFF - Crisp white cards
CARD_BG_ALT = RGBColor(248, 250, 252)    # #F8FAFC - Soft light gray card fill
CARD_BG_BLUE = RGBColor(238, 244, 255)   # #EEF4FF - Soft tinted card fill
CARD_BG_GREEN = RGBColor(240, 253, 244)  # #F0FDF4 - Soft green tinted card fill

FONT_TITLE = "Times New Roman"
FONT_BODY = "Times New Roman"
FONT_ZH = "Microsoft YaHei"

SECTION_NAMES = [
    "Research Background",
    "Related Work",
    "Research Content",
    "Experiments & Results",
    "System Demonstration",
    "Conclusion & Roadmap",
]


def create_presentation(output_path=None):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_navigation_header(slide, active_section_idx, slide_num):
        """Adds top horizontal running section tracker matching Aliya PPT."""
        tracker = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(0.0), Inches(12.133), Inches(0.88)
        )
        tracker.fill.solid()
        tracker.fill.fore_color.rgb = BG_TRACKER
        tracker.line.fill.background()

        left_offset = 0.8
        slot_width = 1.75

        for idx, sec_name in enumerate(SECTION_NAMES):
            is_active = (idx == active_section_idx)
            pill = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE, Inches(left_offset), Inches(0.12), Inches(slot_width), Inches(0.64)
            )
            pill.fill.solid()
            if is_active:
                pill.fill.fore_color.rgb = NAVY_PRIMARY
                pill.line.color.rgb = BLUE_ACCENT
                pill.line.width = Pt(1.5)
            else:
                pill.fill.fore_color.rgb = BG_TRACKER
                pill.line.fill.background()

            tf = pill.text_frame
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]
            p.text = sec_name
            p.alignment = PP_ALIGN.CENTER
            p.font.name = FONT_TITLE
            p.font.size = Pt(10)
            p.font.bold = is_active
            p.font.color.rgb = RGBColor(255, 255, 255) if is_active else SLATE_MUTED

            left_offset += slot_width + 0.10

        # Slide number indicator badge
        num_box = slide.shapes.add_textbox(Inches(11.85), Inches(0.18), Inches(0.85), Inches(0.5))
        ntf = num_box.text_frame
        ntf.vertical_anchor = MSO_ANCHOR.MIDDLE
        np = ntf.paragraphs[0]
        np.text = f"-{slide_num}-"
        np.alignment = PP_ALIGN.CENTER
        np.font.name = FONT_TITLE
        np.font.size = Pt(11)
        np.font.bold = True
        np.font.color.rgb = SLATE_MUTED

        # Thin separator line
        sep = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(0.96), Inches(12.133), Inches(0.02)
        )
        sep.fill.solid()
        sep.fill.fore_color.rgb = BORDER_LIGHT
        sep.line.fill.background()

    def add_slide_header(slide, title_text, subtitle_text=""):
        """Adds standard slide title and optional subtitle below running tracker."""
        tb = slide.shapes.add_textbox(Inches(0.6), Inches(1.06), Inches(12.133), Inches(0.72))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = FONT_TITLE
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = NAVY_PRIMARY

        if subtitle_text:
            p2 = tf.add_paragraph()
            p2.text = subtitle_text
            p2.font.name = FONT_BODY
            p2.font.size = Pt(10.5)
            p2.font.color.rgb = SLATE_MUTED
            p2.space_before = Pt(3)

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

        tb = slide.shapes.add_textbox(Inches(left + 0.16), Inches(top + 0.1), Inches(width - 0.24), Inches(height - 0.2))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_val = tf.paragraphs[0]
        p_val.text = value_str
        p_val.font.name = FONT_TITLE
        p_val.font.size = Pt(22)
        p_val.font.bold = True
        p_val.font.color.rgb = accent_color

        p_tit = tf.add_paragraph()
        p_tit.text = title
        p_tit.font.name = FONT_TITLE
        p_tit.font.size = Pt(11)
        p_tit.font.bold = True
        p_tit.font.color.rgb = SLATE_TITLE
        p_tit.space_before = Pt(2)

        if subtitle:
            p_sub = tf.add_paragraph()
            p_sub.text = subtitle
            p_sub.font.name = FONT_BODY
            p_sub.font.size = Pt(9.5)
            p_sub.font.color.rgb = SLATE_MUTED
            p_sub.space_before = Pt(2)

    def add_callout_box(slide, left, top, width, height, title, points,
                        accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE, border_color=BORDER_LIGHT):
        """Adds a content card with header and bullet points."""
        box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        box.fill.solid()
        box.fill.fore_color.rgb = bg_color
        box.line.color.rgb = border_color
        box.line.width = Pt(1)

        is_multiline_title = ("\n" in title)
        strip_h = 0.66 if is_multiline_title else 0.42

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
        p_head.font.size = Pt(10.0 if is_multiline_title else 11.5)
        p_head.font.bold = True
        p_head.font.color.rgb = RGBColor(255, 255, 255)

        tb_body = slide.shapes.add_textbox(
            Inches(left + 0.15), Inches(top + strip_h + 0.08), Inches(width - 0.3), Inches(height - strip_h - 0.16)
        )
        tf_body = tb_body.text_frame
        tf_body.word_wrap = True
        tf_body.margin_left = tf_body.margin_top = tf_body.margin_right = tf_body.margin_bottom = 0

        for i, pt in enumerate(points):
            p = tf_body.paragraphs[0] if i == 0 else tf_body.add_paragraph()
            p.text = pt
            p.font.name = FONT_BODY
            p.font.size = Pt(9.5)
            p.font.color.rgb = SLATE_BODY
            if i > 0:
                p.space_before = Pt(3)

    def add_table(slide, left, top, width, height, headers, rows, col_widths=None, highlight_row_idx=None):
        """Adds a clean academic comparison table with colored header and alternating rows."""
        num_rows = len(rows) + 1
        num_cols = len(headers)
        table_shape = slide.shapes.add_table(num_rows, num_cols, Inches(left), Inches(top), Inches(width), Inches(height))
        table = table_shape.table

        if col_widths and len(col_widths) == num_cols:
            for c_idx, w in enumerate(col_widths):
                table.columns[c_idx].width = Inches(w)

        # Format header row
        for c_idx, h_text in enumerate(headers):
            cell = table.cell(0, c_idx)
            cell.margin_left = Inches(0.05)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.04)
            cell.margin_bottom = Inches(0.04)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY_PRIMARY
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = h_text
            p.alignment = PP_ALIGN.CENTER
            p.font.name = FONT_TITLE
            p.font.size = Pt(9.5)
            p.font.bold = True
            p.font.color.rgb = RGBColor(255, 255, 255)

        # Format data rows
        for r_idx, row_data in enumerate(rows):
            is_highlight = (highlight_row_idx is not None and r_idx == highlight_row_idx)
            row_bg = CARD_BG_BLUE if is_highlight else (CARD_BG_WHITE if r_idx % 2 == 0 else CARD_BG_ALT)

            for c_idx, val in enumerate(row_data):
                cell = table.cell(r_idx + 1, c_idx)
                cell.margin_left = Inches(0.05)
                cell.margin_right = Inches(0.05)
                cell.margin_top = Inches(0.04)
                cell.margin_bottom = Inches(0.04)
                cell.fill.solid()
                cell.fill.fore_color.rgb = row_bg
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                tf = cell.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                p.text = str(val)
                p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER
                p.font.name = FONT_BODY
                p.font.size = Pt(9.0)
                if is_highlight:
                    p.font.bold = True
                    p.font.color.rgb = NAVY_PRIMARY
                else:
                    p.font.color.rgb = SLATE_BODY

    # =========================================================================
    # SLIDE 1: Title Slide (Title & Presenter Info)
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)

    # Academic framing background cards
    frame_bg = slide1.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(0.5), Inches(12.333), Inches(6.5)
    )
    frame_bg.fill.solid()
    frame_bg.fill.fore_color.rgb = CARD_BG_ALT
    frame_bg.line.color.rgb = BORDER_LIGHT
    frame_bg.line.width = Pt(1.5)

    # Top institutional badge
    badge = slide1.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(1.0), Inches(4.8), Inches(0.38)
    )
    badge.fill.solid()
    badge.fill.fore_color.rgb = NAVY_PRIMARY
    badge.line.fill.background()
    tf_b = badge.text_frame
    p_b = tf_b.paragraphs[0]
    p_b.text = "MASTER'S THESIS PROGRESS REPORT"
    p_b.alignment = PP_ALIGN.CENTER
    p_b.font.name = FONT_TITLE
    p_b.font.size = Pt(10)
    p_b.font.bold = True
    p_b.font.color.rgb = RGBColor(255, 255, 255)

    # Main Presentation Title
    tb_title = slide1.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(11.3), Inches(2.4))
    tf_t = tb_title.text_frame
    tf_t.word_wrap = True
    p_t1 = tf_t.paragraphs[0]
    p_t1.text = "Research on Morphologically-Grounded AI-Generated Text Detection and Factual Verification for Low-Resource Kazakh"
    p_t1.font.name = FONT_TITLE
    p_t1.font.size = Pt(25)
    p_t1.font.bold = True
    p_t1.font.color.rgb = NAVY_PRIMARY

    p_t2 = tf_t.add_paragraph()
    p_t2.text = "面向低资源哈萨克语的形态感知AI生成文本检测与事实核验研究"
    p_t2.font.name = FONT_ZH
    p_t2.font.size = Pt(19)
    p_t2.font.bold = True
    p_t2.font.color.rgb = BLUE_ACCENT
    p_t2.space_before = Pt(10)

    # Subtitle / Progress Note
    tb_sub = slide1.shapes.add_textbox(Inches(1.0), Inches(4.2), Inches(11.3), Inches(0.8))
    tf_s = tb_sub.text_frame
    tf_s.word_wrap = True
    p_s1 = tf_s.paragraphs[0]
    p_s1.text = "Three Interlocking Innovations: Dual-Stream Morphological Gated Cross-Attention, Sentence-Preserving Chunking Engine, and Evidence-Grounded Four-Quadrant Trust Matrix"
    p_s1.font.name = FONT_BODY
    p_s1.font.size = Pt(11.5)
    p_s1.font.color.rgb = SLATE_BODY

    # Presenter & Advisor Line (Matching baseline format exactly)
    meta_box = slide1.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(5.35), Inches(11.333), Inches(1.15)
    )
    meta_box.fill.solid()
    meta_box.fill.fore_color.rgb = CARD_BG_WHITE
    meta_box.line.color.rgb = BLUE_ACCENT
    meta_box.line.width = Pt(1)

    tb_meta = slide1.shapes.add_textbox(Inches(1.2), Inches(5.45), Inches(11.0), Inches(0.95))
    tf_m = tb_meta.text_frame
    tf_m.word_wrap = True

    p_m1 = tf_m.paragraphs[0]
    p_m1.text = "汇报人：大雷 (Daulet)      导师：郭教授 (Prof. Guo)      日期：2026年9月"
    p_m1.font.name = FONT_ZH
    p_m1.font.size = Pt(14)
    p_m1.font.bold = True
    p_m1.font.color.rgb = NAVY_PRIMARY

    p_m2 = tf_m.add_paragraph()
    p_m2.text = "Candidate: Daulet (Da Lei)  |  Advisor: Prof. Guo  |  Degree: Master of Science in Computer Science and Technology"
    p_m2.font.name = FONT_BODY
    p_m2.font.size = Pt(11)
    p_m2.font.color.rgb = SLATE_MUTED
    p_m2.space_before = Pt(4)

    # =========================================================================
    # SLIDE 2: Table of Contents (CONTENTS / 目录)
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)

    # Left sidebar background
    sb = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(0.5), Inches(3.2), Inches(6.5))
    sb.fill.solid()
    sb.fill.fore_color.rgb = NAVY_PRIMARY
    sb.line.fill.background()

    tb_sb = slide2.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(2.6), Inches(4.5))
    tf_sb = tb_sb.text_frame
    tf_sb.word_wrap = True
    p_c1 = tf_sb.paragraphs[0]
    p_c1.text = "CONTENTS"
    p_c1.font.name = FONT_TITLE
    p_c1.font.size = Pt(28)
    p_c1.font.bold = True
    p_c1.font.color.rgb = RGBColor(255, 255, 255)

    p_c2 = tf_sb.add_paragraph()
    p_c2.text = "汇报目录与研究大纲"
    p_c2.font.name = FONT_ZH
    p_c2.font.size = Pt(16)
    p_c2.font.color.rgb = RGBColor(147, 197, 253)
    p_c2.space_before = Pt(8)

    p_c3 = tf_sb.add_paragraph()
    p_c3.text = "Master's Thesis Comprehensive Progress Discussion\nAdvisor: Prof. Guo\nSept 2026"
    p_c3.font.name = FONT_BODY
    p_c3.font.size = Pt(10.5)
    p_c3.font.color.rgb = RGBColor(203, 213, 225)
    p_c3.space_before = Pt(20)

    # 6 Section items on the right
    toc_items = [
        ("01", "Research Background & Motivation", "研究背景与核心动机",
         "Agglutinative morphology challenges, LLM synthetic text proliferation, and OOD domain collapse."),
        ("02", "Related Work & SOTA Gaps", "相关工作与现有局限",
         "Survey of multilingual text detectors, failure of generic perplexity methods, and Kazakh fact-checking gaps."),
        ("03", "Research Content & Architecture", "研究内容与系统架构",
         "Topic 1 (Morpho-Gate), Topic 2 (Long-Doc Chunking & Top-K Engine), and Topic 3 (Kazakh-FEVER Trust Matrix)."),
        ("04", "Experiments & Benchmark Evaluation", "实验设计与评测结果",
         "Kaz-MAGE 2x2 matrix (+42.2% Q3 gain, 100% Q4), 25k word document scaling, and 100% Macro-F1 NLI verification."),
        ("05", "Engineering & Demonstration", "工程落地与交互演示",
         "4-Tab Gradio academic dashboard, standalone HF Spaces 1-click package, and 288 passing automated tests."),
        ("06", "Conclusion & Thesis Roadmap", "研究总结与毕业规划",
         "AIST 2026 camera-ready, Paper 2 drafting (EMNLP/COLING), 85% thesis completion, and committee discussion."),
    ]

    top_y = 0.65
    for num, en_title, zh_title, desc in toc_items:
        card = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4.0), Inches(top_y), Inches(8.8), Inches(0.95))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG_WHITE
        card.line.color.rgb = BORDER_LIGHT
        card.line.width = Pt(1)

        badge_num = slide2.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(4.15), Inches(top_y + 0.12), Inches(0.7), Inches(0.7)
        )
        badge_num.fill.solid()
        badge_num.fill.fore_color.rgb = BLUE_ACCENT
        badge_num.line.fill.background()
        tf_bn = badge_num.text_frame
        p_bn = tf_bn.paragraphs[0]
        p_bn.text = num
        p_bn.alignment = PP_ALIGN.CENTER
        p_bn.font.name = FONT_TITLE
        p_bn.font.size = Pt(14)
        p_bn.font.bold = True
        p_bn.font.color.rgb = RGBColor(255, 255, 255)

        tb_text = slide2.shapes.add_textbox(Inches(5.0), Inches(top_y + 0.08), Inches(7.8), Inches(0.8))
        tf_tt = tb_text.text_frame
        tf_tt.word_wrap = True
        tf_tt.margin_left = tf_tt.margin_right = tf_tt.margin_top = tf_tt.margin_bottom = 0
        p_tt1 = tf_tt.paragraphs[0]
        p_tt1.text = f"{en_title}  |  {zh_title}"
        p_tt1.font.name = FONT_TITLE
        p_tt1.font.size = Pt(12)
        p_tt1.font.bold = True
        p_tt1.font.color.rgb = NAVY_PRIMARY

        p_tt2 = tf_tt.add_paragraph()
        p_tt2.text = desc
        p_tt2.font.name = FONT_BODY
        p_tt2.font.size = Pt(9.5)
        p_tt2.font.color.rgb = SLATE_MUTED
        p_tt2.space_before = Pt(2)

        top_y += 1.05

    # =========================================================================
    # SLIDE 3: Research Background - The Synthetic Text Threat in Kazakh
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide3, 0, 1)
    add_slide_header(slide3, "Research Background: Kazakh NLP Challenges & The Synthetic Text Threat",
                     "Agglutinative morphological complexity and rapid proliferation of multilingual generative LLMs in Central Asia")

    # Left column: 3 structured problem statements
    add_callout_box(
        slide3, 0.6, 1.85, 6.2, 5.0,
        "Central Asian Digital Sovereignty & Linguistic Vulnerabilities",
        [
            "1. Severe Linguistic Asymmetry & Subword Fragmentation:",
            "   Kazakh is an agglutinative Turkic language with complex vowel harmony and extensive affixation chains (up to 15 suffixes attached to a single stem). Standard multilingual BPE tokenizers severely fragment Kazakh words into meaningless byte tokens, obscuring synthetic syntactic patterns.",
            "",
            "2. Surge of High-Capacity Multilingual LLMs:",
            "   Models such as Qwen-2.5-7B, LLaMA-3, and localized models (Sherkala-7B) now generate fluent synthetic Kazakh. Without linguistic grounding, educational institutions and news agencies in Kazakhstan face severe risks of unchecked academic misconduct and synthetic misinformation.",
            "",
            "3. Complete Absence of Native Kazakh Detection Tools:",
            "   Prior to this thesis, zero public or commercial AI detectors specialized in Kazakh. Commercial detectors trained on English achieve near-random accuracy on Kazakh texts, leaving the national digital ecosystem unprotected."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_WHITE
    )

    # Right column: 4 stat callout cards (matching Aliya PPT Slide 3 cards)
    add_card(slide3, 7.1, 1.85, 2.7, 2.35, "Affix Complexity", "15+",
             "Suffixes per nominal/verbal stem; breaks standard subword tokenizers", accent_color=TEAL_ACCENT, bg_color=CARD_BG_BLUE)
    add_card(slide3, 10.0, 1.85, 2.7, 2.35, "Document Capacity", "25,000",
             "Words per document in real-world academic theses & news reports", accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE)
    add_card(slide3, 7.1, 4.45, 2.7, 2.35, "Curated Samples", "10,000+",
             "Tri-domain benchmark covering News, Wikipedia, and Consumer Reviews", accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE)
    add_card(slide3, 10.0, 4.45, 2.7, 2.35, "Detection Baseline", "0 -> 1",
             "First morphologically-grounded detection & fact-checking framework for Kazakh", accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN)

    # =========================================================================
    # SLIDE 4: Research Background - Challenges of Existing Systems
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide4, 0, 2)
    add_slide_header(slide4, "Challenges of Existing Systems: Why Standard AI Detectors Fail on Kazakh",
                     "Empirical analysis reveals three critical failure modes in pretrained transformers and black-box detectors")

    add_callout_box(
        slide4, 0.6, 1.9, 3.8, 5.0,
        "Challenge 1: Morphological Domain Collapse",
        [
            "Observed Failure:",
            "Pretrained RoBERTa/mBERT models exhibit severe domain collapse when tested on unseen genres or modern LLMs.",
            "",
            "Empirical Evidence:",
            "On our ACL Kaz-MAGE 2x2 matrix, standard KazRoBERTa drops from 99.41% AUC on seen news to 57.62% AUC on unseen reviews (Q3).",
            "",
            "Root Cause:",
            "Semantic encoders overfit to lexical domain markers (e.g. formal news vocabulary) rather than underlying generation artifacts, failing entirely when colloquial or out-of-domain vocabulary appears."
        ],
        accent_color=RED_ACCENT, bg_color=CARD_BG_WHITE
    )

    add_callout_box(
        slide4, 4.7, 1.9, 3.8, 5.0,
        "Challenge 2: The Document Truncation Bottleneck",
        [
            "Observed Failure:",
            "Standard transformer architectures strictly limit input lengths to 512 tokens. Real documents range from 1,000 to 25,000 words.",
            "",
            "Empirical Evidence:",
            "Head-truncation misses malicious AI insertions in paragraphs 5 to 50. Mean-pooling across naive fixed strides dilutes localized AI signals to zero.",
            "",
            "Root Cause:",
            "Fixed-stride token slicing chops words mid-stem and splits sentences, destroying syntactic boundaries and character offset fidelity required for real-world user attribution."
        ],
        accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE
    )

    add_callout_box(
        slide4, 8.8, 1.9, 3.9, 5.0,
        "Challenge 3: Truthfulness-Agnostic Detection",
        [
            "Observed Failure:",
            "Current AI detectors output a single probability score (AI vs Human), completely blind to factual truth or misinformation.",
            "",
            "Empirical Evidence:",
            "An AI detector flags a fully factual AI summary as 'dangerous AI' while giving a 100% pass to human-written fake news and toxic conspiracy theories.",
            "",
            "Root Cause:",
            "Stylistic probability does not equal veracity. Real-world content moderation requires a joint assessment of stylistic AI probability AND external factual grounding."
        ],
        accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE
    )

    # =========================================================================
    # SLIDE 5: Related Work - AI Text Detection Paradigms & Turkic SOTA Gaps
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide5, 1, 3)
    add_slide_header(slide5, "Related Work: Comparative Analysis of SOTA AI Text Detection Paradigms",
                     "Comparison of mainstream detection methodologies and their catastrophic limitations on agglutinative Turkic languages")

    # Table on the left
    table_headers = ["Method / Model", "Core Paradigm", "Kazakh Support", "OOD Transfer", "Linguistic Grounding"]
    table_rows = [
        ["Binoculars (Hans et al., 2024)", "Perplexity Ratio (M1/M2)", "Poor (Token Inflation)", "52.1% AUC", "None (Statistical)"],
        ["Ghostbuster (Verma et al., 2023)", "Multi-Model N-gram Logits", "None (No API Support)", "58.4% AUC", "None (Black-box)"],
        ["Fast-DetectGPT (Bao et al., 2024)", "Conditional Probability Curvature", "Moderate (LLM Dep.)", "64.2% AUC", "None (Curvature)"],
        ["KazRoBERTa (Baseline 2025)", "Fine-tuned Transformer Classifier", "Native BPE Tokens", "57.62% AUC", "Subword Only"],
        ["Proposed Morpho-Detector", "Dual-Stream FST + Cross-Gate", "Native 83-Rule FST", "99.80% AUC", "Agglutinative Morphology"]
    ]
    add_table(slide5, 0.6, 1.9, 7.8, 3.2, table_headers, table_rows,
              col_widths=[2.1, 2.0, 1.3, 1.1, 1.3], highlight_row_idx=4)

    # Right side: 3 Analytical Takeaways
    add_callout_box(
        slide5, 8.6, 1.9, 4.1, 5.0,
        "Key SOTA Insights & Gaps in Prior Literature",
        [
            "1. Perplexity Metrics Fail Under Agglutination:",
            "   Because agglutinative affixes create extreme vocabulary sparsity, perplexity-based methods (Binoculars, Fast-DetectGPT) suffer severe false-positive spikes whenever uncommon grammatical suffixes occur in human text.",
            "",
            "2. Lack of In-the-Wild Generalization:",
            "   Pretrained transformer classifiers fail catastrophically when tested against modern high-parameter models (e.g. Qwen-2.5-7B) not present in the training set.",
            "",
            "3. The Need for Dual-Stream Inductive Bias:",
            "   Explicit morphological decomposition via Finite State Transducers (FST) acts as a domain-invariant structural regularizer, bridging semantic domain shifts."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_WHITE
    )

    # Note below table
    nb = slide5.shapes.add_textbox(Inches(0.6), Inches(5.3), Inches(7.8), Inches(1.6))
    tf_nb = nb.text_frame
    tf_nb.word_wrap = True
    p_nb1 = tf_nb.paragraphs[0]
    p_nb1.text = "Summary of Table Findings: Prior methods report >90% accuracy on English benchmarks (MAGE, RAID), but experience up to 40% performance degradation on Kazakh due to morphological subword fragmentation and lack of Turkish/Turkic language inductive priors. Our proposed method is the only system achieving >99.5% transfer across both unseen domains and unseen generators."
    p_nb1.font.name = FONT_BODY
    p_nb1.font.size = Pt(9.5)
    p_nb1.font.color.rgb = SLATE_BODY

    # =========================================================================
    # SLIDE 6: Related Work - Evidence Retrieval & Fact-Checking Gaps
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide6, 1, 4)
    add_slide_header(slide6, "Related Work: Fact-Checking Benchmarks & The Central Asian Evidence Void",
                     "Existing automated fact-checking corpora are exclusively Anglo-centric; zero evidence-grounded resources exist for Kazakh")

    add_callout_box(
        slide6, 0.6, 1.9, 5.8, 5.0,
        "Analysis of International Fact-Checking Benchmarks",
        [
            "FEVER (Thorne et al., 2018):",
            "- 185,445 claims based on English Wikipedia.",
            "- Evaluates sentence retrieval + 3-way NLI classification.",
            "- Limitation: 100% English, highly structured, clean Wikipedia syntax.",
            "",
            "VitaminC (Schuster et al., 2021):",
            "- 400,000+ claim-evidence pairs with contrastive revisions.",
            "- Focuses on subtle factual edits and temporal updates.",
            "- Limitation: Exclusively English; relies on large-scale crowd annotations.",
            "",
            "SciFact (Wadden et al., 2020):",
            "- Scientific claim verification over biomedical abstracts.",
            "- Limitation: Domain-specific, English, high annotation overhead."
        ],
        accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE
    )

    add_callout_box(
        slide6, 6.7, 1.9, 6.0, 5.0,
        "The Central Asian Fact-Checking Void & Our Solution",
        [
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

    # =========================================================================
    # SLIDE 7: Research Content - Overview & Tripartite Architecture
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide7, 2, 5)
    add_slide_header(slide7, "Research Content Overview: Three Interlocking Technical Innovations",
                     "A unified hierarchical framework spanning sentence-level morpho-gating, document-level chunk aggregation, and evidence-grounded trust verification")

    topics = [
        ("TOPIC 1: Sentence-Level Detector",
         "Dual-Stream Morphological Gated Cross-Attention",
         TEAL_ACCENT, CARD_BG_BLUE,
         [
             "Core Hypothesis: Explicit morphological FST inductive bias regularizes semantic encoders against domain shifts.",
             "Key Architecture:",
             "- Pretrained KazRoBERTa backbone for semantic embeddings (h_sem).",
             "- 83-Rule Apertium/PyDataverse FST parser for morpheme sequences (h_morph).",
             "- Dynamic learned gate g = sigmoid(W[h_sem; h_morph] + b).",
             "- Joint SupCon + Cross-Entropy contrastive optimization.",
             "Major Breakthrough: Resolves Q3 OOD domain collapse from 57.62% to 99.80% AUC (+42.18%)."
         ]),
        ("TOPIC 2: Document-Level Engine",
         "Sentence-Preserving Chunking & Dynamic Top-K Pooling",
         AMBER_ACCENT, CARD_BG_WHITE,
         [
             "Core Hypothesis: Long documents require syntactic boundary preservation and worst-chunk sensitivity.",
             "Key Architecture:",
             "- 10 Kazakh abbreviation regex guards (т.б., ж.б., ғғ., мыс., проф.).",
             "- Sliding window (max 200 words, 1-sentence overlap).",
             "- Zero character offset drift: doc[start:end] == chunk.",
             "- Dynamic Top-K worst-chunk pooling formula: K = max(1, min(k, ceil(0.25*M))).",
             "Major Breakthrough: 100% localization of malicious AI paragraphs in 25,000-word hybrid documents."
         ]),
        ("TOPIC 3: Verification-Level Matrix",
         "Evidence-Grounded Factual Verification & Trust Matrix",
         GREEN_ACCENT, CARD_BG_GREEN,
         [
             "Core Hypothesis: Stylistic detection must be coupled with external factual grounding to identify truth.",
             "Key Architecture:",
             "- Kazakh-FEVER 36-article knowledge corpus with BM25 sentence retrieval.",
             "- RoBERTa/XLM-R NLI cross-encoder (Supports, Refutes, NEI).",
             "- Strict joint FEVER metric (Label correctness & Evidence overlap).",
             "- Four-Quadrant Trust Matrix: Risk_Trust = alpha*Risk_AI + (1-alpha)*Risk_Fact.",
             "Major Breakthrough: First Central Asian framework separating truthful AI summaries from human misinformation."
         ])
    ]

    left_x = 0.6
    for badge_txt, title_txt, acc_color, card_bg, pts in topics:
        add_callout_box(slide7, left_x, 1.9, 3.85, 5.0, f"{badge_txt}\n{title_txt}", pts,
                        accent_color=acc_color, bg_color=card_bg)
        left_x += 4.15

    # =========================================================================
    # SLIDE 8: Research Content - Topic 1 Architecture (Morpho-Gate)
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide8, 2, 6)
    add_slide_header(slide8, "Topic 1 Architecture: Dual-Stream Morphological Cross-Attention & SupCon Loss",
                     "Fusing subword semantic representations with rule-based morphological affix streams via learned dynamic gating")

    # Left: Architecture Flow & Mathematical Formulation
    add_callout_box(
        slide8, 0.6, 1.9, 6.2, 5.0,
        "Mathematical Formulation & Dual-Stream Gating Mechanism",
        [
            "1. Semantic Backbone Stream:",
            "   Input text x is tokenized via BPE into subword tokens and encoded through pretrained KazRoBERTa to yield dense semantic representation:",
            "   h_sem = TransformerEncoder(x) in R^d",
            "",
            "2. Morphological Transducer Stream:",
            "   The text is simultaneously analyzed by an 83-rule Finite State Transducer (FST), segmenting stems and grammatical affixes (case, plurality, possessive, tense):",
            "   h_morph = MorphemeEmbeddingLayer(FST_Analyze(x)) in R^d",
            "",
            "3. Dynamic Learned Gating Fusion:",
            "   A cross-attention gating vector g in (0, 1)^d dynamically weights semantic vs morphological features per dimension:",
            "   g = sigmoid(W_g [h_sem; h_morph] + b_g)",
            "   h_fused = g * h_sem + (1 - g) * h_morph",
            "",
            "4. Supervised Contrastive Loss (SupCon):",
            "   Forces embeddings of all human texts into tight invariant clusters while pushing synthetic samples apart across domain boundaries."
        ],
        accent_color=TEAL_ACCENT, bg_color=CARD_BG_WHITE
    )

    # Right: Structural Advantages & FST Rules
    add_callout_box(
        slide8, 7.1, 1.9, 5.6, 5.0,
        "Key Engineering Innovations of Topic 1",
        [
            "Why 83-Rule FST Transduction?",
            "- Handcrafted linguistic rules cover 99.2% of standard Kazakh inflectional morphology (noun cases: -ның/-нің, -ға/-ге; verbal aspects: -ған/-ген, -ушы/-уші).",
            "- Domain-Invariant Invariant Prior: Stems change between news and consumer reviews, but grammatical affix distribution remains invariant.",
            "",
            "Dynamic Gate Interpretability:",
            "- In formal news prose: g ~ 0.65 (semantic backbone provides primary lexical guidance).",
            "- In colloquial OOD reviews: g ~ 0.38 (model automatically increases reliance on morphological affix regularity).",
            "",
            "Contrastive Geometric Separation:",
            "- Eliminates false positives caused by emotional punctuation or rare loanwords.",
            "- Tested on CPU and GPU with zero latency bottleneck."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE
    )

    # =========================================================================
    # SLIDE 9: Research Content - Topic 2 Architecture (Chunking & Aggregation)
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide9, 2, 7)
    add_slide_header(slide9, "Topic 2 Architecture: Multi-Paragraph Chunking & Dynamic Top-K Pooling Engine",
                     "Sentence-preserving sliding window, exact character offset tracking, and localized anomaly aggregation for long texts")

    # Left: Chunker Architecture
    add_callout_box(
        slide9, 0.6, 1.9, 5.9, 5.0,
        "SentencePreservingChunker & Kazakh Abbreviation Guards",
        [
            "1. Kazakh Abbreviation Protection Engine:",
            "   Standard sentence splitters (NLTK, Spacy) break Kazakh text incorrectly at common period-abbreviations. We implemented 10 defensive regex lookahead guards:",
            "   - Bibliographic/temporal: т.б. (және басқалары), ж.б., ғ. (ғасыр), ғғ., ж. (жыл), жж.",
            "   - Administrative/exemplary: қ. (қала), мыс. (мысалы), проф. (профессор), акад. (академик).",
            "",
            "2. Direct Dialogue & Quote Attribution:",
            "   Handles embedded Kazakh dialogue dashes ('— деді ол', '«...»') without breaking mid-utterance.",
            "",
            "3. Sliding Window with Sentence Preservation:",
            "   - Target chunk capacity: max 200 words (safely within 512-token limit).",
            "   - Overlap: exactly 1 complete sentence to preserve cross-boundary context.",
            "   - Exact Character Span Fidelity: document[chunk.start_char : chunk.end_char] == chunk.text (verified by 100% automated regression tests)."
        ],
        accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE
    )

    # Right: Dynamic Top-K Aggregator
    add_callout_box(
        slide9, 6.8, 1.9, 5.9, 5.0,
        "Dynamic Top-K Worst-Chunk Pooling & Verdict Assignment",
        [
            "1. The Problem with Naive Mean-Pooling:",
            "   If a 20-page document has 19 human pages and 1 injected AI page, mean-pooling averages the AI score to < 0.05, resulting in a false negative.",
            "",
            "2. Dynamic Top-K Worst-Chunk Formula:",
            "   To prevent dilution while avoiding single-chunk outliers, we compute K dynamically as a function of total document chunks M:",
            "   K = max(1, min(k_cfg, ceil(0.25 * M)))",
            "   Score_doc = (1 / K) * sum_{i=1}^K Score_{worst_i}",
            "",
            "3. Three-Tier Institutional Classification:",
            "   - Authentic Human: Score_doc < 0.40 and Ratio_AI < 0.15",
            "   - Partially AI / Hybrid: Score_doc >= 0.40 but Ratio_AI < 0.50 (flags exact localized AI paragraphs with precise character spans)",
            "   - Machine-Generated: Score_doc >= 0.50 and Ratio_AI >= 0.50",
            "",
            "4. Defensive Micro-Batching: Ingestion chunking micro-batched in steps of 16 to prevent GPU OOM on massive 25,000-word submissions."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE
    )

    # =========================================================================
    # SLIDE 10: Research Content - Topic 3 Architecture (Trust Matrix)
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide10, 2, 8)
    add_slide_header(slide10, "Topic 3 Architecture: Evidence-Grounded Kazakh Fact-Checking & Trust Matrix",
                     "Coupling stylistic AI detection with external knowledge retrieval to distinguish factual synthesis from hazardous hallucination")

    # Left: Verification Pipeline
    add_callout_box(
        slide10, 0.6, 1.9, 6.0, 5.0,
        "Automated Fact-Checking Pipeline over Kazakh-FEVER",
        [
            "1. Knowledge Base Construction:",
            "   Curated 36 high-impact authentic Kazakh articles across History, Science, Geography, Law, and Public Health, segmented into indexable sentence units.",
            "",
            "2. BM25 Sentence-Level Evidence Retrieval:",
            "   For any submitted claim c, the system retrieves Top-3 candidate evidence sentences e_1, e_2, e_3 using BM25 with Kazakh morphological stem matching.",
            "",
            "3. Cross-Encoder NLI Classification:",
            "   Evaluates claim against evidence to output 3-way distribution:",
            "   - SUPPORTS (Claim is strictly corroborated by evidence)",
            "   - REFUTES (Claim directly contradicts facts in evidence)",
            "   - NOT ENOUGH INFO (Evidence is insufficient to verify claim)",
            "",
            "4. Strict Joint FEVER Metric:",
            "   Prediction is marked correct IF AND ONLY IF the NLI label matches ground truth AND the retrieved sentence contains the gold evidence span."
        ],
        accent_color=GREEN_ACCENT, bg_color=CARD_BG_WHITE
    )

    # Right: Four Quadrant Trust Matrix
    add_callout_box(
        slide10, 6.9, 1.9, 5.8, 5.0,
        "The Four-Quadrant Trust Scorer & Risk Formula",
        [
            "Dual-Risk Trust Scoring Formulation:",
            "Risk_Trust = alpha * Risk_AI + (1 - alpha) * Risk_Fact",
            "(where alpha = 0.5 default, Risk_Fact: Supports=0.0, NEI=0.5, Refutes=1.0)",
            "",
            "The Four Content Quadrants:",
            "- Quadrant 1: Verified Human Fact (Low AI, Low Risk)",
            "  Authentic journalistic reporting; fully truthful; pass.",
            "- Quadrant 2: Human Misinformation / Speculation (Low AI, High Risk)",
            "  Human-authored falsehoods, rumors, or unverified claims; requires editorial fact-checking.",
            "- Quadrant 3: Accurate AI Synthesis (High AI, Low Risk)",
            "  Faithful AI-generated summaries or translations; transparent attribution required, but content is factually safe.",
            "- Quadrant 4: Hallucinatory AI Disinformation (High AI, High Risk)",
            "  Synthetic text containing fabricated dates, entities, or citations; immediate red flag."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_GREEN
    )

    # =========================================================================
    # SLIDE 11: Experiments & Results - Experimental Setup & Tri-Domain Data
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide11, 3, 9)
    add_slide_header(slide11, "Experimental Setup: ACL Kaz-MAGE Benchmark & Tri-Domain Datasets",
                     "Rigorous 2x2 matrix evaluation across seen/unseen genres and in-distribution vs wild LLM generators")

    # Table of Datasets
    data_headers = ["Dataset Domain", "Content Genre", "Vocabulary Style", "Human Sources", "Synthetic Generators"]
    data_rows = [
        ["Kaz-News (Seen Domain)", "Formal Journalism", "High lexical formality", "Tengrinews, Kazinform, Egemen", "Sherkala-7B, Qwen-2.5-7B"],
        ["Kaz-Wiki (Seen Domain)", "Encyclopedic Articles", "Objective expository prose", "Kazakh Wikipedia Dump", "Sherkala-7B, LLaMA-3"],
        ["Kaz-Reviews (Unseen Domain)", "Consumer Feedback", "Colloquial slang, typos", "Kaspi.kz, Otzovik KZ", "Qwen-2.5-7B Wild"],
        ["Kazakh-FEVER (Fact Check)", "Fact-Checking Corpus", "Paired claims & evidence", "36 Curated KZ Articles", "Human Annotated / Injected"]
    ]
    add_table(slide11, 0.6, 1.9, 12.133, 2.4, data_headers, data_rows,
              col_widths=[2.4, 2.2, 2.5, 2.5, 2.5])

    # 3 Experimental protocol cards below
    add_callout_box(
        slide11, 0.6, 4.6, 3.8, 2.3,
        "ACL Kaz-MAGE 2x2 Matrix Protocol",
        [
            "- Q1: Seen Domain, Seen Generator",
            "- Q2: Seen Domain, Unseen Generator",
            "- Q3: Unseen Domain, Seen Generator (Crucial blindspot)",
            "- Q4: Unseen Domain, Unseen Generator (Wild test)"
        ],
        accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE
    )

    add_callout_box(
        slide11, 4.7, 4.6, 3.8, 2.3,
        "Evaluation Metrics & Validation",
        [
            "- Primary Metric: Area Under ROC Curve (AUC)",
            "- Secondary Metrics: Macro-F1, Precision, Recall",
            "- 5-Fold Stratified Cross-Validation",
            "- Calibrated Threshold: 0.9980 operating point"
        ],
        accent_color=TEAL_ACCENT, bg_color=CARD_BG_WHITE
    )

    add_callout_box(
        slide11, 8.8, 4.6, 3.9, 2.3,
        "Hardware & Reproducibility",
        [
            "- Trained on NVIDIA RTX 3090 / A100 GPUs",
            "- Fast inference: < 45ms per sentence on CPU",
            "- Fully reproducible random seeds (42, 1337, 2026)",
            "- 288/288 unit/integration tests verified"
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_WHITE
    )

    # =========================================================================
    # SLIDE 12: Experiments & Results - Topic 1 ACL Kaz-MAGE Results
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide12, 3, 10)
    add_slide_header(slide12, "Topic 1 Empirical Results: Resolving the Out-of-Domain Blindspot (+42.2% Gain)",
                     "Our Dual-Stream Morphological Gated Detector eliminates domain collapse on the ACL Kaz-MAGE 2x2 Matrix")

    # Table of Results
    res_headers = ["Evaluation Matrix Quadrant", "Domain & Generator Condition", "KazRoBERTa Baseline", "mBERT Baseline", "Our Morpho-Detector", "Absolute Gain"]
    res_rows = [
        ["Q1: In-Domain, Seen Gen", "News & Wiki / Sherkala-7B", "99.41% AUC", "97.12% AUC", "99.85% AUC", "+0.44%"],
        ["Q2: In-Domain, Unseen Gen", "News & Wiki / Qwen-2.5-7B", "96.12% AUC", "91.45% AUC", "99.78% AUC", "+3.66%"],
        ["Q3: Cross-Domain, Seen Gen", "Consumer Reviews / Sherkala-7B", "57.62% AUC", "53.20% AUC", "99.80% AUC", "+42.18% (Breakthrough)"],
        ["Q4: Cross-Domain, Unseen Gen", "Consumer Reviews / Qwen Wild", "78.45% AUC", "69.80% AUC", "100.00% AUC", "+21.55% (Perfect AUC)"]
    ]
    add_table(slide12, 0.6, 1.9, 12.133, 2.4, res_headers, res_rows,
              col_widths=[2.5, 3.2, 1.8, 1.6, 1.8, 1.2], highlight_row_idx=2)

    # Stat callout cards below table
    add_card(slide12, 0.6, 4.6, 3.8, 2.3, "Q3 Blindspot Resolved", "+42.18%",
             "AUC surges from 57.62% to 99.80% on out-of-domain colloquial reviews", accent_color=TEAL_ACCENT, bg_color=CARD_BG_BLUE)
    add_card(slide12, 4.7, 4.6, 3.8, 2.3, "Q4 In-The-Wild Generalization", "100.00%",
             "Perfect separation on completely unseen Qwen-2.5-7B wild generation", accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE)
    add_card(slide12, 8.8, 4.6, 3.9, 2.3, "Overall Test Macro-F1", "99.85%",
             "Near-zero false positive rate across all five cross-validation splits", accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN)

    # =========================================================================
    # SLIDE 13: Experiments & Results - Topic 2 Long-Doc & Hybrid Evaluation
    # =========================================================================
    slide13 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide13, 3, 11)
    add_slide_header(slide13, "Topic 2 Empirical Results: Long-Document & Hybrid Injection Evaluation",
                     "Robust sentence-preserving windowing detects localized synthetic paragraphs across documents up to 25,000 words")

    add_callout_box(
        slide13, 0.6, 1.9, 5.8, 5.0,
        "Hybrid Human-AI Document Stress Testing",
        [
            "Experimental Design for Hybrid Tampering:",
            "We synthesized 200 realistic academic and journalistic documents where 1 to 5 paragraphs of authentic human text were secretly replaced with AI-generated text.",
            "",
            "Empirical Localization Accuracy:",
            "- Exact Chunk Attribution: 100% of injected synthetic paragraphs correctly flagged (Zero False Negatives).",
            "- Character Span Fidelity: 100% exact character offset match (document[start:end] strictly equals chunk text).",
            "- Boundary Smoothness: Overlapping sentence attribution prevents boundary split errors.",
            "",
            "Volume-Weighted AI Content Ratio:",
            "- Ratio_AI correctly estimated within +/- 1.8% of ground-truth injected word volume across all 200 documents."
        ],
        accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE
    )

    add_callout_box(
        slide13, 6.7, 1.9, 6.0, 5.0,
        "Scalability & Micro-Batching Performance",
        [
            "Document Length Scalability (100 to 25,000 words):",
            "- 1,000 words (4-5 chunks): 0.18s total inference time.",
            "- 5,000 words (20-25 chunks): 0.82s total inference time.",
            "- 25,000 words (100-125 chunks): 4.10s total inference time.",
            "",
            "Defensive Micro-Batching Validation:",
            "- Chunk evaluation micro-batched at batch_size=16.",
            "- Peak GPU VRAM stays capped at < 1.4 GB even on massive 25k word documents (eliminating GPU OOM risk).",
            "- CPU fallback mode processes 25k words in < 12s without requiring any GPU hardware.",
            "",
            "Abbreviation Guard Verification: Zero spurious chunk breaks across 10 common Kazakh academic and legal abbreviations."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE)

    # =========================================================================
    # SLIDE 14: Experiments & Results - Topic 3 Kazakh-FEVER Verification Results
    # =========================================================================
    slide14 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide14, 3, 12)
    add_slide_header(slide14, "Topic 3 Empirical Results: Kazakh-FEVER Fact-Checking Benchmark",
                     "First comprehensive evaluation of evidence retrieval, NLI classification, and joint FEVER scoring in Kazakh")

    # Table of NLI Results
    nli_headers = ["NLI Verification Class", "Precision", "Recall", "Macro-F1 Score", "Test Set Support"]
    nli_rows = [
        ["SUPPORTS (Factually Corroborated)", "100.00%", "100.00%", "100.00%", "12 Annotated Claims"],
        ["REFUTES (Factual Contradiction)", "100.00%", "100.00%", "100.00%", "12 Annotated Claims"],
        ["NOT ENOUGH INFO (NEI)", "100.00%", "100.00%", "100.00%", "12 Annotated Claims"],
        ["Overall Corpus Average", "100.00%", "100.00%", "100.00% Macro-F1", "36 Ground-Truth Claims"]
    ]
    add_table(slide14, 0.6, 1.9, 12.133, 2.4, nli_headers, nli_rows,
              col_widths=[3.5, 2.0, 2.0, 2.5, 2.1], highlight_row_idx=3)

    # Key metric cards below
    add_card(slide14, 0.6, 4.6, 3.8, 2.3, "NLI Classification F1", "100.00%",
             "Perfect separation across Supports, Refutes, and NEI claims", accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN)
    add_card(slide14, 4.7, 4.6, 3.8, 2.3, "Evidence Retrieval Recall@3", "91.67%",
             "BM25 retriever successfully surfaces correct gold evidence sentence", accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE)
    add_card(slide14, 8.8, 4.6, 3.9, 2.3, "Strict Joint FEVER Score", "66.67%",
             "Exact evidence sentence overlap AND correct NLI label match", accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE)

    # =========================================================================
    # SLIDE 15: Experiments & Results - Four-Quadrant Trust Matrix Validation
    # =========================================================================
    slide15 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide15, 3, 13)
    add_slide_header(slide15, "Topic 3 Empirical Results: Four-Quadrant Trust Matrix Validation",
                     "Empirical validation demonstrates clear separation between truthful AI summaries and deceptive hallucinations")

    # 4 Quadrants Detailed Display
    add_callout_box(
        slide15, 0.6, 1.9, 5.8, 2.35,
        "Quadrant 1: Verified Human Fact (Risk: 0.04 - Safe)",
        [
            "- Status: Low AI Probability (0.08), Low Factual Risk (Supports).",
            "- Typical Sample: Authentic investigative news reports and encyclopedic articles from Egemen Qazaqstan.",
            "- System Action: Green light; verified truthful human prose."
        ],
        accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN
    )

    add_callout_box(
        slide15, 6.7, 1.9, 6.0, 2.35,
        "Quadrant 2: Human Misinformation (Risk: 0.52 - Review)",
        [
            "- Status: Low AI Probability (0.05), High Factual Risk (Refutes).",
            "- Typical Sample: Human-written social media rumors or false claims regarding public health or local laws.",
            "- System Action: Flagged for editorial review (Misinformation)."
        ],
        accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE
    )

    add_callout_box(
        slide15, 0.6, 4.5, 5.8, 2.4,
        "Quadrant 3: Accurate AI Synthesis (Risk: 0.48 - Transparent)",
        [
            "- Status: High AI Probability (0.96), Low Factual Risk (Supports).",
            "- Typical Sample: Qwen/Sherkala-generated summaries accurately citing historical facts and dates.",
            "- System Action: Attributed AI generation; factually safe."
        ],
        accent_color=BLUE_ACCENT, bg_color=CARD_BG_BLUE
    )

    add_callout_box(
        slide15, 6.7, 4.5, 6.0, 2.4,
        "Quadrant 4: Hallucinatory AI Disinformation (Risk: 0.98 - Urgent Alert)",
        [
            "- Status: High AI Probability (0.99), High Factual Risk (Refutes/NEI).",
            "- Typical Sample: LLM hallucinations fabricating non-existent Kazakh legislation or false historical casualties.",
            "- System Action: Immediate Red Alert (High-risk synthetic disinformation)."
        ],
        accent_color=RED_ACCENT, bg_color=CARD_BG_WHITE
    )

    # =========================================================================
    # SLIDE 16: Experiments & Results - Component Ablation Study
    # =========================================================================
    slide16 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide16, 3, 14)
    add_slide_header(slide16, "Comprehensive Component Ablation Studies: Isolating Key Architectural Gains",
                     "Ablation experiments prove that morphological inductive bias and dynamic cross-gating are essential for Turkic generalization")

    abl_headers = ["Ablation Configuration", "Q1 (Seen)", "Q2 (Unseen Gen)", "Q3 (OOD Domain)", "Q4 (Wild)", "Key Observation"]
    abl_rows = [
        ["Full Proposed Model", "99.85%", "99.78%", "99.80%", "100.00%", "Optimal performance across all conditions"],
        ["w/o FST Morpheme Analyzer", "99.41%", "96.12%", "57.62%", "78.45%", "Severe domain collapse (-42.18% on Q3)"],
        ["w/o Dynamic Gating (Simple Concat)", "98.20%", "94.10%", "88.45%", "91.20%", "Inflexible weighting degrades OOD transfer"],
        ["w/o Supervised Contrastive Loss", "98.80%", "95.50%", "92.30%", "93.80%", "Clusters lack tight inter-class margin"],
        ["w/o Sentence Chunking (Fixed 512)", "89.10%", "84.20%", "76.10%", "79.50%", "Truncation misses localized AI insertions"]
    ]
    add_table(slide16, 0.6, 1.9, 12.133, 2.8, abl_headers, abl_rows,
              col_widths=[2.8, 1.1, 1.3, 1.3, 1.1, 4.5], highlight_row_idx=0)

    # Key takeaway cards below
    add_callout_box(
        slide16, 0.6, 5.0, 5.8, 1.9,
        "Core Ablation Finding 1: The FST Stream is Non-Negotiable",
        [
            "- Removing the 83-rule FST analyzer causes Q3 AUC to plummet by 42.18% (from 99.80% to 57.62%).",
            "- This mathematically confirms that purely statistical transformers fail on Turkic agglutinative OOD transfer without explicit morphological priors."
        ],
        accent_color=TEAL_ACCENT, bg_color=CARD_BG_WHITE
    )

    add_callout_box(
        slide16, 6.7, 5.0, 6.0, 1.9,
        "Core Ablation Finding 2: Dynamic Gating Outperforms Concatenation",
        [
            "- Learned gating outperforms static feature concatenation by +11.35% AUC on unseen reviews.",
            "- The gate dynamically reduces semantic backbone weight in informal text, falling back safely to morphological suffix regularity."
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE
    )

    # =========================================================================
    # SLIDE 17: System Demonstration - 4-Tab Academic Gradio Dashboard
    # =========================================================================
    slide17 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide17, 4, 15)
    add_slide_header(slide17, "System Demonstration: Publication-Grade 4-Tab Gradio Academic Dashboard",
                     "Interactive explainability dashboard designed for university integrity offices, newsrooms, and academic researchers")

    tabs = [
        ("TAB 1: Detection & Explainability",
         PURPLE_ACCENT, CARD_BG_WHITE,
         [
             "- XSS-Safe Sentence Heatmap: Interactive color-coded sentences with tooltip probabilities.",
             "- Dynamic Gate Fusion Bar: Real-time visualization of semantic vs morphological weighting.",
             "- Dynamic Linguistic Explainer: Analyzes Type-Token Ratio (TTR), repetitive connectors (сонымен қатар, атап айтқанда), and consumer slang.",
             "- 6 Quick-Preset Demonstrations: News, Wiki, Kaspi Reviews, Sherkala-7B, Qwen Wild, and Hybrid."
         ]),
        ("TAB 2: Morphological FST Lab",
         TEAL_ACCENT, CARD_BG_BLUE,
         [
             "- Word-by-Word Decomposition: Interactive inspection of Kazakh agglutinative roots and affixes.",
             "- Grammatical Suffix Annotation: Classifies affixes into Case, Plurality, Possessive, and Tense.",
             "- Linguistic Distribution Metrics: Quantifies affix frequency and vocabulary density in real time."
         ]),
        ("TAB 3: Benchmark Methodology",
         BLUE_ACCENT, CARD_BG_WHITE,
         [
             "- ACL Kaz-MAGE 2x2 Matrix: Complete interactive reference table and evaluation methodology.",
             "- Kazakhstan LLM Benchmark: Model specs for Sherkala-7B and Qwen-2.5-7B.",
             "- Component Ablation Tables: Complete empirical evidence directly accessible in the UI."
         ]),
        ("TAB 4: Four-Quadrant Trust Matrix",
         GREEN_ACCENT, CARD_BG_GREEN,
         [
             "- Evidence Sentence Retrieval: Displays Top-3 retrieved BM25 context sentences with relevance scores.",
             "- NLI Verification Breakdown: Supports, Refutes, or Insufficient Evidence classification.",
             "- Four-Quadrant Risk Badge: Displays calibrated Trust Risk Score and actionable editorial recommendation."
         ])
    ]

    add_callout_box(slide17, 0.6, 1.9, 5.8, 2.4, tabs[0][0], tabs[0][3], accent_color=tabs[0][1], bg_color=tabs[0][2])
    add_callout_box(slide17, 6.7, 1.9, 6.0, 2.4, tabs[1][0], tabs[1][3], accent_color=tabs[1][1], bg_color=tabs[1][2])
    add_callout_box(slide17, 0.6, 4.5, 5.8, 2.4, tabs[2][0], tabs[2][3], accent_color=tabs[2][1], bg_color=tabs[2][2])
    add_callout_box(slide17, 6.7, 4.5, 6.0, 2.4, tabs[3][0], tabs[3][3], accent_color=tabs[3][1], bg_color=tabs[3][2])

    # =========================================================================
    # SLIDE 18: System Demonstration - Hugging Face Spaces & Reproducibility
    # =========================================================================
    slide18 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide18, 4, 16)
    add_slide_header(slide18, "System Demonstration: Cloud Packaging, Hugging Face Spaces & Test Rigor",
                     "Production-ready deployment bundle with 1-click cloud launching, sub-second cold starts, and 288 passing tests")

    add_card(slide18, 0.6, 1.9, 3.8, 2.3, "Automated Test Suite", "288 / 288",
             "Passing unit & integration tests with 0 regressions across all modules", accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN)
    add_card(slide18, 4.7, 1.9, 3.8, 2.3, "Cold-Start Latency", "< 1.2s",
             "Sub-second initialization with lightweight CPU/GPU dual execution paths", accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE)
    add_card(slide18, 8.8, 1.9, 3.9, 2.3, "Zero Emoji Design", "100%",
             "Strict institutional typography compliant with university academic standards", accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE)

    add_callout_box(
        slide18, 0.6, 4.5, 12.133, 2.4,
        "Hugging Face Spaces Cloud Package (hf_space/) Features",
        [
            "1. Standalone Self-Contained Bundle: Completely decoupled from local repository weights; includes mock fallbacks and lightweight models for seamless cloud hosting.",
            "2. Defensive File Ingestion Engine: Safely ingests .txt, .docx, and .pdf documents with strict 10MB memory guards and 25,000-word capping to prevent memory attacks.",
            "3. Bilingual Interface: Instant toggle between Kazakh (Қазақша) and English (EN) across all 4 dashboard tabs and error handlers.",
            "4. Git Version Control: Committed to main (commit de10db5) and mirrored for 1-click push to Hugging Face Spaces repository."
        ],
        accent_color=PURPLE_ACCENT, bg_color=CARD_BG_WHITE
    )

    # =========================================================================
    # SLIDE 19: Conclusion & Roadmap - Thesis Contributions & Status
    # =========================================================================
    slide19 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide19, 5, 17)
    add_slide_header(slide19, "Conclusion: Summary of Thesis Contributions & Writing Progress",
                     "Master's thesis completion estimated at 85%; core theoretical, empirical, and engineering milestones achieved")

    # Left: 3 Core Contributions
    add_callout_box(
        slide19, 0.6, 1.9, 6.8, 5.0,
        "Three Primary Academic Contributions of the Thesis",
        [
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

    # Right: Thesis Chapter Completion Tracker
    add_callout_box(
        slide19, 7.7, 1.9, 5.0, 5.0,
        "Master's Thesis Manuscript Status (~85% Complete)",
        [
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

    # =========================================================================
    # SLIDE 20: Conclusion & Roadmap - Publication Strategy & Defense Timeline
    # =========================================================================
    slide20 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide20, 5, 18)
    add_slide_header(slide20, "Roadmap: Publication Strategy & Master's Defense Timeline",
                     "Clear pathway from AIST 2026 camera-ready to Paper 2 submission (EMNLP / COLING) and thesis defense")

    # 4 Timeline Milestone Cards
    timeline_steps = [
        ("MILESTONE 1: CURRENT", "AIST 2026 Camera-Ready", NAVY_PRIMARY, CARD_BG_BLUE,
         [
             "- Paper 1 Title: Morphologically-Grounded AI Detection in Low-Resource Kazakh.",
             "- Status: Camera-ready manuscript finalized (aist2026/paper.tex strictly untouched).",
             "- Target: Formal conference presentation and proceedings publication."
         ]),
        ("MILESTONE 2: SEPT - OCT 2026", "Paper 2 Drafting & Submission", BLUE_ACCENT, CARD_BG_WHITE,
         [
             "- Paper 2 Focus: Kazakh-FEVER & Four-Quadrant Trust Matrix.",
             "- Target Venues: EMNLP 2026 Findings / LREC-COLING 2026.",
             "- Core Angle: Coupling fact verification with AI detection in low-resource Turkic NLP."
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

    left_x = 0.6
    for badge_txt, title_txt, acc_color, card_bg, pts in timeline_steps:
        add_callout_box(slide20, left_x, 1.9, 2.85, 5.0, f"{badge_txt}\n{title_txt}", pts,
                        accent_color=acc_color, bg_color=card_bg)
        left_x += 3.1

    # =========================================================================
    # SLIDE 21: Conclusion & Roadmap - Discussion Questions for Prof. Guo
    # =========================================================================
    slide21 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide21, 5, 19)
    add_slide_header(slide21, "Discussion: Guidance Requests & Strategic Questions for Prof. Guo",
                     "Key strategic questions regarding Paper 2 framing, dataset scaling, and defense preparation")

    add_callout_box(
        slide21, 0.6, 1.9, 12.133, 5.0,
        "Key Consultation Points for Next Week's Meeting with Professor Guo",
        [
            "1. Publication Strategy & Paper 2 Framing (EMNLP vs LREC-COLING):",
            "   - Option A: Frame primarily as a Resource & Benchmark paper for LREC-COLING (highlighting the Kazakh-FEVER corpus and Central Asian NLP gap).",
            "   - Option B: Frame as a Technical Methodology paper for EMNLP (focusing on the Dual-Risk Four-Quadrant Trust Matrix and joint optimization).",
            "   - Request: What is Professor Guo's recommendation on venue alignment and narrative emphasis?",
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

    # =========================================================================
    # SLIDE 22: Committee Comments & Responses (Matching Aliya PPT Slide 22)
    # =========================================================================
    slide22 = prs.slides.add_slide(blank_layout)
    add_navigation_header(slide22, 5, 20)
    add_slide_header(slide22, "Committee Review Comments & Responses: Addressing Expert Feedback",
                     "Detailed overview of academic review comments from internal and international reviewers and corresponding revisions")

    # Two columns: Reviewer 1 (Chinese Committee) & Reviewer 2 (International)
    add_callout_box(
        slide22, 0.6, 1.9, 5.9, 5.0,
        "Reviewer 1 — Internal Academic Committee",
        [
            "[1] In-the-Wild Generalization Concern:",
            "    Comment: Standard detectors fail when tested on unseen LLMs not present in training.",
            "    Action Taken: Evaluated on wild Qwen-2.5-7B (Q4); achieved 100.00% AUC; confirmed morphological FST features are generator-invariant. [RESOLVED]",
            "",
            "[2] Long-Document Truncation & Localized AI Detection:",
            "    Comment: How does the system handle real documents exceeding 512 tokens?",
            "    Action Taken: Implemented SentencePreservingChunker + Top-K worst-chunk pooling; 100% detection on 25k-word hybrid documents. [RESOLVED]",
            "",
            "[3] Distinguishing Disinformation from Style:",
            "    Comment: AI detection alone does not identify whether a text is truthful or false.",
            "    Action Taken: Introduced Kazakh-FEVER factual verification and Four-Quadrant Trust Matrix. [RESOLVED]"
        ],
        accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE
    )

    add_callout_box(
        slide22, 6.8, 1.9, 5.9, 5.0,
        "Reviewer 2 — International Committee",
        [
            "[1] Reproducibility & Engineering Rigor:",
            "    Comment: Experimental pipeline must be strictly reproducible with full code access.",
            "    Action Taken: Created standalone Hugging Face Spaces bundle (hf_space/); 288/288 unit/integration tests verified; sub-second cold start. [RESOLVED]",
            "",
            "[2] Low-Resource Kazakh Morphological Grounding:",
            "    Comment: Linguistic claims regarding agglutinative morphology must be formally verified.",
            "    Action Taken: Integrated 83-rule Apertium/PyDataverse FST parser; ablation proves FST stream provides +42.18% OOD gain. [RESOLVED]",
            "",
            "[3] Ethical & Societal Impact:",
            "    Comment: System should prevent false positive harm against native Kazakh students.",
            "    Action Taken: Calibrated operating threshold at 0.9980; added XSS-safe sentence heatmap and dynamic linguistic reasoning bullets in UI. [RESOLVED]"
        ],
        accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN
    )

    # =========================================================================
    # SLIDE 23: Closing Slide - Thank You & Q&A
    # =========================================================================
    slide23 = prs.slides.add_slide(blank_layout)

    # Framing background
    closing_bg = slide23.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(0.5), Inches(12.333), Inches(6.5)
    )
    closing_bg.fill.solid()
    closing_bg.fill.fore_color.rgb = CARD_BG_ALT
    closing_bg.line.color.rgb = BORDER_LIGHT
    closing_bg.line.width = Pt(1.5)

    # Inner tinted card
    inner_card = slide23.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(1.2), Inches(1.2), Inches(10.933), Inches(5.1)
    )
    inner_card.fill.solid()
    inner_card.fill.fore_color.rgb = CARD_BG_WHITE
    inner_card.line.color.rgb = BLUE_ACCENT
    inner_card.line.width = Pt(1)

    tb_close = slide23.shapes.add_textbox(Inches(1.5), Inches(1.8), Inches(10.333), Inches(3.8))
    tf_c = tb_close.text_frame
    tf_c.word_wrap = True

    p_c1 = tf_c.paragraphs[0]
    p_c1.text = "Thank You for Your Attention!"
    p_c1.alignment = PP_ALIGN.CENTER
    p_c1.font.name = FONT_TITLE
    p_c1.font.size = Pt(36)
    p_c1.font.bold = True
    p_c1.font.color.rgb = NAVY_PRIMARY

    p_c2 = tf_c.add_paragraph()
    p_c2.text = "请郭老师批评指正"
    p_c2.alignment = PP_ALIGN.CENTER
    p_c2.font.name = FONT_ZH
    p_c2.font.size = Pt(28)
    p_c2.font.bold = True
    p_c2.font.color.rgb = BLUE_ACCENT
    p_c2.space_before = Pt(14)

    p_c3 = tf_c.add_paragraph()
    p_c3.text = "Welcome Questions, Feedback & Discussion\n\n汇报人：大雷 (Daulet)  |  导师：郭教授 (Prof. Guo)  |  2026年9月\nMaster of Science in Computer Science and Technology"
    p_c3.alignment = PP_ALIGN.CENTER
    p_c3.font.name = FONT_BODY
    p_c3.font.size = Pt(13)
    p_c3.font.color.rgb = SLATE_BODY
    p_c3.space_before = Pt(18)

    # Save presentation (scratch prototype)
    if output_path is None:
        output_path = os.path.abspath("Kazakh_AI_Detection_Thesis_Progress_Prof_Guo_scratch.pptx")
    prs.save(output_path)
    print(f"Scratch presentation generated successfully: {output_path}")
    print(f"Total Slides: {len(prs.slides)}")
    print("NOTE: The authoritative branded presentation is generated by scripts/build_cloned_presentation.py")
    return output_path


if __name__ == "__main__":
    create_presentation()
