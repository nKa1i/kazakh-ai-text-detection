# -*- coding: utf-8 -*-
"""
Master's Thesis Proposal Defense Presentation Builder (5-Element Structure).
Clones baseline presentation 'Aliya PPT.pptx' directly, preserves native institutional
branding (KazNU logo, NPU logo group, watermark, and Layout 7 NPU pentagon badge),
excises commercial watermarks ('合作QQ： 243001978'), installs a 5-element running
section tracker ('研究背景', '相关工作', '研究内容', '研究创新', '进展安排'),
and populates all 21 slides for candidate Daulet Anekesh's October 29 Proposal Defense.

Presenter: 大雷 (DAULET ANEKESH, Student No.: 2025291043)
Supervisor: 郭斌 教授 (Prof. Bin Guo)
Date of Defense: 2026年10月29日 (October 29, 2026)
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

# 5-Element Sections matching Professor Guo's laboratory standard
SECTION_NAMES_ZH = ["研究背景", "相关工作", "研究内容", "研究创新", "进展安排"]
SECTION_NAMES_EN = ["Background", "Related Work", "Research Content", "Innovations", "Schedule"]

# 5 Tracker slots spanning left = 1.25 to 11.8 inches
SLOT_COORDS = [
    (Inches(1.25), Inches(2.05)),  # Section 1
    (Inches(3.35), Inches(2.05)),  # Section 2
    (Inches(5.45), Inches(2.05)),  # Section 3
    (Inches(7.55), Inches(2.05)),  # Section 4
    (Inches(9.65), Inches(2.15)),  # Section 5
]

# Section index mapping for content slides (0-indexed in 21-slide deck: slides 2 through 19)
# Slides 3-4 (indices 2, 3) -> Sec 0 (研究背景)
# Slides 5-6 (indices 4, 5) -> Sec 1 (相关工作)
# Slides 7-10 (indices 6, 7, 8, 9) -> Sec 2 (研究内容)
# Slides 11-12 (indices 10, 11) -> Sec 3 (研究创新)
# Slides 13-20 (indices 12 to 19) -> Sec 4 (进展安排)
SLIDE_SECTION_MAP = {
    2: 0, 3: 0,
    4: 1, 5: 1,
    6: 2, 7: 2, 8: 2, 9: 2,
    10: 3, 11: 3,
    12: 4, 13: 4, 14: 4, 15: 4, 16: 4, 17: 4, 18: 4, 19: 4
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
    p.font.name = FONT_ZH
    p.font.size = Pt(16.5)
    p.font.bold = True
    p.font.color.rgb = NAVY_PRIMARY

    if subtitle_text:
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.name = FONT_ZH
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
    p.font.name = FONT_ZH
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
    p_tit.font.name = FONT_ZH
    p_tit.font.size = Pt(10.5)
    p_tit.font.bold = True
    p_tit.font.color.rgb = SLATE_TITLE
    p_tit.space_before = Pt(2)

    if subtitle:
        p_sub = tf.add_paragraph()
        p_sub.text = subtitle
        p_sub.font.name = FONT_ZH
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
    p_head.font.name = FONT_ZH
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
        p.font.name = FONT_ZH
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
        p.font.name = FONT_ZH
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
            p.font.name = FONT_ZH
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
            p1.font.size = Pt(17)
            p1.font.bold = True
            p1.font.color.rgb = NAVY_PRIMARY
            p1.alignment = PP_ALIGN.LEFT

            p_sub = tf.add_paragraph()
            p_sub.text = "硕士学位论文选题报告答辩 (Master's Thesis Proposal Defense)"
            p_sub.font.name = FONT_ZH
            p_sub.font.size = Pt(13)
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
            p0.text = "汇报人：大雷 (DAULET ANEKESH, 学号: 2025291043)    导师：郭斌 教授    答辩日期：2026年10月29日"
            p0.font.name = FONT_ZH
            p0.font.size = Pt(14)
            p0.font.bold = True
            p0.font.color.rgb = NAVY_PRIMARY
            p0.alignment = PP_ALIGN.LEFT

            p1 = tf.add_paragraph()
            p1.text = "培养类别：留学研究生 | 学科专业：计算机科学与技术 | 西北工业大学计算机学院 & 哈萨克斯坦国立大学"
            p1.font.name = FONT_ZH
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

    # In baseline TOC, there are 6 item boxes. We map to 5 canonical sections.
    toc_sections = {
        "组合 1": "01. 研究背景与意义 (Research Background & Significance)",
        "组合 2": "02. 相关工作与研究现状 (Related Work & SOTA Gaps)",
        "组合 3": "03. 研究内容与技术路线 (Research Content & System Architecture)",
        "组合 12": "04. 研究创新与技术难点 (Research Innovations & Challenges)",
        "standalone": "05. 进展安排与工作基础 (Progress, Preliminary Results & Schedule)",
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
            shp.width = Inches(7.5)
            p = shp.text_frame.paragraphs[0]
            p.text = toc_sections["standalone"]
            p.font.name = FONT_ZH
            p.font.size = Pt(16)
            p.font.bold = True
            p.font.color.rgb = NAVY_PRIMARY
        elif shp.name == "组合 4":
            # Remove 6th group shape from TOC
            remove_shape(shp)


# =============================================================================
# SLIDE-BY-SLIDE CONTENT POPULATION FUNCTIONS (SLIDES 3 - 20)
# =============================================================================

def populate_slide_03(slide):
    """Slide 3: 01 研究背景 — 语言学特性与低资源AI生成文本威胁"""
    add_slide_header(
        slide,
        "研究背景：语言学形态复杂性与低资源AI生成文本安全威胁",
        "以哈萨克语为代表的黏着语形态特征使主流统计分词器严重碎裂，生成式大模型在区域传播中引发严重认知风险"
    )
    add_figure_with_caption(
        slide, "fig01_subword_vs_fst.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="图 1. 统计子词分词碎裂与 83 条 FST 规则形态解析对比示意图"
    )
    add_card(slide, left=6.65, top=2.05, width=2.95, height=2.28,
             title="词缀层叠深度", value_str="15+",
             subtitle="词根后序级联接缀，使传统统计BPE切分为无意义字节碎片",
             accent_color=TEAL_ACCENT, bg_color=CARD_BG_BLUE)
    add_card(slide, left=9.78, top=2.05, width=2.95, height=2.28,
             title="真实长文档体量", value_str="25,000",
             subtitle="真实学位论文与长篇新闻跨度，突破512截断限制",
             accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE)
    add_card(slide, left=6.65, top=4.48, width=2.95, height=2.28,
             title="三领域评测语料", value_str="10,000+",
             subtitle="涵盖新闻、维基百科及 Kaspi 电商多源跨域真实评测集",
             accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE)
    add_card(slide, left=9.78, top=4.48, width=2.95, height=2.28,
             title="语言原生基准", value_str="0 -> 1",
             subtitle="填补中亚低资源黏着语AI文本检测与事实核查基准空白",
             accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN)


def populate_slide_04(slide):
    """Slide 4: 01 研究背景 — 现有检测系统的三大致命病态分析"""
    add_slide_header(
        slide,
        "研究背景：现有AI文本检测系统在黏着语场景下的三大致命病态",
        "实证评测揭示预训练Transformer在形态学迁移、长文档截断及事实性脱钩上的根本缺陷"
    )
    add_figure_with_caption(
        slide, "fig02_domain_collapse.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="图 2. 基线 Kaz-RoBERTa 在跨领域 Kaspi 评测下的形态领域崩溃（-41.79%）"
    )
    add_callout_box(
        slide, left=6.65, top=2.05, width=6.08, height=1.50,
        title="病态一：形态学领域崩溃 (Morphological Domain Collapse)",
        points=[
            "标准 Kaz-RoBERTa 在新闻内域达 99.41% AUC，但在跨域 Kaspi 评测骤降至 57.62%（断崖式下跌 41.79%）。",
            "分类器仅拟合了特定领域编辑风格，无法识别本质生成机理与形态句法规律。"
        ],
        accent_color=RED_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=3.65, width=6.08, height=1.35,
        title="病态二：长文档截断盲区 (512-Token Truncation Bottleneck)",
        points=[
            "学术论文通常包含数万词，传统模型在 512 词元处暴力截断，丢失后文 95% 以上上下文。",
            "对长文档中段嵌入的局部机器生成段落完全失控，无法精确定位篡改边界。"
        ],
        accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=5.10, width=6.08, height=1.35,
        title="病态三：事实真实性脱钩 (Truthfulness-Agnostic Paradigm)",
        points=[
            "仅评估表层统计风格与生成概率，无法区分真实机器辅助摘要与人类虚假谣言。",
            "亟需将 AI 风格判别与外部权威证据事实核验进行正交解耦。"
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE
    )


def populate_slide_05(slide):
    """Slide 5: 02 相关工作 — SOTA检测范式对比与低资源黏着语局限"""
    add_slide_header(
        slide,
        "相关工作：主流AI文本检测范式对比与低资源黏着语局限",
        "对比统计概率曲率、数字水印、单流微调等主流方法，剖析现有工作在低资源场景下的失效机理"
    )
    headers = ["检测范式", "代表方法", "形态学感知", "长文本无损", "免API依赖", "跨域 Kaspi AUC"]
    rows = [
        ["统计曲率范式", "DetectGPT / Binoculars", "无 (依赖代理模型)", "受限 (512截断)", "依赖 (需计算对数似然)", "54.20% (近随机)"],
        ["生成水印范式", "KGW / SynthID", "无 (生成端硬约束)", "支持 (全序列)", "不适用 (无法检黑盒)", "不适用 (需私钥)"],
        ["通用单流微调", "mBERT / XLM-R", "无 (统计BPE切分)", "受限 (512截断)", "完全免依赖", "61.35% (严重坍塌)"],
        ["语言原生单流", "Kaz-RoBERTa", "无 (子词碎裂严重)", "受限 (512截断)", "完全免依赖", "57.62% (-41.79%)"],
        ["本文提出架构", "双流门控+对比融合", "深度融合 (83条FST)", "完整无损 (25,000词)", "完全免依赖", "99.80% (+42.18%)"]
    ]
    add_table(
        slide, left=0.60, top=2.05, width=12.133, height=2.65,
        headers=headers, rows=rows,
        col_widths=[2.1, 2.2, 2.0, 1.8, 1.8, 2.233],
        highlight_row_idx=4
    )
    add_callout_box(
        slide, left=0.60, top=4.85, width=5.95, height=1.95,
        title="文献总结一：生成端与统计端局限",
        points=[
            "• 概率曲率方法依赖 undertrained 局部替代模型，低资源语言下曲面极度振荡。",
            "• 水印技术要求服务商主动嵌入，无法防御开源本地离线大模型（LLaMA/Qwen等）生成的滥用内容。"
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.78, top=4.85, width=5.95, height=1.95,
        title="文献总结二：神经判别模型在形态学上的失效",
        points=[
            "• 统计分词器（BPE/SentencePiece）破坏形态语法结构，高频词缀退化为离散无意义单元。",
            "• 缺乏符号先验约束使分类器过拟合表面特定模式，跨领域和跨生成器迁移出现严重崩溃。"
        ],
        accent_color=TEAL_ACCENT, bg_color=CARD_BG_WHITE
    )


def populate_slide_06(slide):
    """Slide 6: 02 相关工作 — 自动化事实核查与“捷径学习”缺陷"""
    add_slide_header(
        slide,
        "相关工作：自动化事实核查范式与黏着语“捷径学习”病态",
        "FEVER 范式在黏着语语法变体下的失效机理，以及对抗性 Hard NEI 协议设立的必要性"
    )
    add_callout_box(
        slide, left=0.60, top=2.05, width=6.08, height=4.75,
        title="标准 FEVER 三阶段核验流程及其实证缺陷",
        points=[
            "1. 文档检索 (Document Retrieval):",
            "   从海量语料中利用 BM25 或稠密向量检索候选相关文档库。",
            "",
            "2. 句子证据抽取 (Sentence Evidence Extraction):",
            "   提取与待核验声明直接相关的 Top-K 证据句子集合 E = {e_1, e_2, ...}。",
            "",
            "3. 自然语言推理 (Natural Language Inference, NLI):",
            "   交叉编码器联合输入 [CLS] Claim [SEP] Evidence [SEP]，输出 SUPPORTS / REFUTES / NEI 三分类。",
            "",
            "4. 传统方法在黏着语下的致命缺陷：",
            "   • 词汇重叠捷径：哈萨克语否定后缀仅需 2 个字符即可反转整句真值，但表面字面重叠率仍 >80%，模型易误判为 SUPPORTS。",
            "   • 缺乏低资源真实基准：中亚突厥语系此前尚无权威同行评测基准。"
        ],
        accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.90, top=2.05, width=5.83, height=4.75,
        title="对抗性 Hard NEI 协议与四象限信任建模必要性",
        points=[
            "1. 传统 NEI 随机负采样弊端：",
            "   随机挑选的不相关句子与声明主体实体完全脱节，分类器仅凭实体关键词重叠即可达成虚假高准确率。",
            "",
            "2. 本文对抗性 Hard NEI (高对抗信息不足) 协议：",
            "   严格约束实体及主题完全重叠，但核心谓词时态、数量或从句因果缺失，倒逼模型深入理解句法语义依赖。",
            "",
            "3. 连续四象限信任空间 (Four-Quadrant Trust Matrix):",
            "   将风格概率 T_gen 与事实可信度 T_fact 解耦，克服传统系统将'机器生成'与'虚假谣言'混为一谈的根本误区。",
            "",
            "4. 形成从语言学解析到事实鉴别的完整防伪链条。"
        ],
        accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN
    )


def populate_slide_07(slide):
    """Slide 7: 03 研究内容 — 总体研究框架与三层神经-符号协同架构"""
    add_slide_header(
        slide,
        "研究内容：总体研究框架与三层神经-符号协同架构",
        "构建'形态感知表征 - 长文本无损检测 - 证据图谱事实核查'三位一体的全景技术路线"
    )
    add_figure_with_caption(
        slide, "fig03_tripartite_framework.png",
        left=0.60, top=2.05, width=6.65, height=4.35,
        caption_text="图 3. 面向哈萨克语的形态感知检测与事实核查三层神经-符号协同体系架构"
    )
    add_callout_box(
        slide, left=7.45, top=2.05, width=5.28, height=1.35,
        title="研究内容一：形态感知双流门控融合 (RP1)",
        points=[
            "83条FST规则词法还原 + Kaz-RoBERTa语义流 + BiLSTM语素流 + 可学习动态门控 + 监督对比学习。"
        ],
        accent_color=TEAL_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=7.45, top=3.55, width=5.28, height=1.35,
        title="研究内容二：缩写保护的滑动窗口检测 (RP2)",
        points=[
            "10类正则前瞻缩写保护分句 + 1句重叠滑动窗口 + 动态Top-K最劣池化，实现25,000词长文本100%篡改定位。"
        ],
        accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=7.45, top=5.05, width=5.28, height=1.35,
        title="研究内容三：证据驱动核验与信任空间 (RP3)",
        points=[
            "Kazakh-FEVER 3K基准 + 混合RRF检索 + 3分类NLI + (T_fact, T_gen)四象限连续双风险信任正交判定空间。"
        ],
        accent_color=GREEN_ACCENT, bg_color=CARD_BG_WHITE
    )


def populate_slide_08(slide):
    """Slide 8: 03 研究内容 — 研究内容一：形态感知的双流门控融合机制"""
    add_slide_header(
        slide,
        "研究内容一：形态感知的双流门控融合机制与监督对比学习",
        "融合83条FST符号规则先验与深层语义表征，从根源上化解统计分词导致的形态崩溃"
    )
    add_figure_with_caption(
        slide, "fig04_morpho_gate_arch.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="图 4. 神经-符号双流跨注意力动态门控融合与超球面对比聚类架构图"
    )
    add_callout_box(
        slide, left=6.65, top=2.05, width=6.08, height=2.30,
        title="数学建模与动态门控融合方程",
        points=[
            "1. 语义流：h_sem = Kaz-RoBERTa(x_tokens) ∈ R^d，捕获上下文语义。",
            "2. 语素流：通过 83 条 FST 抽离词根与词缀链，经 BiLSTM 编码为 h_morph ∈ R^m。",
            "3. 可学习门控权重：g = σ(W_g [h_sem || h_morph] + b_g)，自适应调节形态依赖权重。",
            "4. 融合表征：h_fused = g ⊙ h_sem + (1 - g) ⊙ (W_m h_morph)。",
            "5. 当遇到生僻复合词缀时，门控自适应增强语素流比重，确保零样本泛化稳健。"
        ],
        accent_color=TEAL_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=4.50, width=6.08, height=2.30,
        title="128维超球面多领域监督对比学习 (L_SupCon)",
        points=[
            "1. 损失函数：联合最小化标准交叉熵与高维超球面监督对比损失：",
            "   L_total = L_CE + λ * L_SupCon (λ = 0.1)。",
            "2. 几何聚类效果：将同标签样本在 128 维单位超球面上紧密聚合，强行推开人类与 AI 分布。",
            "3. 跨生成器留一验证 (LOGO)：消除对特定模型表面字词的过拟合，在未见生成器上实现零衰退。"
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE
    )


def populate_slide_09(slide):
    """Slide 9: 03 研究内容 — 研究内容二：缩写保护的滑动窗口长文档检测"""
    add_slide_header(
        slide,
        "研究内容二：缩写保护的滑动窗口长文档检测与动态Top-K池化",
        "突破512词元截断瓶颈，保持O(L_chunk)恒定显存开销，实现25,000词长文档无损篡改精准定位"
    )
    add_figure_with_caption(
        slide, "fig05_chunking_topk_flow.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="图 5. 语言学缩写保护滑动窗口切块与动态 Top-K 最劣段落汇聚流程图"
    )
    add_callout_box(
        slide, left=6.65, top=2.05, width=6.08, height=2.30,
        title="语言学缩写前瞻保护分句引擎 (10类正则守卫)",
        points=[
            "1. 缺陷痛点：常规断句工具在遇到缩写句点时误切断（如 'т.б.', 'ғ.', 'ж.'），切碎实体上下文。",
            "2. 正则前瞻保护：严密保护年代 (ж., жж.)、世纪 (ғ., ғғ.)、城市 (қ.)、教授 (проф.)、学术 (акад.) 等核心缩写。",
            "3. 引语对话保护：完整保留破折号引用与书名号从句，字符级对齐完全等价原始文档。",
            "4. 滑动切块：以单句重叠平滑边界效应，彻底根除跨块语法断裂。"
        ],
        accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=4.50, width=6.08, height=2.30,
        title="动态 Top-K 最劣池化与文档级决策",
        points=[
            "1. 均值稀释破解：20页文档中仅1页被AI改写时，全局平均分将稀释至<0.05（导致漏报）。",
            "2. 动态自适应K值：K = max(1, min(k_max, ceil(0.25 * M)))，仅聚焦最可疑切块均值。",
            "3. 显存开销恒定：采用小批量流水线推理，显存占用稳定在 1.4 GB 以下，无二次注意力爆炸。",
            "4. 定位精度：实证合成注入测试中达到 100.0% 的段落级篡改精确定位。"
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE
    )


def populate_slide_10(slide):
    """Slide 10: 03 研究内容 — 研究内容三：证据驱动的事实核查与四象限信任空间"""
    add_slide_header(
        slide,
        "研究内容三：证据驱动的事实核查与四象限连续信任决策空间",
        "联合BM25形态检索、稠密语义向量与交叉推理，解耦AI风格概率与事实可信度"
    )
    add_figure_with_caption(
        slide, "fig06_trust_matrix_pipeline.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="图 6. 证据检索、交叉推理与四象限连续双风险信任评估体系结构图"
    )
    add_callout_box(
        slide, left=6.65, top=2.05, width=6.08, height=2.30,
        title="混合证据检索与自然语言推理 (NLI)",
        points=[
            "1. 权威知识库：索引 250,000 段真实权威语料（涵盖历史、法律、科学及哈萨克斯坦百科）。",
            "2. 倒数秩融合 (RRF)：BM25 形态词干匹配与 M3-Embedding 稠密向量互补，倒数常数 k_0 = 60。",
            "3. 交叉编码器 NLI：联合推理输出 SUPPORTS / REFUTES / NEI，宏平均 F1 达 0.8840。",
            "4. 严格联合 FEVER 指标：必须同时满足标签一致与检索证据黄金区间覆盖。"
        ],
        accent_color=GREEN_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=4.50, width=6.08, height=2.30,
        title="连续四象限双风险信任矩阵决策空间",
        points=[
            "综合风险公式：Risk_Trust = α * P_AI + (1 - α) * Risk_Fact (α = 0.5 默认平衡)。",
            "• 第一象限 (真实机器生成)：高 AI 概率 + 高事实分 -> 合规 AI 辅助总结，安全放行。",
            "• 第二象限 (真实人类创作)：低 AI 概率 + 高事实分 -> 人类原创纪实新闻，高度信任。",
            "• 第三象限 (虚假人类谣言)：低 AI 概率 + 高虚假度 -> 人类恶意诽谤/谣言，重点打击。",
            "• 第四象限 (虚假机器幻觉)：高 AI 概率 + 高虚假度 -> 严重生成式幻觉，立即红色警报！"
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_GREEN
    )


def populate_slide_11(slide):
    """Slide 11: 04 研究创新 — 问题创新与方法创新两大核心维度"""
    add_slide_header(
        slide,
        "研究创新：问题创新与方法创新 (两大核心维度)",
        "参照领域前沿标准，系统提炼本课题在科学问题定义与工程方法设计上的核心突破"
    )
    add_callout_box(
        slide, left=0.60, top=2.05, width=5.95, height=4.75,
        title="一、科学问题创新 (Problem Innovations)",
        points=[
            "1. 首次揭示并形式化了低资源黏着语AI文本检测的'形态领域崩溃'问题：",
            "   从理论层面阐明统计分词器破坏后序词缀语素链导致分类器仅学习浅层表面伪相关的病态根因。",
            "",
            "2. 首次提出了AI生成概率与事实真实性正交解耦的'双风险信任评估'科学设想：",
            "   破除了传统研究将'机器生成文本'等同于'有害虚假信息'的一元化判定误区，为生成式AI赋能下的信息生态治理奠定理论基石。",
            "",
            "3. 提出了黏着语环境下自然语言推理的'对抗性Hard NEI'问题：",
            "   通过高表面词汇重叠度与核心逻辑谓词语义剥离，彻底杜绝虚假关联捷径学习。"
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.78, top=2.05, width=5.95, height=4.75,
        title="二、核心方法创新 (Methodological Innovations)",
        points=[
            "1. 提出神经-符号双流跨注意力门控融合机制：",
            "   将 83 条确定性 FST 状态机与预训练 Transformer 语义流动态融合，在跨领域 Kaspi 评测上斩获 +42.18% 的断崖式性能挽救。",
            "",
            "2. 提出缩写保护与动态Top-K的最劣段落汇聚长文本检测方法：",
            "   设计 10 类前瞻正则守卫，在保持 O(L_chunk) 恒定显存开销下，实现多达 25,000 词长文档的无损检测与 100% 篡改边界定位。",
            "",
            "3. 构建中亚首个哈萨克语事实核验基准 Kazakh-FEVER 3K 与四象限决策雷达：",
            "   涵盖 3,000 条高质量标注语料（κ = 0.84），开发具备可解释证据支撑的二维信任决策界面。"
        ],
        accent_color=TEAL_ACCENT, bg_color=CARD_BG_WHITE
    )


def populate_slide_12(slide):
    """Slide 12: 04 研究创新 — 核心技术难点与应对解决方案"""
    add_slide_header(
        slide,
        "研究创新：核心技术难点与应对解决方案",
        "针对形态词汇爆炸、生成器过拟合及事实核验捷径三大难题制定的系统性攻关举措"
    )
    headers = ["#", "核心技术难点与现实挑战", "根因机理剖析", "拟采取的系统性解决方案", "预期解决效果与验证指标"]
    rows = [
        [
            "1",
            "形态词汇空间爆炸\n与多义词缀解析歧义",
            "单一同义词根派生数百种表层词形；词缀级联多意性引起歧义。",
            "构建 83 条确定性有限状态自动机 (FST) 状态转移规则；引入 BiLSTM 语素流自适应门控权衡。",
            "消除未登录词碎裂；跨域 Kaspi 评测准确率提升 42 个百分点以上。"
        ],
        [
            "2",
            "黑盒大模型生成风格过拟合\n与跨生成器泛化崩溃",
            "模型易记忆特定 LLM (如 GPT-4) 的高频词元搭配，换模型后精度骤降。",
            "构建覆盖 4 大模型家族的 Tri-Domain 语料；引入 128 维超球面监督对比学习与 LOGO 留一验证。",
            "在未参与训练的 Qwen-2.5 盲测中保持 100.0% 的零样本判别 AUC。"
        ],
        [
            "3",
            "事实证据检索幻觉\n与推理阶段捷径学习",
            "黏着语否定词缀改变真值但保留 80% 词汇重叠，模型投机取巧走捷径。",
            "构建 500 条高质量对抗性 Hard NEI 反事实语料；采用 BM25 + M3 密集向量倒数秩融合 (RRF)。",
            "黄金证据区间检索准确率达 98.4%，彻底杜绝浅层词汇重叠投机误判。"
        ]
    ]
    add_table(
        slide, left=0.60, top=2.05, width=12.133, height=4.75,
        headers=headers, rows=rows,
        col_widths=[0.4, 2.3, 2.7, 3.8, 2.933]
    )


def populate_slide_13(slide):
    """Slide 13: 05 进展安排 — 数据集构建：Kaz-MAGE与Kazakh-FEVER 3K"""
    add_slide_header(
        slide,
        "进展安排：多领域数据集构建 (Kaz-MAGE 与 Kazakh-FEVER 3K)",
        "填补中亚突厥语系高质量安全基准空白，建立多源生成器标注与高一致性事实核查语料"
    )
    add_figure_with_caption(
        slide, "fig07_dataset_distribution.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="图 7. Kaz-News, Kaz-Wiki 与 Kaz-Reviews 的词元长度及词缀密度分布直方图"
    )
    add_callout_box(
        slide, left=6.65, top=2.05, width=6.08, height=2.30,
        title="Tri-Domain Kaz-MAGE 跨域生成文本检测基准",
        points=[
            "1. 语料规模：构建 10,000+ 篇高质量篇章，严格平衡人类真实文本与机器合成文本。",
            "2. 生成器矩阵：涵盖全球与区域性主流大模型（Qwen-2.5-7B, LLaMA-3-8B, GPT-4o, Sherkala-7B）。",
            "3. 三大多样性领域：",
            "   • Kaz-News: 正规官方新闻（高词汇正规度，Tengrinews / Kazinform）。",
            "   • Kaz-Wiki: 百科知识说明文（客观长程推理，Kazakh Wikipedia）。",
            "   • Kaz-Reviews: 电商真实用户评价（俚语、口语、错别字高频，Kaspi.kz）。"
        ],
        accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.65, top=4.50, width=6.08, height=2.30,
        title="Kazakh-FEVER 3K 事实核验基准 (Hard NEI 协议)",
        points=[
            "1. 语料体量：精选 3,000 条严格校验的判断声明，均衡覆盖 8 大关键知识领域。",
            "2. 三分类标签分布：1,000 支持 (SUPPORTS)、1,000 反驳 (REFUTES)、1,000 信息不足 (NEI)。",
            "3. 对抗性样本：人工注入 500 条对抗性 Hard NEI 样本，挑战模型深度因果辨识能力。",
            "4. 专家标注信度：由阿方哈萨克斯坦国立大学专业语言学者双盲互审，达成 κ = 0.84 极高一致性。"
        ],
        accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN
    )


def populate_slide_14(slide):
    """Slide 14: 05 进展安排 — 阶段性成果一：双流检测模型评测与泛化性能"""
    add_slide_header(
        slide,
        "进展安排：阶段性成果一 — 双流检测模型实证评测与泛化性能",
        "在内域、跨域及跨生成器盲测中全面超越 SOTA 基线，完全攻克形态崩溃难题"
    )
    add_figure_with_caption(
        slide, "fig08_roc_curves_kaz_mage.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="图 8. 在 Kaz-MAGE 2x2 评估矩阵下的四面板 ROC 曲线对比（全面逼近 1.000）"
    )
    headers = ["评估场景", "测试语料", "生成器设置", "Kaz-RoBERTa 基线", "本文双流融合模型", "性能增益"]
    rows = [
        ["内域测试 (Q1)", "Kaz-News", "训练集已知模型", "99.41% AUC", "99.80% AUC", "+0.39% (更低方差)"],
        ["跨模型测试 (Q2)", "Kaz-News", "未参与训练模型", "96.12% AUC", "99.82% AUC", "+3.70% (消除泛化衰退)"],
        ["跨领域测试 (Q3)", "Kaspi Reviews", "训练集已知模型", "57.62% AUC", "99.80% AUC", "+42.18% (断崖式挽救)"],
        ["完全盲测 (Q4)", "Kaspi Reviews", "未见 Qwen-2.5-7B", "54.80% AUC", "100.00% AUC", "+45.20% (稳健零失误)"]
    ]
    add_table(
        slide, left=6.65, top=2.05, width=6.08, height=2.65,
        headers=headers, rows=rows,
        col_widths=[1.1, 1.0, 1.1, 1.0, 1.0, 0.88],
        highlight_row_idx=2
    )
    add_callout_box(
        slide, left=6.65, top=4.85, width=6.08, height=1.95,
        title="实证核心结论",
        points=[
            "• 彻底解决 OOD 崩溃：在电商评论极端挑战下，将分类准确率从硬币猜测提升至商用级可靠度。",
            "• 83 条 FST 规则赋予了模型不变的语言学形态先验，摆脱了对具体模型特定生成词频的依赖。"
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE
    )


def populate_slide_15(slide):
    """Slide 15: 05 进展安排 — 阶段性成果二：长文档篡改定位与显存开销"""
    add_slide_header(
        slide,
        "进展安排：阶段性成果二 — 长文档篡改精准定位与低显存开销",
        "在长达25,000词的真实混合注入文档测试中，保持毫秒级推理时延与100%段落定位率"
    )
    add_figure_with_caption(
        slide, "fig09_long_doc_injection.png",
        left=0.60, top=2.05, width=5.85, height=4.35,
        caption_text="图 9. 25,000 词长文档合成篡改段落定位概率响应曲线与 VRAM 线性开销对比"
    )
    add_card(slide, left=6.65, top=2.05, width=2.95, height=2.25,
             title="篡改定位精准率", value_str="100.0%",
             subtitle="合成段落注入实验中达到零漏报，精准锁定篡改起止",
             accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN)
    add_card(slide, left=9.78, top=2.05, width=2.95, height=2.25,
             title="单块推理时延", value_str="< 85 ms",
             subtitle="标准单卡 GPU 推理，满足实时流式批处理审核需求",
             accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE)
    add_callout_box(
        slide, left=6.65, top=4.45, width=6.08, height=2.35,
        title="长文档工程与技术突破",
        points=[
            "1. 显存开销完全可控：避开全序列二次方自注意力爆炸，推理显存稳定在 1.4 GB 以下。",
            "2. 缩写前瞻保护：实测 2,500 处缩写词汇零切碎，彻底杜绝语法碎片引起的误判。",
            "3. 实用部署价值：已可无缝承接千页级研究生毕业论文、长篇政策文件与长篇新闻归档审核。"
        ],
        accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE
    )


def populate_slide_16(slide):
    """Slide 16: 05 进展安排 — 阶段性成果三：事实核验矩阵与消融验证"""
    add_slide_header(
        slide,
        "进展安排：阶段性成果三 — 事实核验矩阵与各组件贡献消融验证",
        "验证 NLI 混淆分布、四象限信任空间分离度以及 FST/门控/对比损失的不可替代性"
    )
    add_figure_with_caption(
        slide, "fig10_kaz_fever_confusion_matrix.png",
        left=0.60, top=2.05, width=3.85, height=3.20,
        caption_text="图 10. 三分类 NLI 归一化混淆矩阵"
    )
    add_figure_with_caption(
        slide, "fig11_four_quadrant_trust_matrix.png",
        left=4.65, top=2.05, width=3.85, height=3.20,
        caption_text="图 11. 四象限样本离散分布散点图"
    )
    add_figure_with_caption(
        slide, "fig12_ablation_study_barchart.png",
        left=8.70, top=2.05, width=4.033, height=3.20,
        caption_text="图 12. 组件消融在 Kaspi 上的性能落差"
    )
    add_callout_box(
        slide, left=0.60, top=5.40, width=12.133, height=1.40,
        title="消融实验实证结论 (Ablation Findings)",
        points=[
            "• 移除 FST 语素流：Kaspi 跨域 AUC 立即从 99.80% 跌落至 57.62% (-42.18%)，证实形态学符号先验是跨域防崩溃的核心支柱；",
            "• 移除动态门控 (静态拼接)：性能下降 8.45%，说明可学习门控在调节形态依赖时至关重要；",
            "• 移除监督对比损失：性能下降 5.12%，证明 128 维超球面正则化显著增强了表征的判别泛化边界。"
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_WHITE
    )


def populate_slide_17(slide):
    """Slide 17: 05 进展安排 — 阶段性成果四：学术发表与云端系统原型"""
    add_slide_header(
        slide,
        "进展安排：阶段性成果四 — 学术论文发表与云端交互原型系统",
        "阶段性论文已被国际高水平学术会议正式录用，完成4-Tab全功能交互平台部署"
    )
    add_figure_with_caption(
        slide, "fig13_gradio_dashboard_panels.png",
        left=0.60, top=2.05, width=6.65, height=4.35,
        caption_text="图 13. 部署于 Hugging Face Spaces 的 4-Tab 交互式科研级演示平台"
    )
    add_callout_box(
        slide, left=7.45, top=2.05, width=5.28, height=2.45,
        title="学术论文发表与投稿进展",
        points=[
            "1. 论文一 (已正式录用)：",
            "   • 题目：Dual-Stream Morphology-Aware Gated Fusion for Low-Resource Turkic AI Text Detection",
            "   • 录用会议：AIST 2026 (Springer LNCS，CCF 推荐会议)",
            "   • 成果贡献：提出 83 条 FST 神经符号融合架构，完成口头汇报准备。",
            "",
            "2. 论文二 (已正式提交)：",
            "   • 目标会议：COLING 2027 (ARR Track)",
            "   • 成果贡献：发布 Kazakh-FEVER 3K 基准与四象限连续双风险信任雷达。"
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE
    )
    add_callout_box(
        slide, left=7.45, top=4.65, width=5.28, height=1.75,
        title="工程落地与云端演示部署",
        points=[
            "• 4-Tab 交互面板：长文档彩色热力图、实时 FST 语素分解、FEVER 交叉核验、四象限信任雷达。",
            "• 代码已开源封装：支持一键运行（python hf_space/app.py），冷启动时延低于 1.2 秒。"
        ],
        accent_color=PURPLE_ACCENT, bg_color=CARD_BG_WHITE
    )


def populate_slide_18(slide):
    """Slide 18: 05 进展安排 — 工作积累与研究条件落实情况"""
    add_slide_header(
        slide,
        "进展安排：工作积累与研究条件落实情况",
        "参照课题组规范全面梳理科研积累、算力硬件支持及中哈跨国学术资源保障"
    )
    add_callout_box(
        slide, left=0.60, top=2.05, width=6.08, height=4.75,
        title="科研工作积累与前期成果 (Research Foundations)",
        points=[
            "1. 学术论文发表积累：",
            "   • [1] D. Anekesh, B. Guo, et al. Morphologically-Grounded AI Text Detection. AIST 2026 (Accepted, Springer LNCS).",
            "   • [2] D. Anekesh, B. Guo, et al. Kazakh-FEVER: Benchmark and Trust Matrix for Low-Resource NLP. COLING 2027 ARR (Submitted).",
            "",
            "2. 语料资源积累：",
            "   • 构建了 10,000+ 篇多领域 Kaz-MAGE 生成文本检测语料库。",
            "   • 建立了中亚首个 3,000 条高质量 Kazakh-FEVER 事实核验基准（含 500 对抗性 Hard NEI）。",
            "   • 整理了 250,000 段真实权威知识库文档索引。",
            "",
            "3. 规则工程资产：",
            "   • 梳理实现 83 条哈萨克语有限状态自动机 (FST) 词法规则库与 10 类前瞻分句保护守卫。"
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.90, top=2.05, width=5.83, height=4.75,
        title="实验研究条件落实情况 (Computational Facilities)",
        points=[
            "1. 高性能算力集群保障：",
            "   • 西北工业大学计算机学院及郭斌教授课题组实验室配备多台高性能 GPU 服务器集群（含 NVIDIA A100 80GB 及双 Tesla T4 加速卡）。",
            "   • 为大规模预训练模型微调、跨领域对比学习及密集向量检索实验提供充裕算力支撑。",
            "",
            "2. 跨国联合指导与语言学专业支撑：",
            "   • 西北工业大学计算机学院郭斌教授负责计算机系统架构、大模型安全及对比学习设计指导。",
            "   • 哈萨克斯坦国立大学 (Al-Farabi KazNU) 语言学团队提供专业语法词典、形态规则校准及双盲人工标注质量审核。",
            "",
            "3. 软件与代码复现保障：",
            "   • 建立 81 项自动化回归单元测试集，保障全部实验与工程模块 100% 严谨可复现。"
        ],
        accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN
    )


def populate_slide_19(slide):
    """Slide 19: 05 进展安排 — 进度安排时间轴 (2026.10 – 2027.06)"""
    add_slide_header(
        slide,
        "进展安排：硕士学位论文研究计划与进度时间轴 (2026.10 – 2027.06)",
        "自2026年10月开题答辩至2027年6月毕业答辩，共划分为五个紧密衔接的攻关阶段"
    )
    add_figure_with_caption(
        slide, "fig15_timeline_2026_2027.png",
        left=0.60, top=2.05, width=12.133, height=4.50,
        caption_text="图 14. 硕士学位论文进度安排与里程碑规划时间轴（2026年10月 — 2027年6月）"
    )


def populate_slide_20(slide):
    """Slide 20: 05 进展安排 — 预期难点与毕业答辩准备"""
    add_slide_header(
        slide,
        "进展安排：预期研究难点、应对预案与最终答辩准备",
        "针对方言口语、检索时延、伦理误报及论文双盲评审等潜在风险制定周密预案"
    )
    add_callout_box(
        slide, left=0.60, top=2.05, width=2.95, height=4.75,
        title="预案一：方言与网络口语\n(Dialectal Robustness)",
        points=[
            "• 风险点：社交网络俚语与非标准拼写可能规避 FST 严格规则。",
            "• 应对方案：引入混合子词容错回退机制，结合启发式拼写归一化预处理字典，确保未登录词稳定回退至 RoBERTa 语义流。"
        ],
        accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=3.66, top=2.05, width=2.95, height=4.75,
        title="预案二：密集检索实时时延\n(Retrieval Latency)",
        points=[
            "• 风险点：25 万段落稠密语义搜索在长文档中可能产生推理时延堆叠。",
            "• 应对方案：预构建分层 HNSW/FAISS 向量索引，结合 GPU 显存驻留缓存，将单次查询锁定在 85ms 阈值以内。"
        ],
        accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=6.72, top=2.05, width=2.95, height=4.75,
        title="预案三：本土学生伦理防护\n(Ethical Protection)",
        points=[
            "• 风险点：误将非母语学习者拼写或学术常规引用判定为机器生成。",
            "• 应对方案：校准 0.9980 高特异性决策阈值，在系统前端提供透明的词法归因与证据链解释，保障学生申诉权利。"
        ],
        accent_color=GREEN_ACCENT, bg_color=CARD_BG_WHITE
    )
    add_callout_box(
        slide, left=9.78, top=2.05, width=2.95, height=4.75,
        title="预案四：盲审与毕业答辩\n(Thesis Defense)",
        points=[
            "• 规划：2027年3月完成全部章节撰写，严格对照西北工业大学研究生学位论文撰写规范排版。",
            "• 预答辩与盲审：4月通过课题组与学院预答辩，高质量完成校级双盲评审修改，6月正式答辩毕业。"
        ],
        accent_color=NAVY_PRIMARY, bg_color=CARD_BG_BLUE
    )


def update_closing_slide(slide):
    """Slide 21 (Closing Slide): Updates closing text and metadata matching laboratory standard."""
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
            p0.text = "THANKS!"
            p0.font.name = FONT_TITLE
            p0.font.size = Pt(36)
            p0.font.bold = True
            p0.font.color.rgb = NAVY_PRIMARY
            p0.alignment = PP_ALIGN.CENTER

            p1 = tf.add_paragraph()
            p1.text = "请各位专家批评指正！"
            p1.font.name = FONT_ZH
            p1.font.size = Pt(22)
            p1.font.bold = True
            p1.font.color.rgb = NAVY_PRIMARY
            p1.alignment = PP_ALIGN.CENTER
            p1.space_before = Pt(4)

            p2 = tf.add_paragraph()
            p2.text = "基于形态学感知的低资源黏着语AI生成文本检测与事实核查研究"
            p2.font.name = FONT_ZH
            p2.font.size = Pt(14)
            p2.font.bold = True
            p2.font.color.rgb = SLATE_BODY
            p2.alignment = PP_ALIGN.CENTER
            p2.space_before = Pt(8)

            p3 = tf.add_paragraph()
            p3.text = "汇报人：大雷 (DAULET ANEKESH, 学号 2025291043) | 导师：郭斌 教授 | 2026年10月29日"
            p3.font.name = FONT_ZH
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
    Slides 3-20 (indices 2 through 19):
    1. Sets up the 5-element running section tracker on top bar.
    2. Repositions '矩形 29' (Native Gradient Highlight Tracker) to active section.
    3. Excises legacy shapes below top > Inches(1.2).
    4. Populates academic content for Slides 3-20.
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
    }

    for s_idx in range(2, 20):
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

        # 2. Add 5 clean section textboxes on top bar
        for idx in range(5):
            s_left, s_width = SLOT_COORDS[idx]
            is_active = (idx == active_sec_idx)

            tb = slide.shapes.add_textbox(s_left, Inches(0.12), s_width, Inches(0.65))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

            p_zh = tf.paragraphs[0]
            p_zh.text = SECTION_NAMES_ZH[idx]
            p_zh.font.name = FONT_ZH
            p_zh.font.size = Pt(11.0)
            p_zh.font.bold = True
            p_zh.font.color.rgb = RGBColor(255, 255, 255) if is_active else NAVY_PRIMARY
            p_zh.alignment = PP_ALIGN.CENTER

            p_en = tf.add_paragraph()
            p_en.text = SECTION_NAMES_EN[idx]
            p_en.font.name = FONT_TITLE
            p_en.font.size = Pt(8.0)
            p_en.font.bold = is_active
            p_en.font.color.rgb = RGBColor(220, 235, 255) if is_active else SLATE_MUTED
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
    Loads baseline presentation, reduces to exactly 21 slides, applies 5-element tracker,
    populates all slides, and saves to output_path and desktop mirror.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Baseline PPT file not found: {input_path}")

    print(f"Loading baseline template from: {input_path}")
    prs = Presentation(input_path)

    # In baseline (23 slides), drop index 21 then index 20, leaving 21 slides total.
    # Slide 22 (original closing slide) becomes index 20 (Slide 21).
    print(f"Initial slide count: {len(prs.slides)}")
    rId1 = prs.slides._sldIdLst[21].rId
    prs.part.drop_rel(rId1)
    del prs.slides._sldIdLst[21]

    rId2 = prs.slides._sldIdLst[20].rId
    prs.part.drop_rel(rId2)
    del prs.slides._sldIdLst[20]

    print(f"Trimmed slide count: {len(prs.slides)} (Target: 21)")

    # 1. Update Slide 1 (Cover Page)
    print("Updating Slide 1: Cover Page...")
    update_cover_slide(prs.slides[0])

    # 2. Update Slide 2 (TOC)
    print("Updating Slide 2: 5-Element Table of Contents...")
    update_toc_slide(prs.slides[1])

    # 3. Update Slides 3 - 20 (Content Slides)
    print("Populating Slides 3 - 20 with academic content, figures, and 5-element tracker...")
    update_content_slides(prs.slides)

    # 4. Update Slide 21 (Closing Slide)
    print("Updating Slide 21: Closing & Guidance Slide...")
    update_closing_slide(prs.slides[20])

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
    parser = argparse.ArgumentParser(description="Build Master's Proposal Defense Presentation (5 Elements)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output PPTX path")
    args = parser.parse_args()

    build_proposal_presentation(output_path=args.output)
