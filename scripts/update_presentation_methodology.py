# -*- coding: utf-8 -*-
"""
Comprehensive Presentation Updater for Kazakh AI Detection Master's Thesis.
Applies Professor Guo's academic feedback across all target slides in:
  - AnekeshD_Progress.pptx
  - Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx

Key Revisions:
1. Slide 01: Cleans template presenter notes (removes Fan Qianyue / Chen Yaxing; sets Daulet's credentials).
2. Slide 02: Sequential reordering of shapes in XML tree: 01, 02, 03, 04, 05, 06.
3. Slide 05: Excises promotional/catastrophic terms and 'Cross-Gate' in related work table.
4. Slide 07: Comprehensive Methodological Framework (Figure 3, 300 DPI, full-width).
5. Slide 08: Topic 1 Architecture — Dual-Stream Morphology-Aware Gated Fusion & SupCon Loss.
6. Slide 09: Topic 2 Architecture — Robust Generalization & Adversarial Rewriting Defense.
7. Slide 10: Topic 3 Architecture — Evidence-Grounded Kazakh Claim Verification & Trust Matrix.
8. Slide 11: Experimental Setup — Kazakh Adaptation of MAGE Evaluation Protocol (314 Tests).
9. Slide 12: Topic 1 Results — Resolving Out-of-Domain Blindspot (+42.18 pp; AUC vs Macro-F1).
10. Slide 13: Topic 2 Results — Cross-Generator Robustness & Long-Document Evaluation.
11. Slide 14: Topic 3 Results — Kazakh-FEVER Pilot Verification Benchmark (Bottleneck analysis).
12. Slide 15: Topic 3 Results — Four-Quadrant Trust Matrix Validation (2D Risk Coordinates).
13. Slide 16: Comprehensive Component Ablation Studies (Disentangling sentence vs document).
14. Slide 17: System Demonstration — Interactive 4-Tab Gradio Academic Prototype.
15. Slide 18: System Demonstration — Cloud Packaging, Hugging Face Spaces & Test Rigor (314 Tests).
16. Slide 19: Conclusion — Four Primary Academic Contributions & Manuscript Status (~85%).
17. Slide 20: Current Progress — Springer LNCS Acceptance & September Meeting Agenda.
18. Slide 21: Discussion — Three Core Strategic Guidance Questions for Prof. Guo.
19. Slide 22: Committee Comments & Responses — Academic Progress Status Indicators.
"""

import os
import sys
import shutil
import argparse
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE

# -----------------------------------------------------------------------------
# Color Palette & Typography (Matching Presentation Theme)
# -----------------------------------------------------------------------------
NAVY_PRIMARY = RGBColor(30, 58, 138)       # #1E3A8A - Deep Academic Navy Blue
BLUE_ACCENT = RGBColor(37, 99, 235)        # #2563EB - Royal Blue Accent
TEAL_ACCENT = RGBColor(13, 148, 136)       # #0D9488 - Topic 1 / FST Morphology
AMBER_ACCENT = RGBColor(217, 119, 6)       # #D97706 - Amber Accent
GREEN_ACCENT = RGBColor(22, 163, 74)       # #16A34A - Green Accent
PURPLE_ACCENT = RGBColor(124, 58, 237)     # #7C3AED - System Engineering & Cloud
CORAL_ACCENT = RGBColor(220, 38, 38)       # #DC2626 - Alert / Disinformation
SLATE_TITLE = RGBColor(15, 23, 42)         # #0F172A - Slide title & primary headers
SLATE_BODY = RGBColor(51, 65, 85)          # #334155 - Standard body prose
SLATE_MUTED = RGBColor(100, 116, 139)      # #64748B - Subtitles & captions
WHITE = RGBColor(255, 255, 255)            # #FFFFFF - Crisp white

CARD_BG_WHITE = RGBColor(255, 255, 255)
CARD_BG_BLUE = RGBColor(239, 246, 255)     # #EFF6FF - Soft blue tinted card fill
CARD_BG_GREEN = RGBColor(240, 253, 244)    # #F0FDF4 - Subtle mint
CARD_BG_AMBER = RGBColor(255, 251, 235)    # #FFFBEB - Warm amber fill
CARD_BG_CORAL = RGBColor(254, 242, 242)    # #FEE2E2 - Soft alert fill
BORDER_LIGHT = RGBColor(226, 232, 240)     # #E2E8F0 - Clean border

FONT_TITLE = "Arial"
FONT_BODY = "Arial"
FONT_ZH = "Microsoft YaHei"

PROTECTED_SHAPE_NAMES = {
    "灯片编号占位符 10",
    "直接连接符 6",
    "矩形 4",
    "矩形 29",
}

# -----------------------------------------------------------------------------
# Reusable Slide Helper Functions
# -----------------------------------------------------------------------------
def remove_shape(shape):
    """Remove a shape element cleanly from its slide XML."""
    sp = shape._element
    sp.getparent().remove(sp)


def update_slide_header(slide, title_text, subtitle_text):
    """
    Finds existing header textbox or creates a new one at top=1.30.
    Updates title and subtitle with consistent typography.
    """
    header_tb = None
    for shape in slide.shapes:
        if shape.has_text_frame:
            if shape.left < Inches(1.0) and Inches(1.20) <= shape.top <= Inches(1.60) and shape.width > Inches(10):
                header_tb = shape
                break

    if header_tb is None:
        header_tb = slide.shapes.add_textbox(Inches(0.60), Inches(1.30), Inches(12.133), Inches(0.68))

    tf = header_tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    tf.text = ""

    p0 = tf.paragraphs[0]
    p0.text = title_text
    p0.font.name = FONT_TITLE
    p0.font.size = Pt(16.5)
    p0.font.bold = True
    p0.font.color.rgb = NAVY_PRIMARY

    if subtitle_text:
        p1 = tf.add_paragraph()
        p1.text = subtitle_text
        is_chinese = any('\u4e00' <= char <= '\u9fff' for char in subtitle_text)
        p1.font.name = FONT_ZH if is_chinese else FONT_BODY
        p1.font.size = Pt(9.8)
        p1.font.bold = False
        p1.font.color.rgb = SLATE_MUTED
        p1.space_before = Pt(2)

    return header_tb


def add_callout_box(slide, left, top, width, height, title, items,
                    accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE, border_color=BORDER_LIGHT):
    """Adds a structured content card with colored header strip and formatted items."""
    # 1. Background box
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    box.fill.solid()
    box.fill.fore_color.rgb = bg_color
    box.line.color.rgb = border_color
    box.line.width = Pt(1)

    # 2. Header strip
    is_multiline = ("\n" in title)
    strip_h = 0.58 if is_multiline else 0.38

    header_strip = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(strip_h)
    )
    header_strip.fill.solid()
    header_strip.fill.fore_color.rgb = accent_color
    header_strip.line.fill.background()

    # 3. Header title textbox
    tb_head = slide.shapes.add_textbox(
        Inches(left + 0.12), Inches(top + 0.04), Inches(width - 0.24), Inches(strip_h - 0.08)
    )
    tf_head = tb_head.text_frame
    tf_head.word_wrap = True
    tf_head.margin_left = tf_head.margin_top = tf_head.margin_right = tf_head.margin_bottom = 0
    p_head = tf_head.paragraphs[0]
    p_head.text = title
    p_head.font.name = FONT_TITLE
    p_head.font.size = Pt(9.5 if is_multiline else 10.5)
    p_head.font.bold = True
    p_head.font.color.rgb = WHITE

    # 4. Body content textbox
    tb_body = slide.shapes.add_textbox(
        Inches(left + 0.14), Inches(top + strip_h + 0.06), Inches(width - 0.28), Inches(height - strip_h - 0.12)
    )
    tf_body = tb_body.text_frame
    tf_body.word_wrap = True
    tf_body.margin_left = tf_body.margin_top = tf_body.margin_right = tf_body.margin_bottom = 0

    for idx, item in enumerate(items):
        if isinstance(item, tuple):
            heading, description = item
            p_h = tf_body.paragraphs[0] if idx == 0 else tf_body.add_paragraph()
            p_h.text = heading
            p_h.font.name = FONT_TITLE
            p_h.font.size = Pt(8.5)
            p_h.font.bold = True
            p_h.font.color.rgb = accent_color
            if idx > 0:
                p_h.space_before = Pt(3.5)

            p_d = tf_body.add_paragraph()
            p_d.text = description
            p_d.font.name = FONT_BODY
            p_d.font.size = Pt(7.8)
            p_d.font.bold = False
            p_d.font.color.rgb = SLATE_BODY
            p_d.space_before = Pt(1.0)
        else:
            p = tf_body.paragraphs[0] if idx == 0 else tf_body.add_paragraph()
            p.text = item
            p.font.name = FONT_BODY
            p.font.size = Pt(8.2)
            p.font.color.rgb = SLATE_BODY
            if idx > 0:
                p.space_before = Pt(2.0)


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
    p_val.font.size = Pt(18)
    p_val.font.bold = True
    p_val.font.color.rgb = accent_color

    p_tit = tf.add_paragraph()
    p_tit.text = title
    p_tit.font.name = FONT_TITLE
    p_tit.font.size = Pt(10.0)
    p_tit.font.bold = True
    p_tit.font.color.rgb = SLATE_TITLE
    p_tit.space_before = Pt(2)

    if subtitle:
        p_sub = tf.add_paragraph()
        p_sub.text = subtitle
        p_sub.font.name = FONT_BODY
        p_sub.font.size = Pt(8.5)
        p_sub.font.color.rgb = SLATE_MUTED
        p_sub.space_before = Pt(2)


# -----------------------------------------------------------------------------
# Slide 01: Title Slide Presenter Notes Cleanup
# -----------------------------------------------------------------------------
def update_slide_01(slide):
    """Cleans up template notes mentioning Fan Qianyue / Chen Yaxing."""
    if not slide.has_notes_slide:
        slide.notes_slide
    tf = slide.notes_slide.notes_text_frame
    tf.text = (
        "Respected committee members, advisor Professor Guo, and fellow researchers: "
        "I am Daulet Anekesh. Today I am presenting my Master's thesis progress report on "
        "'Morphologically-Grounded AI-Generated Text Detection and Evidence-Based Verification in Low-Resource Kazakh'."
    )


# -----------------------------------------------------------------------------
# Slide 02: Sequential Reordering & Harmonization
# -----------------------------------------------------------------------------
def update_slide_02(slide):
    """
    Reorders Slide 2 chapter shapes in the XML tree so that iteration order is strictly:
    01, 02, 03, 04, 05, 06.
    """
    title_map = {
        1: "01. Research Background & Motivation",
        2: "02. Related Work & SOTA Challenges",
        3: "03. Research Content & Methodological Framework",
        4: "04. Experiments & Benchmark Evaluation",
        5: "05. Engineering Implementation & Prototype Demonstration",
        6: "06. Conclusion & Thesis Roadmap",
    }

    def get_chapter_num(s):
        if s.shape_type == MSO_SHAPE_TYPE.GROUP:
            for sub in s.shapes:
                if sub.has_text_frame and sub.text_frame.text.strip()[:2].isdigit():
                    return int(sub.text_frame.text.strip()[:2])
        elif s.has_text_frame and s.text_frame.text.strip()[:2].isdigit():
            return int(s.text_frame.text.strip()[:2])
        return None

    chapter_shapes = {}
    other_05_shapes = []
    for s in slide.shapes:
        c_num = get_chapter_num(s)
        if c_num is not None:
            chapter_shapes[c_num] = s
            if s.shape_type == MSO_SHAPE_TYPE.GROUP:
                for sub in s.shapes:
                    if sub.has_text_frame and sub.text_frame.text.strip()[:2].isdigit():
                        sub.text_frame.text = title_map[c_num]
            elif s.has_text_frame:
                s.text_frame.text = title_map[c_num]
        else:
            if 4.7 <= s.top / 914400 <= 5.6 and s.left / 914400 >= 4.0:
                other_05_shapes.append(s)

    sp_tree = slide.shapes._spTree
    ordered_elements = []
    if 1 in chapter_shapes: ordered_elements.append(chapter_shapes[1]._element)
    if 2 in chapter_shapes: ordered_elements.append(chapter_shapes[2]._element)
    if 3 in chapter_shapes: ordered_elements.append(chapter_shapes[3]._element)
    if 4 in chapter_shapes: ordered_elements.append(chapter_shapes[4]._element)
    if 5 in chapter_shapes:
        for o in other_05_shapes:
            ordered_elements.append(o._element)
        ordered_elements.append(chapter_shapes[5]._element)
    if 6 in chapter_shapes: ordered_elements.append(chapter_shapes[6]._element)

    for elem in ordered_elements:
        sp_tree.append(elem)


# -----------------------------------------------------------------------------
# Slide 05: Related Work Detection Cleanup
# -----------------------------------------------------------------------------
def update_slide_05(slide):
    """Excises 'catastrophic' and 'Cross-Gate' from Slide 5."""
    for s in slide.shapes:
        if s.has_text_frame:
            tf = s.text_frame
            for p in tf.paragraphs:
                if "catastrophic limitations" in p.text:
                    p.text = p.text.replace("catastrophic limitations", "critical performance limitations")
                if "fail catastrophically" in p.text:
                    p.text = p.text.replace("fail catastrophically", "experience severe degradation")
                if "catastrophic" in p.text.lower():
                    p.text = p.text.replace("catastrophic", "severe").replace("Catastrophic", "Severe")
                if "cross-gate" in p.text.lower():
                    p.text = p.text.replace("Cross-Gate", "Gated Fusion")
        if s.has_table:
            for row in s.table.rows:
                for cell in row.cells:
                    if "Cross-Gate" in cell.text:
                        cell.text = cell.text.replace("Cross-Gate", "Gated Fusion")


# -----------------------------------------------------------------------------
# Slide 07: Comprehensive Methodological Framework (Figure 3)
# -----------------------------------------------------------------------------
def update_slide_07(slide, fig_path=None):
    """Embeds Figure 3 (Methodological Innovations Framework) full width."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fig_path is None:
        fig_path = os.path.join(project_root, "presentation_figures", "fig15_methodological_innovations_framework.png")
        if not os.path.isfile(fig_path):
            fig_path = os.path.join(project_root, "presentation_figures", "fig03_methodological_innovation_framework.png")
        if not os.path.isfile(fig_path):
            fig_path = os.path.join(project_root, "presentation_figures", "fig03_end_to_end_pipeline.png")

    if not os.path.isfile(fig_path):
        raise FileNotFoundError(f"Framework figure not found at: {fig_path}")

    header_tb = update_slide_header(
        slide,
        "Research Content Overview: Comprehensive Methodological Framework",
        "A unified 3-column architecture spanning edge ingestion, dual-stream feature fusion with ablated baselines, and 3-tier portal verification"
    )

    for shape in list(slide.shapes):
        if shape == header_tb or shape.name in PROTECTED_SHAPE_NAMES or shape.top < Inches(1.20):
            continue
        remove_shape(shape)

    left = Inches(0.60)
    top = Inches(1.85)
    width = Inches(12.133)
    height = Inches(4.85)
    slide.shapes.add_picture(fig_path, left, top, width, height)

    caption_box = slide.shapes.add_textbox(left, Inches(6.75), width, Inches(0.30))
    tf = caption_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = "Figure 3. Overall Methodological Innovation Framework for Kazakh AI-Generated Text Detection and Factual Verification."
    p.font.name = FONT_TITLE
    p.font.size = Pt(8.5)
    p.font.italic = True
    p.font.color.rgb = SLATE_MUTED
    p.alignment = PP_ALIGN.CENTER


# -----------------------------------------------------------------------------
# Slide 08: Topic 1 Architecture (Morphology-Aware Gated Fusion)
# -----------------------------------------------------------------------------
def update_slide_08(slide, fig_path=None):
    """Updates Slide 8 with Morphology-Aware Gated Fusion, removing Cross-Attention."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fig_path is None:
        fig_path = os.path.join(project_root, "presentation_figures", "fig04_morpho_gate_arch.png")

    header_tb = update_slide_header(
        slide,
        "Topic 1 Architecture: Dual-Stream Morphology-Aware Gated Fusion & SupCon Loss",
        "Fusing contextual semantic representations with 83-rule FST morphological inductive bias via dynamic gating"
    )

    # Remove left column picture and old cards below top 1.8 in
    for shape in list(slide.shapes):
        if shape == header_tb or shape.name in PROTECTED_SHAPE_NAMES or shape.top < Inches(1.20):
            continue
        remove_shape(shape)

    # Insert Figure 4
    left = Inches(0.60)
    top = Inches(2.05)
    width = Inches(5.85)
    height = Inches(4.35)
    slide.shapes.add_picture(fig_path, left, top, width, height)

    caption_box = slide.shapes.add_textbox(left, Inches(6.44), width, Inches(0.30))
    tf = caption_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = "Figure 4. Dual-Stream Morphology-Aware Gated Fusion Architecture with SupCon Optimization."
    p.font.name = FONT_TITLE
    p.font.size = Pt(8.5)
    p.font.italic = True
    p.font.color.rgb = SLATE_MUTED
    p.alignment = PP_ALIGN.CENTER

    # Right Card 1: Morphology-Aware Gated Fusion
    card1_items = [
        ("1. Dynamic Morphology-Aware Gating Formulation:",
         "Computes element-wise gate g = sigma(W_g [h_sem; W_proj h_morph] + b_g), yielding h_fused = g * h_sem + (1 - g) * (W_proj h_morph). Automatically routes regular Turkic inflectional affix cues."),
        ("2. Grounding the 83-Rule FST Transducer:",
         "Built on rule-based finite-state morphology validated on 5,000 dictionary headwords across nominal and verbal paradigms (99.2% inflectional coverage; case markers -ның/-нің, verbal -ған/-ген)."),
        ("3. Dynamic Gating Parameter Distribution:",
         "Illustrative sample gate g = 0.682 (KazRoBERTa 68%, FST 32%). Across morphologically rich colloquial spans, empirical mean gate values range g in [0.55, 0.78].")
    ]
    add_callout_box(slide, left=6.65, top=2.05, width=6.08, height=2.35,
                    title="Dual-Stream Morphology-Aware Gated Fusion",
                    items=card1_items, accent_color=NAVY_PRIMARY, bg_color=CARD_BG_WHITE)

    # Right Card 2: Supervised Contrastive Loss
    card2_items = [
        ("1. Latent Separation Objective in R^128:",
         "Optimizes total loss L_total = L_BCE + lambda * L_SupCon (lambda = 0.10). Enforces tight intra-class clustering for Human vs. AI latent representations regardless of domain shifts."),
        ("2. Generator-Agnostic Invariance:",
         "Decouples lexical surface patterns from underlying morphological regularity, substantially mitigating domain collapse (+42.18 pp OOD gain)."),
        ("3. Planned Paper 2 Contrastive Expansion:",
         "Extending SupCon positive pairs to explicitly align original AI text with paraphrased and perturbed variants for adversarial rewriting defense.")
    ]
    add_callout_box(slide, left=6.65, top=4.50, width=6.08, height=2.30,
                    title="Supervised Contrastive (SupCon) Representation Learning",
                    items=card2_items, accent_color=TEAL_ACCENT, bg_color=CARD_BG_WHITE)


# -----------------------------------------------------------------------------
# Slide 09: Topic 2 Architecture (Robust Generalization & Adversarial Defense)
# -----------------------------------------------------------------------------
def update_slide_09(slide, fig_path=None):
    """Restores Topic 2 to Robust Generalization & Adversarial Rewriting Defense."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fig_path is None:
        fig_path = os.path.join(project_root, "presentation_figures", "fig05_chunking_topk_flow.png")

    header_tb = update_slide_header(
        slide,
        "Topic 2 Architecture: Robust Generalization & Adversarial Rewriting Defense",
        "Mitigating generator and domain shift with adversarial robustness and supporting sentence-preserving chunk aggregation"
    )

    for shape in list(slide.shapes):
        if shape == header_tb or shape.name in PROTECTED_SHAPE_NAMES or shape.top < Inches(1.20):
            continue
        remove_shape(shape)

    # Insert Figure 5
    left = Inches(0.60)
    top = Inches(2.05)
    width = Inches(5.85)
    height = Inches(4.35)
    slide.shapes.add_picture(fig_path, left, top, width, height)

    caption_box = slide.shapes.add_textbox(left, Inches(6.44), width, Inches(0.30))
    tf = caption_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = "Figure 5. Multi-Paragraph Sliding Window Chunking and Dynamic Top-K Worst-Chunk Pooling Pipeline."
    p.font.name = FONT_TITLE
    p.font.size = Pt(8.5)
    p.font.italic = True
    p.font.color.rgb = SLATE_MUTED
    p.alignment = PP_ALIGN.CENTER

    # Right Card 1: Robust Generalization & Adversarial Rewriting Defense
    card1_items = [
        ("1. Cross-Generator Latent Invariance:",
         "Learns generator-agnostic structural representations that withstand model shifts from seen LLMs (Sherkala-7B) to held-out architectures (Qwen-2.5-7B, Qwen Wild)."),
        ("2. Adversarial Rewriting & Perturbation Defense:",
         "Formulated to defend against paraphrasing attacks, synonym substitution, and spelling noise by exploiting invariant morphological transition regularities."),
        ("3. Planned Contrastive Formulations for Paper 2:",
         "Developing positive pairs between original AI text and AI-paraphrased text to directly benchmark robustness against modern generative rewriters.")
    ]
    add_callout_box(slide, left=6.65, top=2.05, width=6.08, height=2.35,
                    title="Robust Generalization & Adversarial Rewriting Defense",
                    items=card1_items, accent_color=AMBER_ACCENT, bg_color=CARD_BG_WHITE)

    # Right Card 2: Supporting Practical Extension: SentencePreservingChunker & Top-K
    card2_items = [
        ("1. Kazakh Abbreviation Protection Guards:",
         "SentencePreservingChunker preserves boundary integrity across 10 common Kazakh abbreviations (ж., ғ., т.б., б.з.д.), avoiding spurious mid-sentence splits."),
        ("2. Dynamic Top-K Worst-Chunk Pooling (K = max(1, floor(0.25N))):",
         "Prevents naive mean dilution where isolated AI paragraphs in 20-page documents go undetected. Pools top 25% highest-risk chunks."),
        ("3. Sub-Linear Micro-Batching Scalability:",
         "Maintains bounded peak VRAM < 1.4 GB and sub-linear latency (<4.1s for 25,000 words), enabling practical document-level auditing.")
    ]
    add_callout_box(slide, left=6.65, top=4.50, width=6.08, height=2.30,
                    title="Supporting Practical Extension: Sentence Chunking & Dynamic Top-K",
                    items=card2_items, accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE)


# -----------------------------------------------------------------------------
# Slide 10: Topic 3 Architecture (Evidence-Grounded Claim Verification & Trust)
# -----------------------------------------------------------------------------
def update_slide_10(slide, fig_path=None):
    """Updates Slide 10 with Kazakh-FEVER pilot benchmark & 2D trust matrix coordinates."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fig_path is None:
        fig_path = os.path.join(project_root, "presentation_figures", "fig06_trust_matrix_pipeline.png")

    header_tb = update_slide_header(
        slide,
        "Topic 3 Architecture: Evidence-Grounded Kazakh Claim Verification & Trust Matrix",
        "Decoupling stylistic AI generation from factual veracity via curated reference grounding and dual-risk scoring"
    )

    for shape in list(slide.shapes):
        if shape == header_tb or shape.name in PROTECTED_SHAPE_NAMES or shape.top < Inches(1.20):
            continue
        remove_shape(shape)

    # Insert Figure 6
    left = Inches(0.60)
    top = Inches(2.05)
    width = Inches(5.85)
    height = Inches(4.35)
    slide.shapes.add_picture(fig_path, left, top, width, height)

    caption_box = slide.shapes.add_textbox(left, Inches(6.44), width, Inches(0.30))
    tf = caption_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = "Figure 6. Automated Fact-Checking Pipeline and Four-Quadrant Dual-Risk Trust Matrix Architecture."
    p.font.name = FONT_TITLE
    p.font.size = Pt(8.5)
    p.font.italic = True
    p.font.color.rgb = SLATE_MUTED
    p.alignment = PP_ALIGN.CENTER

    # Right Card 1: Curated Kazakh-FEVER Pilot Verification Pipeline
    card1_items = [
        ("1. Curated Pilot Knowledge Base (36 Articles):",
         "Constructed from 36 authoritative encyclopedic articles spanning 4 national domains: Kazakh History, Public Law, Natural Science, and Healthcare."),
        ("2. 120 Annotated Verification Claims (40/40/40):",
         "Balanced split across 40 SUPPORTS, 40 REFUTES, and 40 NOT ENOUGH INFO (NEI) claims paired with sentence-level gold evidence annotations."),
        ("3. BM25 Retrieval & mDeBERTa-v3 3-Way NLI Verifier:",
         "Surfaces Top-K candidate evidence passages and performs cross-lingual NLI to determine evidentiary support with calibrated epistemic uncertainty.")
    ]
    add_callout_box(slide, left=6.65, top=2.05, width=6.08, height=2.35,
                    title="Curated Kazakh-FEVER Pilot Verification Pipeline",
                    items=card1_items, accent_color=GREEN_ACCENT, bg_color=CARD_BG_WHITE)

    # Right Card 2: Two-Dimensional Risk Coordinates & Decision Quadrants
    card2_items = [
        ("1. Primary Output: Two-Dimensional Coordinates (y_AI, R_fact):",
         "Maps every text to the unit square [0, 1]^2, preserving orthogonal risk dimensions rather than collapsing them into an uninterpretable single metric."),
        ("2. Auxiliary Composite Risk Score:",
         "R_Trust = alpha * y_AI + (1 - alpha) * R_fact (alpha = 0.5 default), used strictly for sorting and administrative ranking."),
        ("3. Epistemic Uncertainty & Quadrant Separation:",
         "NEI claims (R_fact = 0.50) represent evidence absence rather than falsehood. Clearly distinguishes Q2 (Human Rumor) from Q3 (Accurate AI).")
    ]
    add_callout_box(slide, left=6.65, top=4.50, width=6.08, height=2.30,
                    title="Two-Dimensional Risk Coordinates & Decision Quadrants",
                    items=card2_items, accent_color=PURPLE_ACCENT, bg_color=CARD_BG_WHITE)


# -----------------------------------------------------------------------------
# Slide 11: Experimental Setup (MAGE-Style Protocol & 314 Tests)
# -----------------------------------------------------------------------------
def update_slide_11(slide, fig_path=None):
    """Updates Slide 11 with MAGE protocol adaptation, 5-fold CV, and 314 tests."""
    header_tb = update_slide_header(
        slide,
        "Experimental Setup: Kazakh Adaptation of MAGE Evaluation Protocol",
        "Rigorous 2x2 matrix evaluation across seen/unseen genres and generators with cross-validation isolation"
    )

    for s in slide.shapes:
        if s.has_text_frame:
            for p in s.text_frame.paragraphs:
                if "288" in p.text:
                    p.text = p.text.replace("288/288", "314/314").replace("288", "314")
                if "ACL Kaz-MAGE" in p.text:
                    p.text = p.text.replace("ACL Kaz-MAGE", "Kazakh MAGE-Style Protocol")


# -----------------------------------------------------------------------------
# Slide 12: Topic 1 Results (Reconciling AUC vs Macro-F1 & pp Gains)
# -----------------------------------------------------------------------------
def update_slide_12(slide, fig_path=None):
    """Updates Slide 12 to reconcile metrics, report pp gains, and remove hype."""
    header_tb = update_slide_header(
        slide,
        "Topic 1 Empirical Results: Resolving the Out-of-Domain Blindspot (+42.18 pp Gain)",
        "Our Dual-Stream Morphological Gated Detector substantially mitigates domain degradation in our evaluation"
    )

    # Update Table 124 cells
    for s in slide.shapes:
        if s.has_table:
            table = s.table
            for row in table.rows:
                for cell in row.cells:
                    if "(Breakthrough)" in cell.text:
                        cell.text = cell.text.replace("+42.18% (Breakthrough)", "+42.18 pp").replace("(Breakthrough)", "")
                    if "(Perfect AUC)" in cell.text:
                        cell.text = cell.text.replace("+21.55% (Perfect AUC)", "+21.55 pp").replace("(Perfect AUC)", "")
        if s.has_text_frame:
            for p in s.text_frame.paragraphs:
                if "Perfect separation" in p.text:
                    p.text = "ROC-AUC 1.000 evaluated on held-out Qwen-2.5-7B web-crawled text"
                if "Overall Test Macro-F1" in p.text:
                    p.text = "Q1 Test ROC-AUC"
                if "Near-zero false positive rate" in p.text:
                    p.text = "Mean AUC: 99.80 +/- 0.15% across 5 folds; Test Macro-F1: 99.24%"
                if "Q3 Blindspot Resolved" in p.text:
                    p.text = "Q3 Blindspot Resolved (+42.18 pp)"
                if "AUC surges from 57.62%" in p.text:
                    p.text = "AUC increases from 57.62% to 99.80% on out-of-domain colloquial reviews"
                if "ACL Kaz-MAGE Benchmark Demonstrating +42.18%" in p.text:
                    p.text = "Figure 8. Multi-Quadrant ROC Curves on Kazakh MAGE-Style Protocol Demonstrating +42.18 pp OOD Gain."


# -----------------------------------------------------------------------------
# Slide 13: Topic 2 Results (Cross-Generator Robustness & Long-Doc Evaluation)
# -----------------------------------------------------------------------------
def update_slide_13(slide, fig_path=None):
    """Updates Slide 13 with cross-generator transfer and hybrid long-doc stress testing."""
    header_tb = update_slide_header(
        slide,
        "Topic 2 Empirical Results: Cross-Generator Robustness & Long-Document Evaluation",
        "Evaluating out-of-distribution transfer and localized synthetic paragraph detection up to 25,000 words"
    )

    for s in slide.shapes:
        if s.has_text_frame:
            for p in s.text_frame.paragraphs:
                if "Hybrid Human-AI Document Stress Testing" in p.text:
                    p.text = "Cross-Generator Robustness & Adversarial Stress Testing"


# -----------------------------------------------------------------------------
# Slide 14: Topic 3 Results (Kazakh-FEVER Pilot Benchmark)
# -----------------------------------------------------------------------------
def update_slide_14(slide, fig_path=None):
    """Updates Slide 14 with pilot benchmark framing and honest bottleneck analysis."""
    header_tb = update_slide_header(
        slide,
        "Topic 3 Empirical Results: Kazakh-FEVER Pilot Verification Benchmark",
        "Empirical evaluation of evidence retrieval, NLI claim verification, and joint FEVER scoring on curated pilot data"
    )

    for s in slide.shapes:
        if s.has_text_frame:
            for p in s.text_frame.paragraphs:
                if "Perfect separation across" in p.text:
                    p.text = "High classification fidelity across Supports, Refutes, and NEI claims in pilot set"
                if "Exact evidence sentence overlap" in p.text:
                    p.text = "Honest bottleneck analysis: Joint metric requires both exact evidence match and NLI label"


# -----------------------------------------------------------------------------
# Slide 15: Topic 3 Results (Trust Matrix Empirical Validation)
# -----------------------------------------------------------------------------
def update_slide_15(slide, fig_path=None):
    """Updates Slide 15 with 2D trust matrix coordinates and distinct risk profiles."""
    header_tb = update_slide_header(
        slide,
        "Topic 3 Empirical Results: Four-Quadrant Trust Matrix Validation",
        "Empirical validation of dual-risk coordinates separating authentic human fact from deceptive hallucination"
    )


# -----------------------------------------------------------------------------
# Slide 16: Comprehensive Component Ablations
# -----------------------------------------------------------------------------
def update_slide_16(slide, fig_path=None):
    """Updates Slide 16 to disentangle sentence-level priors from document chunking."""
    header_tb = update_slide_header(
        slide,
        "Comprehensive Component Ablation Studies: Isolating Key Architectural Gains",
        "Disentangling sentence-level morphological inductive priors from document-level aggregation modules"
    )

    for s in slide.shapes:
        if s.has_text_frame:
            for p in s.text_frame.paragraphs:
                if "FST Stream Non-Negotiable" in p.text:
                    p.text = "Core Finding 1: FST Morphological Inductive Prior (+42.18 pp)"
                if "Dynamic Gating Wins" in p.text:
                    p.text = "Core Finding 2: Dynamic Gating Outperforms Concatenation (+11.35 pp)"


# -----------------------------------------------------------------------------
# Slide 17: Gradio Dashboard Demo (Academic Prototype)
# -----------------------------------------------------------------------------
def update_slide_17(slide, fig_path=None):
    """Updates Slide 17 with academic prototype labeling and squarish Figure 13."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fig_path is None:
        fig_path = os.path.join(project_root, "presentation_figures", "fig13_gradio_dashboard_panels.png")

    header_tb = update_slide_header(
        slide,
        "System Demonstration: Interactive 4-Tab Gradio Academic Prototype",
        "Explainable AI detection and evidence-grounded verification prototype designed for academic integrity"
    )

    for shape in list(slide.shapes):
        if shape.left < Inches(6.50) and shape.top > Inches(1.80) and shape.name not in PROTECTED_SHAPE_NAMES:
            remove_shape(shape)

    left = Inches(0.60)
    top = Inches(2.05)
    width = Inches(5.85)
    height = Inches(4.35)
    slide.shapes.add_picture(fig_path, left, top, width, height)

    caption_box = slide.shapes.add_textbox(left, Inches(6.44), width, Inches(0.30))
    tf = caption_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = "Figure 13. Four-Tab Gradio Academic Explainability Dashboard Interface and Visual Analytics."
    p.font.name = FONT_TITLE
    p.font.size = Pt(8.5)
    p.font.italic = True
    p.font.color.rgb = SLATE_MUTED
    p.alignment = PP_ALIGN.CENTER


# -----------------------------------------------------------------------------
# Slide 18: System Demo (Deployment & Test Rigor — 314 Tests)
# -----------------------------------------------------------------------------
def update_slide_18(slide, fig_path=None):
    """Updates Slide 18 with 314 tests, ~1s cold start, and containerized prototype."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fig_path is None:
        fig_path = os.path.join(project_root, "presentation_figures", "fig14_cloud_deployment_pipeline.png")

    header_tb = update_slide_header(
        slide,
        "System Demonstration: Cloud Packaging, Hugging Face Spaces & Test Rigor",
        "Containerized prototype deployment bundle with sub-1.2s cold start, multi-format ingestion, and 314 passing tests"
    )

    for shape in list(slide.shapes):
        if shape == header_tb or shape.name in PROTECTED_SHAPE_NAMES or shape.top < Inches(1.20):
            continue
        remove_shape(shape)

    left = Inches(0.60)
    top = Inches(2.05)
    width = Inches(5.85)
    height = Inches(4.35)
    slide.shapes.add_picture(fig_path, left, top, width, height)

    caption_box = slide.shapes.add_textbox(left, Inches(6.44), width, Inches(0.30))
    tf = caption_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = "Figure 14. Hugging Face Spaces Cloud Deployment Architecture and Automated Verification Suite."
    p.font.name = FONT_TITLE
    p.font.size = Pt(8.5)
    p.font.italic = True
    p.font.color.rgb = SLATE_MUTED
    p.alignment = PP_ALIGN.CENTER

    add_card(slide, left=6.65, top=2.05, width=1.92, height=1.85,
             title="Automated Test Suite", value_str="314 / 314",
             subtitle="Passing unit & integration tests with 0 regressions",
             accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN)
    add_card(slide, left=8.72, top=2.05, width=1.92, height=1.85,
             title="Cold-Start Latency", value_str="< 1.2s",
             subtitle="Sub-1.2s (~1s) initialization with CPU/GPU dual paths",
             accent_color=BLUE_ACCENT, bg_color=CARD_BG_WHITE)
    add_card(slide, left=10.79, top=2.05, width=1.94, height=1.85,
             title="Multi-Format Ingestion", value_str="3 Formats",
             subtitle="Defensive parsing for .txt, .docx, .pdf with 10MB memory guards",
             accent_color=TEAL_ACCENT, bg_color=CARD_BG_WHITE)

    add_callout_box(
        slide, left=6.65, top=4.05, width=6.08, height=2.75,
        title="Hugging Face Spaces Containerized Prototype Features (hf_space/)",
        items=[
            "1. Standalone Container Bundle: Decoupled from heavy cluster weights; utilizes distilled models and defensive mock fallbacks for resilient cloud hosting.",
            "2. Defensive File Ingestion Engine: Safely ingests .txt, .docx, and .pdf documents with strict 10MB memory guards and 25,000-word capping to prevent memory attacks.",
            "3. Bilingual Interface: Instant toggle between Kazakh (Қазақша) and English (EN) across all 4 dashboard tabs and error handlers.",
            "4. Rigorous Continuous Integration: 314 / 314 passing unit and regression tests verifying all mathematical and engineering invariants."
        ],
        accent_color=PURPLE_ACCENT, bg_color=CARD_BG_WHITE
    )


# -----------------------------------------------------------------------------
# Slide 19: Conclusion — Contributions & Writing Progress (~85%)
# -----------------------------------------------------------------------------
def update_slide_19(slide):
    """Updates Slide 19 with 4 Academic Contributions and Manuscript Status (~85%)."""
    header_tb = update_slide_header(
        slide,
        "Conclusion: Summary of Thesis Contributions & Writing Progress",
        "Master's thesis completion estimated at 85%; core theoretical, empirical, and prototype milestones achieved"
    )

    for shape in list(slide.shapes):
        if shape == header_tb or shape.name in PROTECTED_SHAPE_NAMES or shape.top < Inches(1.20):
            continue
        remove_shape(shape)

    contributions_items = [
        (
            "1. Algorithmic Contribution: Dual-Stream Morphology-Aware Gated Fusion:",
            "Pioneered the integration of rule-based 83-rule FST morphological representations with transformer semantic backbones via learned dynamic gating, substantially mitigating out-of-domain collapse (+42.18 pp OOD gain)."
        ),
        (
            "2. Methodological & Practical Extension: Cross-Generator Robustness & Chunk Aggregation:",
            "Established generator-invariant detection under unseen LLMs and engineered sentence-preserving chunking with 10 Kazakh abbreviation guards and dynamic Top-K pooling for documents up to 25,000 words."
        ),
        (
            "3. Trustworthy Verification Contribution: Evidence-Grounded Kazakh Verification Pilot:",
            "Constructed the first curated Kazakh-FEVER pilot benchmark (36 articles, 120 claims) and two-dimensional Four-Quadrant Trust Matrix decoupling factual veracity from AI style."
        ),
        (
            "4. Engineering & Prototype Contribution: Interactive Dashboard & Cloud Serving Platform:",
            "Delivered a reproducible 4-tab Gradio prototype, multi-format ingestion engine (.txt, .docx, .pdf), and 314 / 314 passing automated regression tests."
        )
    ]

    add_callout_box(
        slide, left=0.60, top=2.05, width=6.65, height=4.75,
        title="Four Primary Academic & System Contributions of the Thesis",
        items=contributions_items, accent_color=NAVY_PRIMARY, bg_color=CARD_BG_WHITE
    )

    manuscript_items = [
        ("Chapter 1: Introduction (100% Complete)", "- Motivation, Turkic linguistic context, problem formulation."),
        ("Chapter 2: Related Work (100% Complete)", "- SOTA detectors, LLM watermarks, fact-checking corpora."),
        ("Chapter 3: Methodology (100% Complete)", "- Topic 1 Morphological Gated Fusion & SupCon formulations."),
        ("Chapter 4: Experiments & Robustness (100% Complete)", "- Topic 2 MAGE-style 2x2 matrix, cross-generator evaluation."),
        ("Chapter 5: Evidence-Grounded Verification (90% Complete)", "- Topic 3 Kazakh-FEVER pilot benchmark, 2D Trust Matrix."),
        ("Chapter 6: System Prototype & Roadmap (70% Complete)", "- Gradio prototype, Paper 2 adversarial rewriting attack roadmap.")
    ]

    add_callout_box(
        slide, left=7.45, top=2.05, width=5.28, height=4.75,
        title="Master's Thesis Manuscript Status (~85% Complete)",
        items=manuscript_items, accent_color=GREEN_ACCENT, bg_color=CARD_BG_GREEN
    )


# -----------------------------------------------------------------------------
# Slide 20: Meeting Agenda & Springer LNCS Acceptance
# -----------------------------------------------------------------------------
def update_slide_20(slide):
    """Updates Slide 20 with Springer LNCS details, Gradio agenda, 314 tests, and next steps."""
    update_slide_header(
        slide,
        "Current Progress: LNCS Acceptance & September Meeting Agenda",
        "论文录用进展、9月课题组汇报演示计划与后续工作推进安排"
    )

    cards_data = [
        {
            "header": "MILESTONE 1: ACCEPTED\nSpringer LNCS (AIST 2026)",
            "bullets": [
                "- Paper 1: Morphologically-Grounded AI Detection in Low-Resource Kazakh.",
                "- Status: Officially accepted to Springer LNCS; camera-ready finalized.",
                "- Contribution: Formally validates dual-stream gated fusion and MAGE-style benchmark.",
            ],
            "x_range": (0.4, 3.2),
            "header_name": "TextBox 126",
            "body_name": "TextBox 127",
        },
        {
            "header": "MILESTONE 2: MEETING AGENDA\nLive System Demonstration",
            "bullets": [
                "- Interactive 4-tab Gradio system prepared for today's meeting demonstration.",
                "- Tab 1: Real-time sentence explainability heatmap and discourse reasoning.",
                "- Tab 2: Morphological FST Lab with live root/affix decomposition.",
                "- Tab 3 & 4: Benchmark explorer and Kazakh-FEVER trust matrix.",
            ],
            "x_range": (3.2, 6.3),
            "header_name": "TextBox 130",
            "body_name": "TextBox 131",
        },
        {
            "header": "MILESTONE 3: SEPTEMBER PROGRESS\nResearch Milestones (~85%)",
            "bullets": [
                "- Curated 10,000+ benchmark dataset across News, Wikipedia, and Consumer Reviews.",
                "- Kazakh-FEVER: Curated 36 articles and 120 verified claims; 100% NLI Macro-F1.",
                "- Engineering: Micro-batching (<1.4 GB VRAM); 314 automated passing tests.",
            ],
            "x_range": (6.3, 9.4),
            "header_name": "TextBox 134",
            "body_name": "TextBox 135",
        },
        {
            "header": "MILESTONE 4: NEXT STEPS\nGuidance Requested from Prof. Guo",
            "bullets": [
                "- Feedback Integration: Incorporate Professor Guo's advice on framework diagrams and thesis draft.",
                "- Paper 2 Preparation: Draft manuscript on Kazakh-FEVER & Four-Quadrant Trust Matrix.",
                "- Chapter Finalization: Schedule internal laboratory review for Chapters 5 & 6.",
            ],
            "x_range": (9.4, 13.0),
            "header_name": "TextBox 138",
            "body_name": "TextBox 139",
        },
    ]

    shapes_by_name = {s.name: s for s in slide.shapes}

    for card in cards_data:
        x_min, x_max = card["x_range"]
        header_tb = shapes_by_name.get(card["header_name"])
        body_tb = shapes_by_name.get(card["body_name"])

        if header_tb is None or body_tb is None:
            col_tbs = [
                s for s in slide.shapes
                if s.has_text_frame and s.top > Inches(1.80)
                and Inches(x_min) <= s.left <= Inches(x_max)
            ]
            col_tbs.sort(key=lambda s: s.top)
            if len(col_tbs) >= 2:
                header_tb = col_tbs[0]
                body_tb = col_tbs[1]

        if header_tb is not None:
            tf_head = header_tb.text_frame
            tf_head.word_wrap = True
            tf_head.margin_left = tf_head.margin_top = tf_head.margin_right = tf_head.margin_bottom = 0
            tf_head.text = ""
            p_head = tf_head.paragraphs[0]
            p_head.text = card["header"]
            p_head.font.name = FONT_TITLE
            p_head.font.size = Pt(9.5)
            p_head.font.bold = True
            p_head.font.color.rgb = WHITE

        if body_tb is not None:
            body_tb.height = Inches(4.05)
            tf_body = body_tb.text_frame
            tf_body.word_wrap = True
            tf_body.margin_left = tf_body.margin_top = tf_body.margin_right = tf_body.margin_bottom = 0
            tf_body.text = ""
            for i, pt in enumerate(card["bullets"]):
                p = tf_body.paragraphs[0] if i == 0 else tf_body.add_paragraph()
                p.text = pt
                p.font.name = FONT_BODY
                p.font.size = Pt(8.8)
                p.font.bold = False
                p.font.color.rgb = SLATE_BODY
                if i > 0:
                    p.space_before = Pt(2.5)


# -----------------------------------------------------------------------------
# Slide 21: Discussion — 3 Core Strategic Guidance Questions for Prof. Guo
# -----------------------------------------------------------------------------
def update_slide_21(slide):
    """Refocuses Slide 21 on 3 core strategic decisions for Prof. Guo."""
    header_tb = update_slide_header(
        slide,
        "Discussion: Guidance Requests & Strategic Questions for Prof. Guo",
        "Three core strategic decisions regarding Topic 2 robustness experiments, Kazakh-FEVER expansion, and Paper 2 target venue"
    )

    for shape in list(slide.shapes):
        if shape == header_tb or shape.name in PROTECTED_SHAPE_NAMES or shape.top < Inches(1.20):
            continue
        remove_shape(shape)

    guidance_items = [
        (
            "1. Topic 2 Adversarial Rewriting Attack Benchmark Execution:",
            "How should we structure the experimental suite for paraphrasing, synonym replacement, and morphological inflection perturbation for Paper 2? Recommended approach: evaluate rule-based synonym substitution and LLM paraphrasing on held-out test splits."
        ),
        (
            "2. Kazakh-FEVER Evidence Retrieval Expansion Strategy:",
            "Our pilot evaluation features 36 curated articles and 120 claims (Macro-F1: 100.0%, Joint FEVER: 66.67%). Should we expand to 100+ articles with dense retrieval (BM25 vs. multilingual dense embeddings) prior to Paper 2 submission?"
        ),
        (
            "3. Paper 2 Framing & Target Venue Alignment:",
            "Option A: Algorithmic methodology paper targeting EMNLP (focusing on the Dual-Risk Trust Matrix and dynamic gated optimization).\nOption B: Benchmark & resource paper targeting LREC/COLING (highlighting the Kazakh-FEVER corpus and Central Asian NLP gap).\nWhat is Professor Guo's advice on venue alignment and narrative emphasis?"
        )
    ]

    add_callout_box(
        slide, left=0.60, top=2.05, width=12.133, height=4.75,
        title="Three Core Strategic Consultation Points for Professor Guo",
        items=guidance_items, accent_color=NAVY_PRIMARY, bg_color=CARD_BG_WHITE
    )


# -----------------------------------------------------------------------------
# Slide 22: Committee Comments & Responses (Academic Status Badges)
# -----------------------------------------------------------------------------
def update_slide_22(slide):
    """Replaces absolute RESOLVED badges with academic progress indicators."""
    update_slide_header(
        slide,
        "Committee Review Comments & Responses: Addressing Expert Feedback",
        "Itemized academic revisions: Addressing internal and international committee feedback with verified empirical rigor"
    )

    status_map_table1 = {
        0: "Status",
        1: "Addressed in Thesis Chapter 4",
        2: "Addressed in Thesis Chapter 4 & 6",
        3: "Addressed in Thesis Chapter 5",
    }

    status_map_table2 = {
        0: "Status",
        1: "Addressed in Thesis Chapter 6 (314 Tests)",
        2: "Addressed in Thesis Chapter 3",
        3: "Addressed in Thesis Chapter 5 & 6",
    }

    tables = [s for s in slide.shapes if s.has_table]
    tables.sort(key=lambda s: s.left)

    if len(tables) >= 1:
        t1 = tables[0].table
        for row_idx, row in enumerate(t1.rows):
            if row_idx in status_map_table1:
                row.cells[3].text = status_map_table1[row_idx]
                for p in row.cells[3].text_frame.paragraphs:
                    p.font.name = FONT_TITLE
                    p.font.size = Pt(8.0)
                    p.font.bold = True
                    p.font.color.rgb = NAVY_PRIMARY

    if len(tables) >= 2:
        t2 = tables[1].table
        for row_idx, row in enumerate(t2.rows):
            if row_idx == 1:
                # Also update 288/288 to 314/314 in action cell
                row.cells[2].text = "Created standalone Hugging Face Spaces bundle (hf_space/); 314/314 passing tests; sub-1.2s cold start."
                for p in row.cells[2].text_frame.paragraphs:
                    p.font.name = FONT_BODY
                    p.font.size = Pt(7.8)
                    p.font.color.rgb = SLATE_BODY
            if row_idx in status_map_table2:
                row.cells[3].text = status_map_table2[row_idx]
                for p in row.cells[3].text_frame.paragraphs:
                    p.font.name = FONT_TITLE
                    p.font.size = Pt(8.0)
                    p.font.bold = True
                    p.font.color.rgb = NAVY_PRIMARY


# -----------------------------------------------------------------------------
# Main Orchestrator Function
# -----------------------------------------------------------------------------
def update_presentation(ppt_path, fig_path=None, fig14_path=None):
    """Updates all relevant slides in the presentation deck."""
    if not os.path.isfile(ppt_path):
        raise FileNotFoundError(f"Presentation file not found: {ppt_path}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fig_path is None:
        fig_path = os.path.join(project_root, "presentation_figures", "fig15_methodological_innovations_framework.png")
        if not os.path.isfile(fig_path):
            fig_path = os.path.join(project_root, "presentation_figures", "fig03_methodological_innovation_framework.png")
        if not os.path.isfile(fig_path):
            fig_path = os.path.join(project_root, "presentation_figures", "fig03_end_to_end_pipeline.png")

    print(f"Loading presentation: {ppt_path}")
    prs = Presentation(ppt_path)
    total_slides = len(prs.slides)
    print(f"Total slides found: {total_slides}")
    if total_slides < 22:
        raise ValueError(f"Presentation must contain at least 22 slides, found {total_slides}")

    # Slide 1: Notes cleanup
    print("Updating Slide 1 (Presenter Notes)...")
    update_slide_01(prs.slides[0])

    # Slide 2: Sequential Table of Contents
    print("Updating Slide 2 (Contents Sequential Reordering)...")
    update_slide_02(prs.slides[1])

    # Slide 5: Related work term cleanup
    print("Updating Slide 5 (Related Work Terminology Cleanup)...")
    update_slide_05(prs.slides[4])

    # Slide 7: Comprehensive Methodological Framework (Figure 3)
    print("Updating Slide 7 (Methodological Framework Figure 3)...")
    update_slide_07(prs.slides[6], fig_path)

    # Slide 8: Topic 1 Architecture (Dual-Stream Gated Fusion)
    print("Updating Slide 8 (Topic 1 Architecture)...")
    update_slide_08(prs.slides[7])

    # Slide 9: Topic 2 Architecture (Robust Generalization & Adversarial Defense)
    print("Updating Slide 9 (Topic 2 Architecture)...")
    update_slide_09(prs.slides[8])

    # Slide 10: Topic 3 Architecture (Claim Verification & Trust Matrix)
    print("Updating Slide 10 (Topic 3 Architecture)...")
    update_slide_10(prs.slides[9])

    # Slide 11: Experimental Setup (MAGE Protocol & 314 Tests)
    print("Updating Slide 11 (Experimental Setup)...")
    update_slide_11(prs.slides[10])

    # Slide 12: Topic 1 Empirical Results (OOD Gain & AUC vs F1)
    print("Updating Slide 12 (Topic 1 Results)...")
    update_slide_12(prs.slides[11])

    # Slide 13: Topic 2 Empirical Results (Cross-Generator Robustness)
    print("Updating Slide 13 (Topic 2 Results)...")
    update_slide_13(prs.slides[12])

    # Slide 14: Topic 3 Empirical Results (Kazakh-FEVER Pilot Benchmark)
    print("Updating Slide 14 (Topic 3 Results: Kazakh-FEVER)...")
    update_slide_14(prs.slides[13])

    # Slide 15: Topic 3 Empirical Results (Trust Matrix Validation)
    print("Updating Slide 15 (Topic 3 Results: Trust Matrix)...")
    update_slide_15(prs.slides[14])

    # Slide 16: Comprehensive Component Ablations
    print("Updating Slide 16 (Component Ablation Studies)...")
    update_slide_16(prs.slides[15])

    # Slide 17: System Demonstration (Gradio Academic Prototype)
    print("Updating Slide 17 (Gradio Prototype Demo)...")
    update_slide_17(prs.slides[16])

    # Slide 18: System Demonstration (Cloud Packaging & 314 Tests)
    print("Updating Slide 18 (Deployment Architecture & 314 Tests)...")
    update_slide_18(prs.slides[17], fig14_path)

    # Slide 19: Conclusion (Contributions & Writing Progress ~85%)
    print("Updating Slide 19 (Contributions & Manuscript Status)...")
    update_slide_19(prs.slides[18])

    # Slide 20: Meeting Agenda & Springer LNCS Acceptance
    print("Updating Slide 20 (LNCS Acceptance & September Meeting Agenda)...")
    update_slide_20(prs.slides[19])

    # Slide 21: Discussion (Guidance Requests for Prof. Guo)
    print("Updating Slide 21 (Strategic Guidance Requests)...")
    update_slide_21(prs.slides[20])

    # Slide 22: Committee Comments & Responses
    print("Updating Slide 22 (Committee Revisions & Academic Status Badges)...")
    update_slide_22(prs.slides[21])

    # Save presentation directly back
    print(f"Saving presentation directly back to: {ppt_path}")
    prs.save(ppt_path)

    # Mirror to Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx in project root
    mirror_path = os.path.join(project_root, "Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx")
    if os.path.abspath(ppt_path) != os.path.abspath(mirror_path):
        try:
            shutil.copyfile(ppt_path, mirror_path)
            print(f"Mirrored presentation to: {mirror_path}")
        except Exception as e:
            print(f"Warning: could not mirror presentation to {mirror_path}: {e}")

    # Mirror to Desktop copies if applicable
    desktop_guo = os.path.join(os.path.expanduser("~"), "Desktop", "Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx")
    desktop_anekesh = os.path.join(os.path.expanduser("~"), "Desktop", "AnekeshD_Progress.pptx")

    for d_path in [desktop_guo, desktop_anekesh]:
        if os.path.abspath(ppt_path) != os.path.abspath(d_path):
            try:
                shutil.copyfile(ppt_path, d_path)
                print(f"Mirrored presentation to Desktop: {d_path}")
            except Exception as e:
                print(f"Warning: could not mirror presentation to {d_path}: {e}")

    print("Presentation update completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Update presentation deck with Professor Guo's feedback.")
    default_ppt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx")
    if not os.path.isfile(default_ppt):
        default_ppt = os.path.join(os.path.expanduser("~"), "Desktop", "Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx")
    if not os.path.isfile(default_ppt):
        default_ppt = os.path.join(os.path.expanduser("~"), "Desktop", "AnekeshD_Progress.pptx")

    parser.add_argument("--ppt-path", default=default_ppt, help="Path to PPTX file")
    parser.add_argument("--fig-path", "--fig15-path", "--fig03-path", dest="fig_path", default=None, help="Path to framework figure")
    parser.add_argument("--fig14-path", default=None, help="Path to cloud deployment figure")
    args = parser.parse_args()

    update_presentation(args.ppt_path, args.fig_path, args.fig14_path)


if __name__ == "__main__":
    main()
