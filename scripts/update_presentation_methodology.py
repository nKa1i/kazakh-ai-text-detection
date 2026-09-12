# -*- coding: utf-8 -*-
"""
Presentation Updater for Slides 19 and 20.
Directly modifies AnekeshD_Progress.pptx to:
1. Embed Figure 15 (Methodological Innovations Framework) on Slide 19.
2. Update Slide 19 header to bilingual technical framework title and add formal caption.
3. Update Slide 20 header to LNCS Acceptance & September Meeting Agenda.
4. Update the 4 milestone cards on Slide 20 with Springer LNCS details, Gradio demo agenda,
   September research milestones, and guidance requests for Prof. Guo.
5. Strictly preserves all other slides (Slides 1-18, 21-23) and existing styling.
"""

import os
import sys
import argparse
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

# -----------------------------------------------------------------------------
# Color Palette & Typography (Matching Presentation Theme)
# -----------------------------------------------------------------------------
NAVY_PRIMARY = RGBColor(30, 58, 138)       # #1E3A8A - Deep Academic Navy Blue
BLUE_ACCENT = RGBColor(37, 99, 235)        # #2563EB - Royal Blue Accent
AMBER_ACCENT = RGBColor(217, 119, 6)       # #D97706 - Amber Accent
GREEN_ACCENT = RGBColor(22, 163, 74)       # #16A34A - Green Accent
SLATE_BODY = RGBColor(51, 65, 85)          # #334155 - Standard body prose
SLATE_MUTED = RGBColor(100, 116, 139)      # #64748B - Subtitles & captions
WHITE = RGBColor(255, 255, 255)            # #FFFFFF - Crisp white

FONT_TITLE = "Arial"
FONT_BODY = "Arial"
FONT_ZH = "Microsoft YaHei"

PROTECTED_SHAPE_NAMES = {
    "灯片编号占位符 10",
    "直接连接符 6",
    "矩形 4",
    "矩形 29",
}


def remove_shape(shape):
    """Remove a shape element cleanly from its slide XML."""
    sp = shape._element
    sp.getparent().remove(sp)


def update_slide_header(slide, title_text, subtitle_text):
    """
    Finds existing header textbox or creates a new one at top=1.30.
    Updates title and bilingual subtitle with consistent typography.
    """
    header_tb = None
    for shape in slide.shapes:
        if shape.has_text_frame:
            # Header textbox is positioned horizontally around 0.6 in and top 1.25-1.5 in
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
        p1.font.name = FONT_ZH
        p1.font.size = Pt(10.0)
        p1.font.bold = False
        p1.font.color.rgb = SLATE_MUTED
        p1.space_before = Pt(2)

    return header_tb


def update_slide_19(slide, fig15_path):
    """
    Updates Slide 19 (Index 18):
    - Updates slide header to Methodological Innovations.
    - Cleans up legacy plain text callout boxes below top > 1.2 in.
    - Embeds Figure 15 diagram at specified coordinates.
    - Adds formal caption below figure.
    """
    if not os.path.isfile(fig15_path):
        raise FileNotFoundError(f"Figure 15 image file not found: {fig15_path}")

    # 1. Update header textbox
    header_tb = update_slide_header(
        slide,
        "Methodological Innovations: Comprehensive Technical Framework",
        "面向低资源哈萨克语的AI生成文本检测与事实核验总体方法创新架构与技术流程"
    )

    # 2. Remove legacy shapes below top > Inches(1.2) while protecting nav/branding elements
    for shape in list(slide.shapes):
        if shape == header_tb:
            continue
        if shape.name in PROTECTED_SHAPE_NAMES:
            continue
        if shape.top < Inches(1.20):
            # Section navigation labels at top bar
            continue
        remove_shape(shape)

    # 3. Insert Figure 15 picture
    left = Inches(0.60)
    top = Inches(1.85)
    width = Inches(12.133)
    height = Inches(4.85)
    slide.shapes.add_picture(fig15_path, left, top, width, height)

    # 4. Add formal caption textbox below picture
    caption_top = top + height + Inches(0.04)  # ~6.74 in
    caption_height = Inches(0.32)
    caption_box = slide.shapes.add_textbox(left, caption_top, width, caption_height)
    tf = caption_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = "Figure 15. Overall Methodological Innovation Framework for Kazakh AI-Generated Text Detection and Factual Verification."
    p.font.name = FONT_TITLE
    p.font.size = Pt(8.5)
    p.font.italic = True
    p.font.color.rgb = SLATE_MUTED
    p.alignment = PP_ALIGN.CENTER


def update_slide_20(slide):
    """
    Updates Slide 20 (Index 19):
    - Updates slide header to LNCS Acceptance & September Meeting Agenda.
    - Updates 4 milestone cards with Springer LNCS details, Gradio demo agenda,
      September milestones, and next research steps for Prof. Guo.
    """
    # 1. Update header textbox
    update_slide_header(
        slide,
        "Current Progress: LNCS Acceptance & September Meeting Agenda",
        "论文录用进展、9月课题组汇报演示计划与后续工作推进安排"
    )

    # Milestone card content definition
    cards_data = [
        {
            "header": "MILESTONE 1: ACCEPTED\nSpringer LNCS (AIST 2026)",
            "bullets": [
                "- Paper 1: Morphologically-Grounded AI Detection in Low-Resource Kazakh.",
                "- Status: Officially accepted to Springer LNCS; camera-ready finalized.",
                "- Contribution: Formally validates dual-stream cross-attention and Kaz-MAGE benchmark.",
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
                "- Tab 3 & 4: Kaz-MAGE benchmark explorer and Kazakh-FEVER trust matrix.",
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
                "- Engineering: Micro-batching (<1.4 GB VRAM); 302 automated passing tests.",
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

    # Find textboxes by name or coordinate range
    shapes_by_name = {s.name: s for s in slide.shapes}

    for card in cards_data:
        x_min, x_max = card["x_range"]
        header_tb = shapes_by_name.get(card["header_name"])
        body_tb = shapes_by_name.get(card["body_name"])

        # Fallback to coordinate matching if names do not match
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


def update_presentation(ppt_path, fig15_path=None):
    """Main function to update Slide 19 and 20 in the presentation."""
    if not os.path.isfile(ppt_path):
        raise FileNotFoundError(f"Presentation file not found: {ppt_path}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fig15_path is None:
        fig15_path = os.path.join(project_root, "presentation_figures", "fig15_methodological_innovations_framework.png")

    if not os.path.isfile(fig15_path):
        raise FileNotFoundError(f"Figure 15 not found at: {fig15_path}")

    print(f"Loading presentation: {ppt_path}")
    prs = Presentation(ppt_path)
    total_slides = len(prs.slides)
    print(f"Total slides found: {total_slides}")
    if total_slides < 20:
        raise ValueError(f"Presentation must contain at least 20 slides, found {total_slides}")

    # Slide 19 (Index 18)
    print("Updating Slide 19 (Index 18: Methodological Innovations)...")
    s19 = prs.slides[18]
    update_slide_19(s19, fig15_path)

    # Slide 20 (Index 19)
    print("Updating Slide 20 (Index 19: LNCS Acceptance & September Meeting Agenda)...")
    s20 = prs.slides[19]
    update_slide_20(s20)

    # Save presentation
    print(f"Saving presentation directly back to: {ppt_path}")
    prs.save(ppt_path)
    print("Presentation update completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Update Slides 19 and 20 of AnekeshD_Progress.pptx.")
    default_ppt = os.path.join(os.path.expanduser("~"), "Desktop", "AnekeshD_Progress.pptx")
    parser.add_argument("--ppt-path", default=default_ppt, help="Path to AnekeshD_Progress.pptx")
    parser.add_argument("--fig15-path", default=None, help="Path to fig15 image file")
    args = parser.parse_args()

    update_presentation(args.ppt_path, args.fig15_path)


if __name__ == "__main__":
    main()
