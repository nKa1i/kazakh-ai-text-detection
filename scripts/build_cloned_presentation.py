# -*- coding: utf-8 -*-
"""
Presentation Template Cloning & Branding Engine.
Clones baseline presentation 'Aliya PPT.pptx' directly, preserves native institutional
branding (KazNU logo, NPU logo group, watermark, and Layout 7 NPU pentagon badge),
removes all commercial watermarks ('合作QQ： 243001978'), repositions the native gradient
highlight tracker ('矩形 29') across sections, clears legacy poultry shapes on content
slides, and saves 'Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx'.

Presenter: 大雷 (Daulet)
Advisor: 郭教授 (Prof. Guo)
Date: 2026年9月
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

# Institutional color tokens
COLOR_NAVY = RGBColor(30, 58, 138)       # #1E3A8A
COLOR_SLATE_BODY = RGBColor(51, 65, 85)   # #334155
COLOR_SLATE_MUTED = RGBColor(100, 116, 139) # #64748B

BASELINE_PPT_PATH = r"C:\Users\Roza\Downloads\Aliya PPT.pptx"
DEFAULT_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx"
)

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


def update_cover_slide(slide):
    """
    Slide 1 (Cover Page):
    - Preserve: Изображение 16 (KazNU logo), LOGO组合 (NPU logo group),
      校徽打底 (NPU watermark), 图书馆照片 (Library photo)
    - Excision: Remove '合作QQ： 243001978' and legacy '汇报人组合' groups
    - Update: 主标题 (Title + Subtitle) and 汇报人 (Presenter + Advisor + Degree)
    """
    shapes_to_remove = []
    for shp in slide.shapes:
        # Commercial watermark removal
        if shp.has_text_frame and ("243001978" in shp.text or "合作QQ" in shp.text):
            shapes_to_remove.append(shp)
        elif shp.name == "合作QQ： 243001978":
            shapes_to_remove.append(shp)
        # Remove legacy poultry student placeholder groups
        elif shp.name == "汇报人组合":
            shapes_to_remove.append(shp)

    for shp in shapes_to_remove:
        remove_shape(shp)

    # Update 主标题
    for shp in slide.shapes:
        if shp.name == "主标题":
            shp.left = Inches(0.34)
            shp.top = Inches(4.05)
            shp.width = Inches(12.60)
            shp.height = Inches(1.50)

            tf = shp.text_frame
            tf.word_wrap = True

            # Paragraph 0: English thesis title
            p0 = tf.paragraphs[0]
            p0.text = "Research on Morphologically-Grounded AI-Generated Text Detection and Factual Verification for Low-Resource Kazakh"
            p0.font.name = "Arial"
            p0.font.size = Pt(22)
            p0.font.bold = True
            p0.font.color.rgb = COLOR_NAVY
            p0.alignment = PP_ALIGN.LEFT

            # Paragraph 1: Chinese thesis subtitle
            p1 = tf.add_paragraph()
            p1.text = "面向低资源哈萨克语的形态感知AI生成文本检测与事实核验研究"
            p1.font.name = "Microsoft YaHei"
            p1.font.size = Pt(17)
            p1.font.bold = True
            p1.font.color.rgb = COLOR_NAVY
            p1.alignment = PP_ALIGN.LEFT

        elif shp.name == "汇报人":
            shp.left = Inches(0.45)
            shp.top = Inches(5.80)
            shp.width = Inches(12.00)
            shp.height = Inches(1.10)

            tf = shp.text_frame
            tf.word_wrap = True

            # Paragraph 0: Presenter, Advisor, Date
            p0 = tf.paragraphs[0]
            p0.text = "汇报人：大雷 (Daulet)      导师：郭教授 (Prof. Guo)      2026年9月"
            p0.font.name = "Microsoft YaHei"
            p0.font.size = Pt(15)
            p0.font.bold = True
            p0.font.color.rgb = COLOR_NAVY
            p0.alignment = PP_ALIGN.LEFT

            # Paragraph 1: Degree & Academic Metadata
            p1 = tf.add_paragraph()
            p1.text = "Degree: Master of Science in Computer Science and Technology | School of Computer Science, NPU & KazNU"
            p1.font.name = "Arial"
            p1.font.size = Pt(12)
            p1.font.bold = False
            p1.font.color.rgb = COLOR_SLATE_MUTED
            p1.alignment = PP_ALIGN.LEFT


def update_toc_slide(slide):
    """
    Slide 2 (Table of Contents / 目录):
    - Preserve: 海天苑, 背景色块, 目录英文 ('CONTENTS'), 打底色块
    - Excision: Remove '合作QQ： 243001978'
    - Update 6 numbered sections
    """
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
                    p.font.name = "Microsoft YaHei"
                    p.font.size = Pt(16)
                    p.font.bold = True
                    p.font.color.rgb = COLOR_NAVY
        elif shp.name == "学科特色鲜明" and shp.shape_type != 6:  # Standalone
            shp.width = Inches(7.2)
            p = shp.text_frame.paragraphs[0]
            p.text = toc_sections["standalone"]
            p.font.name = "Microsoft YaHei"
            p.font.size = Pt(16)
            p.font.bold = True
            p.font.color.rgb = COLOR_NAVY


def update_content_slides(slides):
    """
    Slides 3-22 (Content Slides):
    - Preserve: Layout 7 ('1_标题和内容') with native master slide NPU pentagon badge,
      top bar background ('矩形 4'), section text boxes ('TextBox 6', 'TextBox 9',
      'TextBox 10', 'TextBox 11'...), top separator line ('直接连接符 6'),
      slide number placeholder ('灯片编号占位符 10').
    - Reposition: '矩形 29' (Native Gradient Highlight Tracker) across sections.
    - Clear legacy content: Delete shapes with top > Inches(1.2) except top separator
      and slide number.
    - Watermark excision: Delete any shape containing '243001978' or '合作QQ'.
    """
    for s_idx in range(2, 22):
        slide = slides[s_idx]

        # 1. Reposition 矩形 29
        if s_idx in TRACKER_SCHEDULE:
            target_left, target_width = TRACKER_SCHEDULE[s_idx]
            for shp in slide.shapes:
                if shp.name == "矩形 29":
                    shp.left = target_left
                    shp.width = target_width

        # 2. Clear legacy shapes (top > 1.2 inches) except line and slide number
        shapes_to_remove = []
        for shp in slide.shapes:
            # Check for watermarks first
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

            # Check if legacy content shape below top header
            if shp.top > Inches(1.2):
                if shp.name.startswith("直接连接符"):
                    continue
                if shp.name.startswith("灯片编号"):
                    continue
                shapes_to_remove.append(shp)

        for shp in shapes_to_remove:
            remove_shape(shp)


def update_closing_slide(slide):
    """
    Slide 23 (Closing Slide):
    - Preserve: 背景色块 1, 背景色块 2, 长安校区 photo, 点缀线段
    - Excision: Remove '合作QQ： 243001978'
    - Update: 敬请各位批评指正 with bilingual closing and student/advisor info
    """
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
            p0.font.name = "Arial"
            p0.font.size = Pt(32)
            p0.font.bold = True
            p0.font.color.rgb = COLOR_NAVY
            p0.alignment = PP_ALIGN.CENTER

            p1 = tf.add_paragraph()
            p1.text = "面向低资源哈萨克语的形态感知AI生成文本检测与事实核验研究"
            p1.font.name = "Microsoft YaHei"
            p1.font.size = Pt(17)
            p1.font.bold = True
            p1.font.color.rgb = COLOR_NAVY
            p1.alignment = PP_ALIGN.CENTER

            p2 = tf.add_paragraph()
            p2.text = "汇报人：大雷 (Daulet) | 导师：郭教授 (Prof. Guo)"
            p2.font.name = "Microsoft YaHei"
            p2.font.size = Pt(15)
            p2.font.bold = True
            p2.font.color.rgb = COLOR_SLATE_BODY
            p2.alignment = PP_ALIGN.CENTER

            p3 = tf.add_paragraph()
            p3.text = "请郭老师批评指正"
            p3.font.name = "Microsoft YaHei"
            p3.font.size = Pt(15)
            p3.font.bold = False
            p3.font.color.rgb = COLOR_SLATE_MUTED
            p3.alignment = PP_ALIGN.CENTER


def build_cloned_presentation(input_path=BASELINE_PPT_PATH, output_path=DEFAULT_OUTPUT_PATH):
    """
    Loads baseline presentation, processes all 23 slides according to Task 2 specifications,
    and saves output cloned presentation.
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

    # 3. Slides 3-22 (Content Slides)
    print("Updating Slides 3-22 (Content Slides & Tracker Repositioning)...")
    update_content_slides(prs.slides)

    # 4. Slide 23 (Closing Slide)
    print("Updating Slide 23 (Closing Slide)...")
    update_closing_slide(prs.slides[22])

    # Save output
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving cloned presentation to: {output_path}")
    prs.save(output_path)
    print("Presentation successfully cloned and saved.")
    return output_path


if __name__ == "__main__":
    out = build_cloned_presentation()
    print(f"Completed build: {out}")
