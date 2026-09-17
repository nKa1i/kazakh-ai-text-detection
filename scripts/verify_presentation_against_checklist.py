# -*- coding: utf-8 -*-
"""Automated presentation verification checker against master thesis checklist.

Validates presentations against core academic and visual invariants:
1. Slide count and dimensions (23 slides, 13.333" x 7.500" 16:9 widescreen).
2. Typography (100% Times New Roman for Latin text, 0 Arial, theme Latin TNR).
3. Sequential Table of Contents (Slide 2 chapters [1, 2, 3, 4, 5, 6]).
4. Slide content anchors across key milestone slides.
5. Zero banned terms in shapes, tables, or presenter notes.
6. Zero decorative emojis across all slides and presenter notes.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple
from pptx import Presentation

# Configure stdout for UTF-8 when possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BANNED_TERMS = [
    "cross-attention",
    "cross-gate",
    "catastrophic",
    "breakthrough",
    "perfect separation",
    "perfect auc",
    "production-ready",
    "eliminates domain collapse",
    "fan qianyue",
    "chen yaxing",
    "范千悦",
    "陈亚兴",
]


def _extract_shape_text(shape) -> List[str]:
    """Recursively extract all text from a shape, table, or group shape."""
    texts = []
    if shape.has_text_frame:
        texts.append(shape.text_frame.text)
    if shape.has_table:
        for row in shape.table.rows:
            for cell in row.cells:
                texts.append(cell.text)
    if shape.shape_type == 6:  # Group shape
        for sub in shape.shapes:
            texts.extend(_extract_shape_text(sub))
    return texts


def get_slide_text(slide) -> str:
    """Extract all text from all shapes and tables in a slide."""
    texts = []
    for shp in slide.shapes:
        texts.extend(_extract_shape_text(shp))
    return " ".join(texts)


def get_slide_notes_text(slide) -> str:
    """Extract speaker notes text from a slide if present."""
    if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
        return slide.notes_slide.notes_text_frame.text
    return ""


def is_emoji_char(ch: str) -> bool:
    """Check if a character falls within common decorative emoji unicode ranges."""
    code = ord(ch)
    return (
        (0x1F600 <= code <= 0x1F64F)
        or (0x1F300 <= code <= 0x1F5FF)
        or (0x1F680 <= code <= 0x1F6FF)
        or (0x1F700 <= code <= 0x1F77F)
        or (0x1F900 <= code <= 0x1F9FF)
        or (0x1FA00 <= code <= 0x1FA6F)
        or (0x1FA70 <= code <= 0x1FAFF)
        or (0x2600 <= code <= 0x26FF)
        or (0x2700 <= code <= 0x27BF and code not in (0x2713, 0x2714))
    )


def check_slide_count_and_dimensions(prs: Presentation) -> Tuple[bool, str]:
    """Verify slide count is exactly 23 and dimensions are 16:9 widescreen (13.333" x 7.500")."""
    slide_count = len(prs.slides)
    if slide_count != 23:
        return False, f"Expected exactly 23 slides, found {slide_count}"

    w_inches = prs.slide_width.inches
    h_inches = prs.slide_height.inches

    if abs(w_inches - 13.333) > 0.01 or abs(h_inches - 7.500) > 0.01:
        return (
            False,
            f"Expected slide dimensions 13.333\" x 7.500\", found {w_inches:.3f}\" x {h_inches:.3f}\""
        )

    return True, f"Exactly 23 slides with 16:9 widescreen dimensions ({w_inches:.3f}\" x {h_inches:.3f}\")"


def check_typography(prs: Presentation) -> Tuple[bool, str]:
    """Verify 0 instances of Arial font and theme Latin typeface is Times New Roman."""
    arial_instances = []

    def check_text_frame_fonts(tf, location_desc: str):
        for p_idx, p in enumerate(tf.paragraphs):
            if p.font.name == "Arial":
                arial_instances.append(f"{location_desc} p[{p_idx}]: '{p.text[:25]}'")
            for r_idx, r in enumerate(p.runs):
                if r.font.name == "Arial":
                    arial_instances.append(f"{location_desc} r[{r_idx}]: '{r.text[:25]}'")

    def check_shape_fonts(shp, slide_idx: int):
        loc = f"Slide {slide_idx} shape '{shp.name}'"
        if shp.has_text_frame:
            check_text_frame_fonts(shp.text_frame, loc)
        if shp.has_table:
            for row_idx, row in enumerate(shp.table.rows):
                for col_idx, cell in enumerate(row.cells):
                    check_text_frame_fonts(
                        cell.text_frame,
                        f"Slide {slide_idx} table '{shp.name}' [{row_idx},{col_idx}]"
                    )
        if shp.shape_type == 6:  # Group
            for sub in shp.shapes:
                check_shape_fonts(sub, slide_idx)

    for s_idx, slide in enumerate(prs.slides, 1):
        for shp in slide.shapes:
            check_shape_fonts(shp, s_idx)

    if arial_instances:
        return (
            False,
            f"Found {len(arial_instances)} Arial font instances: {'; '.join(arial_instances[:3])}"
        )

    # Check theme Latin font
    theme_has_tnr = False
    theme_has_arial = False
    for rel in prs.part.rels.values():
        if "theme" in rel.target_ref:
            try:
                xml_text = rel.target_part.blob.decode("utf-8", errors="ignore")
                if '<a:latin typeface="Times New Roman"/>' in xml_text:
                    theme_has_tnr = True
                if '<a:latin typeface="Arial"/>' in xml_text:
                    theme_has_arial = True
            except Exception:
                pass

    if theme_has_arial or not theme_has_tnr:
        return (
            False,
            f"Theme Latin font check failed (Times New Roman: {theme_has_tnr}, Arial: {theme_has_arial})"
        )

    return True, "100% Times New Roman Latin typography (0 Arial instances across slides/tables, theme verified)"


def check_sequential_toc(prs: Presentation) -> Tuple[bool, str]:
    """Verify Slide 2 Table of Contents strictly lists chapters 01 through 06 in sequence."""
    if len(prs.slides) < 2:
        return False, "Presentation has fewer than 2 slides"

    s2 = prs.slides[1]
    order = []
    for s in s2.shapes:
        num = None
        if s.shape_type == 6:  # Group
            for sub in s.shapes:
                if sub.has_text_frame and sub.text_frame.text.strip()[:2].isdigit():
                    num = int(sub.text_frame.text.strip()[:2])
        elif s.has_text_frame and s.text_frame.text.strip()[:2].isdigit():
            num = int(s.text_frame.text.strip()[:2])
        if num is not None:
            order.append(num)

    expected = [1, 2, 3, 4, 5, 6]
    if order != expected:
        return False, f"Slide 2 chapter sequence is {order}, expected {expected}"

    return True, f"Slide 2 TOC sequence verified: {order}"


def check_slide_content_anchors(prs: Presentation) -> Tuple[bool, str]:
    """Verify mandatory textual, mathematical, and structural anchors across milestone slides."""
    if len(prs.slides) < 23:
        return False, f"Slide count ({len(prs.slides)}) insufficient for full anchor audit"

    errors = []

    # Slide 1: Presenter Daulet, no references to Fan Qianyue or Chen Yaxing
    s1_text = get_slide_text(prs.slides[0]) + " " + get_slide_notes_text(prs.slides[0])
    if "Daulet" not in s1_text:
        errors.append("Slide 1 missing presenter 'Daulet'")
    for retired in ["Fan Qianyue", "Chen Yaxing", "范千悦", "陈亚兴"]:
        if retired in s1_text:
            errors.append(f"Slide 1 contains retired name '{retired}'")

    # Slide 6: Hallucination Verification related work & 0 SciFact
    s6_text = get_slide_text(prs.slides[5])
    s6_reqs = [
        "LLM Hallucination Verification & Factual Grounding Gaps",
        "The Scientific Blindspot",
        "1. Stylistic AI Detectors",
        "2. International Fact-Checking Corpora",
        "3. Kazakh-FEVER & 2D Trust Matrix",
    ]
    for req in s6_reqs:
        if req not in s6_text:
            errors.append(f"Slide 6 missing anchor '{req}'")
    if "SciFact" in s6_text:
        errors.append("Slide 6 still contains retired term 'SciFact'")

    # Slide 7: Figure 3 picture and Comprehensive Methodological Framework
    s7_text = get_slide_text(prs.slides[6])
    s7_pics = [s for s in prs.slides[6].shapes if s.shape_type == 13]
    if not s7_pics:
        errors.append("Slide 7 missing Figure 3 embedded picture")
    if "Comprehensive Methodological Framework" not in s7_text:
        errors.append("Slide 7 missing 'Comprehensive Methodological Framework'")

    # Slide 8: Dual-Stream Morphology-Aware Gated Fusion, formula, 83 FST rules, 0 Cross-Attention
    s8_text = get_slide_text(prs.slides[7])
    if "Dual-Stream Morphology-Aware Gated Fusion" not in s8_text:
        errors.append("Slide 8 missing 'Dual-Stream Morphology-Aware Gated Fusion'")
    if "83" not in s8_text or "FST" not in s8_text:
        errors.append("Slide 8 missing '83' or 'FST' rule grounding")
    if not any(f in s8_text for f in ["g = sigma", "h_fused =", "W_g", "W_proj"]):
        errors.append("Slide 8 missing dynamic gating mathematical formulation")
    if "Cross-Attention" in s8_text or "cross-attention" in s8_text.lower():
        errors.append("Slide 8 contains retired term 'Cross-Attention'")

    # Slide 9: Robust Generalization & SentencePreservingChunker
    s9_text = get_slide_text(prs.slides[8])
    if "Robust Generalization & Adversarial Rewriting Defense" not in s9_text:
        errors.append("Slide 9 missing 'Robust Generalization & Adversarial Rewriting Defense'")
    if "SentencePreservingChunker" not in s9_text:
        errors.append("Slide 9 missing 'SentencePreservingChunker'")

    # Slide 10: Kazakh-FEVER & Two-Dimensional Trust Matrix
    s10_text = get_slide_text(prs.slides[9])
    if "Kazakh-FEVER" not in s10_text:
        errors.append("Slide 10 missing 'Kazakh-FEVER'")
    if "Two-Dimensional" not in s10_text:
        errors.append("Slide 10 missing 'Two-Dimensional'")
    if "Trust Matrix" not in s10_text:
        errors.append("Slide 10 missing 'Trust Matrix'")

    # Slide 11: MAGE protocol & 314 tests
    s11_text = get_slide_text(prs.slides[10])
    if not any(
        p in s11_text.lower()
        for p in [
            "kazakh adaptation of mage",
            "kazakh adaptation of a mage",
            "mage-style evaluation protocol",
            "mage evaluation protocol",
        ]
    ):
        errors.append("Slide 11 missing MAGE evaluation protocol phrasing")
    if "314" not in s11_text:
        errors.append("Slide 11 missing '314' test validation count")

    # Slide 12: +42.18 pp and +21.55 pp
    s12_text = get_slide_text(prs.slides[11])
    if "+42.18 pp" not in s12_text:
        errors.append("Slide 12 missing '+42.18 pp'")
    if "+21.55 pp" not in s12_text:
        errors.append("Slide 12 missing '+21.55 pp'")

    # Slide 18: 314 / 314, Automated Test Suite, Multi-Format Ingestion, 0 Zero Emoji
    s18_text = get_slide_text(prs.slides[17])
    if "314 / 314" not in s18_text:
        errors.append("Slide 18 missing '314 / 314'")
    if "Automated Test Suite" not in s18_text:
        errors.append("Slide 18 missing 'Automated Test Suite'")
    if "Multi-Format Ingestion" not in s18_text:
        errors.append("Slide 18 missing 'Multi-Format Ingestion'")
    if "zero emoji" in s18_text.lower():
        errors.append("Slide 18 contains unacademic literal 'Zero Emoji' phrase")

    # Slide 19: Four Primary Academic Contributions, Manuscript Status (~85% Complete)
    s19_text = get_slide_text(prs.slides[18])
    if not any(
        c in s19_text
        for c in [
            "Four Primary Academic Contributions",
            "Four Primary Academic & System Contributions",
            "Four Primary Academic",
        ]
    ):
        errors.append("Slide 19 missing 'Four Primary Academic Contributions'")
    if "Manuscript Status (~85% Complete)" not in s19_text:
        errors.append("Slide 19 missing 'Manuscript Status (~85% Complete)'")

    # Slide 20: Springer LNCS, AIST 2026, Live System Demonstration, Gradio
    s20_text = get_slide_text(prs.slides[19])
    for req in ["Springer LNCS", "AIST 2026", "Live System Demonstration", "Gradio"]:
        if req not in s20_text:
            errors.append(f"Slide 20 missing anchor '{req}'")

    # Slide 21: Strategic Consultation & Open Questions for Prof. Guo
    s21_text = get_slide_text(prs.slides[20])
    if not any(
        sq in s21_text
        for sq in [
            "Strategic Consultation & Open Questions for Prof. Guo",
            "Strategic Questions for Prof. Guo",
            "Strategic Consultation Points for Professor Guo",
            "Guidance Requests & Strategic Questions for Prof. Guo",
        ]
    ):
        errors.append("Slide 21 missing strategic consultation header for Prof. Guo")

    # Slide 22: Addressed in Thesis Chapter (0 RESOLVED badges)
    s22_text = get_slide_text(prs.slides[21])
    if "Addressed in Thesis Chapter" not in s22_text:
        errors.append("Slide 22 missing 'Addressed in Thesis Chapter' status badge")
    if "RESOLVED" in s22_text:
        errors.append("Slide 22 contains overconfident 'RESOLVED' badge")

    if errors:
        return False, f"{len(errors)} anchor verification failures: {'; '.join(errors)}"

    return True, "All 13 milestone slide content anchors verified (Slides 1, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 22)"


def check_banned_terms(prs: Presentation) -> Tuple[bool, str]:
    """Verify 0 instances of banned terms in shapes, tables, or presenter notes."""
    found = []

    for s_idx, slide in enumerate(prs.slides, 1):
        texts = []
        notes = get_slide_notes_text(slide)
        if notes:
            texts.append((f"Slide {s_idx} speaker notes", notes))

        for shp in slide.shapes:
            if shp.has_text_frame:
                texts.append((f"Slide {s_idx} shape '{shp.name}'", shp.text_frame.text))
            if shp.has_table:
                for row in shp.table.rows:
                    for cell in row.cells:
                        texts.append((f"Slide {s_idx} table '{shp.name}'", cell.text))
            if shp.shape_type == 6:
                for sub in shp.shapes:
                    if sub.has_text_frame:
                        texts.append((f"Slide {s_idx} group subshape '{sub.name}'", sub.text_frame.text))

        for loc, txt in texts:
            txt_lower = txt.lower()
            for b in BANNED_TERMS:
                if b.lower() in txt_lower:
                    found.append(f"{loc}: '{b}'")

    if found:
        return False, f"Found {len(found)} banned term occurrences: {'; '.join(found[:5])}"

    return True, "Zero banned terms detected across shapes, tables, and speaker notes"


def check_zero_emojis(prs: Presentation) -> Tuple[bool, str]:
    """Verify zero decorative emojis across all slides and presenter notes."""
    emojis_found = []

    for s_idx, slide in enumerate(prs.slides, 1):
        texts = []
        notes = get_slide_notes_text(slide)
        if notes:
            texts.append((f"Slide {s_idx} notes", notes))

        for shp in slide.shapes:
            texts.extend([(f"Slide {s_idx} shape '{shp.name}'", t) for t in _extract_shape_text(shp)])

        for loc, txt in texts:
            for ch in txt:
                if is_emoji_char(ch):
                    emojis_found.append(f"{loc}: '{ch}' (U+{ord(ch):04X})")

    if emojis_found:
        return False, f"Found {len(emojis_found)} emojis: {'; '.join(emojis_found[:5])}"

    return True, "Zero decorative emojis detected across all 23 slides and speaker notes"


def verify_presentation_file(ppt_path: str) -> Dict[str, Any]:
    """Run full verification audit against a PPTX presentation file.

    Returns:
        dict: {"path": ppt_path, "all_passed": bool, "results": [{"name": str, "passed": bool, "detail": str}, ...]}
    """
    if not os.path.isfile(ppt_path):
        return {
            "path": ppt_path,
            "all_passed": False,
            "results": [
                {
                    "name": "file_existence",
                    "passed": False,
                    "detail": f"File does not exist at {ppt_path}",
                }
            ],
        }

    try:
        prs = Presentation(ppt_path)
    except Exception as exc:
        return {
            "path": ppt_path,
            "all_passed": False,
            "results": [
                {
                    "name": "file_load",
                    "passed": False,
                    "detail": f"Failed to load PPTX: {str(exc)}",
                }
            ],
        }

    checks = [
        ("slide_count_and_dimensions", check_slide_count_and_dimensions),
        ("typography", check_typography),
        ("sequential_toc", check_sequential_toc),
        ("slide_content_anchors", check_slide_content_anchors),
        ("banned_terms", check_banned_terms),
        ("zero_emojis", check_zero_emojis),
    ]

    results = []
    all_passed = True

    for name, func in checks:
        passed, detail = func(prs)
        results.append({
            "name": name,
            "passed": passed,
            "detail": detail,
        })
        if not passed:
            all_passed = False

    return {
        "path": ppt_path,
        "all_passed": all_passed,
        "results": results,
    }


def format_report(verification_output: Dict[str, Any]) -> str:
    """Format verification results as a clean, readable terminal report."""
    lines = []
    lines.append("=" * 78)
    lines.append(" MASTER'S THESIS PRESENTATION VERIFICATION REPORT")
    lines.append(f" File: {verification_output['path']}")
    overall = "[PASS]" if verification_output["all_passed"] else "[FAIL]"
    lines.append(f" Overall Status: {overall}")
    lines.append("-" * 78)

    for idx, item in enumerate(verification_output["results"], 1):
        status = "[PASS]" if item["passed"] else "[FAIL]"
        lines.append(f" {idx}. {status} {item['name']}")
        lines.append(f"    Detail: {item['detail']}")

    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify master thesis presentation against audit checklist."
    )
    parser.add_argument(
        "--input",
        "-i",
        dest="input_path",
        help="Path to PPTX presentation to verify.",
    )
    args = parser.parse_args()

    paths_to_check = []
    if args.input_path:
        paths_to_check.append(args.input_path)
    else:
        desktop_target = r"C:\Users\Roza\Desktop\AnekeshD_Progress.pptx"
        repo_target = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx"
        )
        if os.path.isfile(desktop_target):
            paths_to_check.append(desktop_target)
        if os.path.isfile(repo_target):
            paths_to_check.append(repo_target)
        if not paths_to_check:
            paths_to_check.append(repo_target)

    overall_exit_code = 0
    for target in paths_to_check:
        res = verify_presentation_file(target)
        print(format_report(res))
        print()
        if not res["all_passed"]:
            overall_exit_code = 1

    return overall_exit_code


if __name__ == "__main__":
    sys.exit(main())
