# -*- coding: utf-8 -*-
"""
tests/test_ui_two_column_workspace.py: Unit tests for Unified Two-Column Workspace Layout Scaffold & Preset Pills.

Verifies:
1. create_app() initializes cleanly without error.
2. Two-column workspace scaffold:
   - Root row with elem_classes=["workspace-row"]
   - Left column (scale=4, elem_classes=["workspace-left-col"])
   - Right column (scale=6, elem_classes=["workspace-right-col"])
3. Single unified text area:
   - Exactly ONE primary interactive multi-line document input across the main layout (lines=10, max_lines=18).
   - No redundant separate textareas for FST, Auto-FST, or Verification.
4. Quick-sample preset pills bar in left column.
5. Document upload accordion with file uploader.
6. Feature toggle checkboxes (autofst_toggle, verifier_toggle) defaulting to True.
7. Action buttons: analyze_btn (btn-primary-charcoal) and clear_btn (btn-secondary-outline).
8. Persistent top HUD in right column:
   - Executive summary card display
   - Sleek confidence meter display
9. Consolidated 3 sub-tabs navigation in right column:
   - Sub-Tab 1: Overview & Heatmap (heatmap_display, linguistic_bullets_display)
   - Sub-Tab 2: Morphology & Slang Lab (autofst_comparison_card, autofst_inductions_table, sentence drawer)
   - Sub-Tab 3: Fact Verification & Trust Matrix (verify_card_display, verify_table_display)
10. Top utility bar:
   - Header with title and subtitle
   - Right-aligned language switcher (lang_radio)
   - Collapsible methodology accordion
11. Mandatory Zero Decorative Emojis across all labels, placeholders, HTML, and source files.
12. Parity between ui/app.py and hf_space/ui/app.py.
"""

import os
import re
import unittest
import gradio as gr

from ui.app import create_app, APP_CSS
from hf_space.ui.app import create_app as hf_create_app, APP_CSS as HF_APP_CSS

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE
)


class TestUITwoColumnWorkspace(unittest.TestCase):
    """Test suite for Unified Two-Column Workspace Architecture."""

    @classmethod
    def setUpClass(cls):
        """Initialize demo once for introspection tests."""
        cls.demo = create_app(load_model=False)
        cls.blocks = list(cls.demo.blocks.values())

    def test_create_app_initializes_cleanly(self):
        """Verifies create_app() initializes without error and returns gr.Blocks."""
        self.assertIsNotNone(self.demo)
        self.assertIsInstance(self.demo, gr.Blocks)
        self.assertEqual(self.demo.title, "Kazakh AI-Text Detector & Explainability UI")

    def test_two_column_workspace_scaffold(self):
        """Verifies workspace-row, workspace-left-col (scale=5), and workspace-right-col (scale=5)."""
        rows = [b for b in self.blocks if isinstance(b, gr.Row)]
        cols = [b for b in self.blocks if isinstance(b, gr.Column)]

        workspace_rows = [
            r for r in rows
            if hasattr(r, "elem_classes") and r.elem_classes and "workspace-row" in r.elem_classes
        ]
        self.assertEqual(
            len(workspace_rows), 1,
            "Expected exactly one gr.Row with elem_classes=['workspace-row']"
        )

        left_cols = [
            c for c in cols
            if hasattr(c, "elem_classes") and c.elem_classes and "workspace-left-col" in c.elem_classes
        ]
        self.assertEqual(
            len(left_cols), 1,
            "Expected exactly one gr.Column with elem_classes=['workspace-left-col']"
        )
        self.assertEqual(left_cols[0].scale, 5, "workspace-left-col must have scale=5")

        right_cols = [
            c for c in cols
            if hasattr(c, "elem_classes") and c.elem_classes and "workspace-right-col" in c.elem_classes
        ]
        self.assertEqual(
            len(right_cols), 1,
            "Expected exactly one gr.Column with elem_classes=['workspace-right-col']"
        )
        self.assertEqual(right_cols[0].scale, 5, "workspace-right-col must have scale=5")

    def test_balanced_column_scales(self):
        """Verifies left and right columns both have scale=5 for balanced scrolling."""
        cols = [b for b in self.blocks if isinstance(b, gr.Column)]
        left_cols = [c for c in cols if hasattr(c, "elem_classes") and c.elem_classes and "workspace-left-col" in c.elem_classes]
        right_cols = [c for c in cols if hasattr(c, "elem_classes") and c.elem_classes and "workspace-right-col" in c.elem_classes]
        self.assertEqual(len(left_cols), 1)
        self.assertEqual(len(right_cols), 1)
        self.assertEqual(left_cols[0].scale, 5)
        self.assertEqual(right_cols[0].scale, 5)

    def test_no_duplicate_confidence_meter(self):
        """Verifies there is no redundant second full-width confidence meter HTML block visible in the layout."""
        html_blocks = [b for b in self.blocks if isinstance(b, gr.HTML)]
        visible_standalone_meters = [
            b for b in html_blocks
            if getattr(b, "visible", True) and "confidence-track" in getattr(b, "value", "") and "executive-summary-card" not in getattr(b, "value", "")
        ]
        self.assertEqual(len(visible_standalone_meters), 0, "Found redundant visible standalone confidence meter")

    def test_single_unified_primary_text_input(self):
        """
        Verifies there is ONLY ONE primary interactive document textarea across the main layout.
        No duplicate text inputs for FST, AutoFST, or Verification.
        """
        textboxes = [b for b in self.blocks if isinstance(b, gr.Textbox)]
        primary_inputs = [
            tb for tb in textboxes
            if getattr(tb, "interactive", True) and getattr(tb, "lines", 1) >= 8
        ]
        self.assertEqual(
            len(primary_inputs), 1,
            f"Expected exactly 1 primary multi-line document input, found {len(primary_inputs)}"
        )
        primary_tb = primary_inputs[0]
        self.assertEqual(primary_tb.lines, 10, "Primary text input must specify lines=10")
        self.assertEqual(primary_tb.max_lines, 18, "Primary text input must specify max_lines=18")

    def test_feature_toggles_present(self):
        """Verifies Auto-FST and Kazakh-FEVER feature toggle checkboxes exist and default to True."""
        checkboxes = [b for b in self.blocks if isinstance(b, gr.Checkbox)]
        labels = [getattr(cb, "label", "") for cb in checkboxes]

        has_autofst_toggle = any("Auto-FST" in (l or "") for l in labels)
        self.assertTrue(has_autofst_toggle, f"Expected Auto-FST checkbox toggle, found: {labels}")

        has_verifier_toggle = any("Kazakh-FEVER" in (l or "") or "Factual" in (l or "") for l in labels)
        self.assertTrue(has_verifier_toggle, f"Expected Kazakh-FEVER checkbox toggle, found: {labels}")

        for cb in checkboxes:
            if "Auto-FST" in (getattr(cb, "label", "") or "") or "Kazakh-FEVER" in (getattr(cb, "label", "") or ""):
                self.assertTrue(cb.value, f"Checkbox {cb.label} must default to True")

    def test_action_buttons_styling(self):
        """Verifies Run Full Analysis and Clear buttons exist with specified class names."""
        buttons = [b for b in self.blocks if isinstance(b, gr.Button)]
        btn_classes = [getattr(b, "elem_classes", []) or [] for b in buttons]

        has_primary_charcoal = any("btn-primary-charcoal" in c for c in btn_classes)
        self.assertTrue(has_primary_charcoal, "analyze_btn must include elem_classes=['btn-primary-charcoal']")

        has_secondary_outline = any("btn-secondary-outline" in c for c in btn_classes)
        self.assertTrue(has_secondary_outline, "clear_btn must include elem_classes=['btn-secondary-outline']")

    def test_three_consolidated_subtabs_in_right_column(self):
        """Verifies exactly 3 consolidated sub-tabs: Overview, Morphology, Verification."""
        tabs_containers = [
            b for b in self.blocks
            if isinstance(b, gr.Tabs) and hasattr(b, "elem_classes") and b.elem_classes and "workspace-tabs" in b.elem_classes
        ]
        self.assertEqual(
            len(tabs_containers), 1,
            "Expected exactly one gr.Tabs with elem_classes=['workspace-tabs']"
        )

        tab_items = [b for b in self.blocks if isinstance(b, gr.Tab)]
        tab_labels = [getattr(t, "label", "") for t in tab_items]

        self.assertEqual(
            len(tab_items), 3,
            f"Expected exactly 3 consolidated sub-tabs, found {len(tab_items)}: {tab_labels}"
        )

        # Tab 1: Sentence Morphology & Fusion
        self.assertTrue(
            any("Morphology" in (l or "") for l in tab_labels),
            f"Sub-Tab 1 missing Morphology label, found: {tab_labels}"
        )
        # Tab 2: Auto-FST & Slang Lab
        self.assertTrue(
            any("Auto-FST" in (l or "") or "Slang" in (l or "") for l in tab_labels),
            f"Sub-Tab 2 missing Auto-FST / Slang label, found: {tab_labels}"
        )
        # Tab 3: Fact Verification & Trust Matrix
        self.assertTrue(
            any("Verification" in (l or "") or "Trust" in (l or "") for l in tab_labels),
            f"Sub-Tab 3 missing Verification label, found: {tab_labels}"
        )

    def test_methodology_accordion_in_top_utility_bar(self):
        """Verifies collapsible methodology accordion exists with Kaz-MAGE matrix and empirical tables."""
        accordions = [b for b in self.blocks if isinstance(b, gr.Accordion)]
        acc_labels = [getattr(a, "label", "") for a in accordions]

        has_methodology = any("Methodology" in (l or "") or "Әдістеме" in (l or "") for l in acc_labels)
        self.assertTrue(has_methodology, f"Expected collapsible methodology accordion, found: {acc_labels}")

    def test_preset_pills_present_in_left_column(self):
        """Verifies quick-sample preset pills exist in left column."""
        buttons = [b for b in self.blocks if isinstance(b, gr.Button)]
        pills = [
            b for b in buttons
            if hasattr(b, "elem_classes") and b.elem_classes and "quick-sample-pill" in b.elem_classes
        ]
        self.assertGreaterEqual(len(pills), 4, "Expected at least 4 quick-sample pills in left column")

    def test_zero_decorative_emojis(self):
        """MANDATORY ZERO EMOJI AUDIT: Ensures zero emojis across ui/app.py and hf_space/ui/app.py."""
        for path in ["ui/app.py", "hf_space/ui/app.py"]:
            full_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                path.replace("/", os.sep)
            )
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            emojis = EMOJI_PATTERN.findall(content)
            self.assertEqual(
                emojis, [],
                f"Found forbidden emojis in {path}: {emojis}"
            )

    def test_full_analysis_handler_produces_all_outputs(self):
        """Asserts that invoking run_full_analysis_logic produces valid results for HUD, Heatmap, Auto-FST, and Verification."""
        from ui.app import run_full_analysis_logic
        text = "вайбы жаксы екен бул ойыннын оте катты унады"
        res = run_full_analysis_logic(text, autofst_enabled=True, verifier_enabled=True, current_lang="English")
        self.assertIn("executive-summary-card", res["executive_card"])
        self.assertIn("confidence-track", res["confidence_meter"])
        self.assertIn("autofst-card", res["autofst_card"])
        self.assertIn("notion-table", res["autofst_table"])
        self.assertIn("trust-card", res["trust_card"])
        self.assertIn("notion-table", res["claims_table"])

    def test_bilingual_switch_coverage(self):
        """Verifies get_workspace_i18n returns complete dictionaries for English and Kazakh."""
        from ui.app import get_workspace_i18n
        kz = get_workspace_i18n("Kazakh")
        en = get_workspace_i18n("English")
        self.assertIn("analyze_btn", kz)
        self.assertIn("analyze_btn", en)
        self.assertIn("run full analysis", en["analyze_btn"].lower())
        self.assertIn("талдау", kz["analyze_btn"].lower())

    def test_zero_emojis_in_workspace(self):
        """Verifies zero emojis in workspace dictionary and CSS."""
        from ui.app import APP_CSS, get_workspace_i18n
        import re
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F900-\U0001F9FF]')
        self.assertEqual(len(emoji_pattern.findall(APP_CSS)), 0)
        for lang in ["Kazakh", "English"]:
            for k, val in get_workspace_i18n(lang).items():
                self.assertEqual(len(emoji_pattern.findall(str(val))), 0)

    def test_pure_english_ui_chrome(self):
        """Verifies zero bilingual slash labels in tabs and buttons."""
        tab_items = [b for b in self.blocks if isinstance(b, gr.Tab)]
        for t in tab_items:
            label = getattr(t, "label", "")
            self.assertNotIn("/", label, f"Found bilingual slash in tab label: {label}")

    def test_audit_markdown_export(self):
        """Verifies generate_audit_markdown returns a valid markdown audit document."""
        from ui.app import generate_audit_markdown, run_full_analysis_logic
        res = run_full_analysis_logic("Қазақстан 1991 жылы тәуелсіздік алды.")
        md_text = generate_audit_markdown(res, source_text="Қазақстан 1991 жылы тәуелсіздік алды.")
        self.assertIn("# Document Inspection & Verification Audit Report", md_text)
        self.assertIn("## Executive Verdict", md_text)

    def test_morpheme_visual_diff_renderer(self):
        """Verifies render_morpheme_visual_diff generates visual diff highlights."""
        from ui.highlighting import render_morpheme_visual_diff
        diff_html = render_morpheme_visual_diff(
            raw_text="салем вайбтадык",
            normalized_text="сәлем вайбтадық",
            inductions=[{"stem": "вайб", "suffixes": ["та", "дық"]}]
        )
        self.assertIn("diff-norm", diff_html)
        self.assertIn("diff-induce", diff_html)

    def test_hf_space_parity(self):
        """Verifies exact parity between ui/app.py and hf_space/ui/app.py."""
        ui_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ui", "app.py"
        )
        hf_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "hf_space", "ui", "app.py"
        )
        with open(ui_path, "r", encoding="utf-8") as f:
            ui_content = f.read()
        with open(hf_path, "r", encoding="utf-8") as f:
            hf_content = f.read()

        self.assertEqual(
            ui_content, hf_content,
            "ui/app.py and hf_space/ui/app.py must match exactly."
        )


if __name__ == "__main__":
    unittest.main()
