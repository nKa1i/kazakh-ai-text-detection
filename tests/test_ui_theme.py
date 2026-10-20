# -*- coding: utf-8 -*-
"""
tests/test_ui_theme.py: Unit tests for Minimalist Apple / Notion Theme and Typography System.

Verifies:
- get_apple_notion_theme() returns a valid gr.Theme object.
- APP_CSS contains modern Apple / Notion class selectors and font stacks:
    - .hero-card
    - .quick-sample-pill
    - SFMono-Regular
    - .tabs > .tab-nav
- Zero decorative emojis in APP_CSS, get_apple_notion_theme(), and ui/app.py.
- Parity between ui/app.py and hf_space/ui/app.py.
- Gradio Blocks initializes with get_apple_notion_theme() and APP_CSS.
"""

import os
import re
import unittest
import gradio as gr

from ui.app import (
    get_apple_notion_theme,
    APP_CSS,
    create_app
)
from hf_space.ui.app import (
    get_apple_notion_theme as hf_get_apple_notion_theme,
    APP_CSS as HF_APP_CSS
)

# Regex matching standard emoji ranges
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
    "]+",
    flags=re.UNICODE
)


class TestUITheme(unittest.TestCase):
    """Test suite for Apple / Notion theme and typography system in Kazakh AI text detection UI."""

    def test_get_apple_notion_theme_returns_theme(self):
        """Verifies get_apple_notion_theme() returns a valid Base theme instance."""
        theme = get_apple_notion_theme()
        self.assertIsNotNone(theme)
        self.assertIsInstance(theme, gr.themes.Base)

    def test_hf_space_theme_returns_theme(self):
        """Verifies hf_space/ui/app.py get_apple_notion_theme() returns a valid Base theme."""
        theme = hf_get_apple_notion_theme()
        self.assertIsNotNone(theme)
        self.assertIsInstance(theme, gr.themes.Base)

    def test_app_css_contains_apple_notion_selectors(self):
        """Verifies APP_CSS contains essential Apple / Notion class names and typography."""
        required_substrings = [
            ".hero-card",
            ".hero-title",
            ".hero-subtitle",
            ".feature-badge",
            ".quick-sample-pill",
            "SFMono-Regular",
            ".tabs > .tab-nav"
        ]
        for sub in required_substrings:
            self.assertIn(
                sub,
                APP_CSS,
                f"Expected substring '{sub}' not found in APP_CSS"
            )

    def test_hf_space_app_css_parity(self):
        """Verifies APP_CSS in ui/app.py matches hf_space/ui/app.py exactly."""
        self.assertEqual(
            APP_CSS,
            HF_APP_CSS,
            "APP_CSS in ui/app.py and hf_space/ui/app.py must match exactly."
        )

    def test_zero_emojis_in_theme_and_css(self):
        """MANDATORY ZERO EMOJI AUDIT: Ensures zero emojis in APP_CSS and app source files."""
        css_emojis = EMOJI_PATTERN.findall(APP_CSS)
        self.assertEqual(
            css_emojis,
            [],
            f"Found forbidden emojis in APP_CSS: {css_emojis}"
        )

        # Check ui/app.py file content
        ui_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ui",
            "app.py"
        )
        with open(ui_path, "r", encoding="utf-8") as f:
            content = f.read()
        file_emojis = EMOJI_PATTERN.findall(content)
        self.assertEqual(
            file_emojis,
            [],
            f"Found forbidden emojis in ui/app.py: {file_emojis}"
        )

    def test_create_app_initializes_with_theme(self):
        """Verifies create_app initializes cleanly with custom theme applied."""
        demo = create_app(load_model=False)
        self.assertIsNotNone(demo)
        self.assertIsInstance(demo, gr.Blocks)
        self.assertEqual(demo.title, "Kazakh AI-Text Detector & Explainability UI")
        self.assertIsNotNone(demo.theme)


if __name__ == "__main__":
    unittest.main()
