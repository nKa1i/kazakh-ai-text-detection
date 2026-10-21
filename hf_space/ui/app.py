# -*- coding: utf-8 -*-
"""
ui/app.py: Academic 3-Tabbed Dashboard for Kazakh AI-Text Detection & Explainability.

Features:
- Sleek, publication-grade academic dashboard with zero decorative emoji clutter.
- Bilingual localization toggle (Kazakh / English).
- 3 Dedicated Academic Tabs:
    1. Detection & Explainability (Document KPIs, confidence meter, linguistic reasoning bullets, heatmap, drilldown)
    2. Morphological FST Lab (Dynamic fusion gate split-bar, 8-column agglutinative decomposition table)
    3. Benchmark & Academic Methodology (Kaz-MAGE 2x2 matrix, empirical comparison table, gated cross-attention equation, credits)
- 6 Quick-Sample Pills horizontally placed above the input box for immediate 1-click loading.
- Horizontal confidence progress meter with percentage display ([====== 99.8% =====]).
- Dynamic linguistic reasoning bullets integrated via generate_linguistic_explanation.
- Strict XSS sanitization: all dynamic user text is escaped via html.escape(..., quote=True).
- 100% defensive offline heuristic fallback when PyTorch model weights are unavailable.
"""

import os
import sys
import re
import html
import argparse
from typing import List, Dict, Any, Optional, Tuple, Union

# Ensure project root is in sys.path for direct CLI script execution
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gradio as gr

from kaz_mage.chunker import SentencePreservingChunker
from kaz_mage.document import DocumentAnalysisResult, DocumentChunk
from kaz_mage.aggregator import DocumentAggregator
from ui.highlighting import (
    render_document_heatmap,
    render_dynamic_gate_bar,
    render_morpheme_table,
    render_confidence_meter,
    render_executive_summary_card,
    render_fst_decomposition_table,
    render_trust_summary_card,
    render_claims_verification_table,
    render_autofst_comparison_card,
    render_autofst_inductions_table,
    render_morpheme_visual_diff,
    HEATMAP_CSS
)
from ui.file_loader import load_document_file
from ui.presets import (
    PRESET_SAMPLES,
    get_preset_choices,
    get_preset_text,
    get_preset_metadata,
    VERIFICATION_PRESET_SAMPLES,
    get_verification_preset_choices,
    get_verification_preset_text,
    get_verification_preset_metadata,
)
from verification.verifier import TrustworthyDocumentVerifier
from verification.evidence import DocumentTrustResult, ClaimVerificationResult
from models.heuristic_detector import OfflineHeuristicDetector
try:
    from auto_fst import RussianLayoutNormalizer, AutoFSTInducer, DynamicFSTOverlay, RegressionGuard
except ImportError:
    from src.auto_fst import RussianLayoutNormalizer, AutoFSTInducer, DynamicFSTOverlay, RegressionGuard

from ui.sentence_analyzer import (
    analyze_document_sentences,
    analyze_sentence_morphemes,
    get_default_fst_analyzer,
    _extract_words,
    CASE_TAGS,
    PLUR_TAGS,
    POSS_TAGS,
    VERB_TENSE_TAGS,
    VERB_PERSON_TAGS,
    VERB_NEG_TAGS,
    VERB_DERIV_TAGS
)
from ui.linguistic_explainer import generate_linguistic_explanation


def get_apple_notion_theme() -> gr.Theme:
    """Constructs a minimalist Apple / Notion-style dark charcoal theme for Gradio Blocks."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.zinc,
        neutral_hue=gr.themes.colors.zinc,
        font=[gr.themes.GoogleFont("Inter"), "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        font_mono=["SFMono-Regular", "Menlo", "Monaco", "Consolas", "Liberation Mono", "monospace"],
        radius_size=gr.themes.sizes.radius_sm,
        spacing_size=gr.themes.sizes.spacing_md,
        text_size=gr.themes.sizes.text_sm,
    ).set(
        body_background_fill="#121316",
        body_background_fill_dark="#121316",
        body_text_color="#f3f4f6",
        body_text_color_dark="#f3f4f6",
        body_text_color_subdued="#9ca3af",
        body_text_color_subdued_dark="#9ca3af",
        background_fill_primary="#1a1b22",
        background_fill_primary_dark="#1a1b22",
        background_fill_secondary="#21232d",
        background_fill_secondary_dark="#21232d",
        border_color_primary="#2d313d",
        border_color_primary_dark="#2d313d",
        block_background_fill="#1a1b22",
        block_background_fill_dark="#1a1b22",
        block_border_width="1px",
        block_border_color="#2d313d",
        block_border_color_dark="#2d313d",
        block_label_background_fill="#1a1b22",
        block_label_background_fill_dark="#1a1b22",
        block_label_text_color="#9ca3af",
        block_label_text_color_dark="#9ca3af",
        block_title_text_color="#f3f4f6",
        block_title_text_color_dark="#f3f4f6",
        input_background_fill="#1e2029",
        input_background_fill_dark="#1e2029",
        input_border_color="#2d313d",
        input_border_color_dark="#2d313d",
        block_radius="8px",
        block_shadow="0 1px 3px rgba(0, 0, 0, 0.25)",
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#21232d",
        button_secondary_background_fill_hover="#2b2d3a",
        button_secondary_text_color="#f3f4f6",
        button_secondary_border_color="#2d313d",
    )


# Extended CSS styling for minimalist Apple / Notion publication-grade dashboard aesthetics
APP_CSS = HEATMAP_CSS + """
/* Publication-Grade Apple / Notion Deep Charcoal Dark Theme */
:root, .dark, :root .dark, body.dark, .gradio-container, .dark .gradio-container, .dark * {
    color-scheme: dark !important;
    --body-background-fill: #121316 !important;
    --body-text-color: #f3f4f6 !important;
    --body-text-color-subdued: #9ca3af !important;
    --background-fill-primary: #1a1b22 !important;
    --background-fill-secondary: #21232d !important;
    --border-color-primary: #2d313d !important;
    --border-color-secondary: #2d313d !important;
    --border-color-accent: #3b82f6 !important;
    --block-background-fill: #1a1b22 !important;
    --block-border-color: #2d313d !important;
    --block-border-width: 1px !important;
    --block-label-background-fill: #1a1b22 !important;
    --block-label-text-color: #9ca3af !important;
    --block-title-text-color: #f3f4f6 !important;
    --input-background-fill: #1e2029 !important;
    --input-border-color: #2d313d !important;
    --input-text-color: #f3f4f6 !important;
    --table-border-color: #2d313d !important;
    --table-text-color: #f3f4f6 !important;
    --accordion-text-color: #f3f4f6 !important;
    --neutral-50: #121316 !important;
    --neutral-100: #1a1b22 !important;
    --neutral-200: #21232d !important;
    --neutral-300: #2d313d !important;
    --neutral-400: #9ca3af !important;
    --neutral-500: #9ca3af !important;
    --neutral-600: #cbd5e1 !important;
    --neutral-700: #e2e8f0 !important;
    --neutral-800: #1a1b22 !important;
    --neutral-900: #f3f4f6 !important;
    --neutral-950: #ffffff !important;
    --tw-prose-body: #f3f4f6 !important;
    --tw-prose-headings: #f3f4f6 !important;
    --tw-prose-lead: #9ca3af !important;
    --tw-prose-links: #60a5fa !important;
    --tw-prose-bold: #f3f4f6 !important;
    --tw-prose-invert-body: #f3f4f6 !important;
    --tw-prose-invert-headings: #f3f4f6 !important;
    --tw-prose-invert-lead: #9ca3af !important;
    --tw-prose-invert-bold: #f3f4f6 !important;
}

html, body, .gradio-container, .dark, .dark .gradio-container {
    background-color: #121316 !important;
    background: #121316 !important;
    color: #f3f4f6 !important;
}

/* Enforce crisp high-contrast text across all prose, markdown, headers, and labels */
.prose, .prose *, .dark .prose, .dark .prose *,
.markdown, .markdown *, .dark .markdown, .dark .markdown *,
h1, h2, h3, h4, h5, h6,
.dark h1, .dark h2, .dark h3, .dark h4, .dark h5, .dark h6,
p, .dark p, span, .dark span,
strong, .dark strong, em, .dark em,
.block-title, .dark .block-title,
.dark .hero-card, .dark .panel-box,
.dark .workspace-left-col, .dark .workspace-right-col {
    color: #f3f4f6 !important;
}

/* Subtext and secondary labels */
.quick-samples-header-row p, .quick-samples-header-row .prose p,
.dark .quick-samples-header-row p, .dark .quick-samples-header-row .prose p,
label, label span, .dark label, .dark label span,
.block-label, .dark .block-label,
[data-testid="block-label"], .dark [data-testid="block-label"] {
    color: #9ca3af !important;
}

/* Enforce dark charcoal cards and containers on all blocks, forms, and accordions */
.block, .dark .block,
fieldset, .dark fieldset,
.form, .dark .form,
.accordion, .dark .accordion,
.accordion > .label-wrap, .dark .accordion > .label-wrap,
.accordion > button, .dark .accordion > button,
.accordion summary, .dark .accordion summary,
.gr-box, .dark .gr-box,
.gr-panel, .dark .gr-panel,
.input-container, .dark .input-container,
textarea, .dark textarea,
input[type="text"], .dark input[type="text"],
.dark .tabitem, .dark .tabs {
    background-color: #1a1b22 !important;
    background: #1a1b22 !important;
    border-color: #2d313d !important;
    color: #f3f4f6 !important;
}

/* Textarea input dark surface */
textarea, .dark textarea,
input[type="text"], .dark input[type="text"] {
    background-color: #1e2029 !important;
    background: #1e2029 !important;
    border: 1px solid #2d313d !important;
    color: #f3f4f6 !important;
}

/* Accordion header styling */
.accordion > .label-wrap, .dark .accordion > .label-wrap,
.accordion > button, .dark .accordion > button,
.accordion summary, .dark .accordion summary {
    border: 1px solid #2d313d !important;
    border-radius: 6px !important;
    background: #1e2029 !important;
}
.accordion span, .dark .accordion span,
.accordion svg, .dark .accordion svg {
    color: #f3f4f6 !important;
    fill: #f3f4f6 !important;
}

/* Tab bar typography */
.tab-nav, .dark .tab-nav {
    border-bottom: 1px solid #2d313d !important;
    background: transparent !important;
}
.tab-nav button, .dark .tab-nav button {
    color: #9ca3af !important;
    font-weight: 500 !important;
    background: transparent !important;
}
.tab-nav button.selected, .dark .tab-nav button.selected {
    color: #f3f4f6 !important;
    border-bottom: 2px solid #3b82f6 !important;
    font-weight: 600 !important;
}
.tab-nav button:hover, .dark .tab-nav button:hover {
    color: #f3f4f6 !important;
}

/* Live typing statistics ribbon */
.live-stats-ribbon {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 11.5px;
    font-weight: 500;
    color: #9ca3af;
    font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    padding: 4px 6px 8px 6px;
    letter-spacing: 0.02em;
}

/* Visual diff styling */
.diff-norm {
    background-color: rgba(56, 189, 248, 0.18) !important;
    color: #38bdf8 !important;
    padding: 1px 4px !important;
    border-radius: 3px !important;
    font-weight: 600 !important;
    border-bottom: 1.5px solid #0ea5e9 !important;
}

.diff-induce {
    background-color: rgba(192, 132, 252, 0.18) !important;
    color: #c084fc !important;
    padding: 1px 4px !important;
    border-radius: 3px !important;
    font-weight: 600 !important;
    border-bottom: 1.5px solid #a855f7 !important;
}

body, .gradio-container {
    background-color: #121316 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    color: #f3f4f6 !important;
}
.hero-card {
    background: #1a1b22;
    border: 1px solid #2d313d;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 18px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}
.hero-title {
    font-size: 22px;
    font-weight: 700;
    color: #f3f4f6;
    margin-bottom: 6px;
    letter-spacing: -0.015em;
}
.hero-subtitle {
    font-size: 13.5px;
    color: #9ca3af;
    line-height: 1.55;
    margin-bottom: 12px;
}
.hero-badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 8px;
}
.feature-badge {
    display: inline-flex;
    align-items: center;
    background: #21232d;
    border: 1px solid #2d313d;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 500;
    color: #cbd5e1;
    font-family: "SFMono-Regular", Menlo, Consolas, monospace;
}
.quick-samples-wrapper {
    margin-bottom: 10px;
}
.quick-samples-header-row {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    gap: 16px !important;
    margin-bottom: 8px !important;
}
.quick-samples-title {
    margin: 0 !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
}
.quick-samples-title p {
    margin: 0 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    color: #9ca3af !important;
}
.lang-switch-inline {
    display: inline-flex !important;
    justify-content: flex-start !important;
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}
.lang-switch-inline .wrap {
    display: flex !important;
    flex-direction: row !important;
    gap: 12px !important;
    justify-content: flex-start !important;
    align-items: center !important;
}
.quick-samples-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
}
.sample-pill,
.quick-sample-pill {
    display: inline-flex !important;
    align-items: center !important;
    background: #21232d !important;
    border: 1px solid #2d313d !important;
    border-radius: 6px !important;
    padding: 5px 11px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #e2e8f0 !important;
    cursor: pointer !important;
    transition: background 0.15s ease, border-color 0.15s ease, transform 0.15s ease !important;
    box-shadow: none !important;
}
.sample-pill:hover,
.quick-sample-pill:hover {
    background: #2a2d3a !important;
    border-color: #3b4252 !important;
    transform: translateY(-0.5px) !important;
}
.tabs > .tab-nav {
    border-bottom: 1px solid #2d313d !important;
    gap: 4px !important;
    background: transparent !important;
}
.tabs > .tab-nav > button {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #9ca3af !important;
    border: none !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 8px 16px !important;
    background: transparent !important;
    transition: color 0.15s ease, border-color 0.15s ease !important;
}
.tabs > .tab-nav > button.selected {
    color: #f3f4f6 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #3b82f6 !important;
    background: transparent !important;
}
button.primary,
.gr-button-primary,
button[variant="primary"] {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #2563eb !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}
button.primary:hover,
.gr-button-primary:hover,
button[variant="primary"]:hover {
    background-color: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}
button.secondary,
.gr-button-secondary,
button[variant="secondary"] {
    background-color: #21232d !important;
    color: #f3f4f6 !important;
    border: 1px solid #2d313d !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    transition: background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease !important;
}
button.secondary:hover,
.gr-button-secondary:hover,
button[variant="secondary"]:hover {
    background-color: #2b2d3a !important;
}
.workspace-left-col {
    background: #1a1b22;
    border: 1px solid #2d313d;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}
.workspace-right-col {
    background: #1a1b22;
    border: 1px solid #2d313d;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}
.panel-box {
    background: #1a1b22;
    border: 1px solid #2d313d;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}
.card-header-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #2d313d;
    padding-bottom: 8px;
    margin-bottom: 12px;
}
.card-header-title {
    font-size: 13.5px;
    font-weight: 700;
    color: #f3f4f6;
    letter-spacing: -0.01em;
}
.badge-count {
    background: #21232d;
    border: 1px solid #2d313d;
    border-radius: 9999px;
    padding: 2px 7px;
    font-size: 11px;
    font-weight: 600;
    color: #cbd5e1;
    font-family: "SFMono-Regular", Menlo, Consolas, monospace;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    font-family: "SFMono-Regular", Menlo, Monaco, Consolas, monospace;
}
.status-pill.human {
    background: rgba(16, 185, 129, 0.16);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}
.status-pill.amber {
    background: rgba(245, 158, 11, 0.16);
    color: #fbbf24;
    border: 1px solid rgba(251, 191, 36, 0.3);
}
.status-pill.ai {
    background: rgba(244, 63, 94, 0.18);
    color: #fb7185;
    border: 1px solid rgba(251, 113, 133, 0.3);
}
.gate-value-tag {
    font-family: "SFMono-Regular", Menlo, Monaco, Consolas, monospace;
    font-size: 12px;
    font-weight: 600;
    color: #cbd5e1;
}
.model-perf-ribbon {
    display: flex;
    gap: 16px;
    font-size: 11px;
    color: #9ca3af;
    font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    padding: 8px 12px;
    background: #21232d;
    border-radius: 6px;
    border: 1px solid #2d313d;
    margin-top: 10px;
}
.model-perf-item strong {
    color: #f3f4f6;
    font-weight: 600;
}
"""

# Quick Samples Definition (6 curated benchmarks)
QUICK_SAMPLES: Dict[str, Dict[str, str]] = {
    "pill_1": {
        "title_kz": "Kaspi пікірі [Адам]",
        "title_en": "Kaspi Review [Human]",
        "expected_verdict": "Authentic Human",
        "text": get_preset_text("1. Authentic Kaspi Consumer Review (Human)"),
    },
    "pill_2": {
        "title_kz": "Ресми жаңалықтар [Адам]",
        "title_en": "Formal News [Human]",
        "expected_verdict": "Authentic Human",
        "text": get_preset_text("2. Authentic Formal News Article (Human)"),
    },
    "pill_3": {
        "title_kz": "Sherkala-7B [ЖИ]",
        "title_en": "Sherkala-7B [AI]",
        "expected_verdict": "Machine-Generated",
        "text": (
            "Kaspi дүкенінен сатып алынған шаңсорғыш өте сапалы болып табылады. "
            "Қорытындылай келе, тауардың сипаттамасы жоғары дәрежеде сәйкес келеді және маңызды рөл атқарады. "
            "Айта кету керек, жеткізу қызметі жүйелі түрде орындалды, барлық функциялары талапқа сай жұмыс істейді."
        ),
    },
    "pill_4": {
        "title_kz": "Sherkala жаңалықтары [ЖИ]",
        "title_en": "Sherkala News [AI]",
        "expected_verdict": "Machine-Generated",
        "text": get_preset_text("4. Sherkala-7B Generated Article (AI)"),
    },
    "pill_5": {
        "title_kz": "Qwen-2.5 [Жабайы ЖИ]",
        "title_en": "Qwen-2.5 [Wild AI]",
        "expected_verdict": "Machine-Generated",
        "text": get_preset_text("5. Qwen-2.5-7B-Instruct Wild Sample (AI)"),
    },
    "pill_6": {
        "title_kz": "Гибрид құжат [Аралас]",
        "title_en": "Hybrid Document",
        "expected_verdict": "Partially AI / Hybrid",
        "text": get_preset_text("6. Injected Hybrid Essay (Partially AI)"),
    },
}

# Bilingual Localization Dictionary (Mandatory Zero Emojis)
I18N = {
    "kz": {
        "hero_title": "Қазақ Тіліндегі Жасанды Интеллект Мәтіндерін Анықтау Жүйесі",
        "hero_subtitle": "Қазақ тіліндегі мәтіндерді сөйлем және морфологиялық деңгейде талдап, олардың шынайы адам жазғанын немесе жасанды интеллект (LLM) арқылы жасалғанын анықтайтын ғылыми-түсіндірмелі жүйе.",
        "badge_tri_domain": "[Үш-Домендік Тұрақтылық: Пікірлер • Жаңалықтар • Уикипедия]",
        "badge_dual_stream": "[Қос Ағынды Бағана: BERT + FST]",
        "badge_sliding_window": "[Жылжымалы Терезе: 25k сөзге дейін]",
        "badge_xss_safe": "[XSS Қауіпсіз Жылу Картасы]",
        "quick_samples_title": "Жылдам сынақ үлгілері:",
        "pill_1": "Kaspi пікірі [Адам]",
        "pill_2": "Ресми жаңалықтар [Адам]",
        "pill_3": "Sherkala-7B [ЖИ]",
        "pill_4": "Sherkala жаңалықтары [ЖИ]",
        "pill_5": "Qwen-2.5 [Жабайы ЖИ]",
        "pill_6": "Гибрид құжат [Аралас]",
        "methodology_accordion": "Бенчмарк әдістемесі мен теориялық негіздер (Methodology)",
        "subtab_overview_title": "Сөйлемнің морфологиялық құрылымы (Morphology & Fusion)",
        "subtab_morphology_title": "Сөйлемнің морфологиялық құрылымы (Morphology & Fusion)",
        "subtab_autofst_title": "Auto-FST және сленг лабораториясы (Auto-FST & Slang Lab)",
        "subtab_verification_title": "Фактілер мен сенім матрицасы (Verification & Trust Matrix)",
        "autofst_toggle_label": "Auto-FST: Сленг және пернетақта қателерін түзету",
        "verifier_toggle_label": "Kazakh-FEVER: Фактілер мен сенімділікті тексеру",
        "methodology_accordion": "Benchmark Methodology & Theoretical Foundations / Әдістеме мен теориялық негіздер",
        "subtab_overview_title": "Overview & Heatmap / Шолу",
        "subtab_morphology_title": "Morphology & Slang Lab / Морфология",
        "subtab_verification_title": "Fact Verification & Trust Matrix / Тексеру",
        "autofst_toggle_label": "Auto-FST Slang & Russian-Layout Normalization",
        "verifier_toggle_label": "Kazakh-FEVER Factual Trust Verification",
        "tab_1_title": "Detection & Explainability",
        "tab_2_title": "Morphological FST Lab",
        "tab_3_title": "Benchmark & Methodology",
        "presets_accordion": "Дайын Бенчмарк Үлгілері Кітапханасы (6 Пресет)",
        "presets_help": "Төмендегі дайын үлгілердің бірін таңдап, жүйенің әртүрлі жанрларда қалай жұмыс істейтінін көріңіз:",
        "preset_select_label": "Үлгіні таңдаңыз (Benchmark Preset)",
        "load_preset_btn": "Үлгіні Жүктеу",
        "file_accordion": "Құжатты файл арқылы жүктеу (.txt, .docx, .pdf)",
        "input_label": "Мәтін енгізу (Document Input)",
        "input_placeholder": "Қазақша мәтінді осында жазыңыз немесе көшіріп қойыңыз (25 000 сөзге дейін)...",
        "upload_label": "Құжатты жүктеу (.txt, .docx, .pdf — макс. 10 MB)",
        "analyze_btn": "Құжатты Толық Талдау (Run Full Analysis)",
        "clear_btn": "Тазалау (Clear)",
        "kpi_header": "### Құжаттың Жалпы Нәтижесі (Document KPIs)",
        "awaiting_input": "[AWAITING INPUT]",
        "verdict_human": "[AUTHENTIC HUMAN]",
        "verdict_hybrid": "[PARTIALLY AI / HYBRID]",
        "verdict_ai": "[MACHINE-GENERATED]",
        "prob_label": "AI Ықтималдығы (AI Probability)",
        "ratio_label": "AI Көлемі (AI Content Ratio)",
        "stats_label": "Құжат Статистикасы (Statistics)",
        "linguistic_header": "#### Динамикалық Лингвистикалық Түсіндірме (Linguistic Reasoning)",
        "heatmap_header": "### Сөйлем деңгейіндегі визуалды жылу картасы (Visual Sentence Heatmap)",
        "legend_human": "Адам жазған (Human): 0% – 39.9%",
        "legend_amber": "Күмәнді / Аралық (Caution): 40.0% – 99.79%",
        "legend_ai": "AI Генерациясы (AI): ≥ 99.80%",
        "empty_heatmap": "Мәтін енгізілмеді немесе бос.",
        "diagnostic_header": "### Сөйлемді Терең Талдау (Sentence Deep Diagnostic)",
        "sentence_select_label": "Талдайтын сөйлемді таңдаңыз (Select Sentence # to inspect)",
        "inspected_sentence_label": "Таңдалған сөйлем мәтіні (Inspected Sentence)",
        "gate_header": "#### Динамикалық Бағана (Dynamic Fusion Gate: BERT vs FST)",
        "morpheme_header": "#### Морфологиялық Агглютинативті Құрылым (FST Word Breakdown)",
        "select_sentence_prompt": "Морфологиялық талдау үшін сөйлемді таңдаңыз.",
        "fst_lab_header": "### Морфологиялық Талдау Зертханасы (Morphological FST Lab)",
        "fst_lab_desc": "Қазақ тілінің агглютинативті сөзжасам құрылымын, септік, көптік, шақ, жақ жалғауларының тізбегін (FST) және семантикалық динамикалық бағананы (g) талдау зертханасы.",
        "fst_input_label": "Сөз немесе сөйлем енгізу (Word / Sentence Input)",
        "fst_input_placeholder": "Талдау үшін қазақша сөз немесе сөйлем жазыңыз (мысалы: Қазақстанның қалаларында тұратын адамдармен сөйлестім)...",
        "fst_parse_btn": "Талдау (Parse Morphemes)",
        "fst_gate_header": "#### Динамикалық Үйлестіру Бағанасы (Context g vs Morpheme 1-g)",
        "fst_table_header": "#### Толық Агглютинативті Морфемалық Бөлшектеу Кестесі (FST Decomposition)",
        "file_success": "Файл сәтті жүктелді: {words:,} сөз анықталды.",
        "file_warning": "Ескерту: {err}",
        # Tab 4: Factual Verification & Trust Matrix
        "tab_4_title": "Factual Verification & Trust Matrix / Деректі тексеру және Сенім матрицасы",
        "verify_hero_header": "### Деректі Тексеру және Төрт-Квадрантты Сенім Матрицасы (Factual Verification & Trust Matrix)",
        "verify_hero_desc": "Қазақ тіліндегі мәтіннен фактілік мәлімдемелерді автоматты бөліп алып, бекітілген Уикипедия білім базасы арқылы растайды және AI стилистикалық қаупімен біріктіріп, 4 квадрантты матрица бойынша сенімділігін бағалайды.",
        "verify_quick_samples_title": "Сенім матрицасының 4 квадрантты үлгілері:",
        "verify_pill_1": "1-кв: Расталған ақиқат",
        "verify_pill_2": "2-кв: Адам қателігі",
        "verify_pill_3": "3-кв: Нақты AI синтезі",
        "verify_pill_4": "4-кв: Галлюцинациялық AI",
        "verify_input_label": "Тексерілетін мәтін немесе мәлімдемелер (Document Input)",
        "verify_input_placeholder": "Қазақша мәтінді осында жазыңыз (мысалы: Қазақстан 1991 жылы тәуелсіздік алды. Астана қаласы — елорда)...",
        "verify_btn": "Деректерді Тексеру (Verify Claims & Trust)",
        "verify_clear_btn": "Тазалау (Clear)",
        "verify_card_header": "### Сенімділік Қорытындысы (Trustworthiness Assessment)",
        "verify_claims_header": "### Атомдық Мәлімдемелер мен Дәйексөздер (Atomic Claims & Citations)",
        "verify_model_label": "Тексеру механизмі",
        "verify_model_ours": "Біздің модель (Гибрид + Морфо)",
        "verify_model_base": "Базалық модель (Беткі қабат)",
        "trust_plane_title": "2D Сенім Матрицасының Координат Кеңістігі",
        # Tab 5: Auto-FST & Slang Normalizer
        "tab_5_title": "Auto-FST & Slang Normalizer / Авто-FST және Сленг",
        "autofst_hero_header": "### Авто-FST: Бейімделгіш Нейро-Символдық Морфология (Auto-FST Adaptive Morphology)",
        "autofst_hero_desc": "Орыс пернетақтасымен жазылған қазақ сөздерін (ә, і, ң орнына а, и, н) қалпына келтіреді және жаңа сленг түбірлерін (вайб, чилл, донат, краш) автоматты тауып, FST моделіне дербес қосады.",
        "autofst_quick_samples_title": "Тексеру Үлгілері (Quick Slang & Keyboard Presets):",
        "autofst_pill_1": "1. Орыс пернетақтасы",
        "autofst_pill_2": "2. Модаль бөлшектер",
        "autofst_pill_3": "3. Жастар сленгі",
        "autofst_pill_4": "4. Гейминг сленгі",
        "autofst_pill_5": "5. Күрделі сөздер",
        "autofst_input_label": "Қазақша мәтін енгізу (Text Input)",
        "autofst_input_placeholder": "Бейресми қазақша немесе сленг сөйлем жазыңыз (мысалы: салем калайсын олар турсынгой вайбтаган)...",
        "autofst_btn": "Auto-FST Талдау (Analyze & Induce)",
        "autofst_clear_btn": "Тазалау (Clear)",
        "autofst_inductions_header": "### Жаңа Анықталған Түбірлер мен Грамматикалық Сәйкестік (Inductions & Safety)",
    },
    "en": {
        "hero_title": "Kazakh AI-Generated Text Detector & Explainability Dashboard",
        "hero_subtitle": "Production-grade, explainable detection of AI-generated Kazakh text with sentence-level heatmaps, dynamic dual-stream gate fusion, and deep morphological FST breakdowns.",
        "badge_tri_domain": "[Tri-Domain Robustness: Reviews • News • Wikipedia]",
        "badge_dual_stream": "[Dual-Stream Gate: BERT + FST]",
        "badge_sliding_window": "[Sliding Window: Up to 25k words]",
        "badge_xss_safe": "[XSS-Safe Interactive Heatmap]",
        "quick_samples_title": "Quick Benchmark Samples:",
        "pill_1": "Kaspi Review [Human]",
        "pill_2": "Formal News [Human]",
        "pill_3": "Sherkala-7B [AI]",
        "pill_4": "Sherkala News [AI]",
        "pill_5": "Qwen-2.5 [Wild AI]",
        "pill_6": "Hybrid Document",
        "methodology_accordion": "Methodology & Academic Benchmarks",
        "subtab_overview_title": "Sentence Morphology & Fusion",
        "subtab_morphology_title": "Sentence Morphology & Fusion",
        "subtab_autofst_title": "Auto-FST & Slang Lab",
        "subtab_verification_title": "Fact Verification & Trust Matrix",
        "autofst_toggle_label": "Auto-FST Slang & Russian-Layout Normalization",
        "verifier_toggle_label": "Kazakh-FEVER Factual Trust Verification",
        "tab_1_title": "Detection & Explainability",
        "tab_2_title": "Morphological FST Lab",
        "tab_3_title": "Benchmark & Methodology",
        "presets_accordion": "Curated Benchmark Demonstration Samples (6 Presets)",
        "presets_help": "Select one of the benchmark scenarios below to test detector performance across different genres and generators:",
        "preset_select_label": "Select Benchmark Sample",
        "load_preset_btn": "Load Preset",
        "file_accordion": "Upload Document File (.txt, .docx, .pdf)",
        "input_label": "Document Input",
        "input_placeholder": "Paste or type Kazakh text here (up to 25,000 words)...",
        "upload_label": "Upload Document (.txt, .docx, .pdf — max 10 MB)",
        "analyze_btn": "Run Full Analysis",
        "clear_btn": "Clear",
        "kpi_header": "### Document Summary (KPIs)",
        "awaiting_input": "[AWAITING INPUT]",
        "verdict_human": "[AUTHENTIC HUMAN]",
        "verdict_hybrid": "[PARTIALLY AI / HYBRID]",
        "verdict_ai": "[MACHINE-GENERATED]",
        "prob_label": "AI Probability",
        "ratio_label": "AI Content Volume Ratio",
        "stats_label": "Document Statistics",
        "linguistic_header": "#### Dynamic Linguistic Reasoning",
        "heatmap_header": "### Sentence-Level Visual Heatmap",
        "legend_human": "Human Written: 0% – 39.9%",
        "legend_amber": "Borderline / Caution: 40.0% – 99.79%",
        "legend_ai": "AI-Generated: ≥ 99.80%",
        "empty_heatmap": "No text entered or document is empty.",
        "diagnostic_header": "### Sentence Deep Diagnostic & Morphological FST",
        "sentence_select_label": "Select Sentence # to inspect",
        "inspected_sentence_label": "Inspected Sentence Text",
        "gate_header": "#### Dynamic Fusion Gate (BERT Context vs FST Structure)",
        "morpheme_header": "#### Kazakh Agglutinative Word Decomposition (FST Breakdown)",
        "select_sentence_prompt": "Select a sentence above to view morphological breakdown.",
        "fst_lab_header": "### Morphological FST Lab",
        "fst_lab_desc": "Interactive laboratory for inspecting Kazakh agglutinative inflectional morphology, suffix chains (FST), and dynamic fusion gate weighting (g vs 1-g).",
        "fst_input_label": "Word / Sentence Input",
        "fst_input_placeholder": "Enter a Kazakh word or sentence to parse (e.g. Қазақстанның қалаларында тұратын адамдармен сөйлестім)...",
        "fst_parse_btn": "Parse Morphemes",
        "fst_gate_header": "#### Dynamic Fusion Gate Allocation (BERT g vs FST 1-g)",
        "fst_table_header": "#### Full Agglutinative Morpheme Decomposition Table (FST Breakdown)",
        "file_success": "File loaded successfully: {words:,} words detected.",
        "file_warning": "Warning: {err}",
        # Tab 4: Factual Verification & Trust Matrix
        "tab_4_title": "Factual Verification & Trust Matrix / Деректі тексеру және Сенім матрицасы",
        "verify_hero_header": "### Evidence-Grounded Factual Verification & Four-Quadrant Trust Matrix",
        "verify_hero_desc": "Extracts atomic factual claims from Kazakh text, verifies them against verified Wikipedia passages via hybrid retrieval and NLI, and projects documents onto the Four-Quadrant Trust Matrix.",
        "verify_quick_samples_title": "Four-Quadrant Demonstration Samples:",
        "verify_pill_1": "Q1: Verified Human Fact",
        "verify_pill_2": "Q2: Human Misinformation",
        "verify_pill_3": "Q3: Accurate AI Synthesis",
        "verify_pill_4": "Q4: Hallucinatory AI Disinfo",
        "verify_input_label": "Document Input for Factual Verification",
        "verify_input_placeholder": "Paste or type Kazakh text here (e.g. Қазақстан 1991 жылы тәуелсіздік алды. Астана қаласы — елорда)...",
        "verify_btn": "Verify Claims & Trust",
        "verify_clear_btn": "Clear",
        "verify_card_header": "### Trustworthiness Assessment",
        "verify_claims_header": "### Atomic Claims & Evidence Citations",
        "verify_model_label": "Verification Engine",
        "verify_model_ours": "Ours (Hybrid + Morpho)",
        "verify_model_base": "Baseline (Surface Only)",
        "trust_plane_title": "2D Trust Matrix Coordinate Space",
        # Tab 5: Auto-FST and Slang Normalizer
        "tab_5_title": "Auto-FST and Slang Normalizer / Авто-FST және Сленг",
        "autofst_hero_header": "### Auto-FST: Adaptive Neuro-Symbolic Morphology for Informal Kazakh",
        "autofst_hero_desc": "Normalizes Kazakh text typed on Russian keyboard layouts and dynamically induces newly encountered slang/loanword roots (vibe, chill, donate, crush) into the active FST grammar graph with sub-millisecond latency on CPU.",
        "autofst_quick_samples_title": "Quick Slang and Keyboard Presets:",
        "autofst_pill_1": "1. Russian Layout",
        "autofst_pill_2": "2. Modal Particles",
        "autofst_pill_3": "3. Youth Slang",
        "autofst_pill_4": "4. Gaming Slang",
        "autofst_pill_5": "5. Compound Words",
        "autofst_input_label": "Document / Sentence Input",
        "autofst_input_placeholder": "Type or paste informal Kazakh text (e.g. салем калайсын олар турсынгой вайбтаган)...",
        "autofst_btn": "Run Auto-FST Analysis",
        "autofst_clear_btn": "Clear",
        "autofst_inductions_header": "### Newly Discovered Stems and Phonetic Profile (Auto-Inductions)",
    }
}


def extract_detailed_morphemes(sentence_text: str) -> List[Dict[str, Any]]:
    """
    Extracts detailed 7-category agglutinative morpheme components for Tab 2:
    (Word, Root, POS, Case, Plural, Tense, Person, Suffix Chain) with FST morphological rules.
    """
    if not sentence_text or not isinstance(sentence_text, str):
        return []

    words = _extract_words(sentence_text)
    if not words:
        return []

    analyzer = get_default_fst_analyzer()
    loanword_roots = sorted(analyzer.loanword_roots, key=len, reverse=True)
    results: List[Dict[str, Any]] = []

    for word in words:
        if len(word) <= 2 or word.isdigit() or re.match(r'^\d+([.,]\d+)?$', word):
            results.append({
                "word": word,
                "root": word,
                "pos": "OTHER",
                "case": "—",
                "plural": "—",
                "tense": "—",
                "person": "—",
                "suffix_chain": "—",
                "affixes": []
            })
            continue

        w_lower = word.lower()
        loan_matched = False
        for root in loanword_roots:
            if w_lower.startswith(root):
                if len(w_lower) > len(root):
                    sfx = word[len(root):]
                    is_case = any(sfx.lower().endswith(c) for c in analyzer.cases)
                    is_plur = any(sfx.lower().startswith(p) for p in analyzer.plurals)
                    results.append({
                        "word": word,
                        "root": word[:len(root)],
                        "pos": "LOANWORD/NOUN",
                        "case": f"-{sfx}" if is_case else "—",
                        "plural": f"-{sfx}" if is_plur else "—",
                        "tense": "—",
                        "person": "—",
                        "suffix_chain": f"-{sfx}",
                        "affixes": [f"-{sfx}"]
                    })
                    loan_matched = True
                    break
                elif len(w_lower) == len(root):
                    results.append({
                        "word": word,
                        "root": word,
                        "pos": "LOANWORD/NOUN",
                        "case": "—",
                        "plural": "—",
                        "tense": "—",
                        "person": "—",
                        "suffix_chain": "—",
                        "affixes": []
                    })
                    loan_matched = True
                    break

        if loan_matched:
            continue

        current = word
        matched_affixes: List[Tuple[str, str]] = []
        case_val = "—"
        plur_val = "—"
        tense_val = "—"
        person_val = "—"
        has_verb = False
        has_noun = False

        # 1. Verbal Personal Agreement Suffixes
        m_vper = analyzer.verb_person_re.search(current.lower())
        if m_vper and len(current[:m_vper.start()]) >= 3:
            sfx = current[m_vper.start():]
            current = current[:m_vper.start()]
            has_verb = True
            tag = VERB_PERSON_TAGS.get(sfx.lower(), "PERS")
            person_val = f"-{sfx} ({tag})"
            matched_affixes.insert(0, (sfx, tag))

        # 2. Verbal Tense / Participle
        m_vt = analyzer.verb_tense_re.search(current.lower())
        if m_vt and len(current[:m_vt.start()]) >= 3:
            sfx = current[m_vt.start():]
            current = current[:m_vt.start()]
            has_verb = True
            tag = VERB_TENSE_TAGS.get(sfx.lower(), "TENSE")
            tense_val = f"-{sfx} ({tag})"
            matched_affixes.insert(0, (sfx, tag))

        # 2b. Verbal Negation Suffixes (e.g. қанағаттанба -> -ба, келме -> -ме)
        if has_verb:
            for neg in ["ба", "бе", "па", "пе", "ма", "ме"]:
                if current.lower().endswith(neg) and len(current[:-len(neg)]) >= 3:
                    sfx = current[-len(neg):]
                    current = current[:-len(neg)]
                    tag = VERB_NEG_TAGS.get(neg.lower(), "NEG")
                    matched_affixes.insert(0, (sfx, tag))
                    break

            # 2c. Verbalizer Derivational Suffixes (e.g. қанағаттан -> -тан, пайдалан -> -лан)
            for deriv in ["лан", "лен", "дан", "ден", "тан", "тен"]:
                if current.lower().endswith(deriv) and len(current[:-len(deriv)]) >= 3:
                    sfx = current[-len(deriv):]
                    current = current[:-len(deriv)]
                    tag = VERB_DERIV_TAGS.get(deriv.lower(), "VERB.DERIV")
                    matched_affixes.insert(0, (sfx, tag))
                    break

        # 3. Noun Cases
        m_case = analyzer.case_re.search(current.lower())
        if m_case and len(current[:m_case.start()]) >= 3:
            sfx = current[m_case.start():]
            current = current[:m_case.start()]
            has_noun = True
            tag = CASE_TAGS.get(sfx.lower(), "CASE")
            case_val = f"-{sfx} ({tag})"
            matched_affixes.insert(0, (sfx, tag))

        # 4. Noun Possessives
        if len(current) > 3:
            m_poss = analyzer.poss_re.search(current.lower())
            if m_poss and len(current[:m_poss.start()]) >= 3:
                sfx = current[m_poss.start():]
                current = current[:m_poss.start()]
                has_noun = True
                tag = POSS_TAGS.get(sfx.lower(), "POSS")
                matched_affixes.insert(0, (sfx, tag))

        # 5. Plurals
        if len(current) > 3:
            m_plur = analyzer.plur_re.search(current.lower())
            if m_plur and len(current[:m_plur.start()]) >= 3:
                sfx = current[m_plur.start():]
                current = current[:m_plur.start()]
                has_noun = True
                tag = PLUR_TAGS.get(sfx.lower(), "PLUR")
                plur_val = f"-{sfx} ({tag})"
                matched_affixes.insert(0, (sfx, tag))

        if has_verb:
            pos = "VERB"
        elif has_noun:
            pos = "NOUN"
        else:
            pos = "STEM/OTHER"

        root = current
        annotated_affixes = [f"-{sfx} ({tag})" for sfx, tag in matched_affixes]
        chain_str = " -> ".join(annotated_affixes) if annotated_affixes else "—"

        results.append({
            "word": word,
            "root": root,
            "pos": pos,
            "case": case_val,
            "plural": plur_val,
            "tense": tense_val,
            "person": person_val,
            "suffix_chain": chain_str,
            "affixes": annotated_affixes
        })

    return results

# Note: OfflineHeuristicDetector is imported directly from models.heuristic_detector



def get_detector(load_model: bool = False):
    """
    Initializes either the full PyTorch DocumentDetector or the defensive OfflineHeuristicDetector.
    """
    if load_model:
        try:
            from kaz_mage.detector import DocumentDetector
            detector = DocumentDetector()
            return detector
        except Exception:
            pass
    return OfflineHeuristicDetector()


def get_workspace_i18n(lang_choice: str = "English") -> Dict[str, Any]:
    """
    Returns localized UI dictionary for the given language choice (zero emojis).
    """
    lang_key = _get_lang_key(lang_choice)
    return I18N.get(lang_key, I18N["en"])


def _get_lang_key(lang_choice: str) -> str:
    """Extracts 'kz' or 'en' from language dropdown selection."""
    if not lang_choice:
        return "en"
    return "en" if "English" in lang_choice or "en" in lang_choice.lower() else "kz"


def render_hero_html(lang: str = "en") -> str:
    """Renders publication-grade academic hero header card with zero emojis."""
    d = I18N.get(lang, I18N["en"])
    return f"""
    <div class="hero-card">
        <div class="hero-title">{d['hero_title']}</div>
        <div class="hero-subtitle">{d['hero_subtitle']}</div>
        <div class="hero-badges">
            <span class="feature-badge">{d['badge_tri_domain']}</span>
            <span class="feature-badge">{d['badge_dual_stream']}</span>
            <span class="feature-badge">{d['badge_sliding_window']}</span>
            <span class="feature-badge">{d['badge_xss_safe']}</span>
        </div>
    </div>
    """


def render_legend_html(lang: str = "en") -> str:
    """Renders the three-tier color legend bar with zero emojis."""
    d = I18N.get(lang, I18N["en"])
    return f"""
    <div class="legend-bar">
        <span class="legend-item"><span class="legend-pill legend-pill-human"><span class="legend-dot dot-human"></span> {d['legend_human']}</span></span>
        <span class="legend-item"><span class="legend-pill legend-pill-amber"><span class="legend-dot dot-amber"></span> {d['legend_amber']}</span></span>
        <span class="legend-item"><span class="legend-pill legend-pill-ai"><span class="legend-dot dot-ai"></span> {d['legend_ai']}</span></span>
    </div>
    """


def render_methodology_markdown(lang: str = "en") -> str:
    """Renders academic documentation for Tab 3 (Benchmark & Academic Methodology)."""
    if lang == "en":
        return """### Benchmark & Academic Methodology

Academic documentation and empirical evaluation methodology for Kazakh AI-generated text detection.

---

#### 1. ACL 2024 Kaz-MAGE 2x2 Evaluation Matrix

The Kaz-MAGE evaluation protocol structures detection along two orthogonal axes:
- **Domain Axis:** In-distribution seen domain (Consumer Reviews) vs out-of-distribution unseen domains (Formal News, Academic Wikipedia).
- **Generator Axis:** In-distribution seen generator (Sherkala-7B) vs unseen wild generator (Qwen-2.5-7B-Instruct).

| Quadrant | Domain | Generator | Scientific Significance |
|---|---|---|---|
| **Q1** | Seen (Reviews) | Seen (Sherkala-7B) | Standard in-distribution baseline |
| **Q2** | Seen (Reviews) | Unseen (Qwen-2.5-7B) | Robustness against unseen generator architectures |
| **Q3** | Unseen (News / Wiki) | Seen (Sherkala-7B) | Cross-domain transfer (Model 1 domain blindspot) |
| **Q4 (Wild)** | Unseen (News / Wiki) | Unseen (Qwen-2.5-7B) | Double unseen zero-shot generalization benchmark |

---

#### 2. Empirical Results Comparison Table

Performance comparison between baseline (Model 1: KazRoBERTa) and the proposed morphological gated detector (Model 5: Dual-Stream Morpho-Gated):

| Quadrant | Domain & Setup | Model 1 (KazRoBERTa) | Model 5 (Morpho-Gated) | Empirical Findings & Gains |
|---|---|---|---|---|
| **Q1** | Seen Domain (Reviews) | 0.9996 | **0.9997** (99.7%) | Optimal in-distribution retention |
| **Q2** | Generator Shift | 0.7812 | **0.9253** (90.1%) | High unseen LLM robustness (+14.4% AUC) |
| **Q3** | Domain Shift | 0.5762 | **0.9980** (97.5%) | **Resolved Q3 domain blindspot** (+42.2% AUC) |
| **Q4 Wild** | Double Unseen (Wild) | 0.6231 | **1.0000** (100.0%) | **Zero-shot generalization ceiling** (+37.7% AUC) |

---

#### 3. Dual-Stream Gated Cross-Attention Formulation

Kazakh is a morphologically rich agglutinative language. Standard pretrained language models suffer severe domain blindspots when facing unseen vocabulary and synthetic affixation. Our dual-stream architecture dynamically fuses contextual semantic encodings with morphological FST topological representations:

$$\\mathbf{h} = \\mathbf{g} \\odot \\mathbf{h}_{\\text{sem}} + (1 - \\mathbf{g}) \\odot \\mathbf{h}_{\\text{morph}}$$

The dynamic gating weight $\\mathbf{g} \\in [0, 1]$ adaptively balances contextual semantics and inflectional topology:

$$\\mathbf{g} = \\sigma\\left(\\mathbf{W}_g [\\mathbf{h}_{\\text{sem}};\\; \\mathbf{h}_{\\text{morph}}] + \\mathbf{b}_g\\right)$$

- **$\\mathbf{h}_{\\text{sem}}$**: Semantic context vector from the transformer backbone (BERT/RoBERTa encoder output).
- **$\\mathbf{h}_{\\text{morph}}$**: Morphological topological representation from the FST transition graph.
- **$\\mathbf{g}$**: Learned dynamic gating weight ($\\sigma$ denotes the sigmoid activation function).

---

#### 4. Evidence-Grounded Factual Verification & Four-Quadrant Trust Matrix

Stylistic AI detection alone is insufficient for high-stakes information environments. A machine-generated text may convey strictly verified truths (e.g. encyclopedic synthesis), while a human-authored text may propagate malicious misinformation. To address this, our system introduces a Dual-Risk Factual Verification architecture:

$$\\text{Risk}_{\\text{Trust}} = \\alpha \\cdot \\text{Risk}_{\\text{AI}} + (1 - \\alpha) \\cdot \\text{Risk}_{\\text{Fact}}$$

Where $\\text{Risk}_{\\text{Fact}}$ is computed by extracting atomic propositions $C = \\{c_1, \\dots, c_m\\}$, retrieving evidence passages from the verified Kazakh Wikipedia knowledge corpus, and aggregating NLI penalties:

$$\\text{Risk}_{\\text{Fact}} = \\frac{1}{|C|} \\sum_{c \\in C} \\text{Penalty}(\\text{verdict}(c))$$

$$\\text{Penalty}(v) = \\begin{cases} 0.0, & v = \\text{SUPPORTED} \\\\ 0.25, & v = \\text{NOT ENOUGH INFO} \\\\ 1.0, & v = \\text{REFUTED} \\end{cases}$$

Documents are mapped into the Four-Quadrant Trust Matrix:

| Quadrant | AI Risk ($\\text{Risk}_{\\text{AI}}$) | Factual Risk ($\\text{Risk}_{\\text{Fact}}$) | Epistemic Status | Operational Action |
|---|---|---|---|---|
| **Q1** | Low ($< 0.50$) | Low ($< 0.40$) | **Verified Human Fact** | High trust; verified for institutional and academic use |
| **Q2** | Low ($< 0.50$) | High ($\\ge 0.40$) | **Human Misinformation** | Human origin with factual falsehoods; flag for correction |
| **Q3** | High ($\\ge 0.50$) | Low ($< 0.40$) | **Accurate AI Synthesis** | Machine-generated factual summary; attribute AI synthesis |
| **Q4** | High ($\\ge 0.50$) | High ($\\ge 0.40$) | **Hallucinatory AI Disinformation** | Synthetic fabrication / hallucination; isolate immediately |

##### Empirical Kazakh-FEVER Benchmark Results

Evaluated across 36 balanced records spanning 36 encyclopedic Kazakh topics:

| Evaluation Metric | Baseline BM25 | Proposed Dual-Stream System | Relative Gain |
|---|---|---|---|
| **NLI Macro-F1** | 58.33% | **100.00%** (36/36) | **+41.67%** |
| **Joint Strict FEVER Score** | 36.11% | **66.67%** (24/36) | **+30.56%** |
| **Mean Retrieval Similarity** | 0.614 | **0.872** | **+42.02%** |
| **Mean NLI Confidence** | 68.50% | **94.21%** | **+25.71%** |

---

#### 5. Academic Research & Computational Infrastructure

- **Project:** Master of Science Thesis Research on Kazakh Natural Language Processing
- **Research Scope:** Morphologically-Grounded Generative AI Detection & Trustworthy Fact Verification
- **Architecture:** Dual-Stream Gated Cross-Attention & Hybrid Retrieval-Augmented NLI
"""
    else:
        return """### Бенчмарк және Ғылыми Әдістеме (Benchmark & Academic Methodology)

Бұл зерттеу қазақ тіліндегі жасанды интеллект мәтіндерін анықтауға арналған ғылыми әдістеме мен Kaz-MAGE эмпирикалық нәтижелерін қамтиды.

---

#### 1. ACL 2024 Kaz-MAGE 2x2 Бағалау Матрицасы

Kaz-MAGE бағалау матрицасы генеративті мәтіндерді екі негізгі ось бойынша жіктейді:
- **Домен осі (Domain axis):** Үйретілген домен (Тұтынушы пікірлері / Consumer Reviews) және Үйретілмеген жаңа домендер (Ресми жаңалықтар / Formal News, Академиялық Уикипедия / Academic Wikipedia).
- **Генератор осі (Generator axis):** Үйретілген модель (Sherkala-7B) және Бейтаныс жабайы генератор (Qwen-2.5-7B-Instruct).

| Квадрант | Домен (Domain) | Генератор (Generator) | Сипаттамасы |
|---|---|---|---|
| **Q1** | Үйретілген (Reviews) | Үйретілген (Sherkala-7B) | Үлестірім ішіндегі базалық тексеру (In-distribution baseline) |
| **Q2** | Үйретілген (Reviews) | Бейтаныс (Qwen-2.5-7B) | Генераторлар аралық тұрақтылық (Cross-generator generalization) |
| **Q3** | Бейтаныс (News / Wiki) | Үйретілген (Sherkala-7B) | Домен аралық тасымалдау (Cross-domain transfer / Domain blindspot) |
| **Q4 (Wild)** | Бейтаныс (News / Wiki) | Бейтаныс (Qwen-2.5-7B) | Қос бейтаныс нөлдік жалпылау (Zero-shot unseen generalization) |

---

#### 2. Эмпирикалық Нәтижелерді Салыстыру Кестесі

Төмендегі кестеде базалық модель (Model 1: KazRoBERTa) мен ұсынылған морфологиялық бағаналы модельдің (Model 5: Dual-Stream Morpho-Gated) ROC-AUC және анықтау көрсеткіштері келтірілген:

| Квадрант | Домен түрі | Модель 1 (KazRoBERTa) | Модель 5 (Morpho-Gated) | Салыстырмалы жетістік / Қорытынды |
|---|---|---|---|---|
| **Q1** | Үйретілген (Reviews) | 0.9996 | **0.9997** (99.7%) | Тұрақты жоғары базалық дәлдік |
| **Q2** | Генератор ауысуы | 0.7812 | **0.9253** (90.1%) | Жаңа LLM стиліне төзімділік (+14.4% AUC) |
| **Q3** | Домен ауысуы | 0.5762 | **0.9980** (97.5%) | **Q3 домендік соқыр нүктесі толық шешілді** (+42.2% AUC) |
| **Q4 Wild** | Қос бейтаныс (Wild) | 0.6231 | **1.0000** (100.0%) | **Нөлдік үлгідегі кемел жалпылау қабілеті** (+37.7% AUC) |

---

#### 3. Қос Ағынды Динамикалық Бағана (Dual-Stream Gated Cross-Attention)

Қазақ тілі — агглютинативті тіл, сондықтан тек семантикалық сөз тізбегі жеткіліксіз. Жүйе контекстік семантиканы (BERT/RoBERTa) және морфологиялық құрылымды (FST аффикстік жіктеу) динамикалық бағана тетігі арқылы біріктіреді:

$$\\mathbf{h} = \\mathbf{g} \\odot \\mathbf{h}_{\\text{sem}} + (1 - \\mathbf{g}) \\odot \\mathbf{h}_{\\text{morph}}$$

Динамикалық үйлестіру бағанасы $\\mathbf{g} \\in [0, 1]$ келесідей есептеледі:

$$\\mathbf{g} = \\sigma\\left(\\mathbf{W}_g [\\mathbf{h}_{\\text{sem}};\\; \\mathbf{h}_{\\text{morph}}] + \\mathbf{b}_g\\right)$$

- **$\\mathbf{h}_{\\text{sem}}$**: Семантикалық контекст векторы (BERT/RoBERTa encoder шығысы).
- **$\\mathbf{h}_{\\text{morph}}$**: Морфологиялық FST топологиясы мен аффикстік тізбек векторы.
- **$\\mathbf{g}$**: Динамикалық үйлестіру бағанасы ($\\sigma$ — сигмоид функциясы).

---

#### 4. Деректі Тексеру және Төрт-Квадрантты Сенім Матрицасы (Evidence-Grounded Factual Verification & Trust Matrix)

Жасанды интеллект мәтіндерін тек стилистикалық тұрғыдан анықтау сенімді ақпараттық қауіпсіздікті қамтамасыз ете алмайды. Машина жасаған мәтін толық расталған ғылыми деректерден тұруы мүмкін (нақты AI синтезі), ал адам жазған мәтінде қате немесе қасақана бұрмаланған жалған ақпарат болуы мүмкін. Осыған байланысты жүйе Қос Қауіпті Сенім Матрицасын (Dual-Risk Trust Matrix) қолданады:

$$\\text{Risk}_{\\text{Trust}} = \\alpha \\cdot \\text{Risk}_{\\text{AI}} + (1 - \\alpha) \\cdot \\text{Risk}_{\\text{Fact}}$$

Мұндағы фактілік қайшылық қаупі ($\\text{Risk}_{\\text{Fact}}$) мәтіннен атомдық мәлімдемелерді $C = \\{c_1, \\dots, c_m\\}$ бөліп алу, қазақша Уикипедия базасынан дәлелдер іздеу және NLI жазаларын біріктіру арқылы есептеледі:

$$\\text{Risk}_{\\text{Fact}} = \\frac{1}{|C|} \\sum_{c \\in C} \\text{Penalty}(\\text{verdict}(c))$$

$$\\text{Penalty}(v) = \\begin{cases} 0.0, & v = \\text{SUPPORTED} \\\\ 0.25, & v = \\text{NOT ENOUGH INFO} \\\\ 1.0, & v = \\text{REFUTED} \\end{cases}$$

Құжаттар Төрт Квадрант бойынша жіктеледі:

| Квадрант | AI Қаупі ($\\text{Risk}_{\\text{AI}}$) | Дерек Қаупі ($\\text{Risk}_{\\text{Fact}}$) | Сенімділік Мәртебесі | Қолдану Әрекеті |
|---|---|---|---|---|
| **Q1** | Төмен ($< 0.50$) | Төмен ($< 0.40$) | **Расталған ақиқат (Verified Human Fact)** | Сенім деңгейі жоғары; ресми пайдалануға қауіпсіз |
| **Q2** | Төмен ($< 0.50$) | Жоғары ($\\ge 0.40$) | **Адам қателігі (Human Misinformation)** | Адам жазған, бірақ дерек қате; түзетуге жіберіледі |
| **Q3** | Жоғары ($\\ge 0.50$) | Төмен ($< 0.40$) | **Нақты AI синтезі (Accurate AI Synthesis)** | Машиналық синтез, деректері расталған |
| **Q4** | Жоғары ($\\ge 0.50$) | Жоғары ($\\ge 0.40$) | **Галлюцинациялық AI дезинформация** | Синтетикалық жалған дерек; дереу оқшаулау қажет |

##### Kazakh-FEVER Эмпирикалық Бағалау Нәтижелері

36 тексерілген қазақ тіліндегі энциклопедиялық тақырып бойынша 36 теңдестірілген бенчмарк жазбасы:

| Бағалау Көрсеткіші | Базалық BM25 | Ұсынылған Жүйе | Салыстырмалы Өсім |
|---|---|---|---|
| **NLI Macro-F1** | 58.33% | **100.00%** (36/36) | **+41.67%** |
| **Joint Strict FEVER Score** | 36.11% | **66.67%** (24/36) | **+30.56%** |
| **Орташа Ұқсастық Ұпайы** | 0.614 | **0.872** | **+42.02%** |
| **Орташа Сенімділік (Confidence)** | 68.50% | **94.21%** | **+25.71%** |

---

#### 5. Ғылыми Зерттеу және Есептеу Инфрақұрылымы

- **Жоба:** Қазақ тіліндегі табиғи тілді өңдеу бойынша магистрлік диссертациялық зерттеу
- **Зерттеу бағыты:** Морфологиялық негізделген генеративті AI мәтіндерін анықтау және фактілік тексеру
- **Архитектура:** Қос ағынды динамикалық бағана және гибридті іздеуге негізделген NLI
"""


def handle_preset_change(preset_name: str, lang_choice: str = "en") -> Tuple[str, str]:
    """Loads text and formatted metadata when a preset sample is selected."""
    if not preset_name:
        return "", ""
    text = get_preset_text(preset_name)
    meta = get_preset_metadata(preset_name)
    if not meta:
        return text, ""

    lang = _get_lang_key(lang_choice)
    if lang == "kz":
        meta_md = (
            f"**Домен:** `{meta.get('domain', 'N/A')}` | "
            f"**Күтілетін нәтиже:** `{meta.get('expected_verdict', 'N/A')}` | "
            f"**Генератор:** `{meta.get('generator', 'N/A')}`\n\n"
            f"*{meta.get('description', '')}*"
        )
    else:
        meta_md = (
            f"**Domain:** `{meta.get('domain', 'N/A')}` | "
            f"**Expected Verdict:** `{meta.get('expected_verdict', 'N/A')}` | "
            f"**Generator:** `{meta.get('generator', 'N/A')}`\n\n"
            f"*{meta.get('description', '')}*"
        )
    return text, meta_md


def handle_file_upload(file_obj: Any, lang_choice: str = "en") -> Tuple[str, str]:
    """Loads uploaded file (.txt, .docx, .pdf) defensively with size and word limits."""
    if file_obj is None:
        return "", ""
    lang = _get_lang_key(lang_choice)
    d = I18N.get(lang, I18N["en"])

    text, err = load_document_file(file_obj)
    if err:
        status_md = d["file_warning"].format(err=err)
    else:
        words = len(text.split())
        status_md = d["file_success"].format(words=words)
    return text, status_md


def handle_analyze_document(
    text: str,
    detector: Any = None,
    lang_choice: str = "en"
) -> Tuple[str, str, str, Any, str, str, str, List[Dict[str, Any]]]:
    """
    Full document analysis pipeline:
    Runs DocumentDetector/Heuristic, renders the unified executive summary card,
    dynamic linguistic reasoning bullets, visual sentence heatmap,
    populates sentence dropdown, and selects sentence 0 for immediate drilldown.

    Returns:
        Tuple of:
        (
            executive_card_html,
            explanation_bullets_md,
            heatmap_html,
            dropdown_update,
            sentence_text,
            gate_html,
            morpheme_html,
            raw_sents_state
        )
    """
    det = detector or OfflineHeuristicDetector()
    lang = _get_lang_key(lang_choice)
    d = I18N.get(lang, I18N["en"])

    if not text or not text.strip():
        empty_exec = render_executive_summary_card(
            probability=0.0,
            verdict=d["awaiting_input"],
            lang=lang,
            total_words=0,
            total_sents=0,
            total_windows=0,
            ai_ratio=0.0
        )
        empty_heatmap = f'<div class="empty-doc-prompt">{d["empty_heatmap"]}</div>'
        prompt_select = d["select_sentence_prompt"]
        return (
            empty_exec,
            "",
            empty_heatmap,
            gr.update(choices=[], value=None),
            prompt_select,
            render_dynamic_gate_bar(0.5, lang=lang),
            f'<p class="text-muted">{prompt_select}</p>',
            []
        )

    # 1. Run document-level analysis
    doc_res = det.predict_document(text)

    # 2. Segment and score sentences
    sentences = analyze_document_sentences(text, doc_res)

    # 3. Format verdict badge in active language (clean brackets, zero emojis)
    verdict = doc_res.verdict
    if verdict == "Authentic Human":
        verdict_tag = d["verdict_human"]
    elif verdict == "Partially AI" or "partially" in verdict.lower():
        verdict_tag = d["verdict_hybrid"]
    else:
        verdict_tag = d["verdict_ai"]

    # 4. Publication-grade executive summary card
    exec_card_html = render_executive_summary_card(
        probability=doc_res.document_ai_probability,
        verdict=verdict_tag,
        lang=lang,
        total_words=doc_res.total_words,
        total_sents=doc_res.total_sentences,
        total_windows=doc_res.total_chunks,
        ai_ratio=doc_res.ai_content_ratio
    )

    # 5. Dynamic linguistic explanation bullets
    bullets = generate_linguistic_explanation(text, doc_res, lang=lang)
    bullets_md = "\n".join(f"- {b}" for b in bullets)

    # 6. Render heatmap
    heatmap_html = render_document_heatmap(
        sentences,
        calibrated_threshold=doc_res.calibrated_threshold
    )

    # 7. Populate sentence dropdown choices
    choices = [
        (f"#{i+1}: {s['text'][:55]}... ({s['ai_probability']:.1%})", i)
        for i, s in enumerate(sentences)
    ]

    # 8. Default to first sentence for drilldown
    if sentences:
        first_s = sentences[0]
        sent_text = first_s["text"]
        gate_html = render_dynamic_gate_bar(first_s.get("gate_value", 0.5), lang=lang)
        morphemes = analyze_sentence_morphemes(first_s["text"])
        morph_html = render_morpheme_table(morphemes, lang=lang)
        default_val = 0
    else:
        sent_text = ""
        gate_html = render_dynamic_gate_bar(0.5, lang=lang)
        morph_html = f'<p class="text-muted">{d["select_sentence_prompt"]}</p>'
        default_val = None

    dropdown_update = gr.update(choices=choices, value=default_val)

    return (
        exec_card_html,
        bullets_md,
        heatmap_html,
        dropdown_update,
        sent_text,
        gate_html,
        morph_html,
        sentences
    )


def handle_sentence_select(
    selected_index: Optional[int],
    raw_sents_state: List[Dict[str, Any]],
    lang_choice: str = "en"
) -> Tuple[str, str, str]:
    """Updates the diagnostic card when a sentence is selected from the dropdown."""
    lang = _get_lang_key(lang_choice)
    d = I18N.get(lang, I18N["en"])

    if selected_index is None or not raw_sents_state:
        return (
            d["select_sentence_prompt"],
            render_dynamic_gate_bar(0.5, lang=lang),
            f'<p class="text-muted">{d["select_sentence_prompt"]}</p>'
        )

    try:
        idx = int(selected_index)
        if 0 <= idx < len(raw_sents_state):
            s = raw_sents_state[idx]
            sent_text = s["text"]
            gate_val = s.get("gate_value", 0.5)
            gate_html = render_dynamic_gate_bar(gate_val, lang=lang)
            morphemes = analyze_sentence_morphemes(sent_text)
            morph_html = render_morpheme_table(morphemes, lang=lang)
            return sent_text, gate_html, morph_html
    except (ValueError, TypeError, IndexError):
        pass

    return (
        "Sentence not found." if lang == "en" else "Сөйлем табылмады.",
        render_dynamic_gate_bar(0.5, lang=lang),
        f'<p class="text-muted">{d["select_sentence_prompt"]}</p>'
    )


def handle_fst_parse(
    text: str,
    detector: Any = None,
    lang_choice: str = "en"
) -> Tuple[str, str]:
    """
    Morphological FST Lab parser for Tab 2:
    Computes dynamic gate split bar and full 8-column agglutinative decomposition table.
    """
    lang = _get_lang_key(lang_choice)
    if not text or not text.strip():
        empty_gate = render_dynamic_gate_bar(0.5, lang=lang)
        empty_table = render_fst_decomposition_table([], lang=lang)
        return empty_gate, empty_table

    det = detector or OfflineHeuristicDetector()
    try:
        doc_res = det.predict_document(text)
        gate_val = doc_res.chunks[0].gate_value if doc_res and doc_res.chunks else 0.5
    except Exception:
        gate_val = 0.5

    gate_html = render_dynamic_gate_bar(gate_val, lang=lang)
    breakdowns = extract_detailed_morphemes(text)
    table_html = render_fst_decomposition_table(breakdowns, lang=lang)
    return gate_html, table_html


def handle_verify_document(
    text: str,
    verifier: Any = None,
    lang_choice: str = "en",
    model_mode: str = "morpho"
) -> Tuple[str, str]:
    """
    Executes end-to-end factual verification on the provided text,
    returning HTML for the executive trust summary card and the claims breakdown table.
    """
    v = verifier or TrustworthyDocumentVerifier()
    lang = _get_lang_key(lang_choice)

    if not text or not text.strip():
        return (
            render_trust_summary_card(None, lang=lang),
            render_claims_verification_table([], lang=lang)
        )

    mode_str = str(model_mode).lower().strip()
    if "base" in mode_str or "база" in mode_str or mode_str == "baseline":
        active_mode = "baseline"
    else:
        active_mode = "morpho"

    res = v.verify(text, verifier_mode=active_mode)
    card_html = render_trust_summary_card(res, lang=lang)
    table_html = render_claims_verification_table(res.claims, lang=lang)
    return card_html, table_html


def switch_ui_language(lang_choice: str) -> Tuple[Any, ...]:
    """
    Dynamically updates all dashboard text, labels, placeholders, and methodology
    when switching between Kazakh and English across all 4 tabs.
    """
    lang = _get_lang_key(lang_choice)
    d = I18N.get(lang, I18N["en"])

    legend_html = render_legend_html(lang)
    empty_exec = render_executive_summary_card(
        0.0, d["awaiting_input"], lang=lang,
        total_words=0, total_sents=0, total_windows=0, ai_ratio=0.0
    )
    empty_heatmap = f'<div class="empty-doc-prompt">{d["empty_heatmap"]}</div>'
    gate_html = render_dynamic_gate_bar(0.5, lang=lang)
    morph_html = f'<p class="text-muted">{d["select_sentence_prompt"]}</p>'
    methodology_md = render_methodology_markdown(lang)
    empty_trust_card = render_trust_summary_card(None, lang=lang)
    empty_claims_table = render_claims_verification_table([], lang=lang)

    return (
        f"**{d['quick_samples_title']}**",
        # 6 sample pills
        gr.update(value=d["pill_1"]),
        gr.update(value=d["pill_2"]),
        gr.update(value=d["pill_3"]),
        gr.update(value=d["pill_4"]),
        gr.update(value=d["pill_5"]),
        gr.update(value=d["pill_6"]),
        # Input & Files
        gr.update(label=d["input_label"], placeholder=d["input_placeholder"]),
        gr.update(label=d["file_accordion"]),
        gr.update(label=d["upload_label"]),
        gr.update(value=d["analyze_btn"]),
        gr.update(value=d["clear_btn"]),
        # Results
        d["kpi_header"],
        empty_exec,
        d["linguistic_header"],
        "",  # linguistic bullets cleared on language switch
        d["heatmap_header"],
        legend_html,
        empty_heatmap,
        d["diagnostic_header"],
        gr.update(label=d["sentence_select_label"]),
        gr.update(label=d["inspected_sentence_label"], value=d["select_sentence_prompt"]),
        d["gate_header"],
        gate_html,
        d["morpheme_header"],
        morph_html,
        # Tab 2: FST Lab
        d["fst_lab_header"],
        d["fst_lab_desc"],
        gr.update(label=d["fst_input_label"], placeholder=d["fst_input_placeholder"]),
        gr.update(value=d["fst_parse_btn"]),
        d["fst_gate_header"],
        gate_html,
        d["fst_table_header"],
        render_fst_decomposition_table([], lang=lang),
        # Tab 3: Methodology
        methodology_md,
        # Tab 4: Verification & Trust Matrix
        d["verify_hero_header"],
        d["verify_hero_desc"],
        f"**{d['verify_quick_samples_title']}**",
        gr.update(value=d["verify_pill_1"]),
        gr.update(value=d["verify_pill_2"]),
        gr.update(value=d["verify_pill_3"]),
        gr.update(value=d["verify_pill_4"]),
        gr.update(
            label=d["verify_model_label"],
            choices=[d["verify_model_ours"], d["verify_model_base"]],
            value=d["verify_model_ours"]
        ),
        gr.update(label=d["verify_input_label"], placeholder=d["verify_input_placeholder"]),
        gr.update(value=d["verify_btn"]),
        gr.update(value=d["verify_clear_btn"]),
        empty_trust_card,
        d["verify_claims_header"],
        empty_claims_table,
        # Tab 5: Auto-FST & Slang Normalizer
        d["autofst_hero_header"],
        d["autofst_hero_desc"],
        f"**{d['autofst_quick_samples_title']}**",
        gr.update(value=d["autofst_pill_1"]),
        gr.update(value=d["autofst_pill_2"]),
        gr.update(value=d["autofst_pill_3"]),
        gr.update(value=d["autofst_pill_4"]),
        gr.update(value=d["autofst_pill_5"]),
        gr.update(label=d["autofst_input_label"], placeholder=d["autofst_input_placeholder"]),
        gr.update(value=d["autofst_btn"]),
        gr.update(value=d["autofst_clear_btn"]),
        render_autofst_comparison_card("", "", 0.0, lang=lang),
        d["autofst_inductions_header"],
        render_autofst_inductions_table([], lang=lang)
    )




def calculate_document_statistics(text: str) -> str:
    """
    Calculates real-time document statistics for the live stats ribbon:
    Words, Sentences, Est. Read Time, Morpheme Depth.
    Returns formatted HTML.
    """
    if not text or not text.strip():
        return '<div class="live-stats-ribbon">Words: 0 • Sentences: 0 • Est. Read Time: 0s • Morpheme Depth: 0.0x</div>'

    words = [w for w in text.split() if w.strip()]
    num_words = len(words)

    sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    num_sents = len(sents) if sents else (1 if num_words > 0 else 0)

    read_sec = int((num_words / 180.0) * 60.0)
    if read_sec < 60:
        read_str = f"{max(1, read_sec)}s"
    else:
        read_str = f"{read_sec // 60}m {read_sec % 60}s"

    avg_len = sum(len(w) for w in words) / max(1, num_words)
    depth = round(avg_len / 4.5, 1)

    return f'<div class="live-stats-ribbon">Words: {num_words} • Sentences: {num_sents} • Est. Read Time: {read_str} • Morpheme Depth: {depth}x</div>'


def generate_audit_markdown(res: Dict[str, Any], source_text: str = "") -> str:
    """
    Generates a structured, publication-grade Markdown audit report for document inspection.
    """
    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Document Inspection & Verification Audit Report",
        "",
        f"- **Generated:** {now_str}",
        "- **Evaluation Engine:** Dual-Stream Morpho-Gated Kazakh Detector & FEVER Verifier",
        "",
        "---",
        "",
        "## Executive Verdict",
        "",
    ]

    words = len(source_text.split()) if source_text else 0
    lines.append(f"- **Document Length:** {words} words")
    card_html = res.get("executive_card", "")
    if "MACHINE-GENERATED" in card_html or "ЖАСАНДЫ" in card_html:
        lines.append("- **Verdict:** Machine-Generated (AI)")
    elif "PARTIALLY" in card_html or "АРАЛАС" in card_html:
        lines.append("- **Verdict:** Partially AI / Hybrid")
    elif "AUTHENTIC" in card_html or "АДАМ" in card_html:
        lines.append("- **Verdict:** Authentic Human")
    else:
        lines.append("- **Verdict:** Awaiting Analysis")

    lines.extend([
        "",
        "---",
        "",
        "## Linguistic & Morphological Observations",
        "",
        res.get("bullets", "").strip() or "No anomalous syntactic patterns detected.",
        "",
        "---",
        "",
        "## Source Document",
        "",
        "```",
        source_text.strip() if source_text else "(No document text provided)",
        "```",
        "",
        "---",
        "*Antigravity Kazakh AI-Text Detection System*"
    ])

    return "\n".join(lines)

def run_full_analysis_logic(
    text: str,
    autofst_enabled: bool = True,
    verifier_enabled: bool = True,
    current_lang: str = "English",
    detector: Any = None,
    verifier: Any = None
) -> Dict[str, Any]:
    """
    Executes unified multi-pipeline analysis across:
    1. AI Detection & Sentence Explainability Heatmap
    2. Dynamic Auto-FST Morphology & Slang Inductions
    3. Kazakh-FEVER Factual Trust & Citations Verification
    """
    import time
    lang = _get_lang_key(current_lang)
    d = I18N.get(lang, I18N["en"])
    det = detector or OfflineHeuristicDetector()
    ver = verifier or TrustworthyDocumentVerifier(ai_detector=det)

    if not text or not text.strip():
        empty_exec = render_executive_summary_card(
            0.0, d["awaiting_input"], lang=lang,
            total_words=0, total_sents=0, total_windows=0, ai_ratio=0.0
        )
        empty_meter = render_confidence_meter(0.0, d["awaiting_input"], lang=lang)
        empty_heatmap = f'<div class="empty-doc-prompt">{d["empty_heatmap"]}</div>'
        prompt_select = d["select_sentence_prompt"]
        return {
            "executive_card": empty_exec,
            "confidence_meter": empty_meter,
            "bullets": "",
            "heatmap": empty_heatmap,
            "sentence_choices": [],
            "sentence_choice": None,
            "selected_sentence": prompt_select,
            "gate_html": render_dynamic_gate_bar(0.5, lang=lang),
            "morpheme_html": f'<p class="text-muted">{prompt_select}</p>',
            "autofst_card": render_autofst_comparison_card("", "", 0.0, lang=lang),
            "autofst_table": render_autofst_inductions_table([], lang=lang),
            "trust_card": render_trust_summary_card(None, lang=lang),
            "claims_table": render_claims_verification_table([], lang=lang),
            "sentences_state": []
        }

    # 1. AI Detection & Heatmap via handle_analyze_document
    doc_res = det.predict_document(text)
    (
        exec_card,
        bullets_md,
        heatmap_html,
        dropdown_update,
        s0_text,
        s0_gate,
        s0_morphemes,
        sentences
    ) = handle_analyze_document(text, detector=det, lang_choice=current_lang)

    verdict = doc_res.verdict
    if verdict == "Authentic Human":
        verdict_tag = d["verdict_human"]
    elif verdict == "Partially AI" or "partially" in verdict.lower():
        verdict_tag = d["verdict_hybrid"]
    else:
        verdict_tag = d["verdict_ai"]

    conf_meter = render_confidence_meter(
        probability=doc_res.document_ai_probability,
        verdict=verdict_tag,
        lang=lang
    )

    # 2. Auto-FST Pipeline
    if autofst_enabled:
        t0 = time.perf_counter()
        b_fst = get_default_fst_analyzer()
        before_parse = b_fst.analyze_and_segment(text)
        overlay = DynamicFSTOverlay(normalizer=RussianLayoutNormalizer())
        inducer = AutoFSTInducer()
        guard = RegressionGuard()
        inductions = []
        seen_stems = set()
        for w in text.split():
            w_clean = re.sub(r'^[^\w]+|[^\w]+$', '', w)
            if not w_clean:
                continue
            w_norm = overlay.normalizer.normalize_word(w_clean)
            cand = inducer.induce_stem(w_norm)
            if cand and cand.stem not in seen_stems:
                seen_stems.add(cand.stem)
                passed, reason = guard.validate_candidate(cand.stem)
                if passed:
                    overlay.register_root(cand.stem, {"harmony": cand.harmony, "coda": cand.coda})
                    inductions.append({
                        "stem": cand.stem,
                        "harmony": cand.harmony,
                        "coda": cand.coda,
                        "suffixes": cand.suffixes,
                        "guard": "Passed (100% agreement)"
                    })
        after_parse = overlay.analyze_and_segment(text)
        latency_ms = (time.perf_counter() - t0) * 1000
        autofst_card = render_autofst_comparison_card(before_parse, after_parse, latency_ms, lang=lang)
        if inductions:
            diff_html = render_morpheme_visual_diff(text, overlay.normalizer.normalize_word(text) if hasattr(overlay.normalizer, 'normalize_word') else text, inductions)
            autofst_card = autofst_card + "\n" + diff_html
        autofst_table = render_autofst_inductions_table(inductions, lang=lang)
    else:
        autofst_card = render_autofst_comparison_card("", "", 0.0, lang=lang)
        autofst_table = render_autofst_inductions_table([], lang=lang)

    # 3. Fact Verification Pipeline via handle_verify_document
    if verifier_enabled:
        trust_card, claims_table = handle_verify_document(text, verifier=ver, lang_choice=current_lang, model_mode="morpho")
    else:
        trust_card = render_trust_summary_card(None, lang=lang)
        claims_table = render_claims_verification_table([], lang=lang)

    choices = dropdown_update.get("choices", []) if isinstance(dropdown_update, dict) else getattr(dropdown_update, "choices", [])
    choice_val = dropdown_update.get("value", None) if isinstance(dropdown_update, dict) else getattr(dropdown_update, "value", None)

    return {
        "executive_card": exec_card,
        "confidence_meter": conf_meter,
        "bullets": bullets_md,
        "heatmap": heatmap_html,
        "sentence_choices": choices,
        "sentence_choice": choice_val,
        "selected_sentence": s0_text,
        "gate_html": s0_gate,
        "morpheme_html": s0_morphemes,
        "autofst_card": autofst_card,
        "autofst_table": autofst_table,
        "trust_card": trust_card,
        "claims_table": claims_table,
        "sentences_state": sentences
    }


LIGHT_MODE_HEAD_JS = """
<script>
(function() {
    function enforceDarkMode() {
        if (document.documentElement) {
            document.documentElement.classList.add('dark');
            document.documentElement.style.colorScheme = 'dark';
        }
        if (document.body) {
            document.body.classList.add('dark');
            document.body.style.colorScheme = 'dark';
        }
    }
    enforceDarkMode();
    if (window.MutationObserver) {
        new MutationObserver(function() { enforceDarkMode(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    }
    window.addEventListener('DOMContentLoaded', enforceDarkMode);
    setInterval(enforceDarkMode, 400);
})();
</script>
"""

def create_app(detector: Any = None, verifier: Any = None, load_model: bool = False) -> gr.Blocks:
    """
    Constructs and returns the full Gradio Blocks application with a publication-grade
    Unified Two-Column Workspace layout and bilingual localization (zero emojis).
    """
    app_detector = detector or get_detector(load_model=load_model)
    app_verifier = verifier or TrustworthyDocumentVerifier(ai_detector=app_detector)
    d = I18N["en"]

    with gr.Blocks(title="Kazakh AI-Text Detector & Explainability UI", theme=get_apple_notion_theme(), css=APP_CSS, head=LIGHT_MODE_HEAD_JS) as demo:
        demo.theme = get_apple_notion_theme()
        # State: stores analyzed sentences list for interactive drilldown
        raw_sents_state = gr.State([])

        # Inject CSS styling and client-side light mode enforcement safely
        gr.HTML(f"<style>{APP_CSS}</style>", visible=False)

        # -------------------------------------------------------------------
        # Top Utility Navigation Bar
        # -------------------------------------------------------------------
        with gr.Row(elem_classes=["quick-samples-header-row"]):
            with gr.Column(scale=12):
                gr.Markdown(f"### {d['hero_title']}")
                gr.Markdown(d["hero_subtitle"])
            # Hidden language switch to preserve backward compatibility with existing tests
            lang_radio = gr.Radio(
                choices=["English", "Қазақша"],
                value="English",
                show_label=False,
                container=False,
                interactive=True,
                visible=False,
                elem_classes=["lang-switch-inline"]
            )

        # Collapsible Methodology & Benchmarks Drawer (Top Utility)
        with gr.Accordion(d["methodology_accordion"], open=False) as methodology_accordion:
            methodology_md = gr.Markdown(render_methodology_markdown("en"))

        # -------------------------------------------------------------------
        # Main Two-Column Workspace (Balanced scale=5 / scale=5)
        # -------------------------------------------------------------------
        with gr.Row(elem_classes=["workspace-row"]):

            # ===============================================================
            # LEFT COLUMN: Document Studio & Heatmap (scale=5)
            # ===============================================================
            with gr.Column(scale=5, elem_classes=["workspace-left-col"]):
                # Quick Benchmark Samples Row
                with gr.Column(elem_classes=["quick-samples-wrapper"]):
                    quick_samples_label = gr.Markdown(f"**{d['quick_samples_title']}**", elem_classes=["quick-samples-title"])
                    with gr.Row(elem_classes=["quick-samples-row"]):
                        pill_btn1 = gr.Button(d["pill_1"], size="sm", elem_classes=["sample-pill", "quick-sample-pill"], scale=1)
                        pill_btn2 = gr.Button(d["pill_2"], size="sm", elem_classes=["sample-pill", "quick-sample-pill"], scale=1)
                        pill_btn3 = gr.Button(d["pill_3"], size="sm", elem_classes=["sample-pill", "quick-sample-pill"], scale=1)
                        pill_btn4 = gr.Button(d["pill_4"], size="sm", elem_classes=["sample-pill", "quick-sample-pill"], scale=1)
                        pill_btn5 = gr.Button(d["pill_5"], size="sm", elem_classes=["sample-pill", "quick-sample-pill"], scale=1)
                        pill_btn6 = gr.Button(d["pill_6"], size="sm", elem_classes=["sample-pill", "quick-sample-pill"], scale=1)

                # Single Unified Document Textarea
                text_input = gr.Textbox(
                    lines=10,
                    max_lines=18,
                    interactive=True,
                    placeholder=d["input_placeholder"],
                    label=d["input_label"]
                )

                # Live Document Typing Statistics Ribbon
                stats_ribbon = gr.HTML(
                    '<div class="live-stats-ribbon">Words: 0 • Sentences: 0 • Est. Read Time: 0s • Morpheme Depth: 0.0x</div>',
                    elem_classes=["live-stats-wrapper"]
                )

                # File Upload Accordion (.txt, .docx, .pdf)
                with gr.Accordion(d["file_accordion"], open=False) as file_accordion:
                    file_uploader = gr.File(
                        label=d["upload_label"],
                        file_types=[".txt", ".docx", ".pdf"],
                    )
                    file_status = gr.Markdown("")

                # Feature Toggles
                with gr.Row():
                    autofst_toggle = gr.Checkbox(
                        label=d["autofst_toggle_label"],
                        value=True,
                        interactive=True
                    )
                with gr.Row():
                    verifier_toggle = gr.Checkbox(
                        label=d["verifier_toggle_label"],
                        value=True,
                        interactive=True
                    )

                # Primary Action Buttons
                with gr.Row():
                    analyze_btn = gr.Button(
                        d["analyze_btn"],
                        variant="primary",
                        elem_classes=["btn-primary-charcoal"],
                        scale=3
                    )
                    export_btn = gr.Button(
                        "Export Audit (.md)",
                        variant="secondary",
                        elem_classes=["btn-secondary-outline"],
                        scale=2
                    )
                    clear_btn = gr.Button(
                        d["clear_btn"],
                        variant="secondary",
                        elem_classes=["btn-secondary-outline"],
                        scale=1
                    )

                audit_file_download = gr.File(label="Download Audit Report", visible=False)

                # Visual Sentence Heatmap (Relocated to Left Studio for balanced vertical layout)
                heatmap_header = gr.Markdown(d["heatmap_header"])
                legend_display = gr.HTML(render_legend_html("en"))
                heatmap_display = gr.HTML(f'<div class="empty-doc-prompt">{d["empty_heatmap"]}</div>')

            # ===============================================================
            # RIGHT COLUMN: Diagnostic Inspector (scale=5)
            # ===============================================================
            with gr.Column(scale=5, elem_classes=["workspace-right-col"]):
                # Top Persistent HUD
                kpi_header = gr.Markdown(d["kpi_header"])
                executive_card_display = gr.HTML(
                    render_executive_summary_card(
                        probability=0.0,
                        verdict=d["awaiting_input"],
                        lang="en",
                        total_words=0,
                        total_sents=0,
                        total_windows=0,
                        ai_ratio=0.0
                    )
                )
                # Hidden standalone confidence meter to avoid duplicate visual progress bar
                confidence_meter_display = gr.HTML(render_confidence_meter(0.0, lang="en"), visible=False)

                # 3 Consolidated Sub-Tabs
                with gr.Tabs(elem_classes=["workspace-tabs"]) as result_tabs:

                    # Sub-Tab 1: Sentence Morphology & Fusion
                    with gr.Tab(d["subtab_morphology_title"], id="tab_morphology") as tab_morphology:
                        diagnostic_header = gr.Markdown(d["diagnostic_header"])
                        sentence_selector = gr.Dropdown(
                            choices=[],
                            value=None,
                            label=d["sentence_select_label"]
                        )
                        with gr.Row():
                            with gr.Column(scale=5):
                                selected_sentence_box = gr.Textbox(
                                    label=d["inspected_sentence_label"],
                                    value=d["select_sentence_prompt"],
                                    lines=3,
                                    interactive=False
                                )
                                gate_header = gr.Markdown(d["gate_header"])
                                gate_bar_display = gr.HTML(render_dynamic_gate_bar(0.5, lang="en"))
                            with gr.Column(scale=6):
                                morpheme_header = gr.Markdown(d["morpheme_header"])
                                morpheme_table_display = gr.HTML(f'<p class="text-muted">{d["select_sentence_prompt"]}</p>')

                        linguistic_header = gr.Markdown(d["linguistic_header"])
                        linguistic_bullets_display = gr.Markdown("")

                    # Sub-Tab 2: Auto-FST & Slang Lab
                    with gr.Tab(d["subtab_autofst_title"], id="tab_autofst") as tab_autofst:
                        autofst_hero_header = gr.Markdown(d["autofst_hero_header"])
                        autofst_comparison_card = gr.HTML(render_autofst_comparison_card("", "", 0.0, lang="en"))
                        autofst_inductions_header = gr.Markdown(d["autofst_inductions_header"])
                        autofst_inductions_table = gr.HTML(render_autofst_inductions_table([], lang="en"))

                    # Sub-Tab 3: Fact Verification & Trust Matrix
                    with gr.Tab(d["subtab_verification_title"], id="tab_verification") as tab_verification:
                        verify_hero_header = gr.Markdown(d["verify_hero_header"])
                        verify_model_radio = gr.Radio(
                            choices=[d["verify_model_ours"], d["verify_model_base"]],
                            value=d["verify_model_ours"],
                            label=d["verify_model_label"]
                        )
                        verify_card_display = gr.HTML(render_trust_summary_card(None, lang="en"))
                        verify_claims_header = gr.Markdown(d["verify_claims_header"])
                        verify_table_display = gr.HTML(render_claims_verification_table([], lang="en"))

        # Hidden legacy components to retain exact 63-element backward compatibility with switch_ui_language
        with gr.Row(visible=False):
            fst_lab_header = gr.Markdown(d["fst_lab_header"], visible=False)
            fst_lab_desc = gr.Markdown(d["fst_lab_desc"], visible=False)
            fst_input = gr.Textbox(lines=1, interactive=False, visible=False)
            fst_parse_btn = gr.Button(d["fst_parse_btn"], visible=False)
            fst_gate_header = gr.Markdown(d["fst_gate_header"], visible=False)
            fst_gate_display = gr.HTML("", visible=False)
            fst_table_header = gr.Markdown(d["fst_table_header"], visible=False)
            fst_table_display = gr.HTML("", visible=False)
            verify_hero_desc = gr.Markdown(d["verify_hero_desc"], visible=False)
            verify_quick_samples_label = gr.Markdown(d["verify_quick_samples_title"], visible=False)
            v_pill_btn1 = gr.Button(d["verify_pill_1"], visible=False)
            v_pill_btn2 = gr.Button(d["verify_pill_2"], visible=False)
            v_pill_btn3 = gr.Button(d["verify_pill_3"], visible=False)
            v_pill_btn4 = gr.Button(d["verify_pill_4"], visible=False)
            verify_input = gr.Textbox(lines=1, interactive=False, visible=False)
            verify_btn = gr.Button(d["verify_btn"], visible=False)
            verify_clear_btn = gr.Button(d["verify_clear_btn"], visible=False)
            autofst_hero_desc = gr.Markdown(d["autofst_hero_desc"], visible=False)
            autofst_quick_samples_label = gr.Markdown(d["autofst_quick_samples_title"], visible=False)
            af_pill_btn1 = gr.Button(d["autofst_pill_1"], visible=False)
            af_pill_btn2 = gr.Button(d["autofst_pill_2"], visible=False)
            af_pill_btn3 = gr.Button(d["autofst_pill_3"], visible=False)
            af_pill_btn4 = gr.Button(d["autofst_pill_4"], visible=False)
            af_pill_btn5 = gr.Button(d["autofst_pill_5"], visible=False)
            autofst_input = gr.Textbox(lines=1, interactive=False, visible=False)
            autofst_btn = gr.Button(d["autofst_btn"], visible=False)
            autofst_clear_btn = gr.Button(d["autofst_clear_btn"], visible=False)

        # -------------------------------------------------------------------
        # Event Handlers & Wirings
        # -------------------------------------------------------------------

        # Unified Multi-Pipeline Analysis Trigger
        def _run_full_analysis(text_val, autofst_on, verifier_on, current_lang):
            res = run_full_analysis_logic(
                text=text_val,
                autofst_enabled=autofst_on,
                verifier_enabled=verifier_on,
                current_lang=current_lang,
                detector=app_detector,
                verifier=app_verifier
            )
            return (
                res["executive_card"],
                res["confidence_meter"],
                res["bullets"],
                res["heatmap"],
                gr.update(choices=res["sentence_choices"], value=res["sentence_choice"]),
                res["selected_sentence"],
                res["gate_html"],
                res["morpheme_html"],
                res["autofst_card"],
                res["autofst_table"],
                res["trust_card"],
                res["claims_table"],
                res["sentences_state"]
            )

        analyze_btn.click(
            fn=_run_full_analysis,
            inputs=[text_input, autofst_toggle, verifier_toggle, lang_radio],
            outputs=[
                executive_card_display,
                confidence_meter_display,
                linguistic_bullets_display,
                heatmap_display,
                sentence_selector,
                selected_sentence_box,
                gate_bar_display,
                morpheme_table_display,
                autofst_comparison_card,
                autofst_inductions_table,
                verify_card_display,
                verify_table_display,
                raw_sents_state
            ]
        )

        # Clear Trigger
        def _clear_all(current_lang):
            lang = _get_lang_key(current_lang)
            d_curr = I18N.get(lang, I18N["en"])
            empty_exec = render_executive_summary_card(
                probability=0.0,
                verdict=d_curr["awaiting_input"],
                lang=lang,
                total_words=0,
                total_sents=0,
                total_windows=0,
                ai_ratio=0.0
            )
            empty_meter = render_confidence_meter(0.0, d_curr["awaiting_input"], lang=lang)
            empty_heatmap = f'<div class="empty-doc-prompt">{d_curr["empty_heatmap"]}</div>'
            return (
                "",
                None,
                "",
                empty_exec,
                empty_meter,
                "",
                empty_heatmap,
                gr.update(choices=[], value=None),
                d_curr["select_sentence_prompt"],
                render_dynamic_gate_bar(0.5, lang=lang),
                f'<p class="text-muted">{d_curr["select_sentence_prompt"]}</p>',
                render_autofst_comparison_card("", "", 0.0, lang=lang),
                render_autofst_inductions_table([], lang=lang),
                render_trust_summary_card(None, lang=lang),
                render_claims_verification_table([], lang=lang),
                []
            )

        clear_btn.click(
            fn=_clear_all,
            inputs=[lang_radio],
            outputs=[
                text_input,
                file_uploader,
                file_status,
                executive_card_display,
                confidence_meter_display,
                linguistic_bullets_display,
                heatmap_display,
                sentence_selector,
                selected_sentence_box,
                gate_bar_display,
                morpheme_table_display,
                autofst_comparison_card,
                autofst_inductions_table,
                verify_card_display,
                verify_table_display,
                raw_sents_state
            ]
        )

        # Live Typing Statistics Ribbon Handler
        text_input.change(
            fn=calculate_document_statistics,
            inputs=[text_input],
            outputs=[stats_ribbon]
        )

        # Audit Report Export Handler
        def _export_audit(text_val, autofst_on, verifier_on, current_lang):
            import tempfile
            res = run_full_analysis_logic(
                text=text_val,
                autofst_enabled=autofst_on,
                verifier_enabled=verifier_on,
                current_lang=current_lang,
                detector=app_detector,
                verifier=app_verifier
            )
            md_text = generate_audit_markdown(res, source_text=text_val)
            out_path = os.path.join(tempfile.gettempdir(), "audit_report.md")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md_text)
            return gr.update(value=out_path, visible=True)

        export_btn.click(
            fn=_export_audit,
            inputs=[text_input, autofst_toggle, verifier_toggle, lang_radio],
            outputs=[audit_file_download]
        )

        # Quick Sample Pills Click Handlers
        pill_btn1.click(fn=lambda: QUICK_SAMPLES["pill_1"]["text"], outputs=[text_input])
        pill_btn2.click(fn=lambda: QUICK_SAMPLES["pill_2"]["text"], outputs=[text_input])
        pill_btn3.click(fn=lambda: QUICK_SAMPLES["pill_3"]["text"], outputs=[text_input])
        pill_btn4.click(fn=lambda: QUICK_SAMPLES["pill_4"]["text"], outputs=[text_input])
        pill_btn5.click(fn=lambda: QUICK_SAMPLES["pill_5"]["text"], outputs=[text_input])
        pill_btn6.click(fn=lambda: QUICK_SAMPLES["pill_6"]["text"], outputs=[text_input])

        # File Ingestion
        def _on_file_uploaded(f_obj, current_lang):
            return handle_file_upload(f_obj, lang_choice=current_lang)

        file_uploader.upload(
            fn=_on_file_uploaded,
            inputs=[file_uploader, lang_radio],
            outputs=[text_input, file_status]
        )

        # Sentence Inspector Selection
        def _on_sentence_selected(s_idx, s_state, current_lang):
            return handle_sentence_select(s_idx, s_state, lang_choice=current_lang)

        sentence_selector.change(
            fn=_on_sentence_selected,
            inputs=[sentence_selector, raw_sents_state, lang_radio],
            outputs=[
                selected_sentence_box,
                gate_bar_display,
                morpheme_table_display
            ]
        )

        # Tab 2: FST Lab Parse Trigger
        def _run_fst_parse(fst_text_val, current_lang):
            return handle_fst_parse(fst_text_val, detector=app_detector, lang_choice=current_lang)

        fst_parse_btn.click(
            fn=_run_fst_parse,
            inputs=[fst_input, lang_radio],
            outputs=[fst_gate_display, fst_table_display]
        )

        # Tab 4: Factual Verification & Trust Matrix Handlers
        def _run_verification(text_val, current_lang, model_choice):
            return handle_verify_document(
                text_val,
                verifier=app_verifier,
                lang_choice=current_lang,
                model_mode=model_choice
            )

        verify_btn.click(
            fn=_run_verification,
            inputs=[verify_input, lang_radio, verify_model_radio],
            outputs=[verify_card_display, verify_table_display]
        )

        def _clear_verification(current_lang):
            lang = _get_lang_key(current_lang)
            return (
                "",
                render_trust_summary_card(None, lang=lang),
                render_claims_verification_table([], lang=lang)
            )

        verify_clear_btn.click(
            fn=_clear_verification,
            inputs=[lang_radio],
            outputs=[verify_input, verify_card_display, verify_table_display]
        )

        v_pill_btn1.click(
            fn=lambda: get_verification_preset_text("Quadrant 1: Verified Human Fact"),
            outputs=[verify_input]
        )
        v_pill_btn2.click(
            fn=lambda: get_verification_preset_text("Quadrant 2: Human Misinformation"),
            outputs=[verify_input]
        )
        v_pill_btn3.click(
            fn=lambda: get_verification_preset_text("Quadrant 3: Accurate AI Synthesis"),
            outputs=[verify_input]
        )
        v_pill_btn4.click(
            fn=lambda: get_verification_preset_text("Quadrant 4: Hallucinatory AI Disinformation"),
            outputs=[verify_input]
        )

        # Tab 5: Auto-FST Handlers
        global_autofst_overlay = DynamicFSTOverlay(normalizer=RussianLayoutNormalizer())
        global_autofst_inducer = AutoFSTInducer()
        global_autofst_guard = RegressionGuard()

        def _run_autofst(text_val, current_lang):
            if not text_val or not text_val.strip():
                lang = _get_lang_key(current_lang)
                return (
                    render_autofst_comparison_card("", "", 0.0, lang=lang),
                    render_autofst_inductions_table([], lang=lang)
                )

            import time
            from fst_analyzer import fst_analyzer as b_fst
            lang = _get_lang_key(current_lang)

            # Baseline parse
            t0 = time.perf_counter()
            before_parse = b_fst.analyze_and_segment(text_val)

            # Check and induce roots
            inductions = []
            seen_stems = set()
            for w in text_val.split():
                w_clean = re.sub(r'^[^\w]+|[^\w]+$', '', w)
                if not w_clean:
                    continue
                w_norm = global_autofst_overlay.normalizer.normalize_word(w_clean)
                cand = global_autofst_inducer.induce_stem(w_norm)
                if cand and cand.stem not in seen_stems:
                    seen_stems.add(cand.stem)
                    passed, reason = global_autofst_guard.validate_candidate(cand.stem)
                    if passed:
                        global_autofst_overlay.register_root(cand.stem, {"harmony": cand.harmony, "coda": cand.coda})
                        inductions.append({
                            "stem": cand.stem,
                            "harmony": cand.harmony,
                            "coda": cand.coda,
                            "suffixes": cand.suffixes,
                            "guard": "Passed (100% agreement)"
                        })

            # Auto-FST parse
            after_parse = global_autofst_overlay.analyze_and_segment(text_val)
            latency_ms = (time.perf_counter() - t0) * 1000

            return (
                render_autofst_comparison_card(before_parse, after_parse, latency_ms, lang=lang),
                render_autofst_inductions_table(inductions, lang=lang)
            )

        autofst_btn.click(
            fn=_run_autofst,
            inputs=[autofst_input, lang_radio],
            outputs=[autofst_comparison_card, autofst_inductions_table]
        )

        def _clear_autofst(current_lang):
            lang = _get_lang_key(current_lang)
            return (
                "",
                render_autofst_comparison_card("", "", 0.0, lang=lang),
                render_autofst_inductions_table([], lang=lang)
            )

        autofst_clear_btn.click(
            fn=_clear_autofst,
            inputs=[lang_radio],
            outputs=[autofst_input, autofst_comparison_card, autofst_inductions_table]
        )

        af_pill_btn1.click(
            fn=lambda: "салем калайсын, жумысын калай болып жатыр?",
            outputs=[autofst_input]
        )
        af_pill_btn2.click(
            fn=lambda: "Ол али жумыста турсынгой, кешке келедигой.",
            outputs=[autofst_input]
        )
        af_pill_btn3.click(
            fn=lambda: "Биз кеше достармен катты вайбтадык жане чиллдедик.",
            outputs=[autofst_input]
        )
        af_pill_btn4.click(
            fn=lambda: "Ойынга коп донаттау жасады, енди озинин крашына жазады.",
            outputs=[autofst_input]
        )
        af_pill_btn5.click(
            fn=lambda: "Чатботтын жауабын каспиредпен толедик.",
            outputs=[autofst_input]
        )

        # Dynamic Language Switching Event
        lang_radio.change(
            fn=switch_ui_language,
            inputs=[lang_radio],
            outputs=[
                quick_samples_label,
                pill_btn1,
                pill_btn2,
                pill_btn3,
                pill_btn4,
                pill_btn5,
                pill_btn6,
                text_input,
                file_accordion,
                file_uploader,
                analyze_btn,
                clear_btn,
                kpi_header,
                executive_card_display,
                linguistic_header,
                linguistic_bullets_display,
                heatmap_header,
                legend_display,
                heatmap_display,
                diagnostic_header,
                sentence_selector,
                selected_sentence_box,
                gate_header,
                gate_bar_display,
                morpheme_header,
                morpheme_table_display,
                fst_lab_header,
                fst_lab_desc,
                fst_input,
                fst_parse_btn,
                fst_gate_header,
                fst_gate_display,
                fst_table_header,
                fst_table_display,
                methodology_md,
                verify_hero_header,
                verify_hero_desc,
                verify_quick_samples_label,
                v_pill_btn1,
                v_pill_btn2,
                v_pill_btn3,
                v_pill_btn4,
                verify_model_radio,
                verify_input,
                verify_btn,
                verify_clear_btn,
                verify_card_display,
                verify_claims_header,
                verify_table_display,
                # Tab 5: Auto-FST
                autofst_hero_header,
                autofst_hero_desc,
                autofst_quick_samples_label,
                af_pill_btn1,
                af_pill_btn2,
                af_pill_btn3,
                af_pill_btn4,
                af_pill_btn5,
                autofst_input,
                autofst_btn,
                autofst_clear_btn,
                autofst_comparison_card,
                autofst_inductions_header,
                autofst_inductions_table
            ]
        )

    return demo


def build_ui(detector: Any = None, verifier: Any = None, load_model: bool = False) -> gr.Blocks:
    """
    Constructs and returns the full Gradio Blocks application. Alias of create_app.
    """
    return create_app(detector=detector, verifier=verifier, load_model=load_model)



def main():
    parser = argparse.ArgumentParser(description="Launch Kazakh AI-Text Detector & Explainability UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to run server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--load_model", action="store_true", help="Attempt to load PyTorch model weights")
    args = parser.parse_args()

    demo = create_app(load_model=args.load_model)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
