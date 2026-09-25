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


# Extended CSS styling for academic publication-grade dashboard aesthetics
APP_CSS = HEATMAP_CSS + """
body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    color: #0f172a;
}
.hero-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 18px;
}
.hero-title {
    font-size: 22px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 6px;
    letter-spacing: -0.01em;
}
.hero-subtitle {
    font-size: 14px;
    color: #475569;
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
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
    color: #334155;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
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
    color: #475569 !important;
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
.sample-pill {
    font-size: 12px !important;
    padding: 4px 10px !important;
    border-radius: 9999px !important;
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    color: #1e293b !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
.sample-pill:hover {
    background: #e2e8f0 !important;
    border-color: #94a3b8 !important;
    color: #0f172a !important;
}
.kpi-container {
    display: flex;
    gap: 10px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}
.kpi-card {
    flex: 1;
    min-width: 130px;
    padding: 12px 14px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    text-align: center;
}
.kpi-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: #64748b;
    font-weight: 600;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 22px;
    font-weight: 700;
    color: #0f172a;
}
.kpi-badge {
    padding: 10px 16px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 14px;
    text-align: center;
    margin-bottom: 10px;
    letter-spacing: 0.04em;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}
.badge-neutral {
    background-color: #f1f5f9;
    color: #475569;
    border: 1px solid #cbd5e1;
}
.badge-human {
    background-color: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}
.badge-hybrid {
    background-color: #fffbeb;
    color: #92400e;
    border: 1px solid #fde68a;
}
.badge-ai {
    background-color: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
}
.legend-bar {
    display: flex;
    gap: 10px;
    font-size: 12px;
    color: #334155;
    margin-top: 6px;
    margin-bottom: 12px;
    flex-wrap: wrap;
    background: #f8fafc;
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
}
.legend-item {
    display: inline-flex;
    align-items: center;
}
.legend-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
}
.legend-pill-human {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
}
.legend-pill-amber {
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #92400e;
}
.legend-pill-ai {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
}
.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}
.dot-human { background-color: #10b981; }
.dot-amber { background-color: #f59e0b; }
.dot-ai { background-color: #ef4444; }
.linguistic-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.panel-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
}
"""

# Quick Samples Definition (6 curated benchmarks)
QUICK_SAMPLES: Dict[str, Dict[str, str]] = {
    "pill_1": {
        "title_kz": "Тұтынушы пікірі - Адам",
        "title_en": "Human Consumer Review",
        "expected_verdict": "Authentic Human",
        "text": get_preset_text("1. Authentic Kaspi Consumer Review (Human)"),
    },
    "pill_2": {
        "title_kz": "Жаңалықтар мәтіні - Адам",
        "title_en": "Human Informative News",
        "expected_verdict": "Authentic Human",
        "text": get_preset_text("2. Authentic Formal News Article (Human)"),
    },
    "pill_3": {
        "title_kz": "Sherkala-7B пікірі - ЖИ",
        "title_en": "Sherkala-7B AI Review",
        "expected_verdict": "Machine-Generated",
        "text": (
            "Kaspi дүкенінен сатып алынған шаңсорғыш өте сапалы болып табылады. "
            "Қорытындылай келе, тауардың сипаттамасы жоғары дәрежеде сәйкес келеді және маңызды рөл атқарады. "
            "Айта кету керек, жеткізу қызметі жүйелі түрде орындалды, барлық функциялары талапқа сай жұмыс істейді."
        ),
    },
    "pill_4": {
        "title_kz": "Sherkala-7B жаңалықтары - ЖИ",
        "title_en": "Sherkala-7B AI News",
        "expected_verdict": "Machine-Generated",
        "text": get_preset_text("4. Sherkala-7B Generated Article (AI)"),
    },
    "pill_5": {
        "title_kz": "Qwen-2.5 еркін ЖИ",
        "title_en": "Qwen-2.5 Wild AI",
        "expected_verdict": "Machine-Generated",
        "text": get_preset_text("5. Qwen-2.5-7B-Instruct Wild Sample (AI)"),
    },
    "pill_6": {
        "title_kz": "Аралас / Гибрид құжат",
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
        "pill_1": "Тұтынушы пікірі - Адам",
        "pill_2": "Жаңалықтар мәтіні - Адам",
        "pill_3": "Sherkala-7B пікірі - ЖИ",
        "pill_4": "Sherkala-7B жаңалықтары - ЖИ",
        "pill_5": "Qwen-2.5 еркін ЖИ",
        "pill_6": "Аралас / Гибрид құжат",
        "tab_1_title": "Detection & Explainability / Мәтінді тексеру",
        "tab_2_title": "Morphological FST Lab / Морфологиялық зертхана",
        "tab_3_title": "Benchmark & Methodology / Ғылыми әдістеме",
        "presets_accordion": "Дайын Бенчмарк Үлгілері Кітапханасы (6 Пресет)",
        "presets_help": "Төмендегі дайын үлгілердің бірін таңдап, жүйенің әртүрлі жанрларда қалай жұмыс істейтінін көріңіз:",
        "preset_select_label": "Үлгіні таңдаңыз (Benchmark Preset)",
        "load_preset_btn": "Үлгіні Жүктеу",
        "file_accordion": "Құжатты файл арқылы жүктеу (.txt, .docx, .pdf)",
        "input_label": "Мәтін енгізу (Document Input)",
        "input_placeholder": "Қазақша мәтінді осында жазыңыз немесе көшіріп қойыңыз (25 000 сөзге дейін)...",
        "upload_label": "Құжатты жүктеу (.txt, .docx, .pdf — макс. 10 MB)",
        "analyze_btn": "Құжатты Талдау (Analyze Document)",
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
    },
    "en": {
        "hero_title": "Kazakh AI-Generated Text Detector & Explainability Dashboard",
        "hero_subtitle": "Production-grade, explainable detection of AI-generated Kazakh text with sentence-level heatmaps, dynamic dual-stream gate fusion, and deep morphological FST breakdowns.",
        "badge_tri_domain": "[Tri-Domain Robustness: Reviews • News • Wikipedia]",
        "badge_dual_stream": "[Dual-Stream Gate: BERT + FST]",
        "badge_sliding_window": "[Sliding Window: Up to 25k words]",
        "badge_xss_safe": "[XSS-Safe Interactive Heatmap]",
        "quick_samples_title": "Quick Benchmark Samples:",
        "pill_1": "Human Consumer Review",
        "pill_2": "Human Informative News",
        "pill_3": "Sherkala-7B AI Review",
        "pill_4": "Sherkala-7B AI News",
        "pill_5": "Qwen-2.5 Wild AI",
        "pill_6": "Hybrid Document",
        "tab_1_title": "Detection & Explainability / Мәтінді тексеру",
        "tab_2_title": "Morphological FST Lab / Морфологиялық зертхана",
        "tab_3_title": "Benchmark & Methodology / Ғылыми әдістеме",
        "presets_accordion": "Curated Benchmark Demonstration Samples (6 Presets)",
        "presets_help": "Select one of the benchmark scenarios below to test detector performance across different genres and generators:",
        "preset_select_label": "Select Benchmark Sample",
        "load_preset_btn": "Load Preset",
        "file_accordion": "Upload Document File (.txt, .docx, .pdf)",
        "input_label": "Document Input",
        "input_placeholder": "Paste or type Kazakh text here (up to 25,000 words)...",
        "upload_label": "Upload Document (.txt, .docx, .pdf — max 10 MB)",
        "analyze_btn": "Analyze Document",
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
        empty_claims_table
    )



def create_app(detector: Any = None, verifier: Any = None, load_model: bool = False) -> gr.Blocks:
    """
    Constructs and returns the full Gradio Blocks application with 4 distinct academic tabs
    and bilingual localization (zero emojis).
    """
    app_detector = detector or get_detector(load_model=load_model)
    app_verifier = verifier or TrustworthyDocumentVerifier(ai_detector=app_detector)
    d = I18N["en"]


    with gr.Blocks(title="Kazakh AI-Text Detector & Explainability UI") as demo:
        # State: stores analyzed sentences list for interactive drilldown
        raw_sents_state = gr.State([])

        # Inject CSS styling safely
        gr.HTML(f"<style>{APP_CSS}</style>", visible=False)

        # -------------------------------------------------------------------
        # 3 Main Academic Tabs (Start directly at the top)
        # -------------------------------------------------------------------
        with gr.Tabs() as tabs:

            # ===============================================================
            # TAB 1: Detection & Explainability
            # ===============================================================
            with gr.Tab(d["tab_1_title"], id="tab_detection") as tab_detect:

                # Header row: Quick Samples title on left, inline language switcher immediately adjacent on left
                with gr.Column(elem_classes=["quick-samples-wrapper"]):
                    with gr.Row(elem_classes=["quick-samples-header-row"]):
                        quick_samples_label = gr.Markdown(f"**{d['quick_samples_title']}**", elem_classes=["quick-samples-title"])
                        lang_radio = gr.Radio(
                            choices=["English", "Қазақша"],
                            value="English",
                            show_label=False,
                            container=False,
                            interactive=True,
                            elem_classes=["lang-switch-inline"]
                        )
                    with gr.Row(elem_classes=["quick-samples-row"]):
                        pill_btn1 = gr.Button(d["pill_1"], size="sm", elem_classes=["sample-pill"], scale=1)
                        pill_btn2 = gr.Button(d["pill_2"], size="sm", elem_classes=["sample-pill"], scale=1)
                        pill_btn3 = gr.Button(d["pill_3"], size="sm", elem_classes=["sample-pill"], scale=1)
                        pill_btn4 = gr.Button(d["pill_4"], size="sm", elem_classes=["sample-pill"], scale=1)
                        pill_btn5 = gr.Button(d["pill_5"], size="sm", elem_classes=["sample-pill"], scale=1)
                        pill_btn6 = gr.Button(d["pill_6"], size="sm", elem_classes=["sample-pill"], scale=1)

                # Main Input & KPI Section
                with gr.Row():
                    # Left Column: Document Input & Controls
                    with gr.Column(scale=5):
                        text_input = gr.Textbox(
                            lines=8,
                            max_lines=22,
                            placeholder=d["input_placeholder"],
                            label=d["input_label"]
                        )

                        with gr.Row():
                            analyze_btn = gr.Button(d["analyze_btn"], variant="primary", scale=3)
                            clear_btn = gr.Button(d["clear_btn"], variant="secondary", scale=1)

                        # File Upload Accordion (.txt, .docx, .pdf)
                        with gr.Accordion(d["file_accordion"], open=False) as file_accordion:
                            file_uploader = gr.File(
                                label=d["upload_label"],
                                file_types=[".txt", ".docx", ".pdf"],
                            )
                            file_status = gr.Markdown("")

                    # Right Column: Document Results & Explainability
                    with gr.Column(scale=5):
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

                        # Dynamic Linguistic Reasoning Bullets
                        linguistic_header = gr.Markdown(d["linguistic_header"])
                        linguistic_bullets_display = gr.Markdown("")

                # Visual Sentence Heatmap Section
                gr.Markdown("---")
                heatmap_header = gr.Markdown(d["heatmap_header"])
                legend_display = gr.HTML(render_legend_html("en"))
                heatmap_display = gr.HTML(f'<div class="empty-doc-prompt">{d["empty_heatmap"]}</div>')

                # Sentence Deep Diagnostic & FST Breakdown Section
                gr.Markdown("---")
                diagnostic_header = gr.Markdown(d["diagnostic_header"])

                with gr.Row():
                    sentence_selector = gr.Dropdown(
                        choices=[],
                        value=None,
                        label=d["sentence_select_label"],
                        scale=3
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

            # ===============================================================
            # TAB 2: Morphological FST Lab
            # ===============================================================
            with gr.Tab(d["tab_2_title"], id="tab_fst") as tab_fst:
                fst_lab_header = gr.Markdown(d["fst_lab_header"])
                fst_lab_desc = gr.Markdown(d["fst_lab_desc"])

                with gr.Row():
                    fst_input = gr.Textbox(
                        lines=3,
                        max_lines=6,
                        placeholder=d["fst_input_placeholder"],
                        label=d["fst_input_label"],
                        scale=4
                    )
                    fst_parse_btn = gr.Button(d["fst_parse_btn"], variant="primary", scale=1)

                fst_gate_header = gr.Markdown(d["fst_gate_header"])
                fst_gate_display = gr.HTML(render_dynamic_gate_bar(0.5, lang="en"))

                fst_table_header = gr.Markdown(d["fst_table_header"])
                fst_table_display = gr.HTML(render_fst_decomposition_table([], lang="en"))

            # ===============================================================
            # TAB 3: Benchmark & Academic Methodology
            # ===============================================================
            with gr.Tab(d["tab_3_title"], id="tab_methodology") as tab_methodology:
                methodology_md = gr.Markdown(render_methodology_markdown("en"))

            # ===============================================================
            # TAB 4: Factual Verification & Trust Matrix
            # ===============================================================
            with gr.Tab(d["tab_4_title"], id="tab_verification") as tab_verify:
                verify_hero_header = gr.Markdown(d["verify_hero_header"])
                verify_hero_desc = gr.Markdown(d["verify_hero_desc"])

                # 4-Quadrant Quick Presets
                with gr.Column(elem_classes=["quick-samples-wrapper"]):
                    verify_quick_samples_label = gr.Markdown(
                        f"**{d['verify_quick_samples_title']}**",
                        elem_classes=["quick-samples-title"]
                    )
                    with gr.Row(elem_classes=["quick-samples-row"]):
                        v_pill_btn1 = gr.Button(d["verify_pill_1"], size="sm", elem_classes=["sample-pill"], scale=1)
                        v_pill_btn2 = gr.Button(d["verify_pill_2"], size="sm", elem_classes=["sample-pill"], scale=1)
                        v_pill_btn3 = gr.Button(d["verify_pill_3"], size="sm", elem_classes=["sample-pill"], scale=1)
                        v_pill_btn4 = gr.Button(d["verify_pill_4"], size="sm", elem_classes=["sample-pill"], scale=1)

                verify_model_radio = gr.Radio(
                    choices=[d["verify_model_ours"], d["verify_model_base"]],
                    value=d["verify_model_ours"],
                    label=d["verify_model_label"]
                )

                with gr.Row():
                    with gr.Column(scale=6):
                        verify_input = gr.Textbox(
                            lines=6,
                            max_lines=18,
                            placeholder=d["verify_input_placeholder"],
                            label=d["verify_input_label"]
                        )
                        with gr.Row():
                            verify_btn = gr.Button(d["verify_btn"], variant="primary", scale=3)
                            verify_clear_btn = gr.Button(d["verify_clear_btn"], variant="secondary", scale=1)

                    with gr.Column(scale=6):
                        verify_card_display = gr.HTML(render_trust_summary_card(None, lang="en"))

                gr.Markdown("---")
                verify_claims_header = gr.Markdown(d["verify_claims_header"])
                verify_table_display = gr.HTML(render_claims_verification_table([], lang="en"))


        # -------------------------------------------------------------------
        # Event Handlers & Wirings
        # -------------------------------------------------------------------

        # Analysis Trigger
        def _run_analysis(text_val, current_lang):
            return handle_analyze_document(text_val, detector=app_detector, lang_choice=current_lang)

        analyze_btn.click(
            fn=_run_analysis,
            inputs=[text_input, lang_radio],
            outputs=[
                executive_card_display,
                linguistic_bullets_display,
                heatmap_display,
                sentence_selector,
                selected_sentence_box,
                gate_bar_display,
                morpheme_table_display,
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
            empty_heatmap = f'<div class="empty-doc-prompt">{d_curr["empty_heatmap"]}</div>'
            return (
                "",
                None,
                "",
                empty_exec,
                "",
                empty_heatmap,
                gr.update(choices=[], value=None),
                d_curr["select_sentence_prompt"],
                render_dynamic_gate_bar(0.5, lang=lang),
                f'<p class="text-muted">{d_curr["select_sentence_prompt"]}</p>',
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
                linguistic_bullets_display,
                heatmap_display,
                sentence_selector,
                selected_sentence_box,
                gate_bar_display,
                morpheme_table_display,
                raw_sents_state
            ]
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
                verify_table_display
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
