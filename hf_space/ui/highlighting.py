"""
ui/highlighting.py: XSS-Safe Sentence Heatmap & HTML/CSS Highlighting Engine.

Renders sentence-level heatmaps, dynamic gate split-bars, and FST morpheme
breakdown tables for the interactive Kazakh AI text detection UI.
All user text is strictly sanitized using html.escape(..., quote=True)
to guarantee protection against Stored and Reflected XSS attacks.
"""

from typing import List, Dict, Any, Optional
import html
import math
import re



HEATMAP_CSS = """
<style>
/* Kaz-Explainability UI Highlighting Stylesheet */
.kaz-heatmap-container {
    line-height: 1.85;
    font-size: 1.05rem;
    padding: 18px;
    border-radius: 8px;
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    max-height: 520px;
    overflow-y: auto;
    word-wrap: break-word;
    white-space: pre-wrap;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #111827;
}

.kaz-sentence {
    display: inline;
    padding: 2px 5px;
    margin: 0 2px;
    border-radius: 4px;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    transition: background-color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

.kaz-sentence:hover {
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    filter: brightness(0.96);
}

/* Three-tier classification level colors */
.lvl-human {
    background-color: #d1fae5;
    border-color: #10b981;
    color: #065f46;
}

.lvl-amber {
    background-color: #fef3c7;
    border-color: #f59e0b;
    color: #92400e;
}

.lvl-ai {
    background-color: #fee2e2;
    border-color: #ef4444;
    color: #991b1b;
}

.empty-doc-prompt {
    color: #6b7280;
    font-style: italic;
    padding: 20px;
    text-align: center;
    font-size: 0.95rem;
}

/* Dynamic Gate Split-Bar Styles */
.gate-bar-wrapper {
    margin: 12px 0 16px 0;
    padding: 14px 16px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.gate-bar-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
}

.gate-header-col {
    display: flex;
    flex-direction: column;
}

.gate-header-bert {
    text-align: left;
}

.gate-header-fst {
    text-align: right;
}

.gate-title-text {
    font-size: 0.85rem;
    font-weight: 700;
    color: #0f172a;
}

.gate-subtitle {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 1px;
}

.gate-bar-container {
    display: flex;
    width: 100%;
    height: 24px;
    border-radius: 6px;
    overflow: hidden;
    background-color: #e2e8f0;
    box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.08);
}

.gate-stream-bert {
    background: linear-gradient(90deg, #3b82f6, #2563eb);
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    transition: width 0.3s ease;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: clip;
}

.gate-stream-fst {
    background: linear-gradient(90deg, #8b5cf6, #7c3aed);
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    transition: width 0.3s ease;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: clip;
}

.gate-bar-legend {
    margin-top: 8px;
    font-size: 0.78rem;
    color: #475569;
    line-height: 1.45;
}

.gate-explanation {
    display: block;
}

/* Horizontal Confidence Meter Styles */
.confidence-meter-container {
    margin: 12px 0 16px 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

.confidence-meter-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.875rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 6px;
}

.confidence-label {
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.8rem;
    color: #4b5563;
}

.confidence-value {
    font-weight: 700;
    font-size: 0.95rem;
    color: #111827;
}

.confidence-verdict-tag {
    margin-left: 8px;
    font-size: 0.75rem;
    padding: 2px 7px;
    border-radius: 4px;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    font-weight: 600;
}

.confidence-track {
    width: 100%;
    height: 22px;
    border-radius: 6px;
    background-color: #f3f4f6;
    border: 1px solid #e5e7eb;
    overflow: hidden;
    position: relative;
    box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.06);
}

.confidence-fill {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 700;
    transition: width 0.3s ease;
    text-shadow: 0 1px 1px rgba(0, 0, 0, 0.2);
}

.meter-human {
    background: linear-gradient(90deg, #10b981, #059669);
}

.meter-amber {
    background: linear-gradient(90deg, #f59e0b, #d97706);
}

.meter-ai {
    background: linear-gradient(90deg, #ef4444, #dc2626);
}

/* Executive Summary Card Styles */
.executive-summary-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.executive-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    gap: 12px;
    flex-wrap: wrap;
}

.executive-card-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #475569;
}

.executive-kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 12px;
}

.executive-kpi-tile {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px 12px;
    text-align: center;
}

.executive-kpi-title {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b;
    margin-bottom: 4px;
}

.executive-kpi-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
}

.executive-kpi-sub {
    font-size: 0.7rem;
    font-weight: 500;
    color: #64748b;
    margin-top: 2px;
}

/* FST Morpheme Breakdown Table Styles */
.fst-table-container {
    overflow-x: auto;
    margin: 12px 0;
    border-radius: 6px;
    border: 1px solid #e5e7eb;
    background-color: #ffffff;
}

.fst-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    text-align: left;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.fst-table th {
    background-color: #f9fafb;
    padding: 10px 14px;
    font-weight: 600;
    color: #374151;
    border-bottom: 2px solid #e5e7eb;
}

.fst-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #f3f4f6;
    color: #1f2937;
    vertical-align: middle;
}

.fst-table tr:nth-child(even) {
    background-color: #fcfcfd;
}

.fst-table tr:hover {
    background-color: #f9fafb;
}

.morpheme-badge {
    display: inline-block;
    padding: 2px 7px;
    margin: 1px 3px 1px 0;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    background-color: #ede9fe;
    color: #6d28d9;
    border: 1px solid #ddd6fe;
}

.badge-case {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    background-color: #fef3c7;
    color: #92400e;
    border: 1px solid #fde68a;
}

.badge-plur {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    background-color: #e0f2fe;
    color: #0369a1;
    border: 1px solid #bae6fd;
}

.badge-tense {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    background-color: #fce7f3;
    color: #9d174d;
    border: 1px solid #fbcfe8;
}

.badge-person {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    background-color: #d1fae5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}

.pos-badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    background-color: #e0f2fe;
    color: #0369a1;
    border: 1px solid #bae6fd;
}

.text-muted {
    color: #6b7280;
    font-style: italic;
}

/* Factual Verification & Trust Matrix Styles */
.trust-card-wrapper {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.trust-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.lvl-verified-human {
    background-color: #d1fae5;
    color: #065f46;
    border: 1px solid #10b981;
}

.lvl-human-misinfo {
    background-color: #fef3c7;
    color: #92400e;
    border: 1px solid #f59e0b;
}

.lvl-ai-synthesis {
    background-color: #dbeafe;
    color: #1e40af;
    border: 1px solid #3b82f6;
}

.lvl-ai-disinfo {
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #ef4444;
}

.trust-bars-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 14px 0;
}

.trust-bar-item {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px 12px;
}

.trust-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    font-weight: 600;
    color: #475569;
    margin-bottom: 6px;
}

.trust-bar-track {
    width: 100%;
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
}

.trust-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}

.fill-ai { background: #ef4444; }
.fill-fact { background: #f59e0b; }
.fill-trust { background: #6366f1; }

.trust-stats-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    font-size: 0.8rem;
    color: #334155;
    padding-top: 8px;
    border-top: 1px solid #f1f5f9;
}

.trust-stat-pill {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
}

.stat-pill-total { background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; }
.stat-pill-sup { background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
.stat-pill-ref { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
.stat-pill-nei { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }

/* Claims Verification Table Styles */
.claims-table-container {
    width: 100%;
    overflow-x: auto;
    margin-top: 12px;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background: #ffffff;
}

.claims-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    text-align: left;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.claims-table th {
    background-color: #f8fafc;
    color: #334155;
    font-weight: 600;
    padding: 10px 14px;
    border-bottom: 2px solid #e2e8f0;
}

.claims-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #f1f5f9;
    color: #1e293b;
    vertical-align: top;
}

.claims-table tr:hover {
    background-color: #f8fafc;
}

.verdict-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}

.verdict-supported {
    background-color: #d1fae5;
    color: #065f46;
    border: 1px solid #10b981;
}

.verdict-refuted {
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #ef4444;
}

.verdict-nei {
    background-color: #fef3c7;
    color: #92400e;
    border: 1px solid #f59e0b;
}

/* 2D Continuous Cartesian Trust Plane Styles */
.trust-matrix-plane-container {
    margin: 12px 0 16px 0;
    padding: 14px 16px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}

.trust-hud-bar {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 12px;
}

.hud-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-family: monospace;
    color: #1e293b;
}

.hud-quad-q1 { background: #ecfdf5; border-color: #10b981; color: #065f46; font-weight: 700; }
.hud-quad-q2 { background: #fff1f2; border-color: #f43f5e; color: #9f1239; font-weight: 700; }
.hud-quad-q3 { background: #f0f9ff; border-color: #0ea5e9; color: #0369a1; font-weight: 700; }
.hud-quad-q4 { background: #fffbeb; border-color: #f59e0b; color: #92400e; font-weight: 700; }

.trust-meter-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin: 12px 0;
}

.trust-meter-tile {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 10px;
    text-align: center;
}

/* Morphological Alignment Drawer Styles */
.morpho-drawer-row:hover {
    background: #f8fafc;
}

.morpho-drawer {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 12px;
}

.morpho-drawer summary {
    cursor: pointer;
    font-weight: 600;
    color: #2563eb;
    font-size: 0.82rem;
    user-select: none;
    outline: none;
}

.morpho-drawer[open] summary {
    margin-bottom: 8px;
    color: #1d4ed8;
}

.morpho-evidence-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 8px;
    text-align: left;
}

.morpho-stem-highlight {
    background-color: #fef08a;
    color: #854d0e;
    padding: 1px 3px;
    border-radius: 3px;
    font-weight: 600;
}

.morpho-diag-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 8px;
    margin-top: 8px;
    text-align: left;
}

.morpho-diag-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 0.78rem;
}

.morpho-diag-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    margin-bottom: 4px;
    display: block;
}

.morpho-diag-val {
    font-size: 0.85rem;
    font-weight: 700;
    color: #0f172a;
}

.morpho-badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
}

.badge-ok { background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
.badge-warn { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
.badge-info { background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }

.mini-bar-track {
    width: 100%;
    height: 5px;
    background: #e2e8f0;
    border-radius: 3px;
    overflow: hidden;
    margin-top: 4px;
}

.mini-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #3b82f6, #10b981);
    border-radius: 3px;
}
</style>
""".strip()



def render_document_heatmap(
    sentences_data: List[Dict[str, Any]],
    calibrated_threshold: float = 0.9980,
) -> str:
    """
    Renders an XSS-safe sentence heatmap inside a styled container.

    Args:
        sentences_data: List of dicts with 'index', 'text', 'ai_probability',
                        and optional 'gate_value'.
        calibrated_threshold: Threshold above which a sentence is classified as AI.

    Returns:
        Safe HTML string containing the highlighted spans.
    """
    if not sentences_data:
        return '<div class="empty-doc-prompt">Мәтін енгізілмеді немесе бос.</div>'

    rendered_spans: List[str] = []

    for idx, item in enumerate(sentences_data):
        sentence_idx = item.get("index", idx)
        raw_text = str(item.get("text", ""))
        val = item.get("ai_probability")
        ai_prob = float(val) if val is not None else 0.0
        gate = item.get("gate_value", None)

        # Classification tier determination
        if ai_prob < 0.40:
            lvl_class = "lvl-human"
        elif ai_prob < calibrated_threshold:
            lvl_class = "lvl-amber"
        else:
            lvl_class = "lvl-ai"

        prob_pct = f"{ai_prob * 100:.1f}%"
        title_str = f"Sentence #{sentence_idx}: {prob_pct} AI"

        # Mandatory strict XSS sanitization
        safe_text = html.escape(raw_text, quote=True)
        safe_title = html.escape(title_str, quote=True)
        safe_idx = html.escape(str(sentence_idx), quote=True)

        gate_attr = ""
        if gate is not None:
            safe_gate = f"{float(gate):.4f}"
            gate_attr = f' data-gate="{safe_gate}"'

        span_html = (
            f'<span class="kaz-sentence {lvl_class}" '
            f'data-index="{safe_idx}" '
            f'data-prob="{ai_prob:.4f}"'
            f'{gate_attr} '
            f'title="{safe_title}">{safe_text}</span>'
        )
        rendered_spans.append(span_html)

    inner_content = " ".join(rendered_spans)
    return f'<div class="kaz-heatmap-container">{inner_content}</div>'


def render_dynamic_gate_bar(gate_value: Optional[float] = None, lang: str = "kz") -> str:
    """
    Renders a dual-colored split progress bar indicating dynamic gate allocation
    between the Context Stream (BERT) and Morpheme Stream (FST).

    Args:
        gate_value: Float in [0, 1] representing context stream weight, or None for neutral fusion (0.5).
        lang: 'kz' for Kazakh labels, 'en' for English labels.

    Returns:
        HTML string displaying the responsive split bar.
    """
    # Defensive clamping
    g_val = 0.5 if gate_value is None else float(gate_value)
    g = max(0.0, min(1.0, g_val))
    bert_pct = g * 100.0
    fst_pct = (1.0 - g) * 100.0

    bert_pct_str = f"{bert_pct:.1f}%"
    fst_pct_str = f"{fst_pct:.1f}%"

    bert_label = bert_pct_str if bert_pct >= 8.0 else ""
    fst_label = fst_pct_str if fst_pct >= 8.0 else ""

    if lang == "en":
        bert_title = f"Context Stream (BERT): {bert_pct_str}"
        fst_title = f"Morpheme Stream (FST): {fst_pct_str}"
        bert_sub = "Sentence semantics & discourse flow"
        fst_sub = "Agglutinative affixes & grammar harmony"
        if bert_pct > 60.0:
            expl_text = f"Semantic context dominates detection for this segment ({bert_pct_str} BERT vs {fst_pct_str} FST)."
        elif bert_pct < 40.0:
            expl_text = f"Morphological structure dominates detection for this segment ({fst_pct_str} FST vs {bert_pct_str} BERT)."
        else:
            expl_text = f"Balanced dual-stream fusion: {bert_pct_str} contextual semantics, {fst_pct_str} morphological structure."
    else:
        bert_title = f"Семантикалық Контекст (BERT): {bert_pct_str}"
        fst_title = f"Морфологиялық FST (Тіл Құрылымы): {fst_pct_str}"
        bert_sub = "Мәтінмәндік мағына мен сөйлем құрылымы"
        fst_sub = "Агглютинативті жұрнақтар мен үйлесімділік"
        if bert_pct > 60.0:
            expl_text = f"Бұл бөлікте семантикалық мәтінмән басымдыққа ие ({bert_pct_str} BERT vs {fst_pct_str} FST)."
        elif bert_pct < 40.0:
            expl_text = f"Бұл бөлікте морфологиялық құрылым басымдыққа ие ({fst_pct_str} FST vs {bert_pct_str} BERT)."
        else:
            expl_text = f"Теңгерімді қос ағынды үйлесім: шешім {bert_pct_str} семантикаға және {fst_pct_str} морфологияға негізделген."

    return (
        f'<div class="gate-bar-wrapper">\n'
        f'  <div class="gate-bar-header">\n'
        f'    <div class="gate-header-col gate-header-bert">\n'
        f'      <span class="gate-title-text">{bert_title}</span>\n'
        f'      <span class="gate-subtitle">{bert_sub}</span>\n'
        f'    </div>\n'
        f'    <div class="gate-header-col gate-header-fst">\n'
        f'      <span class="gate-title-text">{fst_title}</span>\n'
        f'      <span class="gate-subtitle">{fst_sub}</span>\n'
        f'    </div>\n'
        f'  </div>\n'
        f'  <div class="gate-bar-container">\n'
        f'    <div class="gate-stream-bert" style="width: {bert_pct:.1f}%;" '
        f'title="{bert_title}">{bert_label}</div>\n'
        f'    <div class="gate-stream-fst" style="width: {fst_pct:.1f}%;" '
        f'title="{fst_title}">{fst_label}</div>\n'
        f'  </div>\n'
        f'  <div class="gate-bar-legend">\n'
        f'    <span class="gate-explanation">{expl_text}</span>\n'
        f'  </div>\n'
        f'</div>'
    )


def render_morpheme_table(word_breakdowns: List[Dict[str, Any]], lang: str = "kz") -> str:
    """
    Renders a responsive HTML table detailing morphological decompositions
    (Word, Stem, POS, Affixes) with strict XSS sanitization.

    Args:
        word_breakdowns: List of dicts containing 'word', 'root', 'pos', and 'affixes'.
        lang: 'kz' for Kazakh headers, 'en' for English headers.

    Returns:
        HTML string representing the table or empty prompt.
    """
    if not word_breakdowns:
        empty_msg = "Morphological analysis data is empty." if lang == "en" else "Морфологиялық талдау деректері жоқ."
        return f'<p class="text-muted">{empty_msg}</p>'

    rows: List[str] = []
    for item in word_breakdowns:
        raw_word = str(item.get("word", ""))
        raw_root = str(item.get("root", ""))
        raw_pos = str(item.get("pos", ""))
        raw_affixes = item.get("affixes", [])

        safe_word = html.escape(raw_word, quote=True)
        safe_root = html.escape(raw_root, quote=True)

        if raw_pos:
            safe_pos = f'<span class="pos-badge">{html.escape(raw_pos, quote=True)}</span>'
        else:
            safe_pos = '<span class="text-muted">—</span>'

        if raw_affixes:
            badge_list = [
                f'<span class="morpheme-badge">{html.escape(str(aff), quote=True)}</span>'
                for aff in raw_affixes
            ]
            affixes_html = " ".join(badge_list)
        else:
            affixes_html = '<span class="text-muted">—</span>'

        rows.append(
            f'      <tr>\n'
            f'        <td><strong>{safe_word}</strong></td>\n'
            f'        <td>{safe_root}</td>\n'
            f'        <td>{safe_pos}</td>\n'
            f'        <td>{affixes_html}</td>\n'
            f'      </tr>'
        )

    table_body = "\n".join(rows)

    if lang == "en":
        th_word = "Word"
        th_stem = "Stem / Root"
        th_pos = "Part of Speech (POS)"
        th_affixes = "Agglutinative Affixes"
    else:
        th_word = "Сөз (Word)"
        th_stem = "Түбір (Stem)"
        th_pos = "Сөз табы (POS)"
        th_affixes = "Жұрнақ / Жалғаулар (Affixes)"

    return (
        f'<div class="fst-table-container">\n'
        f'  <table class="fst-table">\n'
        f'    <thead>\n'
        f'      <tr>\n'
        f'        <th>{th_word}</th>\n'
        f'        <th>{th_stem}</th>\n'
        f'        <th>{th_pos}</th>\n'
        f'        <th>{th_affixes}</th>\n'
        f'      </tr>\n'
        f'    </thead>\n'
        f'    <tbody>\n'
        f'{table_body}\n'
        f'    </tbody>\n'
        f'  </table>\n'
        f'</div>'
    )


def render_confidence_meter(
    probability: Optional[float],
    verdict: str = "",
    lang: str = "kz"
) -> str:
    """
    Renders a horizontal confidence progress meter with percentage display
    and zero emoji clutter, formatted in publication-grade academic style.

    Args:
        probability: AI probability in [0.0, 1.0] or [0.0, 100.0].
        verdict: Verdict string (e.g. '[AUTHENTIC HUMAN]', '[MACHINE-GENERATED]').
        lang: 'kz' for Kazakh labels, 'en' for English labels.

    Returns:
        XSS-safe HTML string representing the horizontal confidence meter.
    """
    if probability is None:
        p_float = 0.0
    else:
        try:
            p_float = float(probability)
        except (ValueError, TypeError):
            p_float = 0.0

    # Defensively clamp probability in [0.0, 1.0] and scale to percentage [0.0, 100.0]
    p_clamped = max(0.0, min(1.0, p_float))
    p_pct = p_clamped * 100.0
    pct_str = f"{p_pct:.1f}%"

    # Color tier matching
    v_clean = (verdict or "").strip()
    v_lower = v_clean.lower()
    if p_pct >= 99.8 or "machine" in v_lower or "жасанды" in v_lower:
        tier_class = "meter-ai"
    elif p_pct >= 40.0 or "partially" in v_lower or "hybrid" in v_lower or "аралас" in v_lower:
        tier_class = "meter-amber"
    else:
        tier_class = "meter-human"

    if lang == "en":
        label_text = "AI Confidence Level"
    else:
        label_text = "AI Сенімділік Деңгейі"

    safe_verdict = html.escape(v_clean, quote=True)
    verdict_badge = f'<span class="confidence-verdict-tag">{safe_verdict}</span>' if safe_verdict else ""

    return (
        f'<div class="confidence-meter-container">\n'
        f'  <div class="confidence-meter-header">\n'
        f'    <span class="confidence-label">{label_text}</span>\n'
        f'    <span class="confidence-value">{pct_str} {verdict_badge}</span>\n'
        f'  </div>\n'
        f'  <div class="confidence-track">\n'
        f'    <div class="confidence-fill {tier_class}" style="width: {p_pct:.1f}%;" '
        f'title="{label_text}: {pct_str}">\n'
        f'      <span class="confidence-bar-text">{pct_str if p_pct >= 10.0 else ""}</span>\n'
        f'    </div>\n'
        f'  </div>\n'
        f'</div>'
    )


def render_executive_summary_card(
    probability: Optional[float] = None,
    verdict: str = "",
    lang: str = "kz",
    total_words: int = 0,
    total_sents: int = 0,
    total_windows: int = 0,
    ai_ratio: float = 0.0,
) -> str:
    """
    Renders an all-in-one publication-grade executive summary card:
    - Prominent classification verdict badge.
    - Sleek modern gradient progress bar (zero ASCII brackets).
    - 3-tile KPI metric grid (AI Probability, AI Content Ratio, Document Volume).
    """
    if probability is None:
        p_float = 0.0
    else:
        try:
            p_float = float(probability)
        except (ValueError, TypeError):
            p_float = 0.0

    p_clamped = max(0.0, min(1.0, p_float))
    p_pct = p_clamped * 100.0
    pct_str = f"{p_pct:.1f}%"

    try:
        r_float = float(ai_ratio)
    except (ValueError, TypeError):
        r_float = 0.0
    ratio_pct = max(0.0, min(100.0, r_float * 100.0 if r_float <= 1.0 else r_float))
    ratio_str = f"{ratio_pct:.1f}%"

    # Color tier and verdict badge matching
    v_clean = (verdict or "").strip()
    v_lower = v_clean.lower()
    if "machine" in v_lower or "жасанды" in v_lower:
        badge_class = "badge-ai"
        tier_class = "meter-ai"
    elif "partially" in v_lower or "hybrid" in v_lower or "аралас" in v_lower:
        badge_class = "badge-hybrid"
        tier_class = "meter-amber"
    elif "authentic" in v_lower or "адам" in v_lower or "human" in v_lower:
        badge_class = "badge-human"
        tier_class = "meter-human"
    elif not v_clean or "awaiting" in v_lower or "күтілуде" in v_lower:
        badge_class = "badge-neutral"
        tier_class = "meter-human"
    elif p_pct >= 99.8:
        badge_class = "badge-ai"
        tier_class = "meter-ai"
    elif p_pct >= 40.0:
        badge_class = "badge-hybrid"
        tier_class = "meter-amber"
    else:
        badge_class = "badge-human"
        tier_class = "meter-human"

    if lang == "en":
        card_title = "Document Analysis Executive Summary"
        conf_label = "AI Confidence Level"
        kpi_prob_title = "AI Probability"
        kpi_ratio_title = "AI Content Ratio"
        kpi_vol_title = "Document Volume"
        kpi_vol_val = f"{total_words} words"
        kpi_vol_sub = f"{total_sents} sentences • {total_windows} window{'s' if total_windows != 1 else ''}"
        empty_default = "Awaiting Input"
    else:
        card_title = "Құжатты Сараптаудың Қорытындысы"
        conf_label = "AI Сенімділік Деңгейі"
        kpi_prob_title = "AI Ықтималдығы"
        kpi_ratio_title = "AI Мазмұн Үлесі"
        kpi_vol_title = "Құжат Көлемі"
        kpi_vol_val = f"{total_words} сөз"
        kpi_vol_sub = f"{total_sents} сөйлем • {total_windows} терезе"
        empty_default = "Мәтін күтілуде"

    display_verdict = v_clean if v_clean else empty_default

    safe_card_title = html.escape(card_title, quote=True)
    safe_conf_label = html.escape(conf_label, quote=True)
    safe_verdict = html.escape(display_verdict, quote=True)
    safe_kpi_prob_title = html.escape(kpi_prob_title, quote=True)
    safe_kpi_ratio_title = html.escape(kpi_ratio_title, quote=True)
    safe_kpi_vol_title = html.escape(kpi_vol_title, quote=True)
    safe_vol_val = html.escape(kpi_vol_val, quote=True)
    safe_vol_sub = html.escape(kpi_vol_sub, quote=True)

    return (
        f'<div class="executive-summary-card">\n'
        f'  <div class="executive-card-header">\n'
        f'    <span class="executive-card-title">{safe_card_title}</span>\n'
        f'    <span class="kpi-badge {badge_class}">{safe_verdict}</span>\n'
        f'  </div>\n'
        f'  <div class="confidence-meter-container" style="margin: 4px 0 12px 0;">\n'
        f'    <div class="confidence-meter-header">\n'
        f'      <span class="confidence-label">{safe_conf_label}</span>\n'
        f'      <span class="confidence-value">{pct_str}</span>\n'
        f'    </div>\n'
        f'    <div class="confidence-track">\n'
        f'      <div class="confidence-fill {tier_class}" style="width: {p_pct:.1f}%;" title="{safe_conf_label}: {pct_str}">\n'
        f'        <span class="confidence-bar-text">{pct_str if p_pct >= 10.0 else ""}</span>\n'
        f'      </div>\n'
        f'    </div>\n'
        f'  </div>\n'
        f'  <div class="executive-kpi-grid">\n'
        f'    <div class="executive-kpi-tile">\n'
        f'      <div class="executive-kpi-title">{safe_kpi_prob_title}</div>\n'
        f'      <div class="executive-kpi-value">{pct_str}</div>\n'
        f'    </div>\n'
        f'    <div class="executive-kpi-tile">\n'
        f'      <div class="executive-kpi-title">{safe_kpi_ratio_title}</div>\n'
        f'      <div class="executive-kpi-value">{ratio_str}</div>\n'
        f'    </div>\n'
        f'    <div class="executive-kpi-tile">\n'
        f'      <div class="executive-kpi-title">{safe_kpi_vol_title}</div>\n'
        f'      <div class="executive-kpi-value">{safe_vol_val}</div>\n'
        f'      <div class="executive-kpi-sub">{safe_vol_sub}</div>\n'
        f'    </div>\n'
        f'  </div>\n'
        f'</div>'
    )


def render_fst_decomposition_table(
    word_breakdowns: List[Dict[str, Any]],
    lang: str = "kz"
) -> str:
    """
    Renders an 8-column morphological FST decomposition table:
    (Word, Root, POS, Case, Plural, Tense, Person, Suffix Chain) with strict XSS sanitization.

    Args:
        word_breakdowns: List of dicts containing 'word', 'root', 'pos', 'case',
                         'plural', 'tense', 'person', and 'suffix_chain'.
        lang: 'kz' for Kazakh headers, 'en' for English headers.

    Returns:
        Safe HTML string containing the responsive table.
    """
    if not word_breakdowns:
        empty_msg = (
            "No morphological decomposition data available. Enter a word or sentence to parse."
            if lang == "en"
            else "Морфологиялық талдау деректері жоқ. Талдау үшін сөз немесе сөйлем енгізіңіз."
        )
        return f'<p class="text-muted">{empty_msg}</p>'

    rows: List[str] = []
    for item in word_breakdowns:
        raw_word = str(item.get("word", ""))
        raw_root = str(item.get("root", ""))
        raw_pos = str(item.get("pos", ""))
        raw_case = str(item.get("case", "—"))
        raw_plur = str(item.get("plural", "—"))
        raw_tense = str(item.get("tense", "—"))
        raw_person = str(item.get("person", "—"))
        raw_chain = str(item.get("suffix_chain", item.get("affixes", "—")))

        safe_word = html.escape(raw_word, quote=True)
        safe_root = html.escape(raw_root, quote=True)

        if raw_pos and raw_pos != "—":
            safe_pos = f'<span class="pos-badge">{html.escape(raw_pos, quote=True)}</span>'
        else:
            safe_pos = '<span class="text-muted">—</span>'

        if raw_case and raw_case != "—":
            safe_case = f'<span class="badge-case">{html.escape(raw_case, quote=True)}</span>'
        else:
            safe_case = '<span class="text-muted">—</span>'

        if raw_plur and raw_plur != "—":
            safe_plur = f'<span class="badge-plur">{html.escape(raw_plur, quote=True)}</span>'
        else:
            safe_plur = '<span class="text-muted">—</span>'

        if raw_tense and raw_tense != "—":
            safe_tense = f'<span class="badge-tense">{html.escape(raw_tense, quote=True)}</span>'
        else:
            safe_tense = '<span class="text-muted">—</span>'

        if raw_person and raw_person != "—":
            safe_person = f'<span class="badge-person">{html.escape(raw_person, quote=True)}</span>'
        else:
            safe_person = '<span class="text-muted">—</span>'

        if isinstance(raw_chain, list):
            if raw_chain:
                safe_chain = " ".join(
                    f'<span class="morpheme-badge">{html.escape(str(c), quote=True)}</span>'
                    for c in raw_chain
                )
            else:
                safe_chain = '<span class="text-muted">—</span>'
        elif raw_chain and raw_chain != "—":
            safe_chain = html.escape(str(raw_chain), quote=True)
        else:
            safe_chain = '<span class="text-muted">—</span>'

        rows.append(
            f'      <tr>\n'
            f'        <td><strong>{safe_word}</strong></td>\n'
            f'        <td>{safe_root}</td>\n'
            f'        <td>{safe_pos}</td>\n'
            f'        <td>{safe_case}</td>\n'
            f'        <td>{safe_plur}</td>\n'
            f'        <td>{safe_tense}</td>\n'
            f'        <td>{safe_person}</td>\n'
            f'        <td>{safe_chain}</td>\n'
            f'      </tr>'
        )

    table_body = "\n".join(rows)

    if lang == "en":
        th_word = "Word"
        th_stem = "Root"
        th_pos = "POS"
        th_case = "Case"
        th_plur = "Plural"
        th_tense = "Tense"
        th_person = "Person"
        th_chain = "Suffix Chain"
    else:
        th_word = "Сөз (Word)"
        th_stem = "Түбір (Root)"
        th_pos = "Сөз табы (POS)"
        th_case = "Септік (Case)"
        th_plur = "Көптік (Plural)"
        th_tense = "Шақ (Tense)"
        th_person = "Жақ (Person)"
        th_chain = "Жұрнақ тізбегі (Suffix Chain)"

    return (
        f'<div class="fst-table-container">\n'
        f'  <table class="fst-table">\n'
        f'    <thead>\n'
        f'      <tr>\n'
        f'        <th>{th_word}</th>\n'
        f'        <th>{th_stem}</th>\n'
        f'        <th>{th_pos}</th>\n'
        f'        <th>{th_case}</th>\n'
        f'        <th>{th_plur}</th>\n'
        f'        <th>{th_tense}</th>\n'
        f'        <th>{th_person}</th>\n'
        f'        <th>{th_chain}</th>\n'
        f'      </tr>\n'
        f'    </thead>\n'
        f'    <tbody>\n'
        f'{table_body}\n'
        f'    </tbody>\n'
        f'  </table>\n'
        f'</div>'
    )


def render_2d_trust_matrix_plane(
    t_fact: float,
    t_gen: float,
    quadrant: str,
    lang: str = "en"
) -> str:
    """
    Renders publication-grade responsive SVG markup of the 2D Cartesian Trust Matrix.
    Vertical axis (T_gen in [0.0, 1.0]) measures Generative Authenticity vs AI Risk.
    Horizontal axis (T_fact in [-1.0, 1.0]) measures Factual Veracity.
    All dynamic inputs are strictly escaped to prevent XSS.
    """
    lang_clean = (lang or "en").strip().lower()
    is_kz = (lang_clean == "kz" or lang_clean == "kk")

    try:
        t_fact_val = float(t_fact)
        if math.isnan(t_fact_val):
            t_fact_val = 0.0
    except (ValueError, TypeError):
        t_fact_val = 0.0

    try:
        t_gen_val = float(t_gen)
        if math.isnan(t_gen_val):
            t_gen_val = 0.5
    except (ValueError, TypeError):
        t_gen_val = 0.5

    t_fact_clamped = max(-1.0, min(1.0, t_fact_val))
    t_gen_clamped = max(0.0, min(1.0, t_gen_val))

    # Coordinate mapping:
    # cx = 200 + t_fact * 160 (clamped [40, 360])
    # cy = 260 - t_gen * 200 (clamped [60, 260])
    cx = max(40.0, min(360.0, 200.0 + t_fact_clamped * 160.0))
    cy = max(60.0, min(260.0, 260.0 - t_gen_clamped * 200.0))

    # Continuous Composite Trust: T(x) = sqrt(0.5 * (max(0, t_fact)^2 + t_gen^2))
    comp_trust = math.sqrt(0.5 * (max(0.0, t_fact_clamped) ** 2 + (t_gen_clamped ** 2)))

    # Determine Quadrant
    raw_quad = str(quadrant or "").strip()
    if "Q1" in raw_quad or "Verified Human" in raw_quad or "Расталған ақиқат" in raw_quad:
        quad_code = "Q1"
        quad_name_en = "Verified Human Fact"
        quad_name_kz = "Расталған ақиқат"
        point_color = "#10b981"
    elif "Q2" in raw_quad or "Human Misinformation" in raw_quad or "Адам қателігі" in raw_quad:
        quad_code = "Q2"
        quad_name_en = "Human Misinformation"
        quad_name_kz = "Адам қателігі"
        point_color = "#f43f5e"
    elif "Q3" in raw_quad or "Accurate AI" in raw_quad or "Нақты AI" in raw_quad:
        quad_code = "Q3"
        quad_name_en = "Accurate AI Synthesis"
        quad_name_kz = "Нақты AI синтезі"
        point_color = "#0ea5e9"
    elif "Q4" in raw_quad or "Disinformation" in raw_quad or "дезинформация" in raw_quad:
        quad_code = "Q4"
        quad_name_en = "Hallucinatory AI Disinformation"
        quad_name_kz = "AI дезинформациясы"
        point_color = "#f59e0b"
    else:
        if t_fact_clamped > 0.0 and t_gen_clamped >= 0.5:
            quad_code = "Q1"
            quad_name_en = "Verified Human Fact"
            quad_name_kz = "Расталған ақиқат"
            point_color = "#10b981"
        elif t_fact_clamped <= 0.0 and t_gen_clamped >= 0.5:
            quad_code = "Q2"
            quad_name_en = "Human Misinformation"
            quad_name_kz = "Адам қателігі"
            point_color = "#f43f5e"
        elif t_fact_clamped > 0.0 and t_gen_clamped < 0.5:
            quad_code = "Q3"
            quad_name_en = "Accurate AI Synthesis"
            quad_name_kz = "Нақты AI синтезі"
            point_color = "#0ea5e9"
        else:
            quad_code = "Q4"
            quad_name_en = "Hallucinatory AI Disinformation"
            quad_name_kz = "AI дезинформациясы"
            point_color = "#f59e0b"

    safe_raw_quad = html.escape(raw_quad, quote=True)
    safe_quad_code = html.escape(quad_code, quote=True)
    active_quad_name = quad_name_kz if is_kz else quad_name_en
    safe_active_name = html.escape(active_quad_name, quote=True)

    if safe_raw_quad and safe_raw_quad not in (safe_quad_code, safe_active_name):
        hud_quad_display = f"{safe_quad_code} ({safe_raw_quad})"
    else:
        hud_quad_display = f"{safe_quad_code} ({safe_active_name})"

    # Quadrant text labels
    if is_kz:
        q1_label = "Q1: Расталған ақиқат"
        q1_sub = "Verified Human Fact"
        q2_label = "Q2: Адам қателігі"
        q2_sub = "Human Misinformation"
        q3_label = "Q3: Нақты AI синтезі"
        q3_sub = "Accurate AI Synthesis"
        q4_label = "Q4: AI дезинформация"
        q4_sub = "Hallucinatory AI Disinformation"

        x_title = "Деректік ақиқаттық (T_fact) →"
        y_title = "↑ Адам жазған / AI қаупі (T_gen)"

        tick_x_neg = "-1.0 (Теріске)"
        tick_x_zero = "0.0 (Бейтарап)"
        tick_x_pos = "+1.0 (Расталған)"

        tick_y_top = "1.0 (Адам)"
        tick_y_mid = "0.5 (Шек)"
        tick_y_bot = "0.0 (AI)"
    else:
        q1_label = "Q1: Verified Human Fact"
        q1_sub = "High Veracity · Human Style"
        q2_label = "Q2: Human Misinformation"
        q2_sub = "Low Veracity · Human Style"
        q3_label = "Q3: Accurate AI Synthesis"
        q3_sub = "High Veracity · AI Generated"
        q4_label = "Q4: Hallucinatory AI Disinformation"
        q4_sub = "Low Veracity · AI Disinformation"

        x_title = "Factual Veracity (T_fact) →"
        y_title = "↑ Human Origin vs AI Risk (T_gen)"

        tick_x_neg = "-1.0 (Refuted)"
        tick_x_zero = "0.0 (Neutral)"
        tick_x_pos = "+1.0 (Supported)"

        tick_y_top = "1.0 (Human)"
        tick_y_mid = "0.5 (Threshold)"
        tick_y_bot = "0.0 (AI)"

    # SVG markup
    svg_markup = (
        f'<svg viewBox="0 0 460 320" xmlns="http://www.w3.org/2000/svg" style="width: 100%; max-width: 580px; height: auto; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;">\n'
        f'  <defs>\n'
        f'    <filter id="glow-2d-filter" x="-50%" y="-50%" width="200%" height="200%">\n'
        f'      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>\n'
        f'      <feMerge>\n'
        f'        <feMergeNode in="coloredBlur"/>\n'
        f'        <feMergeNode in="SourceGraphic"/>\n'
        f'      </feMerge>\n'
        f'    </filter>\n'
        f'  </defs>\n'
        f'\n'
        f'  <!-- 4 Quadrants Pastel Backgrounds -->\n'
        f'  <!-- Top-Right: Q1 Verified Human Fact -->\n'
        f'  <rect x="200" y="60" width="160" height="100" fill="#ecfdf5" fill-opacity="{1.0 if quad_code == "Q1" else 0.5}" stroke="{"#10b981" if quad_code == "Q1" else "#d1fae5"}" stroke-width="{"2.5" if quad_code == "Q1" else "1"}"/>\n'
        f'  <text x="280" y="95" text-anchor="middle" font-size="10.5" font-weight="700" fill="#065f46">{q1_label}</text>\n'
        f'  <text x="280" y="112" text-anchor="middle" font-size="8.5" fill="#047857">{q1_sub}</text>\n'
        f'\n'
        f'  <!-- Top-Left: Q2 Human Misinformation -->\n'
        f'  <rect x="40" y="60" width="160" height="100" fill="#fff1f2" fill-opacity="{1.0 if quad_code == "Q2" else 0.5}" stroke="{"#f43f5e" if quad_code == "Q2" else "#ffe4e6"}" stroke-width="{"2.5" if quad_code == "Q2" else "1"}"/>\n'
        f'  <text x="120" y="95" text-anchor="middle" font-size="10.5" font-weight="700" fill="#9f1239">{q2_label}</text>\n'
        f'  <text x="120" y="112" text-anchor="middle" font-size="8.5" fill="#be123c">{q2_sub}</text>\n'
        f'\n'
        f'  <!-- Bottom-Right: Q3 Accurate AI Synthesis -->\n'
        f'  <rect x="200" y="160" width="160" height="100" fill="#f0f9ff" fill-opacity="{1.0 if quad_code == "Q3" else 0.5}" stroke="{"#0ea5e9" if quad_code == "Q3" else "#e0f2fe"}" stroke-width="{"2.5" if quad_code == "Q3" else "1"}"/>\n'
        f'  <text x="280" y="215" text-anchor="middle" font-size="10.5" font-weight="700" fill="#0369a1">{q3_label}</text>\n'
        f'  <text x="280" y="232" text-anchor="middle" font-size="8.5" fill="#0284c7">{q3_sub}</text>\n'
        f'\n'
        f'  <!-- Bottom-Left: Q4 Hallucinatory AI Disinformation -->\n'
        f'  <rect x="40" y="160" width="160" height="100" fill="#fffbeb" fill-opacity="{1.0 if quad_code == "Q4" else 0.5}" stroke="{"#f59e0b" if quad_code == "Q4" else "#fef3c7"}" stroke-width="{"2.5" if quad_code == "Q4" else "1"}"/>\n'
        f'  <text x="120" y="215" text-anchor="middle" font-size="10.5" font-weight="700" fill="#92400e">{q4_label}</text>\n'
        f'  <text x="120" y="232" text-anchor="middle" font-size="8.5" fill="#b45309">{q4_sub}</text>\n'
        f'\n'
        f'  <!-- Center Crosshairs: Vertical at X=200 (T_fact=0.0), Horizontal at Y=160 (T_gen=0.5) -->\n'
        f'  <line x1="200" y1="60" x2="200" y2="260" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4,4"/>\n'
        f'  <line x1="40" y1="160" x2="360" y2="160" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4,4"/>\n'
        f'\n'
        f'  <!-- Plane Outer Border -->\n'
        f'  <rect x="40" y="60" width="320" height="200" fill="none" stroke="#64748b" stroke-width="1.5"/>\n'
        f'\n'
        f'  <!-- Y-Axis Labels and Ticks -->\n'
        f'  <text x="40" y="44" font-size="10.5" font-weight="700" fill="#1e293b">{y_title}</text>\n'
        f'  <line x1="35" y1="60" x2="40" y2="60" stroke="#64748b" stroke-width="1.5"/>\n'
        f'  <text x="34" y="64" text-anchor="end" font-size="8.5" fill="#475569">{tick_y_top}</text>\n'
        f'  <line x1="35" y1="160" x2="40" y2="160" stroke="#64748b" stroke-width="1.5"/>\n'
        f'  <text x="34" y="164" text-anchor="end" font-size="8.5" fill="#475569">{tick_y_mid}</text>\n'
        f'  <line x1="35" y1="260" x2="40" y2="260" stroke="#64748b" stroke-width="1.5"/>\n'
        f'  <text x="34" y="264" text-anchor="end" font-size="8.5" fill="#475569">{tick_y_bot}</text>\n'
        f'\n'
        f'  <!-- X-Axis Labels and Ticks -->\n'
        f'  <line x1="40" y1="260" x2="40" y2="265" stroke="#64748b" stroke-width="1.5"/>\n'
        f'  <text x="40" y="278" text-anchor="middle" font-size="8.5" fill="#475569">{tick_x_neg}</text>\n'
        f'  <line x1="200" y1="260" x2="200" y2="265" stroke="#64748b" stroke-width="1.5"/>\n'
        f'  <text x="200" y="278" text-anchor="middle" font-size="8.5" fill="#475569">{tick_x_zero}</text>\n'
        f'  <line x1="360" y1="260" x2="360" y2="265" stroke="#64748b" stroke-width="1.5"/>\n'
        f'  <text x="360" y="278" text-anchor="middle" font-size="8.5" fill="#475569">{tick_x_pos}</text>\n'
        f'  <text x="200" y="300" text-anchor="middle" font-size="10.5" font-weight="700" fill="#1e293b">{x_title}</text>\n'
        f'\n'
        f'  <!-- Active Point Indicator: cx={cx:.1f}, cy={cy:.1f} -->\n'
        f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="12" fill="{point_color}" opacity="0.3" class="pulse-ring">\n'
        f'    <animate attributeName="r" values="9;14;9" dur="2s" repeatCount="indefinite"/>\n'
        f'    <animate attributeName="opacity" values="0.4;0.1;0.4" dur="2s" repeatCount="indefinite"/>\n'
        f'  </circle>\n'
        f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" fill="{point_color}" stroke="#ffffff" stroke-width="2" filter="url(#glow-2d-filter)"/>\n'
        f'</svg>'
    )

    return (
        f'<div class="trust-matrix-plane-container">\n'
        f'  <div class="trust-hud-bar">\n'
        f'    <span class="hud-pill"><strong>T_fact:</strong> {t_fact_val:+.2f}</span>\n'
        f'    <span class="hud-pill"><strong>T_gen:</strong> {t_gen_val:.2f}</span>\n'
        f'    <span class="hud-pill"><strong>𝒯(x) [T(x)]:</strong> {comp_trust:.2f}</span>\n'
        f'    <span class="hud-pill hud-quad-{quad_code.lower()}"><strong>Quadrant:</strong> {hud_quad_display}</span>\n'
        f'  </div>\n'
        f'  {svg_markup}\n'
        f'</div>'
    )


def render_trust_summary_card(
    doc_trust_result: Optional[Any],
    lang: str = "en"
) -> str:
    """
    Renders an executive summary card for Document Trust Verification,
    including the Four-Quadrant badge, continuous 2D Cartesian plane, dual risk progress bars, and claim counts.
    """
    lang_clean = (lang or "en").strip().lower()
    is_kz = (lang_clean == "kz" or lang_clean == "kk")

    if doc_trust_result is None:
        title_text = "Деректерді тексеру күтілуде" if is_kz else "Awaiting Verification"
        desc_text = (
            "Құжатты тексеру үшін мәтінді енгізіп, «Тексеру» түймесін басыңыз."
            if is_kz
            else "Enter text and click 'Verify Document' to evaluate factual accuracy and trust risk."
        )
        plane_svg = render_2d_trust_matrix_plane(0.0, 0.5, "Q1", lang=lang)
        return (
            f'<div class="trust-card-wrapper">\n'
            f'  <div style="font-size: 1.05rem; font-weight: 700; color: #334155; margin-bottom: 4px;">{title_text}</div>\n'
            f'  <div class="text-muted" style="font-size: 0.85rem; margin-bottom: 12px;">{desc_text}</div>\n'
            f'  {plane_svg}\n'
            f'</div>'
        )

    ai_risk = max(0.0, min(1.0, float(getattr(doc_trust_result, "ai_risk", 0.0))))
    fact_risk = max(0.0, min(1.0, float(getattr(doc_trust_result, "factual_risk", 0.0))))
    trust_risk = max(0.0, min(1.0, float(getattr(doc_trust_result, "trust_risk", 0.0))))
    quadrant = getattr(doc_trust_result, "quadrant_verdict", "Verified Human Fact")
    total_claims = int(getattr(doc_trust_result, "total_claims", 0))
    sup_count = int(getattr(doc_trust_result, "supported_count", 0))
    ref_count = int(getattr(doc_trust_result, "refuted_count", 0))
    nei_count = int(getattr(doc_trust_result, "nei_count", 0))

    t_fact = getattr(doc_trust_result, "t_fact", 0.0)
    try:
        t_fact = float(t_fact)
    except (ValueError, TypeError):
        t_fact = 0.0

    t_gen = getattr(doc_trust_result, "t_gen", None)
    if t_gen is None:
        t_gen = max(0.0, min(1.0, 1.0 - ai_risk))
    else:
        try:
            t_gen = float(t_gen)
        except (ValueError, TypeError):
            t_gen = max(0.0, min(1.0, 1.0 - ai_risk))

    comp_trust = getattr(doc_trust_result, "composite_trust", None)
    if comp_trust is None:
        comp_trust = math.sqrt(0.5 * (max(0.0, t_fact) ** 2 + t_gen ** 2))
    else:
        try:
            comp_trust = float(comp_trust)
        except (ValueError, TypeError):
            comp_trust = math.sqrt(0.5 * (max(0.0, t_fact) ** 2 + t_gen ** 2))

    quadrant_code = getattr(doc_trust_result, "quadrant_code", "Q1")

    # Determine CSS class and localized badge text
    if quadrant == "Verified Human Fact":
        badge_class = "lvl-verified-human"
        badge_text = "Расталған ақиқат (Verified Human Fact)" if is_kz else "Verified Human Fact"
    elif quadrant == "Human Misinformation":
        badge_class = "lvl-human-misinfo"
        badge_text = "Адам қателігі / Жалған дерек (Human Misinformation)" if is_kz else "Human Misinformation"
    elif quadrant == "Accurate AI Synthesis":
        badge_class = "lvl-ai-synthesis"
        badge_text = "Нақты AI синтезі (Accurate AI Synthesis)" if is_kz else "Accurate AI Synthesis"
    else:
        badge_class = "lvl-ai-disinfo"
        badge_text = "Галлюцинациялық AI дезинформация (Hallucinatory AI Disinformation)" if is_kz else "Hallucinatory AI Disinformation"

    card_title = "Сенімділік бағалауы" if is_kz else "Trustworthiness Assessment"
    ai_label = "AI генерация қаупі" if is_kz else "AI Risk"
    fact_label = "Дерек қайшылығы қаупі" if is_kz else "Factual Risk"
    trust_label = "Жиынтық сенім қаупі" if is_kz else "Trust Risk"

    lbl_total = "Барлық мәлімдемелер" if is_kz else "Total Claims"
    lbl_sup = "Расталған" if is_kz else "Supported"
    lbl_ref = "Теріске шығарылған" if is_kz else "Refuted"
    lbl_nei = "Ақпарат жеткіліксіз" if is_kz else "Not Enough Info"

    ai_pct = f"{ai_risk * 100:.1f}%"
    fact_pct = f"{fact_risk * 100:.1f}%"
    trust_pct = f"{trust_risk * 100:.1f}%"
    comp_pct = f"{comp_trust * 100:.1f}%"
    t_fact_pct = f"{min(100.0, max(0.0, (t_fact + 1.0) / 2.0 * 100.0)):.1f}%"
    t_gen_pct = f"{min(100.0, max(0.0, t_gen * 100.0)):.1f}%"

    lbl_comp_trust = "Үздіксіз сенім 𝒯(x)" if is_kz else "Composite Trust 𝒯(x)"
    lbl_tf_meter = "Деректік ақиқаттық (T_fact)" if is_kz else "Factual Veracity (T_fact)"
    lbl_tg_meter = "Адамдық шығу тегі (T_gen)" if is_kz else "Generative Authenticity (T_gen)"

    plane_html = render_2d_trust_matrix_plane(t_fact, t_gen, quadrant_code or quadrant, lang=lang)

    return (
        f'<div class="trust-card-wrapper">\n'
        f'  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">\n'
        f'    <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a;">{card_title}</div>\n'
        f'    <div class="trust-badge {badge_class}">{html.escape(badge_text, quote=True)}</div>\n'
        f'  </div>\n'
        f'  <div class="trust-meter-row">\n'
        f'    <div class="trust-meter-tile">\n'
        f'      <div class="morpho-diag-label">{lbl_comp_trust}</div>\n'
        f'      <div style="font-size: 1.15rem; font-weight: 700; color: #4338ca;">{comp_pct}</div>\n'
        f'      <div class="mini-bar-track"><div class="mini-bar-fill" style="width: {comp_pct};"></div></div>\n'
        f'    </div>\n'
        f'    <div class="trust-meter-tile">\n'
        f'      <div class="morpho-diag-label">{lbl_tf_meter}</div>\n'
        f'      <div style="font-size: 1.15rem; font-weight: 700; color: {"#059669" if t_fact >= 0 else "#dc2626"};">T_fact: {t_fact:+.2f}</div>\n'
        f'      <div class="mini-bar-track"><div class="mini-bar-fill" style="width: {t_fact_pct}; background: {"#10b981" if t_fact >= 0 else "#ef4444"};"></div></div>\n'
        f'    </div>\n'
        f'    <div class="trust-meter-tile">\n'
        f'      <div class="morpho-diag-label">{lbl_tg_meter}</div>\n'
        f'      <div style="font-size: 1.15rem; font-weight: 700; color: {"#0284c7" if t_gen >= 0.5 else "#d97706"};">T_gen: {t_gen_pct}</div>\n'
        f'      <div class="mini-bar-track"><div class="mini-bar-fill" style="width: {t_gen_pct}; background: {"#0ea5e9" if t_gen >= 0.5 else "#f59e0b"};"></div></div>\n'
        f'    </div>\n'
        f'  </div>\n'
        f'  {plane_html}\n'
        f'  <div class="trust-bars-grid">\n'
        f'    <div class="trust-bar-item">\n'
        f'      <div class="trust-bar-label"><span>{ai_label}</span><span>{ai_pct}</span></div>\n'
        f'      <div class="trust-bar-track"><div class="trust-bar-fill fill-ai" style="width: {ai_pct};"></div></div>\n'
        f'    </div>\n'
        f'    <div class="trust-bar-item">\n'
        f'      <div class="trust-bar-label"><span>{fact_label}</span><span>{fact_pct}</span></div>\n'
        f'      <div class="trust-bar-track"><div class="trust-bar-fill fill-fact" style="width: {fact_pct};"></div></div>\n'
        f'    </div>\n'
        f'    <div class="trust-bar-item">\n'
        f'      <div class="trust-bar-label"><span>{trust_label}</span><span>{trust_pct}</span></div>\n'
        f'      <div class="trust-bar-track"><div class="trust-bar-fill fill-trust" style="width: {trust_pct};"></div></div>\n'
        f'    </div>\n'
        f'  </div>\n'
        f'  <div class="trust-stats-row">\n'
        f'    <span class="trust-stat-pill stat-pill-total">{lbl_total}: {total_claims}</span>\n'
        f'    <span class="trust-stat-pill stat-pill-sup">{lbl_sup}: {sup_count}</span>\n'
        f'    <span class="trust-stat-pill stat-pill-ref">{lbl_ref}: {ref_count}</span>\n'
        f'    <span class="trust-stat-pill stat-pill-nei">{lbl_nei}: {nei_count}</span>\n'
        f'  </div>\n'
        f'</div>'
    )


def render_claims_verification_table(
    claims: Optional[List[Any]],
    lang: str = "en"
) -> str:
    """
    Renders an HTML table listing atomic claims, verdicts, evidence citations, and explanations,
    with an expandable 16-feature morphological diagnostic drawer under each claim row.
    All dynamic texts are strictly escaped to prevent XSS.
    """
    lang_clean = (lang or "en").strip().lower()
    is_kz = (lang_clean == "kz" or lang_clean == "kk")

    if not claims:
        msg = "Әзірге тексерілген атомдық мәлімдемелер жоқ." if is_kz else "No atomic claims extracted or verified yet."
        return f'<div class="empty-doc-prompt">{msg}</div>'

    rows = []
    for idx, c in enumerate(claims, 1):
        claim_obj = getattr(c, "claim", None)
        claim_text = getattr(claim_obj, "text", str(c)) if claim_obj else str(c)
        safe_claim = html.escape(claim_text, quote=True)

        verdict = getattr(c, "verdict", "NOT ENOUGH INFO")
        if verdict == "SUPPORTED":
            v_badge = '<span class="verdict-badge verdict-supported">SUPPORTED</span>'
        elif verdict == "REFUTED":
            v_badge = '<span class="verdict-badge verdict-refuted">REFUTED</span>'
        else:
            v_badge = '<span class="verdict-badge verdict-nei">NOT ENOUGH INFO</span>'

        conf = float(getattr(c, "confidence", 0.5))
        conf_str = f"{conf * 100:.1f}%"

        # Evidence citation in main table row
        evidence_list = getattr(c, "evidence", [])
        if evidence_list:
            top_ev = evidence_list[0]
            ev_title = html.escape(getattr(top_ev, "title", "Wikipedia"), quote=True)
            ev_text = html.escape(getattr(top_ev, "text", ""), quote=True)
            ev_url = getattr(top_ev, "source_url", "")
            if len(ev_text) > 130:
                ev_text = ev_text[:127] + "..."
            ev_html = f"<strong>{ev_title}</strong>: {ev_text}"
            if ev_url:
                safe_url = html.escape(ev_url, quote=True)
                ev_html += f' <a href="{safe_url}" target="_blank" style="color: #2563eb; text-decoration: underline;">[Wiki]</a>'
        else:
            ev_html = '<span class="text-muted">—</span>'

        explanation = getattr(c, "explanation", "")
        safe_exp = html.escape(explanation, quote=True) if explanation else '<span class="text-muted">—</span>'

        # Morphological Drawer Content
        drawer_summary = "Морфологиялық сәйкестік және дәлелдемелер" if is_kz else "Morphological Alignment & Evidence Diagnostics"

        # Highlight matched stems in top evidence text
        top_ev = evidence_list[0] if evidence_list else None
        ev_drawer_html = ""
        if top_ev:
            raw_ev_text = str(getattr(top_ev, "text", ""))
            matched_stems = getattr(top_ev, "matched_stems", []) or []
            safe_ev_text = html.escape(raw_ev_text, quote=True)
            if matched_stems:
                for stem in matched_stems:
                    stem_clean = str(stem).strip()
                    if stem_clean and len(stem_clean) >= 2:
                        safe_stem = html.escape(stem_clean, quote=True)
                        pattern = re.compile(rf"(?i)({re.escape(safe_stem)}[а-яәіңғүұқөһa-z0-9]*)")
                        safe_ev_text = pattern.sub(r'<mark class="morpho-stem-highlight">\1</mark>', safe_ev_text)
            ev_title = html.escape(str(getattr(top_ev, "title", "Wikipedia")), quote=True)
            ev_url = getattr(top_ev, "source_url", "")
            safe_url = html.escape(ev_url, quote=True) if ev_url else ""
            link_html = f' <a href="{safe_url}" target="_blank" style="color: #2563eb; text-decoration: underline;">[Wiki]</a>' if safe_url else ""
            passage_label = "Дереккөз мәтіні (үзінді)" if is_kz else "Evidence Passage"
            ev_drawer_html = (
                f'<div class="morpho-evidence-box">\n'
                f'  <div style="font-weight: 600; font-size: 0.8rem; margin-bottom: 4px; color: #334155;">{passage_label}: <em>{ev_title}</em>{link_html}</div>\n'
                f'  <div style="font-size: 0.82rem; color: #1e293b; line-height: 1.5;">{safe_ev_text}</div>\n'
                f'</div>'
            )

        # Structured 16-feature diagnostic grid
        morpho = getattr(c, "morphological_features", {}) or {}
        if morpho and isinstance(morpho, dict):
            neg_mismatch = float(morpho.get("directional_negation_mismatch", 0.0))
            if neg_mismatch > 0.0:
                neg_badge = '<span class="morpho-badge badge-warn">Mismatch (1.0)</span>'
            else:
                neg_badge = '<span class="morpho-badge badge-ok">Clean (0.0)</span>'

            year_conflict = float(morpho.get("calendar_year_conflict", 0.0))
            if year_conflict > 0.0:
                year_badge = '<span class="morpho-badge badge-warn">Conflict (1.0)</span>'
            else:
                year_badge = '<span class="morpho-badge badge-ok">Matched (0.0)</span>'

            num_mismatch = float(morpho.get("number_mismatch", 0.0))
            if num_mismatch > 0.0:
                num_badge = '<span class="morpho-badge badge-warn">Divergence (1.0)</span>'
            else:
                num_badge = '<span class="morpho-badge badge-ok">Matched (0.0)</span>'

            has_c_evid = bool(morpho.get("has_claim_evidential", False))
            has_e_evid = bool(morpho.get("has_evidence_evidential", False))
            if has_c_evid or has_e_evid:
                evid_badge = '<span class="morpho-badge badge-info">Detected</span>'
            else:
                evid_badge = '<span class="morpho-badge badge-ok">None</span>'

            has_nec = bool(morpho.get("has_modal_necessity", False))
            has_poss = bool(morpho.get("has_modal_possibility", False))
            if has_nec and has_poss:
                modal_badge = '<span class="morpho-badge badge-info">Necessity + Possibility</span>'
            elif has_nec:
                modal_badge = '<span class="morpho-badge badge-info">Necessity (керек)</span>'
            elif has_poss:
                modal_badge = '<span class="morpho-badge badge-info">Possibility (мүмкін)</span>'
            else:
                modal_badge = '<span class="morpho-badge badge-ok">None</span>'

            fst_jaccard = float(morpho.get("fst_root_jaccard", morpho.get("fst_jaccard", 0.0)))
            fst_pct = fst_jaccard * 100.0

            surf_overlap = float(morpho.get("surface_token_overlap", 0.0))
            surf_pct = surf_overlap * 100.0

            lex_div = float(morpho.get("lexical_fst_divergence", 0.0))
            lex_pct = lex_div * 100.0

            case_align = float(morpho.get("case_suffix_alignment", 0.0))
            case_pct = case_align * 100.0

            lbl_neg = "Болымсыздық қайшылығы" if is_kz else "Directional Negation Mismatch"
            lbl_year = "Күнтізбелік жыл қайшылығы" if is_kz else "Calendar Year Conflict"
            lbl_num = "Сан/мөлшер алшақтығы" if is_kz else "Number/Quantity Divergence"
            lbl_evid = "Эвиденциалды қосымшалар (-ыпты/-іпті)" if is_kz else "Evidential Markers"
            lbl_modal = "Эпистемикалық модальдар (керек, мүмкін)" if is_kz else "Epistemic Modals"
            lbl_fst = "FST түбір ұқсастығы (Jaccard)" if is_kz else "FST Root Jaccard Similarity"
            lbl_surf = "Беткі токендер қабаттасуы" if is_kz else "Surface Token Overlap"
            lbl_lex = "Лексикалық-FST алшақтығы (Hard NEI қаупі)" if is_kz else "Lexical-FST Divergence (Hard NEI Risk)"
            lbl_case = "Септік жалғауларының үйлесімі" if is_kz else "Case Suffix Alignment"

            diag_grid_html = (
                f'<div class="morpho-diag-grid">\n'
                f'  <div class="morpho-diag-card"><span class="morpho-diag-label">{lbl_neg}</span><span class="morpho-diag-val">{neg_badge}</span></div>\n'
                f'  <div class="morpho-diag-card"><span class="morpho-diag-label">{lbl_year}</span><span class="morpho-diag-val">{year_badge}</span></div>\n'
                f'  <div class="morpho-diag-card"><span class="morpho-diag-label">{lbl_num}</span><span class="morpho-diag-val">{num_badge}</span></div>\n'
                f'  <div class="morpho-diag-card"><span class="morpho-diag-label">{lbl_evid}</span><span class="morpho-diag-val">{evid_badge}</span></div>\n'
                f'  <div class="morpho-diag-card"><span class="morpho-diag-label">{lbl_modal}</span><span class="morpho-diag-val">{modal_badge}</span></div>\n'
                f'  <div class="morpho-diag-card"><span class="morpho-diag-label">{lbl_fst}</span><span class="morpho-diag-val">{fst_pct:.1f}%</span><div class="mini-bar-track"><div class="mini-bar-fill" style="width: {min(100.0, max(0.0, fst_pct)):.1f}%;"></div></div></div>\n'
                f'  <div class="morpho-diag-card"><span class="morpho-diag-label">{lbl_surf}</span><span class="morpho-diag-val">{surf_pct:.1f}%</span><div class="mini-bar-track"><div class="mini-bar-fill" style="width: {min(100.0, max(0.0, surf_pct)):.1f}%;"></div></div></div>\n'
                f'  <div class="morpho-diag-card"><span class="morpho-diag-label">{lbl_lex}</span><span class="morpho-diag-val">{lex_pct:.1f}%</span><div class="mini-bar-track"><div class="mini-bar-fill" style="width: {min(100.0, max(0.0, lex_pct)):.1f}%;"></div></div></div>\n'
                f'  <div class="morpho-diag-card"><span class="morpho-diag-label">{lbl_case}</span><span class="morpho-diag-val">{case_pct:.1f}%</span><div class="mini-bar-track"><div class="mini-bar-fill" style="width: {min(100.0, max(0.0, case_pct)):.1f}%;"></div></div></div>\n'
                f'</div>'
            )
        else:
            no_morpho_msg = (
                "Морфологиялық талдау деректері бұл мәлімдеме үшін қолжетімсіз."
                if is_kz
                else "Morphological alignment diagnostics not available for this claim."
            )
            diag_grid_html = f'<div class="text-muted" style="font-size: 0.8rem; margin-top: 4px;">{no_morpho_msg}</div>'

        drawer_content = f'{ev_drawer_html}\n{diag_grid_html}' if ev_drawer_html else diag_grid_html

        row = (
            f'      <tr>\n'
            f'        <td style="font-weight: 600; color: #64748b;">{idx}</td>\n'
            f'        <td style="font-weight: 500;">{safe_claim}</td>\n'
            f'        <td>{v_badge}</td>\n'
            f'        <td style="font-family: monospace; font-weight: 600;">{conf_str}</td>\n'
            f'        <td style="font-size: 0.8rem; line-height: 1.45;">{ev_html}</td>\n'
            f'        <td style="font-size: 0.8rem; color: #475569;">{safe_exp}</td>\n'
            f'      </tr>\n'
            f'      <tr class="morpho-drawer-row">\n'
            f'        <td colspan="6" style="padding: 4px 14px 10px 14px; background: #fafafa; border-bottom: 2px solid #e2e8f0;">\n'
            f'          <details class="morpho-drawer">\n'
            f'            <summary style="cursor: pointer; font-weight: 600; color: #2563eb; font-size: 0.82rem; user-select: none;">{drawer_summary}</summary>\n'
            f'            <div style="margin-top: 8px;">\n'
            f'{drawer_content}\n'
            f'            </div>\n'
            f'          </details>\n'
            f'        </td>\n'
            f'      </tr>'
        )
        rows.append(row)

    table_body = "\n".join(rows)

    th_idx = "#"
    th_claim = "Мәлімдеме (Claim)" if is_kz else "Atomic Claim"
    th_verdict = "Үкім (Verdict)" if is_kz else "Verdict"
    th_conf = "Сенімділік (Conf)" if is_kz else "Confidence"
    th_ev = "Дәйексөз (Evidence Source)" if is_kz else "Evidence Source & Passage"
    th_exp = "Түсіндірме (Explanation)" if is_kz else "Reasoning / Contradiction"

    return (
        f'<div class="claims-table-container">\n'
        f'  <table class="claims-table">\n'
        f'    <thead>\n'
        f'      <tr>\n'
        f'        <th style="width: 4%;">{th_idx}</th>\n'
        f'        <th style="width: 28%;">{th_claim}</th>\n'
        f'        <th style="width: 12%;">{th_verdict}</th>\n'
        f'        <th style="width: 9%;">{th_conf}</th>\n'
        f'        <th style="width: 27%;">{th_ev}</th>\n'
        f'        <th style="width: 20%;">{th_exp}</th>\n'
        f'      </tr>\n'
        f'    </thead>\n'
        f'    <tbody>\n'
        f'{table_body}\n'
        f'    </tbody>\n'
        f'  </table>\n'
        f'</div>'
    )


