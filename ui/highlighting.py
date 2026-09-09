"""
ui/highlighting.py: XSS-Safe Sentence Heatmap & HTML/CSS Highlighting Engine.

Renders sentence-level heatmaps, dynamic gate split-bars, and FST morpheme
breakdown tables for the interactive Kazakh AI text detection UI.
All user text is strictly sanitized using html.escape(..., quote=True)
to guarantee protection against Stored and Reflected XSS attacks.
"""

from typing import List, Dict, Any, Optional
import html


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
    margin: 14px 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.gate-bar-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.875rem;
    font-weight: 600;
    margin-bottom: 6px;
    color: #374151;
}

.gate-bar-container {
    display: flex;
    width: 100%;
    height: 26px;
    border-radius: 6px;
    overflow: hidden;
    background-color: #e5e7eb;
    box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
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
    display: flex;
    justify-content: space-between;
    margin-top: 5px;
    font-size: 0.75rem;
    color: #6b7280;
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
        ai_prob = float(item.get("ai_probability", 0.0))
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


def render_dynamic_gate_bar(gate_value: float) -> str:
    """
    Renders a dual-colored split progress bar indicating dynamic gate allocation
    between the Context Stream (BERT) and Morpheme Stream (FST).

    Args:
        gate_value: Float in [0, 1] representing context stream weight.

    Returns:
        HTML string displaying the responsive split bar.
    """
    # Defensive clamping
    g = max(0.0, min(1.0, float(gate_value)))
    bert_pct = g * 100.0
    fst_pct = (1.0 - g) * 100.0

    bert_pct_str = f"{bert_pct:.1f}%"
    fst_pct_str = f"{fst_pct:.1f}%"

    bert_label = bert_pct_str if bert_pct >= 8.0 else ""
    fst_label = fst_pct_str if fst_pct >= 8.0 else ""

    return (
        f'<div class="gate-bar-wrapper">\n'
        f'  <div class="gate-bar-header">\n'
        f'    <span>Семантикалық Контекст (BERT): {bert_pct_str}</span>\n'
        f'    <span>Морфологиялық FST (Тіл Құрылымы): {fst_pct_str}</span>\n'
        f'  </div>\n'
        f'  <div class="gate-bar-container">\n'
        f'    <div class="gate-stream-bert" style="width: {bert_pct:.1f}%;" '
        f'title="Семантикалық Контекст: {bert_pct_str}">{bert_label}</div>\n'
        f'    <div class="gate-stream-fst" style="width: {fst_pct:.1f}%;" '
        f'title="Морфологиялық FST: {fst_pct_str}">{fst_label}</div>\n'
        f'  </div>\n'
        f'  <div class="gate-bar-legend">\n'
        f'    <span>Context Stream (BERT): {bert_pct_str}</span>\n'
        f'    <span>Morpheme Stream (FST): {fst_pct_str}</span>\n'
        f'  </div>\n'
        f'</div>'
    )


def render_morpheme_table(word_breakdowns: List[Dict[str, Any]]) -> str:
    """
    Renders a responsive HTML table detailing morphological decompositions
    (Word, Stem, POS, Affixes) with strict XSS sanitization.

    Args:
        word_breakdowns: List of dicts containing 'word', 'root', 'pos', and 'affixes'.

    Returns:
        HTML string representing the table or empty prompt.
    """
    if not word_breakdowns:
        return '<p class="text-muted">Морфологиялық талдау деректері жоқ.</p>'

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

    return (
        f'<div class="fst-table-container">\n'
        f'  <table class="fst-table">\n'
        f'    <thead>\n'
        f'      <tr>\n'
        f'        <th>Сөз (Word)</th>\n'
        f'        <th>Түбір (Stem)</th>\n'
        f'        <th>Сөз табы (POS)</th>\n'
        f'        <th>Жұрнақ / Жалғаулар (Affixes)</th>\n'
        f'      </tr>\n'
        f'    </thead>\n'
        f'    <tbody>\n'
        f'{table_body}\n'
        f'    </tbody>\n'
        f'  </table>\n'
        f'</div>'
    )
