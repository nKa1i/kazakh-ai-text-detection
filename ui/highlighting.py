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

