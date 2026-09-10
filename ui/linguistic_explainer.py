# -*- coding: utf-8 -*-
"""
ui/linguistic_explainer.py: Dynamic Linguistic Explainer Engine for Kazakh AI Detection.

Computes quantitative and morphological signals from Kazakh text:
- Type-Token Ratio (TTR) for lexical diversity
- LLM formulaic discourse connectors
- Authentic colloquial and e-commerce consumer markers
- Morphological agglutinative suffix complexity peeling via AdvancedKazakhFSTAnalyzer

Produces 2-4 plain-language, evidence-backed reasoning bullets explaining
the detection verdict in publication-grade academic typography (zero emojis).
"""

import os
import sys
import re
from typing import List, Dict, Any, Optional, Tuple, Union

# Ensure project root is in sys.path for direct CLI script execution
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from kaz_mage.document import DocumentAnalysisResult, DocumentChunk
except ImportError:
    DocumentAnalysisResult = None
    DocumentChunk = None


# ---------------------------------------------------------------------------
# Formulaic LLM Discourse Markers (Synthetic / AI indicators)
# ---------------------------------------------------------------------------
FORMULAIC_MARKERS: List[str] = [
    # Multi-word discourse connectors
    "қорытындылай келе",
    "айта кету керек",
    "маңызды рөл атқарады",
    "осыған орай",
    "сонымен қатар",
    "айрықша мәнге ие",
    "назар аударарлық",
    "байқауға болады",
    "болып табылады",
    "көрініс табады",
    "айтуға болады",
    "негізгі ерекшелігі",
    "жоғары дәрежеде",
    "атап өткен жөн",
    "жүйелі түрде",
    "бір жағынан",
    "екінші жағынан",
    "заманауи әлемде",
    "бүгінгі таңда",
    "тиімділігін арттыру",
    # Single-word formal/formulaic markers
    "айтарлықтай",
]

# ---------------------------------------------------------------------------
# Authentic Colloquial / Consumer Markers (Human indicators)
# ---------------------------------------------------------------------------
COLLOQUIAL_MARKERS: List[str] = [
    # Multi-word phrases
    "сатып алған",
    "сатып алдым",
    "өте ұнады",
    # Single-word tokens / stems
    "керемет",
    "жақсы",
    "сапасы",
    "доставка",
    "каспи",
    "рахмет",
    "ұнады",
    "өте",
    "жаман",
    "жылдам",
    "дүкеннен",
    "дүкенімнен",
    "дүкен",
    "алдым",
    "келді",
    "бағасы",
    "арзан",
    "сатушы",
    "тамаша",
    "жеткізу",
    "жарады",
    "курьер",
    "ұнайды",
]


def _compile_markers(markers: List[str]) -> List[Tuple[str, re.Pattern]]:
    """Compiles marker phrases into whitespace-tolerant Unicode regexes, sorted by length."""
    compiled = []
    # Sort markers by length descending so longer phrases match before subphrases
    sorted_markers = sorted(markers, key=lambda m: len(m), reverse=True)
    for m in sorted_markers:
        parts = m.strip().split()
        if len(parts) > 1:
            escaped_parts = [re.escape(p) for p in parts]
            pattern_str = r'\b' + r'\s+'.join(escaped_parts) + r'\b'
        else:
            pattern_str = r'\b' + re.escape(m.strip()) + r'\b'
        compiled.append((m.strip(), re.compile(pattern_str, re.IGNORECASE | re.UNICODE)))
    return compiled


_COMPILED_FORMULAIC = _compile_markers(FORMULAIC_MARKERS)
_COMPILED_COLLOQUIAL = _compile_markers(COLLOQUIAL_MARKERS)


def _extract_tokens(text: str) -> List[str]:
    """
    Extracts lowercased alphabetic tokens from Kazakh/English text,
    handling Cyrillic characters (including Kazakh extensions) and hyphenated words.
    """
    if not text or not isinstance(text, str):
        return []
    raw = re.findall(
        r'[a-zA-Zа-яА-ЯёЁәіңғүұқөһӘІҢҒҮҰҚӨҺ]+(?:-[a-zA-Zа-яА-ЯёЁәіңғүұқөһӘІҢҒҮҰҚӨҺ]+)*',
        text,
    )
    return [t.lower() for t in raw if any(c.isalpha() for c in t)]


def _find_markers(
    text_lower: str,
    marker_tuples: List[Tuple[str, re.Pattern]]
) -> Tuple[int, List[str]]:
    """
    Finds non-overlapping occurrences of markers in text, returning total count
    and list of distinct marker names found.
    """
    total_count = 0
    found_markers: List[str] = []
    matched_spans: List[Tuple[int, int]] = []

    for name, pattern in marker_tuples:
        hits = 0
        for match in pattern.finditer(text_lower):
            span = match.span()
            # Avoid counting substrings inside already matched longer phrases
            if not any(s[0] <= span[0] and span[1] <= s[1] for s in matched_spans):
                matched_spans.append(span)
                hits += 1
        if hits > 0:
            total_count += hits
            if name not in found_markers:
                found_markers.append(name)

    return total_count, found_markers


def _compute_avg_suffix_count(words: List[str]) -> float:
    """Computes average suffix count per word using FST analyzer."""
    if not words:
        return 0.0
    try:
        from ui.sentence_analyzer import analyze_sentence_morphemes
        morphemes = analyze_sentence_morphemes(" ".join(words))
        if not morphemes:
            return 0.0
        total_affixes = sum(len(m.get("affixes", [])) for m in morphemes)
        return float(round(total_affixes / len(words), 4))
    except Exception:
        try:
            from fst_analyzer import AdvancedKazakhFSTAnalyzer
            fst = AdvancedKazakhFSTAnalyzer()
            segmented = fst.analyze_and_segment(" ".join(words))
            tokens = segmented.split()
            affix_count = sum(1 for t in tokens if t.startswith("-"))
            return float(round(affix_count / len(words), 4))
        except Exception:
            return 0.0


def compute_linguistic_features(text: str) -> Dict[str, Any]:
    """
    Computes key linguistic indicators from Kazakh text:
    - total_words: int
    - unique_words: int
    - type_token_ratio: float (unique_words / total_words, 0.0 if empty)
    - formulaic_marker_count: int (occurrences of LLM formulaic discourse markers)
    - formulaic_markers_found: List[str]
    - colloquial_marker_count: int (occurrences of authentic colloquial/consumer markers)
    - colloquial_markers_found: List[str]
    - avg_word_length: float
    - avg_suffix_count: float (using AdvancedKazakhFSTAnalyzer if available)
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {
            "total_words": 0,
            "unique_words": 0,
            "type_token_ratio": 0.0,
            "formulaic_marker_count": 0,
            "formulaic_markers_found": [],
            "colloquial_marker_count": 0,
            "colloquial_markers_found": [],
            "avg_word_length": 0.0,
            "avg_suffix_count": 0.0,
        }

    tokens = _extract_tokens(text)
    total_words = len(tokens)
    if total_words == 0:
        return {
            "total_words": 0,
            "unique_words": 0,
            "type_token_ratio": 0.0,
            "formulaic_marker_count": 0,
            "formulaic_markers_found": [],
            "colloquial_marker_count": 0,
            "colloquial_markers_found": [],
            "avg_word_length": 0.0,
            "avg_suffix_count": 0.0,
        }

    unique_tokens = set(tokens)
    unique_words = len(unique_tokens)
    type_token_ratio = float(round(unique_words / total_words, 4))
    avg_word_length = float(round(sum(len(w) for w in tokens) / total_words, 2))

    text_lower = text.lower()
    formulaic_count, formulaic_found = _find_markers(text_lower, _COMPILED_FORMULAIC)
    colloquial_count, colloquial_found = _find_markers(text_lower, _COMPILED_COLLOQUIAL)

    avg_suffix_count = _compute_avg_suffix_count(tokens)

    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "type_token_ratio": type_token_ratio,
        "formulaic_marker_count": formulaic_count,
        "formulaic_markers_found": formulaic_found,
        "colloquial_marker_count": colloquial_count,
        "colloquial_markers_found": colloquial_found,
        "avg_word_length": avg_word_length,
        "avg_suffix_count": avg_suffix_count,
    }


def _extract_doc_info(doc_result: Any) -> Tuple[str, float, float]:
    """Safely extracts verdict, probability, and AI content ratio from doc_result."""
    if doc_result is None:
        return "", -1.0, 0.0
    if isinstance(doc_result, dict):
        verdict = str(doc_result.get("verdict", ""))
        try:
            prob = float(doc_result.get("document_ai_probability", -1.0))
        except (ValueError, TypeError):
            prob = -1.0
        try:
            ratio = float(doc_result.get("ai_content_ratio", 0.0))
        except (ValueError, TypeError):
            ratio = 0.0
        return verdict, prob, ratio

    verdict = str(getattr(doc_result, "verdict", ""))
    try:
        prob = float(getattr(doc_result, "document_ai_probability", -1.0))
    except (ValueError, TypeError):
        prob = -1.0
    try:
        ratio = float(getattr(doc_result, "ai_content_ratio", 0.0))
    except (ValueError, TypeError):
        ratio = 0.0
    return verdict, prob, ratio


def generate_linguistic_explanation(
    text: str,
    doc_result: Optional[Union[DocumentAnalysisResult, Dict[str, Any]]] = None,
    lang: str = "kz"
) -> List[str]:
    """
    Generates 2 to 4 bullet points explaining the linguistic rationale for the classification.
    Bilingual support:
      - lang="kz": Kazakh explanations (natural academic/informative style)
      - lang="en": English explanations (publication-grade style)
    Strictly zero decorative emojis.
    """
    feats = compute_linguistic_features(text)
    total_words = feats["total_words"]

    # Defensive check: empty or trivial text
    if total_words == 0:
        if lang == "kz":
            return [
                "Талдау үшін мәтін енгізілмеген немесе мазмұны жеткіліксіз.",
                "Лексикалық алуандылық пен морфологиялық құрылым көрсеткіштері есептелмеді.",
            ]
        return [
            "No input text provided or content is too short for reliable linguistic analysis.",
            "Lexical diversity and morphological indicators could not be computed.",
        ]

    verdict, prob, ratio = _extract_doc_info(doc_result)

    # If verdict/probability not supplied, infer dynamically from linguistic features
    if prob < 0:
        if feats["formulaic_marker_count"] >= 2 or (
            feats["formulaic_marker_count"] >= 1 and feats["colloquial_marker_count"] == 0
        ):
            verdict = "Machine-Generated"
            prob = 0.9995
        elif feats["formulaic_marker_count"] >= 1 and feats["colloquial_marker_count"] >= 1:
            verdict = "Partially AI / Hybrid"
            prob = 0.75
        else:
            verdict = "Authentic Human"
            prob = 0.03

    v_lower = verdict.lower()
    is_ai = ("machine" in v_lower or "ai" in v_lower) and "partially" not in v_lower and "hybrid" not in v_lower
    is_hybrid = "partially" in v_lower or "hybrid" in v_lower
    if not is_ai and not is_hybrid:
        if prob >= 0.90:
            is_ai = True
        elif prob >= 0.40:
            is_hybrid = True

    formulaic_count = feats["formulaic_marker_count"]
    formulaic_found = feats["formulaic_markers_found"]
    colloquial_count = feats["colloquial_marker_count"]
    colloquial_found = feats["colloquial_markers_found"]
    ttr = feats["type_token_ratio"]
    avg_suffix = feats["avg_suffix_count"]

    bullets: List[str] = []

    # -----------------------------------------------------------------------
    # Bullet 1: Verdict & Discourse / Stylistic Markers
    # -----------------------------------------------------------------------
    if lang == "kz":
        if is_ai:
            if formulaic_count > 0:
                markers_preview = ", ".join(f"«{m}»" for m in formulaic_found[:3])
                b1 = (
                    f"Жоғары жиілікті формулалық дискурстық маркерлер ({markers_preview}) анықталды: "
                    f"мұндай стандартты клишелер мен байланыстырушы конструкциялар генеративті тілдік модельдерге тән."
                )
            else:
                b1 = (
                    "Синтаксистік біртектілік және генеративті үлгілерге тән формулалық құрылыс байқалады "
                    "(модельдік сенімділік деңгейі жоғары)."
                )
        elif is_hybrid:
            if formulaic_count > 0:
                markers_preview = ", ".join(f"«{m}»" for m in formulaic_found[:2])
                b1 = (
                    f"Аралас синтаксистік стилистика: мәтінде табиғи адам жазбасымен қатар ЖИ-ге тән "
                    f"формулалық дискурстық байланыстырғыштар ({markers_preview}) қатар кездеседі."
                )
            else:
                b1 = (
                    "Аралас мәтін құрылымы: құжатта табиғи жазылған фрагменттер мен жасанды интеллект "
                    "сипаттары бар синтетикалық сөйлемдер араласқан."
                )
        else:  # Authentic Human
            if colloquial_count > 0:
                markers_preview = ", ".join(f"«{m}»" for m in colloquial_found[:3])
                b1 = (
                    f"Шынайы тұтынушылық және ауызекі лексика маркерлері табылды ({markers_preview}): "
                    f"еркін семантикалық құрылым табиғи авторлық мәнерді растайды."
                )
            else:
                b1 = (
                    "Формулалық машиналық клишелер мен жасанды тілдік үлгілер анықталмады: "
                    "синтаксистік құрылымы табиғи, еркін стильге толық сәйкес келеді."
                )
    else:  # English
        if is_ai:
            if formulaic_count > 0:
                markers_preview = ", ".join(f'"{m}"' for m in formulaic_found[:3])
                b1 = (
                    f"Detected high-density formulaic discourse markers ({markers_preview}): "
                    f"standardized transitional phrases characteristic of generative large language models."
                )
            else:
                b1 = (
                    "Structural uniformity and stylistic patterns characteristic of "
                    "generative language models detected (high AI classification confidence)."
                )
        elif is_hybrid:
            if formulaic_count > 0:
                markers_preview = ", ".join(f'"{m}"' for m in formulaic_found[:2])
                b1 = (
                    f"Mixed stylistic strata detected: text exhibits authentic phrasing alongside "
                    f"formulaic synthetic discourse connectors ({markers_preview})."
                )
            else:
                b1 = (
                    "Heterogeneous composition: document contains alternating segments of "
                    "organic human prose and synthetic machine generation."
                )
        else:  # Authentic Human
            if colloquial_count > 0:
                markers_preview = ", ".join(f'"{m}"' for m in colloquial_found[:3])
                b1 = (
                    f"Authentic colloquial and domain-specific terminology detected ({markers_preview}): "
                    f"reflects genuine human sentiment and organic phrasing."
                )
            else:
                b1 = (
                    "Absence of synthetic formulaic connectors and repetitive LLM artifacts: "
                    "syntax exhibits organic structural variation."
                )

    bullets.append(b1)

    # -----------------------------------------------------------------------
    # Bullet 2: Lexical Diversity (Type-Token Ratio / TTR)
    # -----------------------------------------------------------------------
    if lang == "kz":
        if total_words >= 8:
            if ttr >= 0.70:
                b2 = (
                    f"Лексикалық алуандылық деңгейі жоғары (TTR = {ttr:.2f}): "
                    f"сөздік қор бай әрі әралуан, қайталанатын біркелкі лексикалық бірліктер минималды."
                )
            else:
                b2 = (
                    f"Лексикалық қайталану деңгейі (TTR = {ttr:.2f}): "
                    f"мәтінде шектеулі сөздік қор немесе біркелкі құрылымдардың қайталануы көрініс табады."
                )
        else:
            b2 = (
                f"Шағын көлемді мәтіндік үзінді ({total_words} сөз, TTR = {ttr:.2f}): "
                f"қысқа үзінділерде лексикалық көрсеткіштер шоғырланған сипатқа ие."
            )
    else:  # English
        if total_words >= 8:
            if ttr >= 0.70:
                b2 = (
                    f"High lexical diversity (TTR = {ttr:.2f}): demonstrates varied "
                    f"vocabulary usage with minimal synthetic redundancy."
                )
            else:
                b2 = (
                    f"Restricted lexical diversity (TTR = {ttr:.2f}): indicates repetitive "
                    f"vocabulary patterns or constrained lexical distribution."
                )
        else:
            b2 = (
                f"Compact text sample ({total_words} words, TTR = {ttr:.2f}): "
                f"short samples naturally exhibit concentrated lexical indices."
            )

    bullets.append(b2)

    # -----------------------------------------------------------------------
    # Bullet 3: Morphological Complexity (Agglutinative Suffix Density)
    # -----------------------------------------------------------------------
    if lang == "kz":
        if avg_suffix >= 0.75:
            b3 = (
                f"Агглютинативті морфологиялық тығыздық: сөзге орташа есеппен {avg_suffix:.2f} аффикс келеді "
                f"(FST талдауы терең септік, көптік және жіктік жалғаулар тізбегін көрсетті)."
            )
        elif avg_suffix >= 0.30:
            b3 = (
                f"Орташа агглютинативті морфологиялық құрылым: сөзге орташа есеппен {avg_suffix:.2f} жалғау сәйкес келеді "
                f"(стандартты қазақ тілінің грамматикалық деңгейі)."
            )
        else:
            b3 = (
                f"Төмен морфологиялық жалғау тығыздығы: сөзге орташа есеппен {avg_suffix:.2f} аффикс келеді "
                f"(негізгі түбір сөздер немесе кірме сөздер басым)."
            )
    else:  # English
        if avg_suffix >= 0.75:
            b3 = (
                f"Rich agglutinative morphology: average of {avg_suffix:.2f} affixes per word "
                f"(FST analysis confirms multi-tiered case, plural, and inflectional chains)."
            )
        elif avg_suffix >= 0.30:
            b3 = (
                f"Moderate agglutinative morphological density: average of {avg_suffix:.2f} affixes per word, "
                f"consistent with standard Kazakh grammatical patterns."
            )
        else:
            b3 = (
                f"Low morphological affix density: average of {avg_suffix:.2f} affixes per word "
                f"(predominance of uninflected roots or loanwords)."
            )

    bullets.append(b3)

    # -----------------------------------------------------------------------
    # Bullet 4 (Optional): Document-level Volume Attribution
    # -----------------------------------------------------------------------
    if 0.0 < ratio < 1.0:
        if lang == "kz":
            b4 = (
                f"Сегменттік үлес салмағы: құжат көлемінің {ratio * 100:.1f}% бөлігі "
                f"ЖИ шекті деңгейінен асқан фрагменттерден тұрады."
            )
        else:
            b4 = (
                f"Segmental volume: {ratio * 100:.1f}% of document content exceeds the "
                f"calibrated machine-generation probability threshold."
            )
        bullets.append(b4)

    return bullets[:4]