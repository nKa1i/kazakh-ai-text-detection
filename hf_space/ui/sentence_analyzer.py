# -*- coding: utf-8 -*-
"""
ui/sentence_analyzer.py: Sentence-Level Scoring & FST Morphological Analyzer Engine.

Segments long documents into sentences using SentencePreservingChunker, maps
chunk-level AI probabilities and dynamic fusion gate weights to individual sentences
(applying worst-case attribution across overlapping windows for defensive security),
and decomposes Kazakh sentences into root stems, POS categories, and annotated
grammatical affixes (cases, plurals, possessives, verbal tenses, and personal agreements)
using the AdvancedKazakhFSTAnalyzer.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Union, Tuple
import re

# Ensure project root is in sys.path for direct CLI script execution
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from kaz_mage.chunker import SentencePreservingChunker
from kaz_mage.document import DocumentAnalysisResult, DocumentChunk
from fst_analyzer import AdvancedKazakhFSTAnalyzer

# Module-level singleton for the FST analyzer
_DEFAULT_FST_ANALYZER: Optional[AdvancedKazakhFSTAnalyzer] = None


def get_default_fst_analyzer() -> AdvancedKazakhFSTAnalyzer:
    """Returns or lazily creates a singleton AdvancedKazakhFSTAnalyzer."""
    global _DEFAULT_FST_ANALYZER
    if _DEFAULT_FST_ANALYZER is None:
        _DEFAULT_FST_ANALYZER = AdvancedKazakhFSTAnalyzer()
    return _DEFAULT_FST_ANALYZER


# Punctuation characters to strip when tokenizing words
PUNCTUATION_CHARS = '.,!?;:«»"“”—–()[]{}<>/\\|`~@#$%^&*+=_\'\"'

# Grammatical Affix Annotation Dictionaries
CASE_TAGS: Dict[str, str] = {
    # Genitive (Ілік септік)
    "нің": "GEN", "ның": "GEN", "дың": "GEN", "дің": "GEN", "тың": "GEN", "тің": "GEN",
    # Dative (Барыс септік)
    "ға": "DAT", "ге": "DAT", "қа": "DAT", "ке": "DAT", "на": "DAT", "не": "DAT",
    # Accusative (Табыс септік)
    "ны": "ACC", "ні": "ACC", "ды": "ACC", "ді": "ACC", "ты": "ACC", "ті": "ACC",
    # Locative (Жатыс септік)
    "да": "LOC", "де": "LOC", "та": "LOC", "те": "LOC", "нда": "LOC", "нде": "LOC",
    # Ablative (Шығыс септік)
    "дан": "ABL", "ден": "ABL", "тан": "ABL", "тен": "ABL", "нан": "ABL", "нен": "ABL",
    # Instrumental (Көмектес септік)
    "мен": "INS", "бен": "INS", "пен": "INS",
}

PLUR_TAGS: Dict[str, str] = {
    "лар": "PLUR", "лер": "PLUR", "дар": "PLUR", "дер": "PLUR", "тар": "PLUR", "тер": "PLUR",
}

POSS_TAGS: Dict[str, str] = {
    # 1st person singular
    "ым": "POSS.1SG", "ім": "POSS.1SG", "м": "POSS.1SG",
    # 1st person plural
    "ымыз": "POSS.1PL", "іміз": "POSS.1PL", "мыз": "POSS.1PL", "міз": "POSS.1PL",
    "тарымыз": "POSS.1PL", "теріміз": "POSS.1PL", "дарымыз": "POSS.1PL", "деріміз": "POSS.1PL",
    "ларымыз": "POSS.1PL", "леріміз": "POSS.1PL",
    # 2nd person singular
    "ың": "POSS.2SG", "ің": "POSS.2SG", "ң": "POSS.2SG",
    # 2nd person singular polite
    "ыңыз": "POSS.2SG.POL", "іңіз": "POSS.2SG.POL", "ңыз": "POSS.2SG.POL", "ңіз": "POSS.2SG.POL",
    "тарыңыз": "POSS.2SG.POL", "теріңіз": "POSS.2SG.POL", "дарыңыз": "POSS.2SG.POL", "деріңіз": "POSS.2SG.POL",
    "ларыңыз": "POSS.2SG.POL", "леріңіз": "POSS.2SG.POL",
    # 3rd person singular / plural
    "ы": "POSS.3SG", "і": "POSS.3SG", "сы": "POSS.3SG", "сі": "POSS.3SG",
    "тары": "POSS.3PL", "тері": "POSS.3PL", "дары": "POSS.3PL", "дері": "POSS.3PL",
    "лары": "POSS.3PL", "лері": "POSS.3PL",
    "сын": "POSS.3SG", "сін": "POSS.3SG",
    "мын": "POSS.1SG", "мін": "POSS.1SG",
}

VERB_TENSE_TAGS: Dict[str, str] = {
    # Converb of reason
    "ғандықтан": "CVB.REASON", "гендіктен": "CVB.REASON",
    "қандықтан": "CVB.REASON", "кендіктен": "CVB.REASON",
    # Habitual participle
    "атын": "HAB.PART", "етін": "HAB.PART", "йтын": "HAB.PART", "йтін": "HAB.PART",
    # Past participle
    "ған": "PAST.PART", "ген": "PAST.PART", "қан": "PAST.PART", "кен": "PAST.PART",
    # Present / habitual 3rd person
    "ады": "PRES.3SG", "еді": "PRES.3SG", "йды": "PRES.3SG", "йді": "PRES.3SG",
    # Future intention / infinitive
    "мақ": "FUT.INT", "мек": "FUT.INT", "бақ": "FUT.INT", "бек": "FUT.INT",
    "пақ": "FUT.INT", "пек": "FUT.INT",
    # Past tense
    "ды": "PAST", "ді": "PAST", "ты": "PAST", "ті": "PAST",
}

VERB_NEG_TAGS: Dict[str, str] = {
    "ба": "NEG", "бе": "NEG", "па": "NEG", "пе": "NEG", "ма": "NEG", "ме": "NEG",
}

VERB_DERIV_TAGS: Dict[str, str] = {
    "лан": "VERB.DERIV", "лен": "VERB.DERIV",
    "дан": "VERB.DERIV", "ден": "VERB.DERIV",
    "тан": "VERB.DERIV", "тен": "VERB.DERIV",
    "ла": "VERB.DERIV", "ле": "VERB.DERIV",
    "да": "VERB.DERIV", "де": "VERB.DERIV",
    "та": "VERB.DERIV", "те": "VERB.DERIV",
}

VERB_PERSON_TAGS: Dict[str, str] = {
    # 1st person singular
    "мын": "PERS.1SG", "мін": "PERS.1SG", "бын": "PERS.1SG", "бін": "PERS.1SG",
    "пын": "PERS.1SG", "пін": "PERS.1SG",
    # 1st person plural
    "мыз": "PERS.1PL", "міз": "PERS.1PL", "быз": "PERS.1PL", "біз": "PERS.1PL",
    "пыз": "PERS.1PL", "піз": "PERS.1PL",
    # 2nd person singular
    "сың": "PERS.2SG", "сің": "PERS.2SG",
    # 2nd person plural
    "сыздар": "PERS.2PL", "сіздер": "PERS.2PL",
}

AGENT_DERIV_TAGS: Dict[str, str] = {
    "ші": "DERIV.AGENT",
    "шы": "DERIV.AGENT",
}


def _get_field(obj: Any, field_name: str, default: Any = None) -> Any:
    """Safely retrieves an attribute or dictionary key."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


def _extract_words(sentence_text: str) -> List[str]:
    """
    Extracts individual words from a sentence, stripping leading/trailing
    punctuation (quotes, brackets, dashes, periods, commas, etc.).
    """
    if not sentence_text or not isinstance(sentence_text, str):
        return []
    raw_tokens = sentence_text.split()
    words = []
    for raw_token in raw_tokens:
        clean = raw_token.strip(PUNCTUATION_CHARS)
        if not clean:
            continue
        # Split internally attached punctuation (e.g. word1,word2)
        parts = re.split(r'[,;:!?«»"“”—–()[\]{}]+', clean)
        for p in parts:
            p_clean = p.strip(PUNCTUATION_CHARS)
            if p_clean:
                words.append(p_clean)
    return words


def analyze_sentence_morphemes(
    sentence_text: str,
    fst_analyzer: Optional[AdvancedKazakhFSTAnalyzer] = None
) -> List[Dict[str, Any]]:
    """
    Decomposes each word in a Kazakh sentence into root stem, POS category,
    and annotated grammatical affixes using FST morphological rules.

    Args:
        sentence_text: Input sentence string.
        fst_analyzer: Optional AdvancedKazakhFSTAnalyzer instance.

    Returns:
        List of dictionaries with keys:
            - word: Original word extracted from text.
            - root: Stem root identified by the analyzer.
            - pos: Part of speech tag ('NOUN', 'VERB', 'LOANWORD/NOUN', 'STEM/OTHER', 'OTHER').
            - affixes: List of annotated affix strings (e.g. ['-лар (PLUR)', '-дың (GEN)']).
    """
    if not sentence_text or not isinstance(sentence_text, str):
        return []

    words = _extract_words(sentence_text)
    if not words:
        return []

    analyzer = fst_analyzer or get_default_fst_analyzer()
    loanword_roots = sorted(analyzer.loanword_roots, key=len, reverse=True)

    breakdowns: List[Dict[str, Any]] = []

    for word in words:
        # Rule 1: Length <= 2 or purely numeric tokens
        if len(word) <= 2 or word.isdigit() or re.match(r'^\d+([.,]\d+)?$', word):
            breakdowns.append({
                "word": word,
                "root": word,
                "pos": "OTHER",
                "affixes": []
            })
            continue

        # Rule 2: Check loanword roots (e.g. доставкасы -> доставка -сы)
        w_lower = word.lower()
        loan_matched = False
        for root in loanword_roots:
            if w_lower.startswith(root):
                if len(w_lower) > len(root):
                    sfx = word[len(root):]
                    if len(sfx) >= 1:
                        breakdowns.append({
                            "word": word,
                            "root": word[:len(root)],
                            "pos": "LOANWORD/NOUN",
                            "affixes": [f"-{sfx}"]
                        })
                        loan_matched = True
                        break
                elif len(w_lower) == len(root):
                    breakdowns.append({
                        "word": word,
                        "root": word,
                        "pos": "LOANWORD/NOUN",
                        "affixes": []
                    })
                    loan_matched = True
                    break

        if loan_matched:
            continue

        # Rule 3: Native Kazakh Morphology via Suffixes
        current = word
        matched_affixes: List[Tuple[str, str]] = []
        has_verb_person = False
        has_verb_tense = False
        has_case = False
        has_poss = False
        has_plur = False

        # 1. Verbal Personal Agreement Suffixes (e.g. келгенмін -> -мін)
        m_vper = analyzer.verb_person_re.search(current.lower())
        if m_vper and len(current[:m_vper.start()]) >= 3:
            sfx = current[m_vper.start():]
            current = current[:m_vper.start()]
            has_verb_person = True
            tag = VERB_PERSON_TAGS.get(sfx.lower(), "PERS")
            matched_affixes.insert(0, (sfx, tag))

        # 2. Verbal Tense / Participle Suffixes (e.g. жасалғандықтан -> -ғандықтан / -ған)
        m_vt = analyzer.verb_tense_re.search(current.lower())
        if m_vt and len(current[:m_vt.start()]) >= 3:
            sfx = current[m_vt.start():]
            current = current[:m_vt.start()]
            has_verb_tense = True
            tag = VERB_TENSE_TAGS.get(sfx.lower(), "TENSE")
            matched_affixes.insert(0, (sfx, tag))

        # 2b. Verbal Negation Suffixes (e.g. қанағаттанба -> -ба, келме -> -ме)
        if has_verb_person or has_verb_tense:
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
            has_case = True
            tag = CASE_TAGS.get(sfx.lower(), "CASE")
            matched_affixes.insert(0, (sfx, tag))

        # 4. Noun Possessives
        if len(current) > 3:
            m_poss = analyzer.poss_re.search(current.lower())
            if m_poss and len(current[:m_poss.start()]) >= 3:
                poss_cand = m_poss.group(1).lower()
                # If current ends with -ші or -шы and candidate possessive is just 'і' or 'ы',
                # do not mistakenly strip the vowel part of the agentive suffix
                if (current.lower().endswith("ші") and poss_cand == "і") or (current.lower().endswith("шы") and poss_cand == "ы"):
                    pass
                else:
                    sfx = current[m_poss.start():]
                    current = current[:m_poss.start()]
                    has_poss = True
                    tag = POSS_TAGS.get(sfx.lower(), "POSS")
                    matched_affixes.insert(0, (sfx, tag))

        # 5. Plurals
        if len(current) > 3:
            m_plur = analyzer.plur_re.search(current.lower())
            if m_plur and len(current[:m_plur.start()]) >= 3:
                sfx = current[m_plur.start():]
                current = current[:m_plur.start()]
                has_plur = True
                tag = PLUR_TAGS.get(sfx.lower(), "PLUR")
                matched_affixes.insert(0, (sfx, tag))

        # 6. Agentive Derivational Suffix (-ші / -шы, e.g. жетекші -> жетек + -ші)
        has_agent = False
        if len(current) >= 4 and not (has_verb_person or has_verb_tense):
            for ag in ["ші", "шы"]:
                if current.lower().endswith(ag) and len(current[:-len(ag)]) >= 3:
                    sfx = current[-len(ag):]
                    current = current[:-len(ag)]
                    has_agent = True
                    tag = AGENT_DERIV_TAGS.get(ag.lower(), "DERIV.AGENT")
                    matched_affixes.insert(0, (sfx, tag))
                    break

        if has_verb_person or has_verb_tense:
            pos = "VERB"
        elif has_case or has_poss or has_plur or has_agent:
            pos = "NOUN"
        else:
            pos = "STEM/OTHER"

        root = current
        annotated_affixes = [f"-{sfx} ({tag})" for sfx, tag in matched_affixes]

        breakdowns.append({
            "word": word,
            "root": root,
            "pos": pos,
            "affixes": annotated_affixes
        })

    return breakdowns


def analyze_document_sentences(
    text: str,
    document_result: Optional[Union[DocumentAnalysisResult, Dict[str, Any]]] = None,
    fst_analyzer: Optional[AdvancedKazakhFSTAnalyzer] = None
) -> List[Dict[str, Any]]:
    """
    Partitions a Kazakh document into sentences and attributes AI detection
    probabilities and dynamic gate weights with defensive worst-case mapping.

    Args:
        text: Full document Kazakh text.
        document_result: Optional DocumentAnalysisResult instance or serialized dictionary.
        fst_analyzer: Optional AdvancedKazakhFSTAnalyzer instance. When provided,
                      attaches morphological breakdown data to each sentence.

    Returns:
        List of dictionaries with keys:
            - index: Sentence index (0-based).
            - text: Sentence text.
            - start_char: Start character offset in document.
            - end_char: End character offset in document.
            - ai_probability: Attributed AI probability float.
            - gate_value: Attributed dynamic gate context stream weight float.
            - is_ai: Boolean indicating whether ai_probability >= calibrated_threshold.
            - morphemes: (Optional) List of word breakdown dicts when fst_analyzer is provided.
    """
    if not text or not text.strip():
        return []

    chunker = SentencePreservingChunker(max_words=200, overlap_sentences=1)
    sentences = chunker.split_sentences(text)
    if not sentences:
        return []

    # Extract chunk list and document-level defaults defensively
    chunks = []
    doc_ai_prob = 0.0
    calibrated_threshold = 0.9980

    if document_result is not None:
        chunks = _get_field(document_result, "chunks", []) or []
        p = _get_field(document_result, "document_ai_probability", None)
        if p is not None:
            try:
                doc_ai_prob = float(p)
            except (ValueError, TypeError):
                doc_ai_prob = 0.0

        t = _get_field(document_result, "calibrated_threshold", None)
        if t is not None:
            try:
                calibrated_threshold = float(t)
            except (ValueError, TypeError):
                calibrated_threshold = 0.9980

    results: List[Dict[str, Any]] = []

    for idx, (s_text, s_start, s_end) in enumerate(sentences):
        # 1. Look for chunks that fully cover the sentence: c.start <= s.start and s.end <= c.end
        covering_chunks = []
        for c in chunks:
            c_start = _get_field(c, "start_char", 0)
            c_end = _get_field(c, "end_char", 0)
            if c_start <= s_start and s_end <= c_end:
                covering_chunks.append(c)

        if covering_chunks:
            # Defensive worst-case attribution: select chunk with maximum ai_probability
            best_chunk = max(
                covering_chunks,
                key=lambda c: (
                    float(_get_field(c, "ai_probability", 0.0)),
                    float(_get_field(c, "gate_value", 0.5))
                )
            )
            ai_prob = float(_get_field(best_chunk, "ai_probability", 0.0))
            gate_val = float(_get_field(best_chunk, "gate_value", 0.5))
        elif chunks:
            # 2. If no chunk fully covers, find chunks with maximum character overlap
            overlap_chunks = []
            for c in chunks:
                c_start = _get_field(c, "start_char", 0)
                c_end = _get_field(c, "end_char", 0)
                overlap = max(0, min(c_end, s_end) - max(c_start, s_start))
                if overlap > 0:
                    overlap_chunks.append((c, overlap))

            if overlap_chunks:
                max_ov = max(ov for _, ov in overlap_chunks)
                candidates = [c for c, ov in overlap_chunks if ov == max_ov]
                best_chunk = max(
                    candidates,
                    key=lambda c: (
                        float(_get_field(c, "ai_probability", 0.0)),
                        float(_get_field(c, "gate_value", 0.5))
                    )
                )
                ai_prob = float(_get_field(best_chunk, "ai_probability", 0.0))
                gate_val = float(_get_field(best_chunk, "gate_value", 0.5))
            else:
                ai_prob = doc_ai_prob
                gate_val = 0.5
        else:
            ai_prob = doc_ai_prob
            gate_val = 0.5

        is_ai = bool(ai_prob >= calibrated_threshold)

        sent_item: Dict[str, Any] = {
            "index": idx,
            "text": s_text,
            "start_char": s_start,
            "end_char": s_end,
            "ai_probability": ai_prob,
            "gate_value": gate_val,
            "is_ai": is_ai,
        }

        if fst_analyzer is not None:
            sent_item["morphemes"] = analyze_sentence_morphemes(s_text, fst_analyzer=fst_analyzer)

        results.append(sent_item)

    return results
