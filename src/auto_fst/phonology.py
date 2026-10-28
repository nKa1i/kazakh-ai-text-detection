# -*- coding: utf-8 -*-
"""Two-level morphophonological stem-boundary restorer for Kazakh FST segmentation."""

from typing import Optional, Set

VOWEL_INITIAL_SUFFIX_STARTS = set("аәеиоөұүыі")

# Canonical Kazakh roots that undergo final stop voicing (п->б, қ->ғ, к->г) before vowels
VOICING_ALTERNATION_LEXICON: Set[str] = {
    "кітап", "мектеп", "кәсіп", "талап", "себеп", "жауап", "сұрақ", "тарақ",
    "парақ", "қонақ", "бұлақ", "құлақ", "тамақ", "сабақ", "одақ", "аймақ",
    "ұрпақ", "шапақ", "туысқандық", "бостандық", "жаңалық", "құжаттық",
    "жүрек", "тілек", "көмек", "еңбек", "бөбек", "терек", "кезек", "дерек",
    "бөлшек", "келешек", "төсек", "көйлек", "жиек", "cүйек", "сүйек",
    "аяқ", "таяқ", "бояқ", "қаймақ", "ошақ", "пышақ", "ұшақ", "қасық",
    "жарық", "құрық", "балық", "тауық", "құдық", "жастық", "орындық",
    "кірпік", "етік", "есік", "бесік", "тесік", "көлік", "бөлік", "серік",
    "жидек", "кебек", "шелек", "өрнек", "тірек"
}

# Canonical Kazakh roots that undergo second-syllable vowel syncope before vowel suffixes
SYNCOPE_STEM_MAP = {
    "халқ": "халық",
    "орн": "орын",
    "мұрн": "мұрын",
    "қарн": "қарын",
    "мойын": "мойын",
    "мойн": "мойын",
    "ерн": "ерін",
    "ауыз": "ауыз",
    "ауз": "ауыз",
    "баур": "бауыр",
    "көңл": "көңіл",
    "өмр": "өмір",
}

DEVOICING_MAP = {
    "б": "п",
    "ғ": "қ",
    "г": "к",
}


class MorphophonologicalRestorer:
    """Restores underlying canonical lemmas from surface allomorphic stems."""

    def __init__(self, extra_voicing_roots: Optional[Set[str]] = None):
        self.voicing_lexicon = set(VOICING_ALTERNATION_LEXICON)
        if extra_voicing_roots:
            self.voicing_lexicon.update(r.lower() for r in extra_voicing_roots)

    def restore_stem(
        self,
        raw_stem: str,
        first_suffix: str,
        custom_roots: Optional[Set[str]] = None
    ) -> str:
        if not raw_stem or not first_suffix:
            return raw_stem

        sfx_clean = first_suffix.lstrip("-").lower()
        if not sfx_clean or sfx_clean[0] not in VOWEL_INITIAL_SUFFIX_STARTS:
            return raw_stem

        stem_lower = raw_stem.lower()
        active_custom = {r.lower() for r in custom_roots} if custom_roots else set()

        # If raw_stem is already a registered custom root (e.g. 'вайб', 'клуб'), preserve it
        if stem_lower in active_custom:
            return raw_stem

        # 1. Check vowel syncope alternation (e.g. халқ + ы -> халық)
        if stem_lower in SYNCOPE_STEM_MAP:
            restored = SYNCOPE_STEM_MAP[stem_lower]
            return self._preserve_case(raw_stem, restored)

        # 2. Check consonant voicing alternation (б->п, ғ->қ, г->к)
        last_char = stem_lower[-1]
        if last_char in DEVOICING_MAP:
            candidate_devoiced = stem_lower[:-1] + DEVOICING_MAP[last_char]
            # Restore if candidate is in known voicing lexicon, active custom roots,
            # or ends in productive derivational suffixes (-лық/-лік/-дық/-дік/-тық/-тік)
            is_derivational = candidate_devoiced.endswith(
                ("лық", "лік", "дық", "дік", "тық", "тік")
            ) and len(candidate_devoiced) >= 5
            if (
                candidate_devoiced in self.voicing_lexicon
                or candidate_devoiced in active_custom
                or is_derivational
            ):
                return self._preserve_case(raw_stem, candidate_devoiced)

        return raw_stem

    @staticmethod
    def _preserve_case(original: str, restored_lower: str) -> str:
        if original.isupper():
            return restored_lower.upper()
        if original[0].isupper():
            return restored_lower.capitalize()
        return restored_lower
