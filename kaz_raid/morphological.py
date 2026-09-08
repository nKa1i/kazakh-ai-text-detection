"""
Kaz-RAID: Tier 2 Morphological Perturbation Attacks
Implements:
1. SuffixTamperer: Inflectional suffix stripping and front/back vowel harmony inversion using FST.
2. ColloquialContractor: Spoken and SMS review colloquial verb contractions.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple

from kaz_raid.base import BasePerturbator, preserve_case

try:
    from fst_analyzer import AdvancedKazakhFSTAnalyzer
except ImportError:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from fst_analyzer import AdvancedKazakhFSTAnalyzer


def preserve_phrase_case(original: str, replacement: str) -> str:
    """
    Preserve case of single-word or multi-word phrase `original` onto `replacement`.
    Handles UPPERCASE, lowercase, Title Case, and Sentence case.
    """
    if not original or not replacement:
        return replacement
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    if original.istitle():
        return replacement.title()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


class SuffixTamperer(BasePerturbator):
    """
    Attacks Kazakh agglutinative morphology by:
    - Stripping inflectional case, plural, or possessive suffixes (mode="strip").
    - Inverting front/back vowel harmony on suffixes (mode="harmony").
    - Randomly choosing between stripping and harmony inversion (mode="mixed").
    Integrates AdvancedKazakhFSTAnalyzer with regex fallback for robust suffix extraction.
    """

    HARMONY_MAP: ClassVar[Dict[str, str]] = {
        # Plural endings (Көптік)
        "лар": "лер", "лер": "лар",
        "дар": "дер", "дер": "дар",
        "тар": "тер", "тер": "тар",
        # Ablative case (Шығыс)
        "дан": "ден", "ден": "дан",
        "тан": "тен", "тен": "тан",
        "нан": "нен", "нен": "нан",
        # Locative case (Жатыс)
        "да": "де", "де": "да",
        "та": "те", "те": "та",
        "нда": "нде", "нде": "нда",
        # Dative case (Барыс)
        "ға": "ге", "ге": "ға",
        "қа": "ке", "ке": "қа",
        "на": "не", "не": "на",
        # Accusative case (Табыс)
        "ны": "ні", "ні": "ны",
        "ты": "ті", "ті": "ты",
        "ды": "ді", "ді": "ды",
        # Genitive case (Ілік)
        "ның": "нің", "нің": "ның",
        "дың": "дің", "дің": "дың",
        "тың": "тің", "тің": "тың",
        # Personal agreement endings (Жіктік)
        "мын": "мін", "мін": "мын",
        "бын": "бін", "бін": "бын",
        "пын": "пін", "пін": "пын",
        "мыз": "міз", "міз": "мыз",
        "быз": "біз", "біз": "быз",
        "пыз": "піз", "піз": "пыз",
        "сың": "сің", "сің": "сың",
        "сыздар": "сіздер", "сіздер": "сыздар",
        # Verbal tense / participle suffixes
        "ған": "ген", "ген": "ған",
        "қан": "кен", "кен": "қан",
        "ғандықтан": "гендіктен", "гендіктен": "ғандықтан",
        "қандықтан": "кендіктен", "кендіктен": "қандықтан",
        "атын": "етін", "етін": "атын",
        "йтын": "йтін", "йтін": "йтын",
        "ады": "еді", "еді": "ады",
        "йды": "йді", "йді": "йды",
        "мақ": "мек", "мек": "мақ",
        "бақ": "бек", "бек": "бақ",
        "пақ": "пек", "пек": "пақ",
        # Possessive endings
        "лары": "лері", "лері": "лары",
        "дары": "дері", "дері": "дары",
        "тары": "тері", "тері": "тары",
        "сы": "сі", "сі": "сы",
        "ымыз": "іміз", "іміз": "ымыз",
        "ыңыз": "іңіз", "іңіз": "ыңыз",
    }

    FALLBACK_SUFFIXES: ClassVar[List[str]] = sorted(
        [
            "ғандықтан", "гендіктен", "қандықтан", "кендіктен",
            "тарымыз", "теріміз", "дарымыз", "деріміз", "ларымыз", "леріміз",
            "тарыңыз", "теріңіз", "дарыңыз", "деріңіз", "ларыңыз", "леріңіз",
            "сыздар", "сіздер",
            "атын", "етін", "йтын", "йтін",
            "ған", "ген", "қан", "кен",
            "ады", "еді", "йды", "йді",
            "мақ", "мек", "бақ", "бек", "пақ", "пек",
            "мын", "мін", "бын", "бін", "пын", "пін",
            "мыз", "міз", "быз", "біз", "пыз", "піз",
            "сың", "сің",
            "ның", "нің", "дың", "дің", "тың", "тің",
            "дан", "ден", "тан", "тен", "нан", "нен",
            "мен", "бен", "пен",
            "нда", "нде",
            "лар", "лер", "дар", "дер", "тар", "тер",
            "дары", "дері", "тары", "тері", "лары", "лері",
            "ымыз", "іміз", "ыңыз", "іңіз",
            "ға", "ге", "қа", "ке", "на", "не",
            "ны", "ні", "ды", "ді", "ты", "ті",
            "да", "де", "та", "те",
            "сы", "сі",
        ],
        key=len,
        reverse=True,
    )

    WORD_REGEX: ClassVar[re.Pattern] = re.compile(r"\b[^\W\d_]+\b")

    def __init__(
        self,
        mode: str = "mixed",
        name: str = "suffix_tamperer",
        tier: str = "tier2_morphological",
    ):
        super().__init__(name=name, tier=tier)
        mode = mode.lower().strip()
        if mode not in {"strip", "harmony", "mixed"}:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'strip', 'harmony', or 'mixed'.")
        self.mode = mode
        self.fst = AdvancedKazakhFSTAnalyzer()

    def extract_suffixes(self, word: str) -> List[str]:
        """
        Extract suffix list from word using FST analyzer, with regex fallback.
        Returns list of lowercase suffixes without leading hyphens.
        """
        w_lower = word.lower()
        seg = self.fst.analyze_and_segment(w_lower)
        tokens = seg.split()
        suffixes: List[str] = []
        if len(tokens) > 1:
            for t in tokens[1:]:
                if t.startswith("-") and len(t) > 1:
                    suffixes.append(t[1:])

        # Fallback if FST returned no affixes
        if not suffixes:
            for sfx in self.FALLBACK_SUFFIXES:
                if w_lower.endswith(sfx) and len(w_lower) - len(sfx) >= 2:
                    suffixes.append(sfx)
                    break
        return suffixes

    def get_suffix_spans(self, word: str, suffixes: List[str]) -> List[Tuple[int, int, str]]:
        """
        Map extracted suffixes to character slice spans [start, end) within word.
        Traverses from right to left to ensure exact suffix offsets.
        """
        spans: List[Tuple[int, int, str]] = []
        w_lower = word.lower()
        curr = len(word)
        for sfx in reversed(suffixes):
            if w_lower[:curr].endswith(sfx):
                start = curr - len(sfx)
                spans.append((start, curr, sfx))
                curr = start
            else:
                break
        spans.reverse()
        return spans

    def _is_eligible(self, word: str, spans: List[Tuple[int, int, str]]) -> bool:
        if not spans:
            return False
        can_strip = spans[-1][0] >= 2
        can_harmony = any(sfx in self.HARMONY_MAP for _, _, sfx in spans)

        if self.mode == "strip":
            return can_strip
        elif self.mode == "harmony":
            return can_harmony
        else:  # mixed
            return can_strip or can_harmony

    def perturb(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        if not text or rate <= 0.0:
            return text

        # Find all candidate words and their suffix spans
        candidates: List[Tuple[re.Match, List[Tuple[int, int, str]]]] = []
        for m in self.WORD_REGEX.finditer(text):
            word = m.group(0)
            suffixes = self.extract_suffixes(word)
            spans = self.get_suffix_spans(word, suffixes)
            if self._is_eligible(word, spans):
                candidates.append((m, spans))

        budget = self.calculate_budget(len(candidates), rate)
        if budget <= 0:
            return text

        rng = self.get_rng(seed)
        chosen_candidates = rng.sample(candidates, budget)

        # Sort descending by start index to mutate from right to left
        chosen_candidates.sort(key=lambda item: item[0].start(), reverse=True)

        for match_obj, spans in chosen_candidates:
            word = match_obj.group(0)
            can_strip = spans[-1][0] >= 2
            invertible = [sp for sp in spans if sp[2] in self.HARMONY_MAP]
            can_harmony = len(invertible) > 0

            if self.mode == "strip":
                action = "strip"
            elif self.mode == "harmony":
                action = "harmony"
            else:  # mixed
                if can_strip and can_harmony:
                    action = rng.choice(["strip", "harmony"])
                elif can_strip:
                    action = "strip"
                else:
                    action = "harmony"

            if action == "strip":
                start_sfx, end_sfx, _ = spans[-1]
                new_word = word[:start_sfx]
            else:
                # Harmony inversion
                chosen_span = rng.choice(invertible)
                start_sfx, end_sfx, orig_sfx = chosen_span
                repl_sfx = self.HARMONY_MAP[orig_sfx]
                cased_repl = preserve_case(word[start_sfx:end_sfx], repl_sfx)
                new_word = word[:start_sfx] + cased_repl + word[end_sfx:]

            text = text[:match_obj.start()] + new_word + text[match_obj.end():]

        return text


class ColloquialContractor(BasePerturbator):
    """
    Substitutes formal Kazakh verbal structures and phrases with common
    spoken and SMS review contractions (e.g. 'жатырмын' -> 'жатырм',
    'келе жатыр' -> 'кеватр', 'келемін' -> 'келем', 'болып жатыр' -> 'боватр').
    Preserves capitalization and punctuation.
    """

    COLLOQUIAL_PATTERNS: ClassVar[List[Tuple[str, str]]] = [
        # Multi-word verb phrases (sorted by descending length)
        ("деген болатын", "деген еді"),
        ("айтқан болатын", "айтқан еді"),
        ("көрген жоқпын", "көрмедім"),
        ("алған жоқпын", "алмадым"),
        ("келген жоқпын", "келмедім"),
        ("барған жоқпын", "бармадым"),
        ("жасаған жоқпын", "жасамадым"),
        ("білген жоқпын", "білмедім"),
        ("болып жатыр", "боватр"),
        ("келе жатыр", "кеватр"),
        ("бара жатыр", "баратр"),
        ("жасап жатыр", "жасаватр"),
        ("айтып жатыр", "айтаватр"),
        ("беріп жатыр", "бераватр"),
        ("көріп жатыр", "көраватр"),
        ("алып келді", "әпкелді"),
        ("алып кетті", "әпкетті"),
        ("алып кел", "әпкел"),
        ("алып кет", "әпкет"),
        ("алып қойды", "апқойды"),
        ("не істеп", "нестеп"),
        ("не қылған", "неғылған"),
        ("рахмет сізге", "рахмет сізке"),
        ("қандай да бір", "қандайда бір"),
        # Single-word verb forms & personal agreement contractions
        ("жатырмын", "жатырм"),
        ("жатырсың", "жатырсын"),
        ("жатырмыз", "жатырз"),
        ("келемін", "келем"),
        ("барамын", "барам"),
        ("аламын", "алам"),
        ("беремін", "берем"),
        ("көремін", "көрем"),
        ("айтамын", "айтам"),
        ("білемін", "білем"),
        ("жасаймын", "жасайм"),
        ("істеймін", "істейм"),
        ("келесің", "келесін"),
        ("барасың", "барасын"),
        ("келеді", "келед"),
        ("барады", "барад"),
        ("болады", "болад"),
        ("болған", "боған"),
        ("қалған", "қаған"),
        ("болды", "боды"),
        ("жоқпын", "жоқпым"),
    ]

    def __init__(
        self,
        name: str = "colloquial_contractor",
        tier: str = "tier2_morphological",
    ):
        super().__init__(name=name, tier=tier)
        self.PATTERNS = self.COLLOQUIAL_PATTERNS
        self._compiled_patterns: List[Tuple[re.Pattern, str]] = [
            (re.compile(r"\b" + re.escape(formal) + r"\b", re.IGNORECASE), repl)
            for formal, repl in self.COLLOQUIAL_PATTERNS
        ]

    def perturb(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        if not text or rate <= 0.0:
            return text

        # Find non-overlapping matches across all patterns
        covered: set[int] = set()
        matches: List[Tuple[int, int, str, str]] = []

        for regex, repl in self._compiled_patterns:
            for m in regex.finditer(text):
                span_set = set(range(m.start(), m.end()))
                if not (span_set & covered):
                    covered |= span_set
                    matches.append((m.start(), m.end(), m.group(0), repl))

        budget = self.calculate_budget(len(matches), rate)
        if budget <= 0:
            return text

        rng = self.get_rng(seed)
        chosen_matches = rng.sample(matches, budget)

        # Replace from right to left
        chosen_matches.sort(key=lambda item: item[0], reverse=True)
        for start, end, orig_text, repl in chosen_matches:
            cased_repl = preserve_phrase_case(orig_text, repl)
            text = text[:start] + cased_repl + text[end:]

        return text
