"""
Kaz-RAID: Tier 1 Character / Orthographic Perturbation Attacks
Implements:
1. HomoglyphSwap: Cyrillic-to-Latin visual lookalike substitution.
2. KeyboardTypo: Mobile typing diacritic drop/substitution and keyboard adjacency.
3. ZeroWidthInjection: Invisible Unicode codepoints (\u200B, \u200C) inside word stems.
"""

from __future__ import annotations

import re
from typing import ClassVar, Dict, List

from kaz_raid.base import BasePerturbator


class HomoglyphSwap(BasePerturbator):
    """
    Substitutes Cyrillic letters with visually indistinguishable Latin characters.
    Commonly used in Kazakh academia and social media to evade plagiarism and automated filters.
    """

    HOMOGLYPH_MAP: ClassVar[Dict[str, str]] = {
        # Lowercase
        "а": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "х": "x",
        "у": "y",
        "і": "i",
        # Uppercase
        "А": "A",
        "Е": "E",
        "О": "O",
        "Р": "P",
        "С": "C",
        "Х": "X",
        "У": "Y",
        "І": "I",
    }

    def __init__(self, name: str = "homoglyph_swap", tier: str = "tier1_orthographic"):
        super().__init__(name=name, tier=tier)

    def perturb(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        if not text or rate <= 0.0:
            return text

        eligible_indices = [
            i for i, ch in enumerate(text) if ch in self.HOMOGLYPH_MAP
        ]
        budget = self.calculate_budget(len(eligible_indices), rate)
        if budget <= 0:
            return text

        rng = self.get_rng(seed)
        chosen_indices = set(rng.sample(eligible_indices, budget))

        chars = list(text)
        for idx in chosen_indices:
            orig = chars[idx]
            chars[idx] = self.HOMOGLYPH_MAP[orig]

        return "".join(chars)


class KeyboardTypo(BasePerturbator):
    """
    Simulates mobile Kazakh keyboard typing errors, specifically:
    - Diacritic omissions and confusion (қ↔к, ң↔н, ғ↔г, ү↔у, ұ↔у, ә↔а, ө↔о, h↔х).
    - Physical key adjacency swaps on Kazakh standard layouts for other characters.
    Preserves original case.
    """

    DIACRITIC_MAP: ClassVar[Dict[str, List[str]]] = {
        "қ": ["к"],
        "к": ["қ"],
        "ң": ["н"],
        "н": ["ң"],
        "ғ": ["г"],
        "г": ["ғ"],
        "ү": ["у"],
        "ұ": ["у"],
        "у": ["ү", "ұ"],
        "ә": ["а"],
        "а": ["ә"],
        "ө": ["о"],
        "о": ["ө"],
        "h": ["х"],
        "һ": ["х"],
        "х": ["h", "һ"],
    }

    ADJACENT_MAP: ClassVar[Dict[str, List[str]]] = {
        "й": ["ц", "ф"],
        "ц": ["й", "у", "ы"],
        "у": ["ц", "к"],
        "к": ["у", "е"],
        "е": ["к", "н"],
        "н": ["е", "г"],
        "г": ["н", "ш"],
        "ш": ["г", "щ"],
        "щ": ["ш", "з"],
        "з": ["щ", "х"],
        "х": ["з"],
        "ф": ["й", "ы"],
        "ы": ["ф", "в"],
        "в": ["ы", "а"],
        "а": ["в", "п"],
        "п": ["а", "р"],
        "р": ["п", "о"],
        "о": ["р", "л"],
        "л": ["о", "д"],
        "д": ["л", "ж"],
        "ж": ["д", "э"],
        "э": ["ж"],
        "я": ["ч", "с"],
        "ч": ["я", "с"],
        "с": ["ч", "м"],
        "м": ["с", "и"],
        "и": ["м", "т"],
        "т": ["и", "ь"],
        "ь": ["т", "б"],
        "б": ["ь", "ю"],
        "ю": ["б"],
    }

    def __init__(
        self,
        include_adjacent: bool = True,
        name: str = "keyboard_typo",
        tier: str = "tier1_orthographic",
    ):
        super().__init__(name=name, tier=tier)
        self.include_adjacent = include_adjacent

    def _get_candidates(self, ch: str) -> List[str]:
        lower = ch.lower()
        if lower in self.DIACRITIC_MAP:
            return self.DIACRITIC_MAP[lower]
        if self.include_adjacent and lower in self.ADJACENT_MAP:
            return self.ADJACENT_MAP[lower]
        return []

    def perturb(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        if not text or rate <= 0.0:
            return text

        eligible_indices = [
            i for i, ch in enumerate(text) if self._get_candidates(ch)
        ]
        budget = self.calculate_budget(len(eligible_indices), rate)
        if budget <= 0:
            return text

        rng = self.get_rng(seed)
        chosen_indices = sorted(rng.sample(eligible_indices, budget))

        chars = list(text)
        for idx in chosen_indices:
            orig = chars[idx]
            candidates = self._get_candidates(orig)
            chosen_repl = rng.choice(candidates)
            chars[idx] = self.preserve_case(orig, chosen_repl)

        return "".join(chars)


class ZeroWidthInjection(BasePerturbator):
    """
    Injects invisible zero-width Unicode codepoints (\u200B zero-width space,
    \u200C zero-width non-joiner) between characters inside word stems.
    Forces SentencePiece / BPE tokenizers to split words into byte-fallback fragments.
    """

    ZERO_WIDTH_CHARS: ClassVar[List[str]] = [
        "\u200B",  # Zero-width space
        "\u200C",  # Zero-width non-joiner
    ]

    # Regex matching alphabetic words (excluding numbers and underscores)
    WORD_REGEX: ClassVar[re.Pattern] = re.compile(r"\b[^\W\d_]+\b")

    def __init__(
        self,
        name: str = "zero_width_injection",
        tier: str = "tier1_orthographic",
    ):
        super().__init__(name=name, tier=tier)

    def perturb(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        if not text or rate <= 0.0:
            return text

        eligible_positions: List[int] = []
        for match in self.WORD_REGEX.finditer(text):
            start, end = match.start(), match.end()
            if end - start >= 2:
                # Interior positions: strictly after char i where start <= i < end - 1
                eligible_positions.extend(range(start, end - 1))

        budget = self.calculate_budget(len(eligible_positions), rate)
        if budget <= 0:
            return text

        rng = self.get_rng(seed)
        chosen_positions = sorted(rng.sample(eligible_positions, budget))
        injections = {
            pos: rng.choice(self.ZERO_WIDTH_CHARS) for pos in chosen_positions
        }

        chars = list(text)
        # Insert from right to left so earlier indices remain untouched
        for pos in sorted(injections.keys(), reverse=True):
            chars.insert(pos + 1, injections[pos])

        return "".join(chars)
