"""
Kaz-RAID: Kazakh Adversarial Robustness Benchmark & Perturbation Engine.
Unified Linguistic-Hierarchy Attack Framework across 4 Tiers (9 Operators).
"""

from typing import Dict

from kaz_raid.base import BasePerturbator, calculate_budget, preserve_case
from kaz_raid.code_switch import DiscourseParticle, LoanwordSwap
from kaz_raid.morphological import ColloquialContractor, SuffixTamperer
from kaz_raid.orthographic import HomoglyphSwap, KeyboardTypo, ZeroWidthInjection
from kaz_raid.semantic import LLMParaphraser, RoundTripTranslator


def get_all_operators() -> Dict[str, BasePerturbator]:
    """
    Returns unified benchmark registry mapping operator identifiers to instantiated perturbators.
    Tier 1 (Orthographic): homoglyph_swap, keyboard_typo, zero_width_injection
    Tier 2 (Morphological): suffix_tamperer, colloquial_contractor
    Tier 3 (Lexical / Code-Switching): loanword_swap, discourse_particle
    Tier 4 (Semantic Interfaces): back_translation, llm_paraphrase
    """
    return {
        "homoglyph_swap": HomoglyphSwap(),
        "keyboard_typo": KeyboardTypo(),
        "zero_width_injection": ZeroWidthInjection(),
        "suffix_tamperer": SuffixTamperer(),
        "colloquial_contractor": ColloquialContractor(),
        "loanword_swap": LoanwordSwap(),
        "discourse_particle": DiscourseParticle(),
        "back_translation": RoundTripTranslator(),
        "llm_paraphrase": LLMParaphraser(),
    }


__all__ = [
    "BasePerturbator",
    "calculate_budget",
    "preserve_case",
    # Tier 1
    "HomoglyphSwap",
    "KeyboardTypo",
    "ZeroWidthInjection",
    # Tier 2
    "SuffixTamperer",
    "ColloquialContractor",
    # Tier 3
    "LoanwordSwap",
    "DiscourseParticle",
    # Tier 4
    "RoundTripTranslator",
    "LLMParaphraser",
    # Registry
    "get_all_operators",
]
