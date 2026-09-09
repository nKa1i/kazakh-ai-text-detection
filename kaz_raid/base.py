"""
Kaz-RAID: Kazakh Adversarial Robustness Benchmark & Perturbation Engine
Base Perturbator abstract class and utility functions.
"""

from __future__ import annotations

import abc
import math
import random
from typing import Optional


def calculate_budget(n_eligible: int, rate: float) -> int:
    """
    Compute perturbation budget:
        Target count = max(1, ceil(N_eligible * rate)) if rate > 0 and N_eligible > 0 else 0
    Capped at N_eligible.
    """
    if rate > 0.0 and n_eligible > 0:
        target = math.ceil(n_eligible * rate)
        return min(n_eligible, max(1, target))
    return 0


def preserve_case(original: str, replacement: str) -> str:
    """
    Preserve the case pattern of `original` onto `replacement`.
    - ALL UPPERCASE: replacement.upper()
    - all lowercase: replacement.lower()
    - Title Case: replacement.title()
    - default: replacement
    """
    if not original or not replacement:
        return replacement
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    if original.istitle():
        return replacement.title()
    return replacement


class BasePerturbator(abc.ABC):
    """
    Abstract base class for all Kaz-RAID adversarial perturbators.

    Attributes:
        name (str): Identifier name of the perturbation operator.
        tier (str): Linguistic tier (e.g., 'tier1_orthographic', 'tier2_morphological').
    """

    def __init__(self, name: str, tier: str):
        self.name = name
        self.tier = tier

    @abc.abstractmethod
    def perturb(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        """
        Perturbs text with a target budget rate in {0.05, 0.10, 0.20}.

        Guarantees:
        1. Deterministic output given the same seed.
        2. Minimum 1 perturbation on short texts when rate > 0 and eligible targets exist.
        3. Preservation of capitalization, punctuation, and whitespace.

        Args:
            text (str): Input text string.
            rate (float): Budget rate fraction.
            seed (int): Random number generator seed.

        Returns:
            str: Adversarially perturbed text string.
        """
        raise NotImplementedError

    def __call__(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        """Callable shorthand for perturb()."""
        return self.perturb(text, rate=rate, seed=seed)

    @staticmethod
    def calculate_budget(n_eligible: int, rate: float) -> int:
        """Calculate perturbation count from eligible targets and budget rate."""
        return calculate_budget(n_eligible, rate)

    @staticmethod
    def get_rng(seed: int = 42) -> random.Random:
        """Return deterministic random.Random instance seeded with `seed`."""
        return random.Random(seed)

    @staticmethod
    def preserve_case(original: str, replacement: str) -> str:
        """Preserve case helper."""
        return preserve_case(original, replacement)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', tier='{self.tier}')"
