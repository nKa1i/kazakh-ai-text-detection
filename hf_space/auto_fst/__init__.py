"""Auto-FST: Adaptive Neuro-Symbolic Morphology for Informal Kazakh."""

from .normalizer import RussianLayoutNormalizer, SUFFIX_EQUIVALENCES
from .inducer import AutoFSTInducer, InductionResult
from .overlay import DynamicFSTOverlay
from .guard import RegressionGuard, GOLDEN_SENTENCES

__all__ = [
    "RussianLayoutNormalizer",
    "SUFFIX_EQUIVALENCES",
    "AutoFSTInducer",
    "InductionResult",
    "DynamicFSTOverlay",
    "RegressionGuard",
    "GOLDEN_SENTENCES",
]
