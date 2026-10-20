"""Automated regression testing sandbox for candidate root induction."""

from typing import List, Tuple

try:
    from .overlay import DynamicFSTOverlay
except (ImportError, ValueError):
    try:
        from auto_fst.overlay import DynamicFSTOverlay
    except ImportError:
        from src.auto_fst.overlay import DynamicFSTOverlay

GOLDEN_SENTENCES: List[str] = [
    "Кеше кешке орталыққа келгенмін және жұмыстарды көргендіктен қайтарды.",
    "Каспиден доставкасы өте тез болды, оплатасын жасадым.",
    "Бұл жазбаларыңыздан AI арқылы жасалғандықтан, оларды тексеру керек.",
    "Балалардың мектептегі оқулықтары өте қызықты.",
    "Тапсырыс берілген тауарлар қоймадан дер кезінде жеткізілді."
]

RESERVED_NATIVE_ROOTS = {
    "бол", "кел", "кет", "ал", "бер", "бар", "көр", "айт", "жаз", "оқы",
    "мен", "сен", "ол", "біз", "сіз", "олар", "бұл", "сол", "осы", "не", "кім",
    "және", "бірақ", "үшін", "туралы", "арқылы", "дейін", "кейін"
}


class RegressionGuard:
    def __init__(self):
        self.baseline_overlay = DynamicFSTOverlay()
        self.golden_parses = [self.baseline_overlay.analyze_and_segment(s) for s in GOLDEN_SENTENCES]

    def validate_candidate(self, candidate_root: str) -> Tuple[bool, str]:
        c_clean = candidate_root.strip().lower()
        if len(c_clean) < 3:
            return False, f"Candidate '{c_clean}' does not meet minimum length requirement (>= 3 chars)."

        if c_clean in RESERVED_NATIVE_ROOTS:
            return False, f"Reserved native root collision: '{c_clean}' is a protected grammatical word."

        # Sandbox testing with isolated overlay
        sandbox_overlay = DynamicFSTOverlay()
        sandbox_overlay.register_root(c_clean)

        for orig_sentence, expected_parse in zip(GOLDEN_SENTENCES, self.golden_parses):
            sandbox_parse = sandbox_overlay.analyze_and_segment(orig_sentence)
            if sandbox_parse != expected_parse:
                return False, f"Regression detected on sentence '{orig_sentence}': Expected '{expected_parse}', got '{sandbox_parse}'."

        return True, "Validation passed: 100% agreement on golden baseline."
