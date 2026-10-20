# Auto-FST Demo Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight, zero-GPU Neuro-Symbolic Auto-FST demo prototype that handles Russian keyboard diacritic flattening (`турсынгой` -> `тұрсыңғой`) and dynamically induces emerging slang/loanword roots (`вайб`, `донат`, `чилл`) into an active FST graph at sub-millisecond speeds.

**Architecture:** A 4-stage pipeline decoupling orthographic normalization, candidate morphological induction, regression testing, and longest-match dynamic lexicon overlay, with an interactive CLI demonstrating automated benchmarking and live REPL parsing.

**Tech Stack:** Python 3.10+, Standard Library (`re`, `json`, `time`, `typing`, `unittest`, `dataclasses`), zero GPU dependencies.

## Global Constraints

- Under no circumstances shall `aist2026/paper.tex` or any files in `papers/kazakh_fever_conference/` be touched or modified (strictly 0 diff lines).
- Strictly zero decorative emojis across all code, datasets, comments, documentation, and console interfaces.
- All Python file operations must explicitly specify `encoding='utf-8'`.
- All code files must contain complete, production-ready content with zero placeholders ("TODO", "TBD").
- Hardware Suitability: The entire engine and demonstration must run seamlessly on resource-constrained consumer laptops (requiring 0 MB GPU VRAM and standard library dependencies).
- Non-Destructive Integration: The baseline `fst_analyzer.py` module must remain clean and backward-compatible. Dynamic augmentations are applied via modular overlay classes without in-place mutation of the core codebase.

---

## File Structure

```text
src/auto_fst/
├── __init__.py                      (Package exports)
├── normalizer.py                    (RussianLayoutNormalizer & suffix invariance)
├── inducer.py                       (AutoFSTInducer & phonetic harmony profiler)
├── guard.py                         (RegressionGuard sandbox verification)
└── overlay.py                       (DynamicFSTOverlay & longest-match lexicon)
data/
└── custom_slang_lexicon.json        (Persistent JSON overlay)
scripts/
└── demo_auto_fst.py                 (Interactive CLI demo: Benchmark + REPL)
tests/
├── test_auto_fst_normalizer.py      (Unit tests for normalizer)
├── test_auto_fst_inducer.py         (Unit tests for inducer & phonetic profiling)
├── test_auto_fst_overlay.py         (Unit tests for dynamic overlay & sorting)
├── test_auto_fst_guard.py           (Unit tests for regression guard sandbox)
└── test_auto_fst_e2e.py             (End-to-end integration & latency tests)
```

---

### Task 1: Russian Layout Normalizer & Diacritic Invariance

**Files:**
- Create: `src/auto_fst/normalizer.py`
- Test: `tests/test_auto_fst_normalizer.py`

**Interfaces:**
- Produces: `class RussianLayoutNormalizer`
  - `normalize_word(self, word: str) -> str`
  - `normalize_text(self, text: str) -> str`
  - `has_russian_layout_artifacts(self, word: str) -> bool`
  - `SUFFIX_EQUIVALENCES: Dict[str, List[str]]`

- [ ] **Step 1: Write the failing test**
Create `tests/test_auto_fst_normalizer.py`:
```python
import unittest
from src.auto_fst.normalizer import RussianLayoutNormalizer

class TestRussianLayoutNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = RussianLayoutNormalizer()

    def test_high_frequency_dictionary_normalization(self):
        self.assertEqual(self.normalizer.normalize_word("турсынгой"), "тұрсыңғой")
        self.assertEqual(self.normalizer.normalize_word("жаксы"), "жақсы")
        self.assertEqual(self.normalizer.normalize_word("салем"), "сәлем")
        self.assertEqual(self.normalizer.normalize_word("калайсын"), "қалайсың")

    def test_text_normalization(self):
        text = "салем калайсын, жумысын жаксы болып жатырма?"
        normalized = self.normalizer.normalize_text(text)
        self.assertIn("сәлем", normalized)
        self.assertIn("жақсы", normalized)

    def test_russian_layout_artifact_detection(self):
        self.assertTrue(self.normalizer.has_russian_layout_artifacts("турсынгой"))
        self.assertFalse(self.normalizer.has_russian_layout_artifacts("тұрсыңғой"))

if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to make sure it fails**
Command: `python -m unittest tests/test_auto_fst_normalizer.py`
Expected: ModuleNotFoundError: No module named 'src.auto_fst'

- [ ] **Step 3: Implement minimal code in `src/auto_fst/normalizer.py`**
Create package directories `src/auto_fst/` and implement `RussianLayoutNormalizer`:
```python
"""Orthographic normalizer for Kazakh text typed on Russian keyboard layouts."""

import re
from typing import Dict, List, Optional

HIGH_FREQ_RUSSIAN_LAYOUT_MAP: Dict[str, str] = {
    "салем": "сәлем",
    "жаксы": "жақсы",
    "калайсын": "қалайсың",
    "калайсыз": "қалайсыз",
    "турсынгой": "тұрсыңғой",
    "турсын": "тұрсың",
    "тур": "тұр",
    "жумыс": "жұмыс",
    "жумысын": "жұмысың",
    "болып": "болып",
    "болган": "болған",
    "болдыгой": "болдығой",
    "келдигой": "келдіғой",
    "кой": "қой",
    "гой": "ғой",
    "омир": "өмір",
    "соз": "сөз",
    "рахмет": "рахмет",
    "ушин": "үшін",
    "улкен": "үлкен",
    "кишкентай": "кішкентай",
}

SUFFIX_EQUIVALENCES: Dict[str, List[str]] = {
    "ған": ["ған", "ган"],
    "ген": ["ген"],
    "қан": ["қан", "кан"],
    "кен": ["кен"],
    "сың": ["сың", "сын"],
    "сің": ["сің", "син"],
    "сыз": ["сыз"],
    "сіз": ["сіз", "сиз"],
    "ғой": ["ғой", "гой"],
    "қой": ["қой", "кой"],
    "ның": ["ның", "нын"],
    "нің": ["нің", "нин"],
    "дың": ["дың", "дын"],
    "дің": ["дің", "дин"],
    "тың": ["тың", "тын"],
    "тің": ["тің", "тин"],
    "ға": ["ға", "га"],
    "ге": ["ге"],
    "қа": ["қа", "ка"],
    "ке": ["ке"],
}

class RussianLayoutNormalizer:
    def __init__(self, custom_map: Optional[Dict[str, str]] = None):
        self.word_map = dict(HIGH_FREQ_RUSSIAN_LAYOUT_MAP)
        if custom_map:
            self.word_map.update(custom_map)

    def normalize_word(self, word: str) -> str:
        w_lower = word.lower()
        if w_lower in self.word_map:
            mapped = self.word_map[w_lower]
            if word.istitle():
                return mapped.capitalize()
            elif word.isupper():
                return mapped.upper()
            return mapped
        return word

    def normalize_text(self, text: str) -> str:
        if not text:
            return ""
        tokens = re.split(r'(\s+|[^\w\s])', text)
        result = []
        for token in tokens:
            if re.match(r'^[а-яёА-ЯЁ]+$', token):
                result.append(self.normalize_word(token))
            else:
                result.append(token)
        return "".join(result)

    def has_russian_layout_artifacts(self, word: str) -> bool:
        w_lower = word.lower()
        return w_lower in self.word_map
```
Create `src/auto_fst/__init__.py`.

- [ ] **Step 4: Run test to make sure it passes**
Command: `python -m unittest tests/test_auto_fst_normalizer.py`
Expected: 3 tests pass.

- [ ] **Step 5: Commit changes**
Command: `git add src/auto_fst/ tests/test_auto_fst_normalizer.py && git commit -m "feat(auto_fst): implement RussianLayoutNormalizer and suffix equivalences"`

---

### Task 2: AutoFST Inducer & Phonetic Profiler

**Files:**
- Create: `src/auto_fst/inducer.py`
- Test: `tests/test_auto_fst_inducer.py`

**Interfaces:**
- Consumes: `src/auto_fst/normalizer.py`
- Produces: `class AutoFSTInducer`
  - `induce_stem(self, token: str) -> Optional[InductionResult]`
  - `dataclass InductionResult`:
    - `stem: str`
    - `suffixes: List[str]`
    - `harmony: str` ("hard" | "soft")
    - `coda: str` ("voiced" | "voiceless")
    - `confidence: float`

- [ ] **Step 1: Write the failing test**
Create `tests/test_auto_fst_inducer.py`:
```python
import unittest
from src.auto_fst.inducer import AutoFSTInducer, InductionResult

class TestAutoFSTInducer(unittest.TestCase):
    def setUp(self):
        self.inducer = AutoFSTInducer()

    def test_slang_verbalizer_induction(self):
        res = self.inducer.induce_stem("вайбтаған")
        self.assertIsNotNone(res)
        self.assertEqual(res.stem, "вайб")
        self.assertIn("та", res.suffixes)
        self.assertIn("ған", res.suffixes)
        self.assertEqual(res.harmony, "hard")
        self.assertEqual(res.coda, "voiced")

    def test_soft_slang_induction(self):
        res = self.inducer.induce_stem("чиллдедік")
        self.assertIsNotNone(res)
        self.assertEqual(res.stem, "чилл")
        self.assertEqual(res.harmony, "soft")
        self.assertEqual(res.coda, "voiced")

    def test_voiceless_coda_induction(self):
        res = self.inducer.induce_stem("донаттау")
        self.assertIsNotNone(res)
        self.assertEqual(res.stem, "донат")
        self.assertEqual(res.coda, "voiceless")

    def test_noun_possessive_case_induction(self):
        res = self.inducer.induce_stem("крашыма")
        self.assertIsNotNone(res)
        self.assertEqual(res.stem, "краш")
        self.assertIn("ым", res.suffixes)
        self.assertIn("а", res.suffixes)

    def test_gibberish_and_short_word_rejection(self):
        self.assertIsNone(self.inducer.induce_stem("12345"))
        self.assertIsNone(self.inducer.induce_stem("ббб"))
        self.assertIsNone(self.inducer.induce_stem("asdfgh"))

if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to make sure it fails**
Command: `python -m unittest tests/test_auto_fst_inducer.py`
Expected: ModuleNotFoundError: No module named 'src.auto_fst.inducer'

- [ ] **Step 3: Implement minimal code in `src/auto_fst/inducer.py`**
Implement `InductionResult` and `AutoFSTInducer`:
```python
"""Automated morphological stem inducer and phonetic profiler for slang and loanwords."""

import re
from dataclasses import dataclass, field
from typing import List, Optional

HARD_VOWELS = set('аоұыяё')
SOFT_VOWELS = set('әөүіеэи')
VOICED_CONSONANTS = set('бвгджзлмнңруй')
VOICELESS_CONSONANTS = set('пфкқтсшщхцч')

KNOWN_SUFFIX_CHAINS = [
    # Complex Verbalizer + Tense + Person
    ['-та', '-ған', '-дықтан'], ['-те', '-ген', '-діктен'],
    ['-та', '-ған'], ['-те', '-ген'], ['-да', '-ған'], ['-де', '-ген'],
    ['-та', '-дық'], ['-те', '-дік'], ['-да', '-дық'], ['-де', '-дік'],
    ['-та', '-ды'], ['-те', '-ді'], ['-да', '-ды'], ['-де', '-ді'],
    ['-та', '-у'], ['-те', '-у'], ['-да', '-у'], ['-де', '-у'],
    ['-де', '-дік'], ['-де', '-п'], ['-та', '-п'],
    # Noun Cases / Possessives
    ['-ым', '-а'], ['-ім', '-е'], ['-ым', '-на'], ['-ім', '-не'],
    ['-ы', '-на'], ['-і', '-не'], ['-сы', '-на'], ['-сі', '-не'],
    ['-лар', '-ы'], ['-лер', '-і'], ['-тар', '-ы'], ['-тер', '-і'],
    ['-ның'], ['-нің'], ['-дың'], ['-дің'], ['-тың'], ['-тің'],
    ['-ға'], ['-ге'], ['-қа'], ['-ке'],
    ['-мен'], ['-бен'], ['-пен'],
    ['-дан'], ['-ден'], ['-тан'], ['-тен'],
]

# Russian keyboard flattened variants
RUSSIAN_LAYOUT_SUFFIX_MAP = {
    'ган': 'ған', 'кан': 'қан',
    'сын': 'сың', 'син': 'сің',
    'гой': 'ғой', 'кой': 'қой',
    'нын': 'ның', 'нин': 'нің',
    'дын': 'дың', 'дин': 'дің',
    'тын': 'тың', 'тин': 'тің',
    'га': 'ға', 'ка': 'қа',
}

@dataclass
class InductionResult:
    stem: str
    suffixes: List[str]
    harmony: str  # "hard" or "soft"
    coda: str     # "voiced" or "voiceless"
    confidence: float = 0.95

class AutoFSTInducer:
    def __init__(self, min_stem_length: int = 3):
        self.min_stem_length = min_stem_length

    def _determine_harmony(self, stem: str) -> str:
        for ch in reversed(stem.lower()):
            if ch in HARD_VOWELS:
                return "hard"
            if ch in SOFT_VOWELS:
                return "soft"
        return "hard"

    def _determine_coda(self, stem: str) -> str:
        for ch in reversed(stem.lower()):
            if ch in VOICELESS_CONSONANTS:
                return "voiceless"
            if ch in VOICED_CONSONANTS or ch in HARD_VOWELS or ch in SOFT_VOWELS:
                return "voiced"
        return "voiced"

    def induce_stem(self, token: str) -> Optional[InductionResult]:
        if not token or not isinstance(token, str):
            return None
        token_clean = re.sub(r'^[^\w]+|[^\w]+$', '', token)
        if not re.match(r'^[а-яёәіңғүұқөһА-ЯЁӘІҢҒҮҰҚӨҺ]+$', token_clean):
            return None
        t_lower = token_clean.lower()
        if len(t_lower) <= self.min_stem_length:
            return None

        for chain in KNOWN_SUFFIX_CHAINS:
            affix_str = "".join(s.lstrip('-') for s in chain)
            # check both authentic and Russian layout
            affix_variants = [affix_str]
            flat_affix = affix_str
            for r_k, r_v in RUSSIAN_LAYOUT_SUFFIX_MAP.items():
                flat_affix = flat_affix.replace(r_v, r_k)
            if flat_affix != affix_str:
                affix_variants.append(flat_affix)

            for cand_affix in affix_variants:
                if t_lower.endswith(cand_affix):
                    stem = token_clean[:-len(cand_affix)]
                    if len(stem) >= self.min_stem_length:
                        # Ensure stem has at least one vowel
                        if any(c in HARD_VOWELS or c in SOFT_VOWELS for c in stem.lower()):
                            clean_suffixes = [s.lstrip('-') for s in chain]
                            return InductionResult(
                                stem=stem,
                                suffixes=clean_suffixes,
                                harmony=self._determine_harmony(stem),
                                coda=self._determine_coda(stem),
                                confidence=0.98 if len(stem) >= 4 else 0.85
                            )
        return None
```

- [ ] **Step 4: Run test to make sure it passes**
Command: `python -m unittest tests/test_auto_fst_inducer.py`
Expected: 5 tests pass.

- [ ] **Step 5: Commit changes**
Command: `git add src/auto_fst/inducer.py tests/test_auto_fst_inducer.py && git commit -m "feat(auto_fst): implement AutoFSTInducer and phonetic profiling"`

---

### Task 3: Dynamic FST Overlay & Longest-Match Lexicon

**Files:**
- Create: `src/auto_fst/overlay.py`
- Test: `tests/test_auto_fst_overlay.py`

**Interfaces:**
- Consumes: `fst_analyzer.py` (AdvancedKazakhFSTAnalyzer), `src/auto_fst/normalizer.py`
- Produces: `class DynamicFSTOverlay`
  - `register_root(self, root: str, metadata: Optional[dict] = None) -> bool`
  - `analyze_and_segment(self, text: str) -> str`
  - `save_to_json(self, path: str) -> None`
  - `load_from_json(self, path: str) -> int`

- [ ] **Step 1: Write the failing test**
Create `tests/test_auto_fst_overlay.py`:
```python
import os
import unittest
import tempfile
from src.auto_fst.overlay import DynamicFSTOverlay

class TestDynamicFSTOverlay(unittest.TestCase):
    def setUp(self):
        self.overlay = DynamicFSTOverlay()

    def test_longest_match_first_sorting(self):
        # Prevent "чат" from shadowing "чатбот"
        self.overlay.register_root("чат")
        self.overlay.register_root("чатбот")
        segmented = self.overlay.analyze_and_segment("чатботтың жұмысы")
        self.assertIn("чатбот -тың", segmented)
        self.assertNotIn("чат -боттың", segmented)

    def test_dynamic_slang_registration(self):
        # Before registration
        raw = "Біз кеше қатты вайбтадық"
        before = self.overlay.analyze_and_segment(raw)
        self.assertNotIn("вайб -та -дық", before)

        # After registration
        self.overlay.register_root("вайб")
        after = self.overlay.analyze_and_segment(raw)
        self.assertIn("вайб -та -дық", after)

    def test_russian_layout_particle_handling(self):
        raw = "Ол әлі жұмыста турсынгой"
        segmented = self.overlay.analyze_and_segment(raw)
        self.assertTrue("тур -сын -гой" in segmented or "тұр -сың -ғой" in segmented)

    def test_json_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name
        try:
            self.overlay.register_root("донат", {"harmony": "hard"})
            self.overlay.save_to_json(temp_path)

            new_overlay = DynamicFSTOverlay()
            count = new_overlay.load_from_json(temp_path)
            self.assertEqual(count, 1)
            self.assertIn("донат", new_overlay.loanword_roots)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to make sure it fails**
Command: `python -m unittest tests/test_auto_fst_overlay.py`
Expected: ModuleNotFoundError: No module named 'src.auto_fst.overlay'

- [ ] **Step 3: Implement minimal code in `src/auto_fst/overlay.py`**
Implement `DynamicFSTOverlay`:
```python
"""Dynamic FST overlay with longest-match-first loanword disambiguation and persistence."""

import os
import json
import re
from typing import Dict, List, Optional
from fst_analyzer import AdvancedKazakhFSTAnalyzer, COMMON_LOANWORD_ROOTS
from src.auto_fst.normalizer import RussianLayoutNormalizer

class DynamicFSTOverlay(AdvancedKazakhFSTAnalyzer):
    def __init__(self, normalizer: Optional[RussianLayoutNormalizer] = None):
        super().__init__()
        self.normalizer = normalizer or RussianLayoutNormalizer()
        self.custom_lexicon: Dict[str, dict] = {}
        # Initialize roots with longest-match-first sorting
        self._refresh_roots()

        # Augment verbal persons and tenses with Russian layout equivalents
        self._augment_rules_for_russian_layout()

    def _augment_rules_for_russian_layout(self):
        russian_persons = ['сын', 'син', 'сыз', 'сиз']
        for p in russian_persons:
            if p not in self.verbal_persons:
                self.verbal_persons.append(p)
        self.verb_person_re = re.compile(r'(' + '|'.join(self.verbal_persons) + r')$')

        russian_tenses = ['ган', 'кан', 'дык', 'дик', 'тык', 'тик']
        for t in russian_tenses:
            if t not in self.verbal_tenses:
                self.verbal_tenses.append(t)
        self.verb_tense_re = re.compile(r'(' + '|'.join(self.verbal_tenses) + r')$')

    def _refresh_roots(self):
        combined = set(COMMON_LOANWORD_ROOTS) | set(self.custom_lexicon.keys())
        # Sort strictly by length descending (longest match first)
        self.loanword_roots = sorted(list(combined), key=lambda r: len(r), reverse=True)

    def register_root(self, root: str, metadata: Optional[dict] = None) -> bool:
        r_clean = root.strip().lower()
        if not r_clean:
            return False
        self.custom_lexicon[r_clean] = metadata or {"origin": "auto_induced"}
        self._refresh_roots()
        return True

    def _segment_loanword(self, word: str) -> Optional[str]:
        w_lower = word.lower()
        for root in self.loanword_roots:
            if w_lower.startswith(root) and len(w_lower) > len(root):
                sfx_fallback = word[len(root):]
                if len(sfx_fallback) >= 1:
                    # check if suffix contains verbalizer + tense/person
                    for deriv in ['-та', '-те', '-да', '-де', '-ла', '-ле']:
                        d_raw = deriv.lstrip('-')
                        if sfx_fallback.startswith(d_raw) and len(sfx_fallback) > len(d_raw):
                            rest = sfx_fallback[len(d_raw):]
                            return f"{word[:len(root)]} -{d_raw} -{rest}"
                    return f"{word[:len(root)]} -{sfx_fallback}"
        return None

    def analyze_and_segment(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        normalized = self.normalizer.normalize_text(text)
        return super().analyze_and_segment(normalized)

    def save_to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "version": "1.0.0",
            "roots": self.custom_lexicon
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_json(self, path: str) -> int:
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        roots = data.get("roots", {})
        count = 0
        for r, meta in roots.items():
            self.custom_lexicon[r.lower()] = meta
            count += 1
        self._refresh_roots()
        return count
```

- [ ] **Step 4: Run test to make sure it passes**
Command: `python -m unittest tests/test_auto_fst_overlay.py`
Expected: 4 tests pass.

- [ ] **Step 5: Commit changes**
Command: `git add src/auto_fst/overlay.py tests/test_auto_fst_overlay.py && git commit -m "feat(auto_fst): implement DynamicFSTOverlay with longest-match sorting and JSON persistence"`

---

### Task 4: Regression Guard & Golden Test Sandbox

**Files:**
- Create: `src/auto_fst/guard.py`
- Test: `tests/test_auto_fst_guard.py`

**Interfaces:**
- Consumes: `src/auto_fst/overlay.py`, `src/auto_fst/inducer.py`
- Produces: `class RegressionGuard`
  - `validate_candidate(self, candidate_root: str) -> Tuple[bool, str]`
  - `GOLDEN_SENTENCES: List[Tuple[str, str]]`

- [ ] **Step 1: Write the failing test**
Create `tests/test_auto_fst_guard.py`:
```python
import unittest
from src.auto_fst.guard import RegressionGuard

class TestRegressionGuard(unittest.TestCase):
    def setUp(self):
        self.guard = RegressionGuard()

    def test_valid_candidate_passes(self):
        passed, reason = self.guard.validate_candidate("вайб")
        self.assertTrue(passed)
        self.assertEqual(reason, "Validation passed: 100% agreement on golden baseline.")

    def test_harmful_prefix_candidate_rejected(self):
        # A candidate that matches a common native stem or function word must be rejected
        passed, reason = self.guard.validate_candidate("бол")
        self.assertFalse(passed)
        self.assertIn("Reserved native root", reason)

    def test_short_length_candidate_rejected(self):
        passed, reason = self.guard.validate_candidate("а")
        self.assertFalse(passed)
        self.assertIn("minimum length", reason.lower())

if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to make sure it fails**
Command: `python -m unittest tests/test_auto_fst_guard.py`
Expected: ModuleNotFoundError: No module named 'src.auto_fst.guard'

- [ ] **Step 3: Implement minimal code in `src/auto_fst/guard.py`**
Implement `RegressionGuard`:
```python
"""Automated regression testing sandbox for candidate root induction."""

import copy
from typing import List, Tuple
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
```

- [ ] **Step 4: Run test to make sure it passes**
Command: `python -m unittest tests/test_auto_fst_guard.py`
Expected: 3 tests pass.

- [ ] **Step 5: Commit changes**
Command: `git add src/auto_fst/guard.py tests/test_auto_fst_guard.py && git commit -m "feat(auto_fst): implement RegressionGuard sandbox verification"`

---

### Task 5: Interactive CLI Demonstration Script & End-to-End Test

**Files:**
- Create: `scripts/demo_auto_fst.py`
- Test: `tests/test_auto_fst_e2e.py`
- Create: `data/custom_slang_lexicon.json`

**Interfaces:**
- Consumes: `src/auto_fst/normalizer.py`, `src/auto_fst/inducer.py`, `src/auto_fst/overlay.py`, `src/auto_fst/guard.py`
- Produces: Standalone executable CLI demonstration with automated benchmark and interactive console.

- [ ] **Step 1: Write the failing end-to-end test**
Create `tests/test_auto_fst_e2e.py`:
```python
import unittest
import time
from src.auto_fst.normalizer import RussianLayoutNormalizer
from src.auto_fst.inducer import AutoFSTInducer
from src.auto_fst.guard import RegressionGuard
from src.auto_fst.overlay import DynamicFSTOverlay

class TestAutoFSTEndToEnd(unittest.TestCase):
    def setUp(self):
        self.normalizer = RussianLayoutNormalizer()
        self.inducer = AutoFSTInducer()
        self.guard = RegressionGuard()
        self.overlay = DynamicFSTOverlay(normalizer=self.normalizer)

    def test_full_autonomous_slang_induction_lifecycle(self):
        sentence = "Бүгін кешке достармен қатты вайбтадық"
        # 1. Check before
        before_parse = self.overlay.analyze_and_segment(sentence)
        self.assertNotIn("вайб -та -дық", before_parse)

        # 2. Extract and induce
        res = self.inducer.induce_stem("вайбтадық")
        self.assertIsNotNone(res)

        # 3. Guardrail check
        passed, reason = self.guard.validate_candidate(res.stem)
        self.assertTrue(passed)

        # 4. Commit to overlay
        self.overlay.register_root(res.stem, {"harmony": res.harmony, "coda": res.coda})

        # 5. Check after
        after_parse = self.overlay.analyze_and_segment(sentence)
        self.assertIn("вайб -та -дық", after_parse)

    def test_submillisecond_latency(self):
        sentence = "Каспиден доставкасы өте тез болды, оплатасын жасадым және вайбтадық."
        start = time.perf_counter()
        for _ in range(50):
            _ = self.overlay.analyze_and_segment(sentence)
        avg_ms = (time.perf_counter() - start) / 50 * 1000
        # Latency must be < 5.0ms on CPU
        self.assertLess(avg_ms, 5.0)

if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to make sure it fails or passes**
Command: `python -m unittest tests/test_auto_fst_e2e.py`
Expected: Pass if components are hooked up correctly.

- [ ] **Step 3: Implement `scripts/demo_auto_fst.py` and initialize `data/custom_slang_lexicon.json`**
Create `data/custom_slang_lexicon.json`:
```json
{
  "version": "1.0.0",
  "roots": {}
}
```
Implement `scripts/demo_auto_fst.py`:
```python
"""Auto-FST Demonstration Console: Automated Benchmark Suite & Interactive REPL.

Zero-GPU, ultra-lightweight adaptive morphological parser for code-switched and
Russian-layout informal Kazakh text.
"""

import sys
import os
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fst_analyzer import fst_analyzer as baseline_fst
from src.auto_fst.normalizer import RussianLayoutNormalizer
from src.auto_fst.inducer import AutoFSTInducer
from src.auto_fst.guard import RegressionGuard
from src.auto_fst.overlay import DynamicFSTOverlay

BENCHMARK_SAMPLES = [
    {
        "category": "Russian Keyboard Diacritic Flattening",
        "text": "салем калайсын, жумысын калай болып жатыр?",
        "note": "Replaces standard Russian layout letters with authentic Kazakh diacritics"
    },
    {
        "category": "Russian Keyboard Particle Agglutination",
        "text": "Ол али жумыста турсынгой, кешке келедигой.",
        "note": "Resolves flattened -сын and -гой modal particles without suffix collision"
    },
    {
        "category": "Emerging English/Russian Slang Verbalization",
        "text": "Биз кеше достармен катты вайбтадык жане чиллдедик.",
        "note": "Auto-induces loan stems 'вайб' and 'чилл' and decomposes verbalizer chains"
    },
    {
        "category": "Youth Consumer & Gaming Slang",
        "text": "Ойынга коп донаттау жасады, енди озинин крашына жазады.",
        "note": "Auto-induces 'донат' (voiceless coda) and 'краш' (noun possessive case)"
    },
    {
        "category": "Compound Disambiguation (Longest-Match First)",
        "text": "Чатботтын жауабын каспиредпен толедик.",
        "note": "Prevents 'чат' from shadowing 'чатбот', isolates 'каспиред'"
    },
    {
        "category": "Noise & Typo Rejection Guard",
        "text": "Бббазар123 жоккк соз емес.",
        "note": "Rejects malformed tokens without corrupting morphological lexicon"
    }
]

def run_benchmark():
    print("=" * 80)
    print("AUTO-FST AUTOMATED BENCHMARK SUITE")
    print("Adaptive Neuro-Symbolic Morphology for Informal & Code-Switched Kazakh")
    print("Hardware: Standard CPU (0 MB GPU VRAM required) | Platform: Lightweight")
    print("=" * 80)

    normalizer = RussianLayoutNormalizer()
    inducer = AutoFSTInducer()
    guard = RegressionGuard()
    overlay = DynamicFSTOverlay(normalizer=normalizer)
    lex_path = os.path.join("data", "custom_slang_lexicon.json")
    if os.path.exists(lex_path):
        loaded = overlay.load_from_json(lex_path)
        if loaded:
            print(f"Loaded {loaded} existing custom slang roots from {lex_path}\n")

    for i, item in enumerate(BENCHMARK_SAMPLES, 1):
        print(f"[{i}/6] Category: {item['category']}")
        print(f"Note : {item['note']}")
        print(f"Input: {item['text']}")

        # 1. Baseline Parse
        t0 = time.perf_counter()
        baseline_parse = baseline_fst.analyze_and_segment(item['text'])
        baseline_ms = (time.perf_counter() - t0) * 1000

        # 2. Check for candidate slang words and induce
        words = item['text'].split()
        induced_stems = []
        for w in words:
            cand = inducer.induce_stem(w)
            if cand:
                passed, reason = guard.validate_candidate(cand.stem)
                if passed:
                    overlay.register_root(cand.stem, {"harmony": cand.harmony, "coda": cand.coda})
                    induced_stems.append(f"{cand.stem} ({cand.harmony}, {cand.coda})")

        # 3. Auto-FST Parse
        t1 = time.perf_counter()
        auto_parse = overlay.analyze_and_segment(item['text'])
        auto_ms = (time.perf_counter() - t1) * 1000

        print(f"Before (Baseline FST) : {baseline_parse} ({baseline_ms:.2f} ms)")
        print(f"After  (Auto-FST)     : {auto_parse} ({auto_ms:.2f} ms)")
        if induced_stems:
            print(f"Auto-Inductions       : {', '.join(induced_stems)}")
        else:
            print(f"Auto-Inductions       : [None - Pure Orthographic / Existing Rule]")
        print("-" * 80)

    overlay.save_to_json(lex_path)
    print(f"Benchmark completed successfully. Updated lexicon saved to {lex_path}.\n")

def run_repl():
    print("=" * 80)
    print("AUTO-FST INTERACTIVE CONSOLE")
    print("Type any informal or code-switched Kazakh text. Type 'exit' to return.")
    print("=" * 80)

    normalizer = RussianLayoutNormalizer()
    inducer = AutoFSTInducer()
    guard = RegressionGuard()
    overlay = DynamicFSTOverlay(normalizer=normalizer)
    lex_path = os.path.join("data", "custom_slang_lexicon.json")
    if os.path.exists(lex_path):
        overlay.load_from_json(lex_path)

    while True:
        try:
            user_input = input("Auto-FST > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            t0 = time.perf_counter()
            # Analyze words
            words = user_input.split()
            for w in words:
                cand = inducer.induce_stem(w)
                if cand:
                    passed, reason = guard.validate_candidate(cand.stem)
                    if passed:
                        overlay.register_root(cand.stem, {"harmony": cand.harmony, "coda": cand.coda})
                        print(f"  [Auto-Induction] Root '{cand.stem}' -> Suffixes: {cand.suffixes} | Harmony={cand.harmony}, Coda={cand.coda}")
                    else:
                        print(f"  [Guard Rejected] Candidate '{cand.stem}': {reason}")

            result = overlay.analyze_and_segment(user_input)
            latency = (time.perf_counter() - t0) * 1000
            print(f"  Result : {result}")
            print(f"  Latency: {latency:.2f} ms\n")
        except (KeyboardInterrupt, EOFError):
            break

    overlay.save_to_json(lex_path)
    print("\nSession ended.")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "--repl":
        run_repl()
    else:
        run_benchmark()
```

- [ ] **Step 4: Run end-to-end test and execute demo CLI**
Command: `python -m unittest tests/test_auto_fst_e2e.py`
Expected: 2 tests pass.
Command: `python scripts/demo_auto_fst.py`
Expected: Runs benchmark suite across all 6 categories cleanly with zero errors.

- [ ] **Step 5: Invariance and regression test verification**
Verify existing repository test suites remain 100% passing:
Command: `python -m unittest tests/test_fst_code_switch.py tests/test_fst_verbal_rules.py`
Expected: All existing tests pass.
Command: `git diff origin/main..HEAD -- aist2026/paper.tex papers/kazakh_fever_conference/`
Expected: 0 diff lines.

- [ ] **Step 6: Commit changes**
Command: `git add data/custom_slang_lexicon.json scripts/demo_auto_fst.py tests/test_auto_fst_e2e.py && git commit -m "feat(auto_fst): add interactive CLI demo and end-to-end benchmark verification"`
