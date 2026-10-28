"""Automated morphological stem inducer and phonetic profiler for slang and loanwords."""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

HARD_VOWELS = set('аоұыяё')
SOFT_VOWELS = set('әөүіеэи')
VOICED_CONSONANTS = set('бвгджзлмнңруй')
VOICELESS_CONSONANTS = set('пфкқтсшщхцч')

KNOWN_SUFFIX_CHAINS = [
    # 3-Stack Plural + Possessive + Case
    ['-тар', '-ымыз', '-дан'], ['-тер', '-іміз', '-ден'],
    ['-дар', '-ымыз', '-дан'], ['-дер', '-іміз', '-ден'],
    ['-лар', '-ымыз', '-дан'], ['-лер', '-іміз', '-ден'],
    ['-тар', '-ыңыз', '-дан'], ['-тер', '-іңіз', '-ден'],
    ['-тар', '-ымыз', '-ға'], ['-тер', '-іміз', '-ге'],
    ['-дар', '-ымыз', '-ға'], ['-дер', '-іміз', '-ге'],
    ['-лар', '-ымыз', '-ға'], ['-лер', '-іміз', '-ге'],
    # 2-Stack Plural + Possessive
    ['-тар', '-ымыз'], ['-тер', '-іміз'],
    ['-дар', '-ымыз'], ['-дер', '-іміз'],
    ['-лар', '-ымыз'], ['-лер', '-іміз'],
    ['-ымыз', '-дан'], ['-іміз', '-ден'],
    ['-ыңыз', '-дан'], ['-іңіз', '-ден'],
    # Complex Verbalizer + Tense + Person
    ['-та', '-ған', '-дықтан'], ['-те', '-ген', '-діктен'],
    ['-та', '-ған'], ['-те', '-ген'], ['-да', '-ған'], ['-де', '-ген'],
    ['-та', '-дық'], ['-те', '-дік'], ['-да', '-дық'], ['-де', '-дік'],
    ['-та', '-ды'], ['-те', '-ді'], ['-да', '-ды'], ['-де', '-ді'],
    ['-та', '-у'], ['-те', '-у'], ['-да', '-у'], ['-де', '-у'],
    ['-де', '-дік'], ['-де', '-п'], ['-та', '-п'],
    # Verbal Person & Tense
    ['-сың'], ['-сің'], ['-сыз'], ['-сіз'],
    ['-ды'], ['-ді'], ['-ты'], ['-ті'],
    ['-ған'], ['-ген'], ['-қан'], ['-кен'],
    ['-ады'], ['-еді'], ['-йды'], ['-йді'],
    # Noun Cases / Compound Possessives
    ['-ым', '-а'], ['-ім', '-е'], ['-ым', '-на'], ['-ім', '-не'],
    ['-ы', '-на'], ['-і', '-не'], ['-сы', '-на'], ['-сі', '-не'],
    ['-лар', '-ы'], ['-лер', '-і'], ['-тар', '-ы'], ['-тер', '-і'],
    ['-ның'], ['-нің'], ['-дың'], ['-дің'], ['-тың'], ['-тің'],
    ['-ға'], ['-ге'], ['-қа'], ['-ке'],
    ['-мен'], ['-бен'], ['-пен'],
    ['-дан'], ['-ден'], ['-тан'], ['-тен'],
    # Simple Noun Possessives
    ['-ым'], ['-ім'], ['-ың'], ['-ің'],
    ['-сы'], ['-сі'], ['-ы'], ['-і'],
]

# Russian keyboard flattened variants to canonical Kazakh equivalents
RUSSIAN_LAYOUT_SUFFIX_MAP = {
    'ган': 'ған', 'кан': 'қан',
    'сын': 'сың', 'син': 'сің',
    'гой': 'ғой', 'кой': 'қой',
    'нын': 'ның', 'нин': 'нің',
    'дын': 'дың', 'дин': 'дің',
    'тын': 'тың', 'тин': 'тің',
    'га': 'ға', 'ка': 'қа',
    'дык': 'дық', 'дик': 'дік',
    'тык': 'тық', 'тик': 'тік',
    'мин': 'мін', 'миз': 'міз',
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
        token_clean = re.sub(r'^[^\w\-]+|[^\w\-]+$', '', token)
        if not token_clean:
            return None

        # 1. Check for hyphenated mixed-script tokens (e.g. AI-дың, ChatGPT-мен, iPhone-ға)
        m_mixed = re.match(r'^([A-Za-z0-9_]+)-([а-яёәіңғүұқөһА-ЯЁӘІҢҒҮҰҚӨҺ]+)$', token_clean)
        if m_mixed:
            root = m_mixed.group(1)
            sfx_raw = m_mixed.group(2).lower()
            sorted_chains = sorted(KNOWN_SUFFIX_CHAINS, key=lambda c: len("".join(s.lstrip('-') for s in c)), reverse=True)
            sfx_chain = None
            for chain in sorted_chains:
                chain_str = "".join(s.lstrip('-') for s in chain)
                if sfx_raw == chain_str:
                    sfx_chain = [s.lstrip('-') for s in chain]
                    break
            if not sfx_chain:
                sfx_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(sfx_raw, sfx_raw)
                sfx_chain = [sfx_canon]
            return InductionResult(
                stem=root,
                suffixes=sfx_chain,
                harmony="hard",
                coda="voiced",
                confidence=0.99
            )

        if not re.match(r'^[а-яёәіңғүұқөһА-ЯЁӘІҢҒҮҰҚӨҺ]+$', token_clean):
            return None
        t_lower = token_clean.lower()
        if len(t_lower) <= self.min_stem_length:
            return None

        # Sort suffix chains by total length descending to ensure longest match
        sorted_chains = sorted(KNOWN_SUFFIX_CHAINS, key=lambda c: len("".join(s.lstrip('-') for s in c)), reverse=True)

        for chain in sorted_chains:
            affix_str = "".join(s.lstrip('-') for s in chain)
            # check both authentic and Russian layout variants
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

    def induce_from_corpus(self, texts: List[str], min_paradigm_size: int = 2) -> List[InductionResult]:
        """Induces roots with confirmed paradigm diversity across corpus sentences."""
        paradigm_map = defaultdict(lambda: {"chains": set(), "results": []})
        for text in texts:
            # Tokenize words, preserving hyphenated tokens
            for token in re.findall(r'[A-Za-z0-9А-Яа-яӘәІіҢңҒғҮүҰұҚқӨөҺһ\-]+', text):
                res = self.induce_stem(token)
                if res:
                    chain_tuple = tuple(res.suffixes)
                    paradigm_map[res.stem]["chains"].add(chain_tuple)
                    paradigm_map[res.stem]["results"].append(res)

        verified = []
        for stem, data in paradigm_map.items():
            if len(data["chains"]) >= min_paradigm_size:
                best_res = data["results"][0]
                verified.append(InductionResult(
                    stem=stem,
                    suffixes=best_res.suffixes,
                    harmony=best_res.harmony,
                    coda=best_res.coda,
                    confidence=0.99
                ))
        return verified
