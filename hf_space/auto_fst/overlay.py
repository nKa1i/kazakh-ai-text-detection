"""Dynamic FST overlay with longest-match-first loanword disambiguation and persistence."""

import os
import json
import re
from typing import Dict, List, Optional
from fst_analyzer import AdvancedKazakhFSTAnalyzer, COMMON_LOANWORD_ROOTS
from src.auto_fst.normalizer import RussianLayoutNormalizer

EMPHATIC_PARTICLES = ['ғой', 'қой', 'гой', 'кой', 'мыс', 'міс', 'ау']

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


class DynamicFSTOverlay(AdvancedKazakhFSTAnalyzer):
    def __init__(self, normalizer: Optional[RussianLayoutNormalizer] = None):
        super().__init__()
        self.normalizer = normalizer or RussianLayoutNormalizer()
        self.custom_lexicon: Dict[str, dict] = {}
        # Initialize roots with longest-match-first sorting
        self._refresh_roots()

        # Augment verbal persons and tenses with Russian layout equivalents
        self._augment_rules_for_russian_layout()
        self.particle_re = re.compile(r'(' + '|'.join(EMPHATIC_PARTICLES) + r')$', re.IGNORECASE)

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
                    # Check if suffix contains verbalizer + tense/person (e.g. вайбтадық -> вайб -та -дық)
                    for deriv in ['-та', '-те', '-да', '-де', '-ла', '-ле']:
                        d_raw = deriv.lstrip('-')
                        if sfx_fallback.startswith(d_raw) and len(sfx_fallback) > len(d_raw):
                            rest = sfx_fallback[len(d_raw):]
                            rest_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(rest.lower(), rest)
                            return f"{word[:len(root)]} -{d_raw} -{rest_canon}"
                    sfx_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(sfx_fallback.lower(), sfx_fallback)
                    return f"{word[:len(root)]} -{sfx_canon}"
        return None

    def analyze_and_segment(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        normalized = self.normalizer.normalize_text(text)
        tokens = normalized.split()
        processed_tokens = []

        for token in tokens:
            m_punct = re.match(r'^([^\w]*)(.*?)([^\w]*)$', token)
            if not m_punct:
                processed_tokens.append(token)
                continue

            leading_punct, raw_word, trailing_punct = m_punct.groups()

            if len(raw_word) <= 3:
                processed_tokens.append(leading_punct + raw_word + trailing_punct)
                continue

            # First check code-switched loanwords
            loanword_seg = self._segment_loanword(raw_word)
            if loanword_seg:
                processed_tokens.append(leading_punct + loanword_seg + trailing_punct)
                continue

            word = raw_word
            suffixes = []
            had_verb = False

            # Check for trailing emphatic particles (e.g. тұрсыңғой -> тұрсың + -ғой)
            m_part = self.particle_re.search(word)
            if m_part and len(word[:m_part.start()]) >= 3:
                p_text = m_part.group(1)
                p_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(p_text.lower(), p_text)
                suffixes.insert(0, p_canon)
                word = word[:m_part.start()]

            # 1. Verbal Personal Agreement Suffixes (e.g. келгенмін -> -мін, тұрсың -> -сың)
            m_vper = self.verb_person_re.search(word)
            if m_vper and len(word[:m_vper.start()]) >= 3:
                s_text = m_vper.group(1)
                s_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(s_text.lower(), s_text)
                suffixes.insert(0, s_canon)
                word = word[:m_vper.start()]
                had_verb = True

            # 2. Verbal Tense / Participle Suffixes (e.g. жасалғандықтан -> -ғандықтан / -ған)
            m_vt = self.verb_tense_re.search(word)
            if m_vt and len(word[:m_vt.start()]) >= 3:
                t_text = m_vt.group(1)
                t_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(t_text.lower(), t_text)
                suffixes.insert(0, t_canon)
                word = word[:m_vt.start()]
                had_verb = True

            # 2b. Verbal Negation Suffixes
            if had_verb:
                for neg in self.verbal_negations:
                    if word.lower().endswith(neg) and len(word[:-len(neg)]) >= 3:
                        sfx = word[-len(neg):]
                        word = word[:-len(neg)]
                        suffixes.insert(0, sfx)
                        break

            # 2c. Verbalizer Derivational Suffixes
            if had_verb:
                for deriv in self.verbalizers:
                    if word.lower().endswith(deriv) and len(word[:-len(deriv)]) >= 3:
                        sfx = word[-len(deriv):]
                        word = word[:-len(deriv)]
                        suffixes.insert(0, sfx)
                        break

            # 3. Noun Cases
            match = self.case_re.search(word)
            if match and len(word[:match.start()]) >= 3:
                c_text = match.group(1)
                c_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(c_text.lower(), c_text)
                suffixes.insert(0, c_canon)
                word = word[:match.start()]

            # 4. Noun Possessives
            if len(word) > 3:
                match = self.poss_re.search(word)
                if match and len(word[:match.start()]) >= 3:
                    poss_cand = match.group(1).lower()
                    if (word.lower().endswith("ші") and poss_cand == "і") or (word.lower().endswith("шы") and poss_cand == "ы"):
                        pass
                    else:
                        suffixes.insert(0, match.group(1))
                        word = word[:match.start()]

            # 5. Plurals
            if len(word) > 3:
                match = self.plur_re.search(word)
                if match and len(word[:match.start()]) >= 3:
                    suffixes.insert(0, match.group(1))
                    word = word[:match.start()]

            # 6. Agentive Derivational Suffixes (-ші / -шы)
            if len(word) >= 4 and not had_verb:
                for ag in self.agentive_suffixes:
                    if word.lower().endswith(ag) and len(word[:-len(ag)]) >= 3:
                        sfx = word[-len(ag):]
                        word = word[:-len(ag)]
                        suffixes.insert(0, sfx)
                        break

            if suffixes:
                segmented = word + " " + " ".join(f"-{s}" for s in suffixes)
                processed_tokens.append(leading_punct + segmented + trailing_punct)
            else:
                processed_tokens.append(leading_punct + raw_word + trailing_punct)

        return " ".join(processed_tokens)

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
