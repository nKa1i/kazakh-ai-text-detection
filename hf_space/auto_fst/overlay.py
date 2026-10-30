"""Dynamic FST overlay with longest-match-first loanword disambiguation and persistence."""

import os
import json
import re
from typing import Dict, List, Optional
from fst_analyzer import AdvancedKazakhFSTAnalyzer, COMMON_LOANWORD_ROOTS

try:
    from .normalizer import RussianLayoutNormalizer
except (ImportError, ValueError):
    try:
        from auto_fst.normalizer import RussianLayoutNormalizer
    except ImportError:
        from src.auto_fst.normalizer import RussianLayoutNormalizer

try:
    from .phonology import MorphophonologicalRestorer, SYNCOPE_STEM_MAP
except (ImportError, ValueError):
    try:
        from auto_fst.phonology import MorphophonologicalRestorer, SYNCOPE_STEM_MAP
    except ImportError:
        from src.auto_fst.phonology import MorphophonologicalRestorer, SYNCOPE_STEM_MAP

EMPHATIC_PARTICLES = ['ғой', 'қой', 'гой', 'кой', 'мыс', 'міс', 'ау']

RUSSIAN_LAYOUT_SUFFIX_MAP = {
    'ган': 'ған', 'кан': 'қан',
    'сын': 'сың', 'син': 'сің',
    'гой': 'ғой', 'кой': 'қой',
    'нын': 'ның', 'нин': 'нің',
    'дын': 'дың', 'дин': 'дің',
    'тын': 'тың', 'тин': 'тің',
    'га': 'ға', 'ка': 'қа',
    'лығ': 'лық', 'ліг': 'лік',
    'дығ': 'дық', 'діг': 'дік',
    'тығ': 'тық', 'тіг': 'тік',
    'дык': 'дық', 'дик': 'дік',
    'тык': 'тық', 'тик': 'тік',
    'мин': 'мін', 'миз': 'міз',
}

VALID_VERBAL_TAILS = {
    'ған', 'ген', 'қан', 'кен', 'ган', 'кан',
    'дық', 'дік', 'тық', 'тік', 'дык', 'дик', 'тык', 'тик',
    'ды', 'ді', 'ты', 'ті', 'у', 'п', 'йды', 'йді', 'ады', 'еді',
    'ғандықтан', 'гендіктен', 'қандықтан', 'кендіктен'
}

COMPOUND_LOANWORD_TAILS = {
    'терді': ('тер', 'ді'), 'тарды': ('тар', 'ды'),
    'дерді': ('дер', 'ді'), 'дарды': ('дар', 'ды'),
    'лерді': ('лер', 'ді'), 'ларды': ('лар', 'ды'),
    'ыма': ('ым', 'а'), 'іме': ('ім', 'е'),
    'ына': ('ы', 'на'), 'іне': ('і', 'не'),
}

TWO_LETTER_NATIVE_ROOTS = {'үй', 'ұл', 'ат', 'ер', 'от', 'ас', 'іс', 'ел', 'аң', 'ақ', 'ой', 'үш', 'он'}

UNINFLECTED_BASE_WORDS = {
    'дәптер', 'мектеп', 'кітап', 'қалам', 'құлақ', 'жүрек',
    'сұрақ', 'жауап', 'мәдениет', 'мемлекет', 'құрылыс'
}


class DynamicFSTOverlay(AdvancedKazakhFSTAnalyzer):
    def __init__(self, normalizer: Optional[RussianLayoutNormalizer] = None):
        super().__init__()
        self.normalizer = normalizer or RussianLayoutNormalizer()
        self.restorer = MorphophonologicalRestorer()
        self.custom_lexicon: Dict[str, dict] = {}
        # Initialize roots with longest-match-first sorting
        self._refresh_roots()

        # Augment verbal persons and tenses with Russian layout equivalents
        self._augment_rules_for_russian_layout()
        self.particle_re = re.compile(r'(' + '|'.join(EMPHATIC_PARTICLES) + r')$', re.IGNORECASE)

        # Refined case endings: include сында, сінде, ын, ін, and proper ordering
        self.cases = [
            'сында', 'сінде',
            'нің', 'ның', 'дың', 'дің', 'тың', 'тің',
            'ға', 'ге', 'қа', 'ке', 'на', 'не',
            'ны', 'ні', 'ды', 'ді', 'ты', 'ті',
            'дан', 'ден', 'тан', 'тен', 'нан', 'нен',
            'мен', 'бен', 'пен',
            'нда', 'нде', 'да', 'де', 'та', 'те',
            'ын', 'ін',
        ]
        self.case_re = re.compile(r'(' + '|'.join(self.cases) + r')$')

        # Refined possessive endings: purely possessive (no fused plurals like тарымыз, теріміз)
        raw_poss = [
            'ымыз', 'іміз', 'ыңыз', 'іңіз',
            'мыз', 'міз', 'ңыз', 'ңіз',
            'сын', 'сін',
            'ым', 'ім', 'ың', 'ің',
            'сы', 'сі', 'м', 'ң', 'ы', 'і'
        ]
        self.possessives = sorted(raw_poss, key=lambda s: len(s), reverse=True)
        self.poss_re = re.compile(r'(' + '|'.join(self.possessives) + r')$')

        # Derivational adjectival & adverbial endings
        self.deriv_adjectives = [
            'ғандықтан', 'гендіктен', 'қандықтан', 'кендіктен',
            'лық', 'лік', 'дық', 'дік', 'тық', 'тік',
            'лығ', 'ліг', 'дығ', 'діг', 'тығ', 'тіг',
            'дык', 'дик', 'тык', 'тик',
            'ша', 'ше'
        ]
        self.deriv_adj_re = re.compile(r'(' + '|'.join(sorted(self.deriv_adjectives, key=len, reverse=True)) + r')$')

    @staticmethod
    def _valid_stem_len(stem_cand: str) -> bool:
        if not stem_cand:
            return False
        return len(stem_cand) >= 3 or stem_cand.lower() in TWO_LETTER_NATIVE_ROOTS

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

    def _decompose_nominal_tail(self, sfx: str) -> List[str]:
        """Decomposes a nominal suffix chain (Plural + Possessive + Case) on a loanword root."""
        sfx_lower = sfx.lower()
        if sfx_lower in COMPOUND_LOANWORD_TAILS:
            p1, p2 = COMPOUND_LOANWORD_TAILS[sfx_lower]
            return [
                RUSSIAN_LAYOUT_SUFFIX_MAP.get(p1, p1),
                RUSSIAN_LAYOUT_SUFFIX_MAP.get(p2, p2),
            ]
        if sfx_lower == 'сын':
            return ['сын']

        rem = sfx_lower
        parts: List[str] = []

        # 1. Strip outer Case ending if remaining stem is non-empty
        m_case = self.case_re.search(rem)
        if m_case and len(rem[:m_case.start()]) >= 1:
            c_raw = m_case.group(1)
            c_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(c_raw, c_raw)
            parts.insert(0, c_canon)
            rem = rem[:m_case.start()]

        # 2. Strip Possessive ending if remaining stem is non-empty
        if rem:
            m_poss = self.poss_re.search(rem)
            if m_poss and len(rem[:m_poss.start()]) >= 2:
                p_raw = m_poss.group(1)
                p_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(p_raw, p_raw)
                parts.insert(0, p_canon)
                rem = rem[:m_poss.start()]

        # 3. Check if remaining prefix is a Plural marker (-лар/-лер/-дар/-дер/-тар/-тер)
        if rem in self.plurals:
            parts.insert(0, rem)
            return parts

        if not rem and parts:
            return parts

        if rem and parts:
            rem_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(rem, rem)
            parts.insert(0, rem_canon)
            return parts

        canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(sfx_lower, sfx)
        return [canon]

    def _segment_loanword(self, word: str) -> Optional[str]:
        w_lower = word.lower()
        for root in self.loanword_roots:
            matched = False
            root_len = len(root)
            actual_root = root
            if w_lower.startswith(root) and len(w_lower) > len(root):
                matched = True
            elif root[-1] in ('п', 'қ', 'к') and len(w_lower) > len(root):
                # Voiced root variant: п->б, қ->ғ, к->г (or к->ғ for back loanwords like тикток->тиктоғ)
                voiced_chars = {'п': ('б',), 'қ': ('ғ',), 'к': ('г', 'ғ')}[root[-1]]
                for vc in voiced_chars:
                    voiced_prefix = root[:-1] + vc
                    if w_lower.startswith(voiced_prefix):
                        matched = True
                        root_len = len(voiced_prefix)
                        actual_root = root
                        break

            if matched:
                sfx_fallback = word[root_len:]
                if len(sfx_fallback) >= 1:
                    sfx_lower = sfx_fallback.lower()

                    # Check if suffix contains verbalizer + tense/person (e.g. вайбтадық -> вайб -та -дық)
                    for deriv in ['-та', '-те', '-да', '-де', '-ла', '-ле']:
                        d_raw = deriv.lstrip('-')
                        if sfx_lower.startswith(d_raw) and len(sfx_lower) > len(d_raw):
                            rest = sfx_lower[len(d_raw):]
                            if rest in VALID_VERBAL_TAILS:
                                rest_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(rest, rest)
                                return f"{actual_root} -{d_raw} -{rest_canon}"

                    tail_parts = self._decompose_nominal_tail(sfx_fallback)
                    return f"{actual_root} " + " ".join(f"-{p}" for p in tail_parts)
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

            if raw_word.lower() in UNINFLECTED_BASE_WORDS:
                processed_tokens.append(leading_punct + raw_word + trailing_punct)
                continue

            if len(raw_word) <= 3 and raw_word.lower() not in TWO_LETTER_NATIVE_ROOTS:
                processed_tokens.append(leading_punct + raw_word + trailing_punct)
                continue

            # Hyphenated mixed-script tokens (e.g. AI-дың, ChatGPT-мен, iPhone-ға)
            m_mixed = re.match(r'^([A-Za-z0-9_]+)-([а-яёәіңғүұқөһА-ЯЁӘІҢҒҮҰҚӨҺ]+)$', raw_word)
            if m_mixed:
                root_str = m_mixed.group(1)
                sfx_str = m_mixed.group(2)
                sfx_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(sfx_str.lower(), sfx_str)
                processed_tokens.append(leading_punct + f"{root_str} -{sfx_canon}" + trailing_punct)
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
            if m_part and self._valid_stem_len(word[:m_part.start()]):
                p_text = m_part.group(1)
                p_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(p_text.lower(), p_text)
                suffixes.insert(0, p_canon)
                word = word[:m_part.start()]

            # 1. Verbal Personal Agreement Suffixes (e.g. келгенмін -> -мін, тұрсың -> -сың)
            m_vper = self.verb_person_re.search(word)
            if m_vper:
                stem_cand = word[:m_vper.start()]
                s_text = m_vper.group(1).lower()

                # Disambiguation A: 2nd-person verbal -сың/-сің never attaches directly to a bare stem
                # ending in -ы/-і unless preceded by adjectival -ды/-ді/-ты/-ті/-лы/-лі or agentive -шы/-ші.
                # Thus stems ending in bare -ы/-і (жұмы-сың, дауы-сың, табы-сың, жеңі-сің, құрылы-сың) or known
                # monosyllabic s-stems (дос, бас, тас, құс, жас, ас, іс) are s-final nouns + possessive -ың/-ің.
                is_adj_or_agent = (
                    stem_cand.lower().endswith(('ды', 'ді', 'ты', 'ті', 'шы', 'ші', 'сы', 'сі'))
                    or (stem_cand.lower().endswith(('лы', 'лі')) and not stem_cand.lower().endswith(('ылы', 'ілі')))
                )
                is_s_final_noun = (
                    (stem_cand.lower().endswith(('ы', 'і')) and not is_adj_or_agent)
                    or (stem_cand + 'с').lower() in ('дос', 'бас', 'тас', 'құс', 'жас', 'ас', 'іс')
                )
                if s_text in ('сың', 'сің', 'сын', 'син') and is_s_final_noun:
                    word = stem_cand + 'с'
                    p_sfx = 'ың' if s_text in ('сың', 'сын') else 'ің'
                    suffixes.insert(0, p_sfx)
                # Disambiguation B: if s_text is мыз/міз and preceded by vowel ы/і (e.g. мектебіміз, қолдарымыз),
                # this is noun possessive -ымыз/-іміз, handled in Step 4
                elif s_text in ('мыз', 'міз') and stem_cand.endswith(('ы', 'і')):
                    pass
                elif self._valid_stem_len(stem_cand):
                    s_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(s_text, s_text)
                    suffixes.insert(0, s_canon)
                    word = stem_cand
                    had_verb = True

            # 2. Verbal Tense / Participle Suffixes (e.g. жасалғандықтан -> -ғандықтан / -ған)
            m_vt = self.verb_tense_re.search(word)
            if m_vt and self._valid_stem_len(word[:m_vt.start()]):
                t_text = m_vt.group(1)
                t_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(t_text.lower(), t_text)
                suffixes.insert(0, t_canon)
                word = word[:m_vt.start()]
                had_verb = True

            # 2b. Verbal Negation Suffixes
            if had_verb:
                for neg in self.verbal_negations:
                    if word.lower().endswith(neg) and self._valid_stem_len(word[:-len(neg)]):
                        sfx = word[-len(neg):]
                        word = word[:-len(neg)]
                        suffixes.insert(0, sfx)
                        break

            # 2c. Verbalizer Derivational Suffixes
            if had_verb:
                for deriv in self.verbalizers:
                    if word.lower().endswith(deriv) and self._valid_stem_len(word[:-len(deriv)]):
                        sfx = word[-len(deriv):]
                        word = word[:-len(deriv)]
                        suffixes.insert(0, sfx)
                        break

            # 3. Noun Cases
            # Do not strip accusative -ны/-ні if word is a syncopated stem + 3rd-person possessive -ы/-і
            # (e.g. мұрны -> мұрн + ы, қарны -> қарн + ы, ерні -> ерн + і, орны -> орн + ы, көңлі -> көңл + і)
            is_syncope_possessive = (
                len(word) >= 4
                and word[-1].lower() in ('ы', 'і')
                and word[:-1].lower() in SYNCOPE_STEM_MAP
            )
            match = None if is_syncope_possessive else self.case_re.search(word)
            if match:
                matched_sfx = match.group(1)
                stem_cand = word[:match.start()]
                if matched_sfx in ('нда', 'нде'):
                    # If preceded by possessive vowel or сы/сі (e.g. қолында, үйінде, қаласында), strip нда/нде
                    if stem_cand.endswith(('ы', 'і', 'сы', 'сі')) and self._valid_stem_len(stem_cand):
                        c_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(matched_sfx.lower(), matched_sfx)
                        suffixes.insert(0, c_canon)
                        word = stem_cand
                    # Otherwise, if stem ends in 'н' or 'ң' (e.g. күнде, түнде), the suffix is actually -де/-да
                    elif len(word) >= 4 and word[:-2].endswith(('н', 'ң')):
                        actual_sfx = 'де' if matched_sfx == 'нде' else 'да'
                        actual_stem = word[:-2]
                        if self._valid_stem_len(actual_stem):
                            suffixes.insert(0, actual_sfx)
                            word = actual_stem
                elif self._valid_stem_len(stem_cand):
                    c_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(matched_sfx.lower(), matched_sfx)
                    suffixes.insert(0, c_canon)
                    word = stem_cand

            # 4. Noun Possessives
            if self._valid_stem_len(word):
                match = self.poss_re.search(word)
                if match and self._valid_stem_len(word[:match.start()]):
                    poss_cand = match.group(1).lower()
                    if (word.lower().endswith("ші") and poss_cand == "і") or (word.lower().endswith("шы") and poss_cand == "ы"):
                        pass
                    else:
                        suffixes.insert(0, match.group(1))
                        word = word[:match.start()]

            # 5. Plurals
            if self._valid_stem_len(word):
                match = self.plur_re.search(word)
                if match and self._valid_stem_len(word[:match.start()]):
                    suffixes.insert(0, match.group(1))
                    word = word[:match.start()]

            # 5b. Verbal Participles / Tenses after Plurals / Cases (e.g. айтқандар -> айт -қан -дар, оқығандарға -> оқы -ған -дар -ға)
            if not had_verb and self._valid_stem_len(word):
                m_vt2 = self.verb_tense_re.search(word)
                if m_vt2 and self._valid_stem_len(word[:m_vt2.start()]):
                    t_text = m_vt2.group(1)
                    t_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(t_text.lower(), t_text)
                    suffixes.insert(0, t_canon)
                    word = word[:m_vt2.start()]
                    had_verb = True

            # 6. Derivational Adjectives / Adverbs (-лық / -лік / -дық / -дік / -тық / -тік / -ша / -ше)
            if not had_verb and self._valid_stem_len(word):
                m_adj = self.deriv_adj_re.search(word)
                if m_adj and len(word[:m_adj.start()]) >= 3 and word[:m_adj.start()].lower() not in ('орта', 'ақ'):
                    adj_text = m_adj.group(1)
                    adj_canon = RUSSIAN_LAYOUT_SUFFIX_MAP.get(adj_text.lower(), adj_text)
                    suffixes.insert(0, adj_canon)
                    word = word[:m_adj.start()]

            # 7. Agentive Derivational Suffixes (-ші / -шы)
            if self._valid_stem_len(word) and not had_verb:
                for ag in getattr(self, "agentive_suffixes", ["ші", "шы"]):
                    if word.lower().endswith(ag) and self._valid_stem_len(word[:-len(ag)]):
                        sfx = word[-len(ag):]
                        word = word[:-len(ag)]
                        suffixes.insert(0, sfx)
                        break

            if suffixes:
                # Apply morphophonological stem restoration (voicing & syncope)
                word = self.restorer.restore_stem(word, suffixes[0], set(self.loanword_roots))
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
