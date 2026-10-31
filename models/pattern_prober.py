# -*- coding: utf-8 -*-
"""
models/pattern_prober.py: Phase 3 Deep Morphosyntactic & Syntactic Pattern Prober.

Measures why frontier, reasoning, and sovereign LLM families systematically diverge
from authentic Kazakh human prose across 9 agglutinative and syntactic dimensions:
  1. chain_len_entropy        - Shannon entropy (bits) of productive morpheme chain depths
  2. chain_len_variance       - Variance of productive morpheme count per word
  3. copula_ratio             - Analytic copula & bureaucratic light-verb crutches per 100 words
  4. synthetic_verb_ratio     - Genuine synthetic finite verbal predicates per 100 words
  5. discourse_rigidity_score - Stereotyped sentence/clause framing formula density [0, 1]
  6. sov_compliance_score     - Head-final Kazakh SOV predicate compliance rate [0, 1]
  7. loan_assimilation_score  - Foreign/loanword stem-suffix vowel harmony compliance [0, 1]
  8. lexical_ttr              - Moving-Average Type-Token Ratio (100-word sliding window) [0, 1]
  9. burstiness               - Sentence length coefficient of variation (sigma / mu)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    from src.auto_fst.overlay import DynamicFSTOverlay
except ImportError:
    try:
        from auto_fst.overlay import DynamicFSTOverlay
    except ImportError:
        _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        from src.auto_fst.overlay import DynamicFSTOverlay


PROBE_METRIC_NAMES: Tuple[str, ...] = (
    "chain_len_entropy",
    "chain_len_variance",
    "copula_ratio",
    "synthetic_verb_ratio",
    "discourse_rigidity_score",
    "sov_compliance_score",
    "loan_assimilation_score",
    "lexical_ttr",
    "burstiness",
)

RADAR_AXIS_KEYS: Tuple[str, ...] = (
    "chain_len_entropy",
    "copula_ratio",
    "synthetic_verb_ratio",
    "discourse_rigidity_score",
    "sov_compliance_score",
    "loan_assimilation_score",
    "lexical_ttr",
    "burstiness",
)

RADAR_AXIS_LABELS: Dict[str, Dict[str, str]] = {
    "chain_len_entropy": {"en": "Chain Entropy", "kz": "Тізбек энтропиясы"},
    "copula_ratio": {"en": "Copula Crutch", "kz": "Көмекші етістік"},
    "synthetic_verb_ratio": {"en": "Synthetic Verbs", "kz": "Синтетикалық етістік"},
    "discourse_rigidity_score": {"en": "Discourse Rigidity", "kz": "Дискурс қатаңдығы"},
    "sov_compliance_score": {"en": "SOV Compliance", "kz": "SOV сәйкестігі"},
    "loan_assimilation_score": {"en": "Loan Harmony", "kz": "Кірме үндестігі"},
    "lexical_ttr": {"en": "Lexical TTR", "kz": "Лексикалық TTR"},
    "burstiness": {"en": "Sent. Burstiness", "kz": "Сөйлем серпіні"},
}

# Empirically calibrated authentic human Kazakh baseline (mean and floor-guarded std)
HUMAN_BASELINE_STATS: Dict[str, Dict[str, float]] = {
    "chain_len_entropy": {"mean": 1.945, "std": 0.180},
    "chain_len_variance": {"mean": 1.154, "std": 0.300},
    "copula_ratio": {"mean": 0.250, "std": 0.650},
    "synthetic_verb_ratio": {"mean": 12.350, "std": 1.950},
    "discourse_rigidity_score": {"mean": 0.030, "std": 0.080},
    "sov_compliance_score": {"mean": 0.985, "std": 0.050},
    "loan_assimilation_score": {"mean": 0.990, "std": 0.050},
    "lexical_ttr": {"mean": 0.975, "std": 0.045},
    "burstiness": {"mean": 0.220, "std": 0.095},
}

# Reference linguistic signatures for known and wild calque LLM families
FAMILY_REFERENCE_PROBES: Dict[str, Dict[str, float]] = {
    "ISSAI-KazLLM-70B": {
        "chain_len_entropy": 1.653,
        "chain_len_variance": 0.946,
        "copula_ratio": 1.749,
        "synthetic_verb_ratio": 6.612,
        "discourse_rigidity_score": 0.278,
        "sov_compliance_score": 1.000,
        "loan_assimilation_score": 1.000,
        "lexical_ttr": 0.990,
        "burstiness": 0.085,
    },
    "Sherkala-8B": {
        "chain_len_entropy": 1.729,
        "chain_len_variance": 0.857,
        "copula_ratio": 7.305,
        "synthetic_verb_ratio": 1.754,
        "discourse_rigidity_score": 0.093,
        "sov_compliance_score": 1.000,
        "loan_assimilation_score": 1.000,
        "lexical_ttr": 0.975,
        "burstiness": 0.112,
    },
    "DeepSeek-R1": {
        "chain_len_entropy": 1.603,
        "chain_len_variance": 0.802,
        "copula_ratio": 2.346,
        "synthetic_verb_ratio": 7.966,
        "discourse_rigidity_score": 0.463,
        "sov_compliance_score": 0.931,
        "loan_assimilation_score": 0.971,
        "lexical_ttr": 0.975,
        "burstiness": 0.102,
    },
    "GPT-4o": {
        "chain_len_entropy": 1.832,
        "chain_len_variance": 0.991,
        "copula_ratio": 7.618,
        "synthetic_verb_ratio": 1.251,
        "discourse_rigidity_score": 0.056,
        "sov_compliance_score": 0.944,
        "loan_assimilation_score": 0.985,
        "lexical_ttr": 0.993,
        "burstiness": 0.104,
    },
    "Qwen-2.5-72B": {
        "chain_len_entropy": 1.787,
        "chain_len_variance": 0.986,
        "copula_ratio": 7.802,
        "synthetic_verb_ratio": 2.901,
        "discourse_rigidity_score": 0.111,
        "sov_compliance_score": 1.000,
        "loan_assimilation_score": 1.000,
        "lexical_ttr": 0.997,
        "burstiness": 0.082,
    },
    "Llama-3.3-70B-Calque": {
        "chain_len_entropy": 1.688,
        "chain_len_variance": 0.896,
        "copula_ratio": 10.165,
        "synthetic_verb_ratio": 0.245,
        "discourse_rigidity_score": 0.722,
        "sov_compliance_score": 0.819,
        "loan_assimilation_score": 1.000,
        "lexical_ttr": 0.964,
        "burstiness": 0.077,
    },
}

_RADAR_CEILINGS: Dict[str, float] = {
    "chain_len_entropy": 2.40,
    "copula_ratio": 11.00,
    "synthetic_verb_ratio": 15.00,
    "discourse_rigidity_score": 1.00,
    "sov_compliance_score": 1.00,
    "loan_assimilation_score": 1.00,
    "lexical_ttr": 1.00,
    "burstiness": 0.55,
}

_WORD_RE = re.compile(r"[а-яёәіңғүұқөһА-ЯЁӘІҢҒҮҰҚӨҺa-zA-Z0-9\-]+", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"[.!?]+")
_PURE_LATIN_WORD_RE = re.compile(r"^[a-zA-Z]{2,}$")
_KAZ_CHARS = set("әіңғүұқөһӘІҢҒҮҰҚӨҺ")
_BACK_VOWELS = set("аоұыяуaou")
_FRONT_VOWELS = set("әеіөүэиei")
_ALL_VOWELS = _BACK_VOWELS | _FRONT_VOWELS | set("юё")

# Bare monomorphemic roots, particles, clitics, pronouns, numerals, and adverbs (0 affixes)
_BARE_MONOMORPHEMIC = {
    "ғой", "қой", "екен", "бірақ", "алайда", "әрі", "мен", "пен", "бен", "және",
    "үшін", "дейін", "кейін", "сайын", "бойы", "арқылы", "туралы", "сәйкес", "қарсы",
    "бірге", "өте", "тым", "аса", "кеше", "бүгін", "ертең", "биыл", "қазір", "әлі",
    "сәл", "әзер", "соң", "біз", "сіз", "олар", "сен", "ол", "бұл", "сол", "осы",
    "әр", "өз", "екі", "бір", "он", "көп", "біраз", "шағын", "үлкен", "жаңа", "ескі",
    "жылы", "керемет", "әдемі", "жақсы", "мол", "тез", "ерте", "қатты", "тура",
    "керек", "қажет", "мүмкін", "сияқты", "деп", "емес", "жоқ", "бар", "еді", "да",
    "де", "та", "те", "ма", "ме", "ба", "бе", "па", "пе", "ауа", "су", "ой", "үй",
    "тау", "жаз", "қыс", "көктем", "күз", "күн", "түн", "жыл", "ай", "жол", "гүл",
    "дос", "адам", "бала", "ана", "аға", "іні", "апа", "ата", "әже", "ел", "жер",
    "қала", "ауыл", "өзен", "ағаш", "кітап", "тарих", "уақыт", "тамақ", "шай", "кофе",
    "дүкен", "базар", "вокзал", "пәтер", "мектеп", "университет", "дәмхана",
    "кітапхана", "саяжай", "аула", "орталық", "көрме", "тауар", "курьер", "мәселе",
    "ұжым", "қолдау", "денсаулық", "серпін", "шаруа", "еңбек", "ойын", "алаң",
    "маман", "интернет", "сыйлық", "өнер", "футбол", "минут", "пойыз", "сағат",
    "сабақ", "қалам", "дәптер", "той", "шаңғы", "мақала", "зат", "кездесу",
    "жаңбыр", "қолшатыр", "сапа", "көңіл", "ғылым", "білім", "ақпарат", "фильм",
    "шаршау", "тосап", "демалыс", "лагерь", "компьютер", "өмір", "алғыс",
}

# Frozen bureaucratic postpositions and single-slot lexicalized templates (depth = 1)
_SINGLE_AFFIX_TEMPLATES = {
    "бойынша", "шеңберінде", "аясында", "барысында", "мақсатында", "нәтижесінде",
    "тұрғысынан", "арқасында", "негізінде", "арасында", "арасындағы", "жөнінде",
    "саласында", "бағытында", "деңгейінде", "режимінде", "форматында", "жағдайында",
    "кезеңінде", "тәртібіндегі", "кезектегі", "деңгейдегі", "бағыттағы", "нарықтағы",
    "қолданыстағы", "аталған", "аталмыш", "тиісті", "қамтамасыз", "айтарлықтай",
    "тұтастай", "заманауи", "болып", "табылады", "саналады", "келеді", "етеді",
    "етіледі", "береді", "беріледі", "болады", "тигізеді", "тигізуде", "ашуда",
    "беруде", "көрсетуде", "жалғасуда", "тартылуда", "енгізілуде", "асырылуда",
    "жасалуда", "күтілуде", "есептеледі", "алынады", "бөлінеді", "жасайды",
    "арттырады", "төмендетеді", "қалайды", "көтерді", "бекітілген", "бағытталған",
    "талданды", "құрастырылды", "дәлелденді", "қолданылды", "сақталды", "расталды",
    "тұжырымдалды", "енгізілді", "күшейтілді", "алғанда", "отырып", "көрсеткендей",
    "қарастырылып", "отырған", "ұсынылған", "қойылған", "қабылданған", "жаңартылатын",
}

_LOAN_STEMS = tuple(sorted([
    "телекоммуникация", "инфрақұрылым", "кибергигиена", "конвергенция",
    "дифференциал", "архитектура", "конференция", "университет", "инвестиция",
    "инновация", "интеграция", "технология", "математика", "статистика",
    "мониторинг", "платформа", "дисперсия", "валидация", "эвристика",
    "компьютер", "экономика", "экология", "медицина", "ведомство", "институт",
    "стандарт", "алгоритм", "параметр", "гипотеза", "когерент", "интернет",
    "капитал", "процесс", "процес", "ресурс", "сектор", "фактор", "формат",
    "клиент", "бизнес", "орган", "модель", "теорема", "теория", "прунинг",
    "кубит", "лагерь", "фильм", "футбол", "вокзал", "курьер", "каспи",
    "режим", "сервер", "парламент", "менеджмент", "документ", "банк", "округ",
    "акті", "акт",
], key=len, reverse=True))

_KNOWN_ROOTS = tuple(sorted(list(set(_LOAN_STEMS) | {
    "коммерциялан", "автоматтан", "әртараптан", "цифрлан", "сәйкестен", "оңтайлан",
    "бағдарлама", "стратегия", "ақпарат", "мемлекет", "республика", "көрсеткіш",
    "қоғам", "қарқын", "жауап", "сынып", "тәжірибе", "қолшатыр", "көкөніс",
    "отбас", "кітапхана", "дәмхана", "орналас", "шүйірке", "көзқарас",
    "кептеліс", "тыныш", "сағын", "қуан", "өкін",
}), key=len, reverse=True))

_ANALYTIC_COPULA_PATTERNS = (
    re.compile(r"\bболып\s+табылады\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bболып\s+келеді\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bболып\s+саналады\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bқызмет\s+атқарады\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bрөл\s+атқарады\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bқамтамасыз\s+етеді\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bқамтамасыз\s+етіледі\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bжүзеге\s+асырылуда\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bіске\s+асырылуда\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bсептігін\s+тигізеді\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bсептігін\s+тигізуде\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bықпалын\s+тигізеді\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bнегіз\s+болады\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bмүмкіндік\s+береді\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bкепілдік\s+береді\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bсерпін\s+беруде\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bжол\s+ашуда\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bмаңызға\s+ие\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bназарға\s+алынады\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bмән\s+беріледі\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bбасымдық\s+беріледі\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bкөңіл\s+бөлінеді\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bқол\s+жеткізуге\s+болады\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bжағдай\s+жасайды\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bжағдайлар\s+жасалуда\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bдеп\s+есептеледі\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bдеп\s+күтілуде\b", re.IGNORECASE | re.UNICODE),
)

_DISCOURSE_FORMULA_PATTERNS = (
    re.compile(r"^\s*(?:қорытындылай\s+келе|атап\s+өткен\s+жөн|атап\s+айтқанда|айта\s+кету\s+керек)\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"^\s*(?:осыған\s+байланысты|осыған\s+орай|сонымен\s+қатар|сонымен\s+бірге)\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"^\s*(?:өз\s+кезегінде|осының\s+арқасында|тұтастай\s+алғанда|сол\s+себепті)\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"^\s*(?:қазіргі\s+уақытта|қазіргі\s+таңда|бүгінгі\s+таңда)\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"^\s*(?:аталған|аталмыш|қарастырылып\s+отырған)\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\b(?:нәтижелері\s+көрсеткендей|талдау\s+көрсеткендей|қорытындыларға\s+сәйкес|теориясына\s+сәйкес|тұрғысынан\s+алғанда)\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\b(?:шеңберінде|аясында|мақсатында|барысында|нәтижесінде)\b", re.IGNORECASE | re.UNICODE),
)

_SENT_END_MODAL_PARTICLES = {
    "ғой", "қой", "екен", "еді", "едік", "керек", "қажет", "сияқты", "шығар", "болар",
}

_SVO_CALQUE_VIOLATION_PATTERNS = (
    re.compile(r"\bмаңызға\s+ие\s*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bдеп\s+(?:күтілуде|есептеледі)\s*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bқол\s+жеткізуге\s+болады", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bболып\s+(?:табылады|келеді|саналады)\s*,\s*\w+", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bкөңіл\s+бөлінеді\s+және\s+\w+", re.IGNORECASE | re.UNICODE),
)

_SYNTHETIC_FINITE_VERB_ENDINGS = (
    "ғанбыз", "генбіз", "қанбыз", "кенбіз",
    "амын", "емін", "ймын", "ймін",
    "амыз", "еміз", "ймыз", "йміз",
    "отырмыз", "жатырмыз", "жүрміз", "тұрмыз",
    "дым", "дім", "тым", "тім",
    "дың", "дің",
    "ады", "еді", "йды", "йді",
    "луда", "луде", "уда", "уде",
)

_FINITE_SENT_FINAL_ENDINGS = _SYNTHETIC_FINITE_VERB_ENDINGS + (
    "ды", "ді", "ты", "ті", "дық", "дік", "тық", "тік",
    "ған", "ген", "қан", "кен", "мын", "мін", "мыз", "міз", "быз", "біз",
)

_KNOWN_SYNTHETIC_VERBS = {
    "шықты", "болды", "айттым", "жазылған", "қайттық", "түсінуіміз", "көрсетті",
    "отырмыз", "байқадым", "өзгерді", "орнаттық", "бөлісті", "қалды", "қойдым",
    "қуандым", "өтті", "келді", "жатырмыз", "болдық", "аламын", "көмектесті",
    "едік", "алды", "бардым", "түсінді", "кетті", "алдық", "тамашаладық",
    "басады", "қойдық", "қалдық", "жібердік", "келдік", "береді", "айттық",
}


@dataclass
class PatternProbeResult:
    """Deep morphosyntactic and syntactic pattern measurements for a Kazakh document."""

    chain_len_entropy: float
    chain_len_variance: float
    copula_ratio: float
    synthetic_verb_ratio: float
    discourse_rigidity_score: float
    sov_compliance_score: float
    loan_assimilation_score: float
    lexical_ttr: float
    burstiness: float
    total_words: int = 0
    total_sentences: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            name: round(float(getattr(self, name)), 4)
            for name in PROBE_METRIC_NAMES
        }


@dataclass
class PatternExplanation:
    """Bilingual diagnostic explanation for an anomalous or authentic linguistic signature."""

    dimension: str
    badge_en: str
    badge_kz: str
    severity: str
    z_score: float
    message_en: str
    message_kz: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "badge_en": self.badge_en,
            "badge_kz": self.badge_kz,
            "severity": self.severity,
            "z_score": round(float(self.z_score), 3),
            "message_en": self.message_en,
            "message_kz": self.message_kz,
        }


class GeneratorPatternProber:
    """
    Probes Kazakh text for deep agglutinative chain entropy, analytic copula crutches,
    discourse transition rigidity, SOV word-order compliance, and loanword assimilation.
    """

    def __init__(self, fst_overlay: Optional[DynamicFSTOverlay] = None) -> None:
        self.fst = fst_overlay or DynamicFSTOverlay()

    @staticmethod
    def _coerce_text(text: Union[str, Sequence[str]]) -> str:
        if text is None:
            return ""
        if isinstance(text, str):
            return text.strip()
        return " ".join(str(chunk).strip() for chunk in text if str(chunk).strip()).strip()

    def _measure_word_chain_depth(self, word: str) -> int:
        w = word.lower().strip("-")
        if len(w) <= 2 or w.isdigit() or w in _BARE_MONOMORPHEMIC:
            return 0
        if not any(ch in _KAZ_CHARS for ch in w) and not _PURE_LATIN_WORD_RE.match(w):
            norm_w = self.fst.normalizer.normalize_word(w)
            if norm_w in _BARE_MONOMORPHEMIC:
                return 0
        if w in _SINGLE_AFFIX_TEMPLATES:
            return 1

        if "-" in w:
            parts = [p for p in w.split("-") if p]
            if len(parts) >= 2:
                if parts[0] in {"іс", "бизнес", "смарт", "көлік", "сауда", "азық", "np", "әл"}:
                    return max(1, min(2, self._measure_word_chain_depth(parts[-1])))
                return min(
                    5,
                    1
                    + self._measure_word_chain_depth(parts[0])
                    + self._measure_word_chain_depth(parts[-1]),
                )

        cur = w
        depth = 0
        root_matched = False
        for rt in _KNOWN_ROOTS:
            if cur == rt:
                return 0
            if cur.startswith(rt) and len(cur) > len(rt):
                cur = cur[len(rt):]
                root_matched = True
                break

        min_rem = 0 if root_matched else 2

        # Layer 1: Relational locatives (-ндағы/-ндегі = 3; -дағы/-дегі/-тағы/-тегі = 2)
        for sfx, val in (
            ("ндағы", 3), ("ндегі", 3),
            ("дағы", 2), ("дегі", 2), ("тағы", 2), ("тегі", 2),
        ):
            if cur.endswith(sfx) and len(cur) - len(sfx) >= min_rem:
                depth += val
                cur = cur[:-len(sfx)]
                break

        # Layer 2: Productive 1st/2nd-person finite verbal & predicative agreement
        for sfx, val in (
            ("ғанбыз", 3), ("генбіз", 3), ("қанбыз", 3), ("кенбіз", 3),
            ("амын", 2), ("емін", 2), ("ймын", 2), ("ймін", 2),
            ("амыз", 2), ("еміз", 2), ("ймыз", 2), ("йміз", 2),
            ("дым", 2), ("дім", 2), ("тым", 2), ("тім", 2),
            ("дың", 2), ("дің", 2),
            ("мын", 1), ("мін", 1), ("бын", 1), ("бін", 1), ("пын", 1), ("пін", 1),
        ):
            if cur.endswith(sfx) and len(cur) - len(sfx) >= min_rem:
                depth += val
                cur = cur[:-len(sfx)]
                break
        else:
            if (
                cur.endswith(("мыз", "міз", "быз", "біз", "пыз", "піз"))
                and not cur.endswith(("ымыз", "іміз"))
                and len(cur) - 3 >= min_rem
            ):
                depth += 1
                cur = cur[:-3]
            elif cur.endswith(("дық", "дік", "тық", "тік")) and len(cur) - 3 >= min_rem:
                stem_c = cur[:-3]
                if (
                    not root_matched
                    and stem_c in {"бар", "қайт", "бол", "ал", "қал", "жібер", "кел", "орнат", "қой", "е", "шық", "айт"}
                ) or stem_c.endswith(("ла", "ле", "да", "де", "та", "те", "дыр", "дір", "тыр", "тір")):
                    depth += 2
                    cur = stem_c

        # Layer 3: Nominal case endings
        for sfx, val in (
            ("ның", 1), ("нің", 1), ("дың", 1), ("дің", 1), ("тың", 1), ("тің", 1),
            ("нда", 2), ("нде", 2),
            ("нан", 1), ("нен", 1), ("дан", 1), ("ден", 1), ("тан", 1), ("тен", 1),
            ("мен", 1), ("бен", 1), ("пен", 1),
            ("ға", 1), ("ге", 1), ("қа", 1), ("ке", 1), ("на", 1), ("не", 1),
            ("ды", 1), ("ді", 1), ("ты", 1), ("ті", 1), ("ны", 1), ("ні", 1),
            ("да", 1), ("де", 1), ("та", 1), ("те", 1),
        ):
            if cur.endswith(sfx) and len(cur) - len(sfx) >= min_rem:
                depth += val
                cur = cur[:-len(sfx)]
                break

        # Layer 4: Possessive suffixes (1st/2nd person plural/singular add deep personal stacking)
        for sfx, val in (
            ("ымыз", 2), ("іміз", 2), ("мыз", 2), ("міз", 2),
            ("ыңыз", 2), ("іңіз", 2),
            ("ым", 1), ("ім", 1), ("ың", 1), ("ің", 1),
            ("сы", 1), ("сі", 1), ("ы", 1), ("і", 1),
        ):
            if cur.endswith(sfx) and len(cur) - len(sfx) >= min_rem:
                if len(sfx) == 1 and not root_matched and len(cur) <= 3 and depth == 0:
                    continue
                depth += val
                cur = cur[:-len(sfx)]
                break

        # Layer 5: Plural suffixes
        for sfx in ("лар", "лер", "дар", "дер", "тар", "тер"):
            if cur.endswith(sfx) and len(cur) - len(sfx) >= min_rem:
                depth += 1
                cur = cur[:-len(sfx)]
                break

        # Layer 6: Converbs, participles, and gerunds
        for sfx, val in (
            ("ғаннан", 2), ("геннен", 2), ("қаннан", 2), ("кеннен", 2),
            ("атын", 2), ("етін", 2), ("йтын", 2), ("йтін", 2),
            ("ған", 1), ("ген", 1), ("қан", 1), ("кен", 1),
            ("ып", 2), ("іп", 2),
            ("у", 1),
        ):
            if cur.endswith(sfx) and len(cur) - len(sfx) >= min_rem:
                if sfx == "у" and not root_matched and len(cur) <= 3 and depth == 0:
                    continue
                depth += val
                cur = cur[:-len(sfx)]
                break
        else:
            if cur.endswith("п") and len(cur) - 1 >= max(2, min_rem) and cur[-2] in _ALL_VOWELS:
                depth += 2
                cur = cur[:-1]

        # Layer 7: Inner voice and derivational suffixes (up to 3 stacked layers)
        for _ in range(3):
            hit = False
            for sfx in (
                "лан", "лен", "дан", "ден", "тан", "тен",
                "лас", "лес", "дас", "дес", "тас", "тес",
                "дыр", "дір", "тыр", "тір", "ғыз", "гіз", "қыз", "кіз",
                "шы", "ші", "кер", "гер",
                "лық", "лік", "дық", "дік", "тық", "тік",
                "ліг", "лығ", "діг", "дығ", "тіг", "тығ",
                "сыз", "сіз", "дай", "дей", "тай", "тей",
                "ыншы", "інші", "ншы", "нші",
                "ыл", "іл", "ын", "ін", "ыс", "іс",
                "лы", "лі", "ғы", "гі", "қы", "кі", "ла", "ле",
            ):
                if cur.endswith(sfx) and len(cur) - len(sfx) >= min_rem:
                    depth += 1
                    cur = cur[:-len(sfx)]
                    hit = True
                    break
            if not hit:
                break

        if root_matched and depth == 0 and len(cur) > 0:
            depth = 1
        elif not root_matched and depth == 0:
            seg = self.fst.analyze_and_segment(w)
            fst_parts = max(0, len(seg.split()) - 1)
            depth = min(fst_parts, 2)

        return min(depth, 5)

    @staticmethod
    def _count_synthetic_verbs(text: str, total_words: int) -> float:
        masked = text
        for pat in _ANALYTIC_COPULA_PATTERNS:
            masked = pat.sub(" ", masked)

        sents = [s.strip() for s in _SENT_SPLIT_RE.split(masked) if s.strip()]
        synth_hits = 0
        for sent in sents:
            clauses = [c.strip() for c in sent.split(",") if c.strip()]
            for cl in clauses:
                c_words = [w.lower() for w in _WORD_RE.findall(cl)]
                if not c_words:
                    continue
                idx = len(c_words) - 1
                while idx >= 0 and c_words[idx] in _SENT_END_MODAL_PARTICLES:
                    idx -= 1
                if idx < 0:
                    continue
                pred = c_words[idx]
                if pred in _KNOWN_SYNTHETIC_VERBS:
                    synth_hits += 1
                elif pred.endswith(_SYNTHETIC_FINITE_VERB_ENDINGS) and len(pred) >= 5:
                    synth_hits += 1
                elif (
                    pred.endswith(("ды", "ді", "ты", "ті", "дық", "дік", "тық", "тік"))
                    and len(pred) >= 4
                    and not pred.endswith(("лық", "лік"))
                ):
                    synth_hits += 1
                elif idx < len(c_words) - 1 and pred.endswith(("ған", "ген", "қан", "кен", "тын", "тін")):
                    synth_hits += 1

        return round((synth_hits / float(max(1, total_words))) * 100.0, 4)

    @staticmethod
    def _compute_sov_compliance(sentences: List[str]) -> float:
        if not sentences:
            return 1.0
        compliant = 0.0
        for sent in sentences:
            s_clean = sent.strip()
            if not s_clean:
                continue
            penalty = 0.0
            for pat in _SVO_CALQUE_VIOLATION_PATTERNS:
                if pat.search(s_clean):
                    penalty += 0.50

            s_words = [w.lower() for w in _WORD_RE.findall(s_clean)]
            if not s_words:
                continue
            idx = len(s_words) - 1
            has_modal_tail = False
            while idx >= 0 and s_words[idx] in _SENT_END_MODAL_PARTICLES:
                has_modal_tail = True
                idx -= 1
            head = s_words[idx] if idx >= 0 else s_words[-1]
            ends_verbal = (
                has_modal_tail
                or head in _KNOWN_SYNTHETIC_VERBS
                or head.endswith(_FINITE_SENT_FINAL_ENDINGS)
                or head in {"бар", "жоқ", "емес", "мүмкін"}
            )
            if not ends_verbal:
                penalty += 0.60

            if len(s_words) >= 9:
                prefix_text = " ".join(s_words[:-3])
                for cpat in _ANALYTIC_COPULA_PATTERNS[:15]:
                    if cpat.search(prefix_text):
                        penalty += 0.35
                        break

            compliant += max(0.0, 1.0 - penalty)

        return round(compliant / float(len(sentences)), 4)

    @staticmethod
    def _compute_loan_assimilation(words: List[str]) -> float:
        loan_total = 0
        loan_valid = 0
        for raw_word in words:
            w = raw_word.strip("-")
            if not w or w.isdigit():
                continue
            # Unassimilated pure English/Latin token
            if _PURE_LATIN_WORD_RE.match(w):
                loan_total += 1
                continue
            # Mixed ASCII-Cyrillic hybrid compound (e.g., NP-толық, cloud-жүйе)
            if "-" in w:
                parts = [p for p in w.split("-") if p]
                if parts and any(c.isascii() and c.isalpha() for c in parts[0]):
                    loan_total += 1
                    continue
                target_token = parts[-1].lower() if parts else w.lower()
            else:
                target_token = w.lower()

            for stem in _LOAN_STEMS:
                if target_token == stem:
                    loan_total += 1
                    loan_valid += 1
                    break
                if target_token.startswith(stem) and len(target_token) > len(stem):
                    tail = target_token[len(stem):].lstrip("-")
                    if not tail:
                        break
                    loan_total += 1
                    stem_vowels = [ch for ch in stem if ch in (_BACK_VOWELS | _FRONT_VOWELS)]
                    is_front_stem = stem.endswith("ь") or (
                        bool(stem_vowels) and stem_vowels[-1] in _FRONT_VOWELS
                    )
                    if stem.endswith(("к", "г")) and not stem.endswith(("қ", "ғ")):
                        is_front_stem = True

                    tail_vowels = [ch for ch in tail if ch in (_BACK_VOWELS | _FRONT_VOWELS)]
                    if not tail_vowels:
                        loan_valid += 1
                    else:
                        is_front_sfx = tail_vowels[0] in _FRONT_VOWELS
                        if is_front_stem == is_front_sfx:
                            loan_valid += 1
                    break

        if loan_total == 0:
            return 1.0
        return round(loan_valid / float(loan_total), 4)

    @staticmethod
    def _compute_sliding_ttr(lower_words: List[str], window: int = 100) -> float:
        n = len(lower_words)
        if n == 0:
            return 0.0
        if n <= window:
            return round(len(set(lower_words)) / float(n), 4)
        ratios = [
            len(set(lower_words[i : i + window])) / float(window)
            for i in range(n - window + 1)
        ]
        return round(sum(ratios) / float(len(ratios)), 4)

    def probe(self, text: Union[str, Sequence[str]]) -> PatternProbeResult:
        """Computes all 9 deep morphosyntactic and syntactic pattern metrics."""
        clean_text = self._coerce_text(text)
        if not clean_text:
            return PatternProbeResult(
                chain_len_entropy=0.0,
                chain_len_variance=0.0,
                copula_ratio=0.0,
                synthetic_verb_ratio=0.0,
                discourse_rigidity_score=0.0,
                sov_compliance_score=1.0,
                loan_assimilation_score=1.0,
                lexical_ttr=0.0,
                burstiness=0.0,
                total_words=0,
                total_sentences=0,
            )

        words = _WORD_RE.findall(clean_text)
        if not words:
            return PatternProbeResult(
                chain_len_entropy=0.0,
                chain_len_variance=0.0,
                copula_ratio=0.0,
                synthetic_verb_ratio=0.0,
                discourse_rigidity_score=0.0,
                sov_compliance_score=1.0,
                loan_assimilation_score=1.0,
                lexical_ttr=0.0,
                burstiness=0.0,
                total_words=0,
                total_sentences=0,
            )

        lower_words = [w.lower() for w in words]
        n_words = len(words)
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(clean_text) if s.strip()]
        n_sents = max(1, len(sentences))

        chains = [self._measure_word_chain_depth(w) for w in words]
        hist = Counter(chains)
        entropy = -sum(
            (cnt / float(n_words)) * math.log2(cnt / float(n_words))
            for cnt in hist.values()
            if cnt > 0
        )
        mean_chain = sum(chains) / float(n_words)
        var_chain = sum((c - mean_chain) ** 2 for c in chains) / float(n_words)

        copula_hits = sum(len(pat.findall(clean_text)) for pat in _ANALYTIC_COPULA_PATTERNS)
        copula_ratio = (copula_hits / float(n_words)) * 100.0
        synth_verb_ratio = self._count_synthetic_verbs(clean_text, n_words)

        disc_hits = 0
        for s in sentences:
            disc_hits += sum(1 for pat in _DISCOURSE_FORMULA_PATTERNS if pat.search(s))
        discourse_rigidity = min(1.0, disc_hits / float(n_sents * 1.5))

        sov_score = self._compute_sov_compliance(sentences)
        loan_score = self._compute_loan_assimilation(words)
        ttr = self._compute_sliding_ttr(lower_words, window=100)

        if len(sentences) >= 2:
            sent_lens = [len(_WORD_RE.findall(s)) for s in sentences]
            mean_sl = sum(sent_lens) / float(len(sent_lens))
            std_sl = math.sqrt(
                sum((sl - mean_sl) ** 2 for sl in sent_lens) / float(len(sent_lens))
            )
            burstiness = std_sl / max(1e-6, mean_sl)
        else:
            burstiness = 0.0

        return PatternProbeResult(
            chain_len_entropy=round(entropy, 4),
            chain_len_variance=round(var_chain, 4),
            copula_ratio=round(copula_ratio, 4),
            synthetic_verb_ratio=round(synth_verb_ratio, 4),
            discourse_rigidity_score=round(discourse_rigidity, 4),
            sov_compliance_score=round(sov_score, 4),
            loan_assimilation_score=round(loan_score, 4),
            lexical_ttr=round(ttr, 4),
            burstiness=round(burstiness, 4),
            total_words=n_words,
            total_sentences=len(sentences),
        )

    def compute_deviation(
        self,
        text: Union[str, Sequence[str]],
        family: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Computes standardized z-scores across all 9 pattern dimensions relative to
        HUMAN_BASELINE_STATS (default) or a specified generator family reference.
        """
        probe_dict = self.probe(text).to_dict()
        ref_means = FAMILY_REFERENCE_PROBES.get(family) if family else None

        deviations: Dict[str, float] = {}
        for metric in PROBE_METRIC_NAMES:
            base = HUMAN_BASELINE_STATS[metric]
            target_mean = ref_means[metric] if ref_means and metric in ref_means else base["mean"]
            scale_std = max(0.04, float(base["std"]))
            z_val = (probe_dict[metric] - target_mean) / scale_std
            deviations[metric] = round(z_val, 3)
        return deviations

    def explain_patterns(
        self,
        text: Union[str, Sequence[str]],
    ) -> List[PatternExplanation]:
        """Generates prioritized bilingual Kazakh/English diagnostic pattern observations."""
        res = self.probe(text)
        if res.total_words == 0:
            return []

        devs = self.compute_deviation(text)
        findings: List[PatternExplanation] = []

        if res.copula_ratio >= 1.5:
            mult = res.copula_ratio / max(0.25, HUMAN_BASELINE_STATS["copula_ratio"]["mean"])
            sev = "high" if res.copula_ratio >= 4.0 else "medium"
            findings.append(
                PatternExplanation(
                    dimension="copula_ratio",
                    badge_en="Excessive Analytic Copula",
                    badge_kz="Шамадан тыс көмекші етістік",
                    severity=sev,
                    z_score=devs["copula_ratio"],
                    message_en=(
                        f"Analytic copula/light-verb density ({res.copula_ratio:.2f}/100w) is "
                        f"{mult:.1f}x higher than authentic human Kazakh prose (z = {devs['copula_ratio']:+.2f})."
                    ),
                    message_kz=(
                        f"Аналитикалық көмекші етістіктер жиілігі ({res.copula_ratio:.2f}/100 сөз) "
                        f"табиғи адам мәтінінен {mult:.1f} есе жоғары (z = {devs['copula_ratio']:+.2f})."
                    ),
                )
            )

        if devs["chain_len_entropy"] <= -0.85:
            sev = "high" if devs["chain_len_entropy"] <= -1.4 else "medium"
            findings.append(
                PatternExplanation(
                    dimension="chain_len_entropy",
                    badge_en="Hyper-Regular Chain Length",
                    badge_kz="Біркелкі морфема тізбегі",
                    severity=sev,
                    z_score=devs["chain_len_entropy"],
                    message_en=(
                        f"Morpheme chain entropy ({res.chain_len_entropy:.2f} bits) is compressed "
                        f"below the human agglutinative baseline ({HUMAN_BASELINE_STATS['chain_len_entropy']['mean']:.2f} bits, "
                        f"z = {devs['chain_len_entropy']:+.2f})."
                    ),
                    message_kz=(
                        f"Морфемалық тізбек энтропиясы ({res.chain_len_entropy:.2f} бит) "
                        f"табиғи агглютинативті деңгейден төмен (z = {devs['chain_len_entropy']:+.2f})."
                    ),
                )
            )

        if res.discourse_rigidity_score >= 0.20:
            sev = "high" if res.discourse_rigidity_score >= 0.40 else "medium"
            findings.append(
                PatternExplanation(
                    dimension="discourse_rigidity_score",
                    badge_en="Rigid Discourse Framing",
                    badge_kz="Қатаң дискурстық қалып",
                    severity=sev,
                    z_score=devs["discourse_rigidity_score"],
                    message_en=(
                        f"Stereotyped sentence-initial transition and postpositional framing "
                        f"score is elevated ({res.discourse_rigidity_score:.2f}, z = {devs['discourse_rigidity_score']:+.2f})."
                    ),
                    message_kz=(
                        f"Қалыптанған кіріспе оралымдар мен септеулік құрылымдар үлесі жоғары "
                        f"({res.discourse_rigidity_score:.2f}, z = {devs['discourse_rigidity_score']:+.2f})."
                    ),
                )
            )

        if devs["synthetic_verb_ratio"] <= -1.5:
            sev = "high" if devs["synthetic_verb_ratio"] <= -2.8 else "medium"
            findings.append(
                PatternExplanation(
                    dimension="synthetic_verb_ratio",
                    badge_en="Depleted Synthetic Predication",
                    badge_kz="Синтетикалық баяндауыш тапшылығы",
                    severity=sev,
                    z_score=devs["synthetic_verb_ratio"],
                    message_en=(
                        f"Genuine synthetic finite verb rate ({res.synthetic_verb_ratio:.2f}/100w) is "
                        f"substantially lower than human narrative predication (z = {devs['synthetic_verb_ratio']:+.2f})."
                    ),
                    message_kz=(
                        f"Табиғи синтетикалық баяндауыштар үлесі ({res.synthetic_verb_ratio:.2f}/100 сөз) "
                        f"адам жазбасынан едәуір төмен (z = {devs['synthetic_verb_ratio']:+.2f})."
                    ),
                )
            )

        if res.sov_compliance_score < 0.92:
            sev = "high" if res.sov_compliance_score < 0.80 else "medium"
            findings.append(
                PatternExplanation(
                    dimension="sov_compliance_score",
                    badge_en="SVO / Calque Word-Order Leakage",
                    badge_kz="SVO / Калька орын тәртібі",
                    severity=sev,
                    z_score=devs["sov_compliance_score"],
                    message_en=(
                        f"Head-final Kazakh SOV compliance dropped to {res.sov_compliance_score:.2f} "
                        f"due to cross-lingual matrix/auxiliary calquing (z = {devs['sov_compliance_score']:+.2f})."
                    ),
                    message_kz=(
                        f"Қазақ тілінің SOV сөз тәртібі сақталу көрсеткіші калькалық құрылымдар әсерінен "
                        f"{res.sov_compliance_score:.2f} дейін төмендеді (z = {devs['sov_compliance_score']:+.2f})."
                    ),
                )
            )

        if res.loan_assimilation_score < 0.95:
            sev = "high" if res.loan_assimilation_score < 0.75 else "medium"
            findings.append(
                PatternExplanation(
                    dimension="loan_assimilation_score",
                    badge_en="Loanword Harmony Violation",
                    badge_kz="Кірме сөз үндестігінің бұзылуы",
                    severity=sev,
                    z_score=devs["loan_assimilation_score"],
                    message_en=(
                        f"Loanword stem-suffix assimilation score ({res.loan_assimilation_score:.2f}) indicates "
                        f"unassimilated Latin lemmas or front/back vowel-harmony mismatches."
                    ),
                    message_kz=(
                        f"Кірме түбірлердің жалғау үндестігі ({res.loan_assimilation_score:.2f}) "
                        f"игерілмеген латын сөздерін немесе буын үндестігі қателерін көрсетеді."
                    ),
                )
            )

        if not findings:
            findings.append(
                PatternExplanation(
                    dimension="chain_len_entropy",
                    badge_en="Organic Human Agglutination",
                    badge_kz="Табиғи агглютинативті құрылым",
                    severity="normal",
                    z_score=devs["chain_len_entropy"],
                    message_en=(
                        f"High morpheme chain entropy ({res.chain_len_entropy:.2f} bits), direct synthetic "
                        f"predication ({res.synthetic_verb_ratio:.2f}/100w), and {res.sov_compliance_score:.0%} SOV compliance align with authentic Kazakh prose."
                    ),
                    message_kz=(
                        f"Жоғары морфемалық энтропия ({res.chain_len_entropy:.2f} бит), синтетикалық баяндауыштар "
                        f"({res.synthetic_verb_ratio:.2f}/100 сөз) және {res.sov_compliance_score:.0%} SOV сәйкестігі табиғи қазақ мәтініне сай."
                    ),
                )
            )

        findings.sort(key=lambda item: abs(item.z_score), reverse=True)
        return findings

    def normalize_for_radar(
        self,
        probe: Union[PatternProbeResult, Dict[str, float]],
    ) -> Dict[str, float]:
        """Normalizes the 8 canonical radar dimensions onto [0.0, 1.0]."""
        raw = probe.to_dict() if isinstance(probe, PatternProbeResult) else dict(probe)
        normalized: Dict[str, float] = {}
        for key in RADAR_AXIS_KEYS:
            val = float(raw.get(key, 0.0))
            if not math.isfinite(val):
                val = 0.0
            ceil = _RADAR_CEILINGS.get(key, 1.0)
            normalized[key] = round(max(0.0, min(1.0, val / ceil)), 4)
        return normalized

    def get_human_radar_baseline(self) -> Dict[str, float]:
        """Returns the normalized [0, 1] 8-axis radar profile for Authentic Human Kazakh."""
        human_means = {k: v["mean"] for k, v in HUMAN_BASELINE_STATS.items()}
        return self.normalize_for_radar(human_means)

    def get_family_radar_profile(self, family: str) -> Optional[Dict[str, float]]:
        """Returns the normalized [0, 1] 8-axis radar profile for a known/wild LLM family."""
        ref = FAMILY_REFERENCE_PROBES.get(family)
        if not ref:
            return None
        return self.normalize_for_radar(ref)
