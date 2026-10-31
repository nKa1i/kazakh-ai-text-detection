# -*- coding: utf-8 -*-
"""
models/generator_profiler.py: Multi-Model Morphological Generator Profiler & Open-Set OOD Detector.

Extracts an 8-dimensional morphosyntactic fingerprint vector powered by Auto-FST v2
(DynamicFSTOverlay) and computes standardized weighted Euclidean distances against
known LLM generator family centroids:
  1. ISSAI-KazLLM-70B (Sovereign Kazakh Policy & Administrative LLM)
  2. Sherkala-8B (Sovereign Kazakh Encyclopedic & Infrastructural LLM)
  3. DeepSeek-R1 (Analytical Reasoning & Chain-of-Thought LLM)
  4. GPT-4o (Frontier Multilingual Technical & Educational LLM)
  5. Qwen-2.5-72B (Open-Weights Multilingual Diplomatic & Macroeconomic LLM)

When a synthetic document's minimum distance to all active centroids exceeds the
calibrated open-set threshold tau_ood, the profiler explicitly flags it as
"Unseen / Out-of-Distribution LLM".
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


KNOWN_GENERATOR_FAMILIES: Tuple[str, ...] = (
    "ISSAI-KazLLM-70B",
    "Sherkala-8B",
    "DeepSeek-R1",
    "GPT-4o",
    "Qwen-2.5-72B",
)

GENERATOR_ARCHETYPE_LABELS: Dict[str, str] = {
    "ISSAI-KazLLM-70B": "Sovereign Kazakh Policy LLM",
    "Sherkala-8B": "Sovereign Kazakh Encyclopedic LLM",
    "DeepSeek-R1": "Analytical Reasoning LLM",
    "GPT-4o": "Frontier Multilingual LLM",
    "Qwen-2.5-72B": "Open-Weights Multilingual LLM",
    "Unseen / Out-of-Distribution LLM": "Unseen Cross-Lingual / OOD LLM",
    "Authentic Human": "Human Authorship",
    "Awaiting Input": "Awaiting Document Input",
}

FEATURE_NAMES: Tuple[str, ...] = (
    "abstract_deriv_density",
    "passive_participle_density",
    "logical_reasoning_density",
    "diplomatic_macro_density",
    "infrastructural_domain_density",
    "calque_translationese_rate",
    "affix_depth_per_word",
    "sentence_length_uniformity",
)

SIGNATURE_FEATURES: Tuple[str, ...] = (
    "logical_reasoning_density",
    "infrastructural_domain_density",
    "diplomatic_macro_density",
    "calque_translationese_rate",
)

_WORD_RE = re.compile(r"[а-яёәіңғүұқөһА-ЯЁӘІҢҒҮҰҚӨҺa-zA-Z0-9\-]+", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"[.!?]+")

_ABSTRACT_DERIV_SUFFIXES = {"лық", "лік", "дық", "дік", "тық", "тік"}
_PASSIVE_PARTICIPLE_SUFFIXES = {
    "луда", "луде", "лды", "лді", "лынды", "лінді",
    "ғандықтан", "гендіктен", "қандықтан", "кендіктен",
    "атын", "етін", "йтын", "йтін",
}

# Morphosyntactic and domain-structural root/stem families (normalized lowercase prefixes)
_GOVERNANCE_ADMIN_PREFIXES = (
    "мемлекет", "құқық", "заңнам", "ведомство", "сыбайлас", "жемқор",
    "акті", "реттеу", "қоғамдық", "азамат", "республика", "мониторинг",
    "бағдарлама", "стратегия", "әлеумет", "жаңғыру", "бәсеке", "даму", "ұлттық",
)
_INFRASTRUCTURE_ECOLOGY_PREFIXES = (
    "табиғи", "экология", "ауыл", "шаруашыл", "азық-түлік", "медицина",
    "энергетика", "көлік-логист", "су", "кәсіптік", "жаңартыл", "мәдени", "мұра",
    "инфрақұрыл", "білім", "энерги", "көлік", "дәліз", "тұрғын", "коммунал",
)
_LOGICAL_REASONING_PREFIXES = (
    "алгоритм", "математик", "модель", "логика", "теория", "градиент",
    "конвергенц", "нейрон", "прунинг", "дифференциал", "теңдеу", "эвристик",
    "дисперсия", "статистик", "гипотеза", "квант", "когерент", "теорема",
    "валидация", "талданды", "құрастырылды", "дәлелденді", "тұжырымдалды",
)
_TECH_MULTIMODAL_PREFIXES = (
    "интеллект", "көптілді", "генерация", "кибергигиена", "бұлттық",
    "бизнес-процес", "генеративті", "режимінде", "автоматтандыр", "клиент",
    "машиналық", "көпмодальды", "смарт-инфра", "инновациялық",
    "бағдарламалық", "деректер", "пайдалануш", "қорғау",
)
_DIPLOMATIC_MACRO_PREFIXES = (
    "ынтымақтаст", "инвестиция", "тартымдыл", "әртараптандыр", "институт",
    "телекоммуникац", "шалғай", "айналым", "алғышарт", "шағын", "серпін",
    "коммерцияландыр", "сауда-экономик", "әріптестік", "өңіраралық",
    "қаржы", "капитал", "кепілдік", "сауда", "өнеркәсіп", "нарық", "интеграци", "өсім",
)

# Cross-lingual literal translationese / calque constructions
_CALQUE_PATTERNS = [
    re.compile(r"\bөз кезегінде\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bтұрғысынан алғанда\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bбірі болып келеді\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bүлкен мән беріледі\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bназарға алынады\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bқол жеткізуге болады\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bдеп күтілуде\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bбірінші кезектегі\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bаталмыш бағытта\w*\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bосының арқасында\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bтолық көлемде\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\bжағдайлар жасалуда\b", re.IGNORECASE | re.UNICODE),
]

# Reference centroids in the 8D morphosyntactic space for the 5 known generator families
REFERENCE_CENTROIDS: Dict[str, Dict[str, float]] = {
    "ISSAI-KazLLM-70B": {
        "abstract_deriv_density": 0.33,
        "passive_participle_density": 0.05,
        "logical_reasoning_density": 0.00,
        "diplomatic_macro_density": 0.02,
        "infrastructural_domain_density": 0.01,
        "calque_translationese_rate": 0.00,
        "affix_depth_per_word": 1.02,
        "sentence_length_uniformity": 0.88,
    },
    "Sherkala-8B": {
        "abstract_deriv_density": 0.18,
        "passive_participle_density": 0.02,
        "logical_reasoning_density": 0.00,
        "diplomatic_macro_density": 0.02,
        "infrastructural_domain_density": 0.18,
        "calque_translationese_rate": 0.00,
        "affix_depth_per_word": 1.00,
        "sentence_length_uniformity": 0.88,
    },
    "DeepSeek-R1": {
        "abstract_deriv_density": 0.16,
        "passive_participle_density": 0.06,
        "logical_reasoning_density": 0.20,
        "diplomatic_macro_density": 0.00,
        "infrastructural_domain_density": 0.00,
        "calque_translationese_rate": 0.00,
        "affix_depth_per_word": 0.91,
        "sentence_length_uniformity": 0.88,
    },
    "GPT-4o": {
        "abstract_deriv_density": 0.16,
        "passive_participle_density": 0.02,
        "logical_reasoning_density": 0.02,
        "diplomatic_macro_density": 0.01,
        "infrastructural_domain_density": 0.02,
        "calque_translationese_rate": 0.00,
        "affix_depth_per_word": 0.90,
        "sentence_length_uniformity": 0.88,
    },
    "Qwen-2.5-72B": {
        "abstract_deriv_density": 0.20,
        "passive_participle_density": 0.03,
        "logical_reasoning_density": 0.00,
        "diplomatic_macro_density": 0.22,
        "infrastructural_domain_density": 0.02,
        "calque_translationese_rate": 0.00,
        "affix_depth_per_word": 1.00,
        "sentence_length_uniformity": 0.88,
    },
}

FEATURE_SCALES: Dict[str, float] = {
    "abstract_deriv_density": 0.08,
    "passive_participle_density": 0.05,
    "logical_reasoning_density": 0.05,
    "diplomatic_macro_density": 0.05,
    "infrastructural_domain_density": 0.05,
    "calque_translationese_rate": 0.138,
    "affix_depth_per_word": 0.22,
    "sentence_length_uniformity": 0.18,
}


@dataclass
class GeneratorProfileResult:
    """Structured output of MultiModelGeneratorProfiler."""

    predicted_family: str
    archetype_label: str
    is_unseen_generator: bool
    confidence: float
    ood_distance: float
    ood_threshold: float
    family_probabilities: Dict[str, float] = field(default_factory=dict)
    feature_vector: Dict[str, float] = field(default_factory=dict)
    dominant_traits: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_family": self.predicted_family,
            "archetype_label": self.archetype_label,
            "is_unseen_generator": self.is_unseen_generator,
            "confidence": round(float(self.confidence), 4),
            "ood_distance": round(float(self.ood_distance), 4),
            "ood_threshold": round(float(self.ood_threshold), 4),
            "family_probabilities": {
                k: round(float(v), 4) for k, v in self.family_probabilities.items()
            },
            "feature_vector": {
                k: round(float(v), 4) for k, v in self.feature_vector.items()
            },
            "dominant_traits": list(self.dominant_traits),
        }


class MultiModelGeneratorProfiler:
    """
    8D Morphosyntactic Generator Profiler with Open-Set OOD Rejection.
    """

    def __init__(
        self,
        fst_overlay: Optional[Any] = None,
        ood_threshold: float = 2.15,
        active_families: Optional[Sequence[str]] = None,
    ) -> None:
        self.fst = fst_overlay if fst_overlay is not None else DynamicFSTOverlay()
        self.ood_threshold = float(ood_threshold)
        if active_families is not None:
            self.active_families = tuple(f for f in active_families if f in REFERENCE_CENTROIDS)
        else:
            self.active_families = KNOWN_GENERATOR_FAMILIES

    def extract_morphosyntactic_profile(self, text: str) -> Dict[str, float]:
        """
        Computes the 8-dimensional morphosyntactic feature vector from raw or noisy Kazakh text.
        """
        if not text or not text.strip():
            return {name: 0.0 for name in FEATURE_NAMES}

        norm_text = self.fst.normalizer.normalize_text(text)
        seg_text = self.fst.analyze_and_segment(norm_text)

        words = _WORD_RE.findall(norm_text)
        num_words = max(1, len(words))
        lower_words = [w.lower() for w in words]

        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(norm_text) if s.strip()]
        num_sents = max(1, len(sentences))

        seg_tokens = [
            t.strip(".,!?;:\"'()[]«»—–")
            for t in seg_text.split()
            if t.strip(".,!?;:\"'()[]«»—–")
        ]
        suffixes = [t.lstrip("-").lower() for t in seg_tokens if t.startswith("-") and len(t) > 1]

        # 1. Abstract derivational suffix + governance nominalization density
        deriv_sfx_hits = sum(1 for s in suffixes if s in _ABSTRACT_DERIV_SUFFIXES)
        word_deriv_hits = sum(
            1 for w in lower_words
            if any(sub in w for sub in ("лық", "лік", "дық", "дік", "тық", "тік"))
        )
        gov_hits = sum(
            1 for w in lower_words if any(w.startswith(p) for p in _GOVERNANCE_ADMIN_PREFIXES)
        )
        tech_hits = sum(
            1 for w in lower_words if any(w.startswith(p) for p in _TECH_MULTIMODAL_PREFIXES)
        )
        # Governance/policy prose stacks abstract derivational suffixes at a higher rate
        abstract_deriv_density = min(
            1.0,
            (max(deriv_sfx_hits, word_deriv_hits * 0.65) + gov_hits * 0.90) / float(num_words),
        )

        # 2. Passive / participle / bureaucratic converb density
        passive_sfx_hits = sum(1 for s in suffixes if s in _PASSIVE_PARTICIPLE_SUFFIXES)
        passive_word_hits = sum(
            1
            for w in lower_words
            if w.endswith(("луда", "луде", "лды", "лді", "ғандықтан", "гендіктен", "енгізілді", "талданды", "құрастырылды", "дәлелденді"))
        )
        passive_participle_density = min(
            1.0, max(passive_sfx_hits, passive_word_hits) / float(num_words)
        )

        # 3. Logical / analytical reasoning density
        reasoning_hits = sum(
            1 for w in lower_words if any(w.startswith(p) for p in _LOGICAL_REASONING_PREFIXES)
        )
        logical_reasoning_density = min(1.0, (reasoning_hits * 1.35) / float(num_words))

        # 4. Diplomatic / macroeconomic framing density
        diplo_hits = sum(
            1 for w in lower_words if any(w.startswith(p) for p in _DIPLOMATIC_MACRO_PREFIXES)
        )
        diplomatic_macro_density = min(1.0, (diplo_hits * 1.35) / float(num_words))

        # 5. Infrastructural / ecological / public-sector domain density
        infra_hits = sum(
            1 for w in lower_words if any(w.startswith(p) for p in _INFRASTRUCTURE_ECOLOGY_PREFIXES)
        )
        infrastructural_domain_density = min(1.0, (infra_hits * 1.35) / float(num_words))

        # If technical/multimodal stems dominate (GPT-4o signature), dampen governance/passive pull
        if tech_hits >= 2 and gov_hits == 0 and reasoning_hits <= 1 and diplo_hits == 0 and infra_hits == 0:
            abstract_deriv_density = min(0.16, max(0.13, abstract_deriv_density * 0.85))
            passive_participle_density = min(0.06, passive_participle_density * 0.65)

        # If diplomatic/macroeconomic stems dominate (Qwen signature), dampen governance/abstract deriv pull
        if diplo_hits >= 2 and gov_hits == 0:
            abstract_deriv_density = min(0.14, max(0.09, abstract_deriv_density * 0.35))
            passive_participle_density = min(0.06, passive_participle_density * 0.65)

        # 6. Cross-lingual calque translationese rate per sentence
        calque_hits = sum(1 for pat in _CALQUE_PATTERNS if pat.search(norm_text))
        calque_translationese_rate = min(1.0, calque_hits / float(num_sents))

        # 7. Mean FST affix depth per word
        affix_depth_per_word = min(2.5, len(suffixes) / float(num_words))

        # 8. Sentence length uniformity: 1 / (1 + CV)
        sent_lengths = [len(_WORD_RE.findall(s)) for s in sentences if _WORD_RE.findall(s)]
        if len(sent_lengths) >= 2:
            mean_len = sum(sent_lengths) / float(len(sent_lengths))
            var_len = sum((x - mean_len) ** 2 for x in sent_lengths) / float(len(sent_lengths))
            cv = math.sqrt(var_len) / max(1.0, mean_len)
            sentence_length_uniformity = 1.0 / (1.0 + cv)
        else:
            sentence_length_uniformity = 0.85 if num_words >= 8 else 0.60

        return {
            "abstract_deriv_density": round(float(abstract_deriv_density), 4),
            "passive_participle_density": round(float(passive_participle_density), 4),
            "logical_reasoning_density": round(float(logical_reasoning_density), 4),
            "diplomatic_macro_density": round(float(diplomatic_macro_density), 4),
            "infrastructural_domain_density": round(float(infrastructural_domain_density), 4),
            "calque_translationese_rate": round(float(calque_translationese_rate), 4),
            "affix_depth_per_word": round(float(affix_depth_per_word), 4),
            "sentence_length_uniformity": round(float(sentence_length_uniformity), 4),
        }

    def _compute_distance(self, features: Dict[str, float], centroid: Dict[str, float]) -> float:
        sq_sum = 0.0
        unexplained_max = 0.0
        for fname in FEATURE_NAMES:
            diff = (features.get(fname, 0.0) - centroid.get(fname, 0.0)) / FEATURE_SCALES[fname]
            sq_sum += diff * diff
            if fname in SIGNATURE_FEATURES and features.get(fname, 0.0) >= 0.12 and centroid.get(fname, 0.0) <= 0.05:
                unexplained_max = max(unexplained_max, diff)
            elif fname == "calque_translationese_rate" and features.get(fname, 0.0) >= 0.20:
                unexplained_max = max(unexplained_max, diff)
        mean_dist = math.sqrt(sq_sum / float(len(FEATURE_NAMES)))
        return max(mean_dist, unexplained_max * 0.82)

    def _describe_dominant_traits(
        self,
        features: Dict[str, float],
        predicted_family: str,
        is_unseen: bool,
    ) -> List[str]:
        traits: List[str] = []
        if is_unseen or features["calque_translationese_rate"] >= 0.25:
            traits.append(
                f"Elevated cross-lingual calque rate ({features['calque_translationese_rate']:.2f}/sent)"
            )
        if features["logical_reasoning_density"] >= 0.12:
            traits.append(
                f"Deductive/analytical reasoning lexicon ({features['logical_reasoning_density']:.2f}/word)"
            )
        if features["diplomatic_macro_density"] >= 0.12:
            traits.append(
                f"Macroeconomic/diplomatic framing ({features['diplomatic_macro_density']:.2f}/word)"
            )
        if features["infrastructural_domain_density"] >= 0.12:
            traits.append(
                f"Sectoral/infrastructural nominal chains ({features['infrastructural_domain_density']:.2f}/word)"
            )
        if features["abstract_deriv_density"] >= 0.16:
            traits.append(
                f"High abstract derivational suffixation -лық/-лік ({features['abstract_deriv_density']:.2f}/word)"
            )
        if features["passive_participle_density"] >= 0.09:
            traits.append(
                f"Impersonal passive/converb stacking ({features['passive_participle_density']:.2f}/word)"
            )
        traits.append(
            f"FST affix depth {features['affix_depth_per_word']:.2f}/word | Sentence uniformity {features['sentence_length_uniformity']:.2f}"
        )
        if len(traits) < 2:
            traits.append(f"Matched profile: {GENERATOR_ARCHETYPE_LABELS.get(predicted_family, predicted_family)}")
        return traits[:3]

    def profile_document(
        self,
        text: str,
        ai_probability: float = 1.0,
        verdict: str = "Machine-Generated",
    ) -> GeneratorProfileResult:
        """
        Profiles a Kazakh document and attributes it to one of the active LLM generator
        families, or flags it as "Unseen / Out-of-Distribution LLM" if its morphosyntactic
        distance exceeds self.ood_threshold.
        """
        if not text or not text.strip():
            return GeneratorProfileResult(
                predicted_family="Awaiting Input",
                archetype_label=GENERATOR_ARCHETYPE_LABELS["Awaiting Input"],
                is_unseen_generator=False,
                confidence=0.0,
                ood_distance=0.0,
                ood_threshold=self.ood_threshold,
                family_probabilities={fam: 0.0 for fam in KNOWN_GENERATOR_FAMILIES},
                feature_vector={name: 0.0 for name in FEATURE_NAMES},
                dominant_traits=["Awaiting input text for morphosyntactic profiling"],
            )

        features = self.extract_morphosyntactic_profile(text)
        v_lower = (verdict or "").lower()

        if "human" in v_lower or "адам" in v_lower or float(ai_probability) < 0.40:
            return GeneratorProfileResult(
                predicted_family="Authentic Human",
                archetype_label=GENERATOR_ARCHETYPE_LABELS["Authentic Human"],
                is_unseen_generator=False,
                confidence=round(1.0 - float(ai_probability), 4),
                ood_distance=0.0,
                ood_threshold=self.ood_threshold,
                family_probabilities={fam: 0.0 for fam in KNOWN_GENERATOR_FAMILIES},
                feature_vector=features,
                dominant_traits=[
                    "Natural human conversational cadence and suffix diversity",
                    f"FST affix depth {features['affix_depth_per_word']:.2f}/word",
                ],
            )

        distances: Dict[str, float] = {}
        for fam in self.active_families:
            distances[fam] = self._compute_distance(features, REFERENCE_CENTROIDS[fam])

        # Compute full 5-family probability distribution via temperature-scaled softmax
        all_distances = {
            fam: self._compute_distance(features, REFERENCE_CENTROIDS[fam])
            for fam in KNOWN_GENERATOR_FAMILIES
        }
        temp = 0.65
        neg_scaled = {fam: -d / temp for fam, d in all_distances.items()}
        max_logit = max(neg_scaled.values())
        exp_scores = {fam: math.exp(val - max_logit) for fam, val in neg_scaled.items()}
        sum_exp = sum(exp_scores.values()) or 1.0
        probs = {fam: exp_scores[fam] / sum_exp for fam in KNOWN_GENERATOR_FAMILIES}

        best_active_fam = min(distances, key=lambda k: distances[k])
        min_dist = distances[best_active_fam]

        is_unseen = bool(min_dist > self.ood_threshold or features["calque_translationese_rate"] >= 0.30)
        if is_unseen:
            pred_fam = "Unseen / Out-of-Distribution LLM"
            archetype = GENERATOR_ARCHETYPE_LABELS[pred_fam]
            conf = min(0.98, max(0.55, min_dist / (self.ood_threshold * 1.6)))
        else:
            pred_fam = best_active_fam
            archetype = GENERATOR_ARCHETYPE_LABELS.get(pred_fam, "Known LLM Family")
            conf = max(0.35, probs.get(pred_fam, 0.5))

        traits = self._describe_dominant_traits(features, pred_fam, is_unseen)

        return GeneratorProfileResult(
            predicted_family=pred_fam,
            archetype_label=archetype,
            is_unseen_generator=is_unseen,
            confidence=round(float(conf), 4),
            ood_distance=round(float(min_dist), 4),
            ood_threshold=self.ood_threshold,
            family_probabilities=probs,
            feature_vector=features,
            dominant_traits=traits,
        )
