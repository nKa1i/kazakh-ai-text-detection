"""
Kaz-RAID: Tier 4 Semantic Perturbation Interfaces
Implements:
1. RoundTripTranslator: Back-translation round-trip (KK -> RU -> KK) with offline simulation fallback.
2. LLMParaphraser: Adversarial LLM rephrasing with offline simulation fallback and HF pipeline loader.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

from kaz_raid.base import BasePerturbator, preserve_case


def preserve_phrase_case(original: str, replacement: str) -> str:
    """Preserve case pattern from original to replacement."""
    if not original or not replacement:
        return replacement
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    if original.istitle():
        return replacement.title()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


class RoundTripTranslator(BasePerturbator):
    """
    Simulates or executes round-trip machine translation (KK -> RU -> KK).
    Induces subtle syntactic and lexical shifts while preserving high-level semantic intent.

    Equipped with an offline deterministic semantic simulator for local testing
    and optional HuggingFace pipeline support for GPU execution.
    """

    OFFLINE_SYNONYMS: ClassVar[Dict[str, List[str]]] = {
        "жақсы": ["керемет", "жақсырақ", "тәуір", "сапалы"],
        "өте": ["аса", "тым", "айрықша", "ерекше"],
        "тауар": ["өнім", "зат", "бұйым"],
        "дүкен": ["маркет", "сауда орны"],
        "жеткізу": ["жеткізіп беру", "алып келу", "жеткізілімі"],
        "тез": ["жылдам", "шапшаң", "дер кезінде"],
        "сапа": ["сапа деңгейі", "сапалылығы"],
        "сапасы": ["сапа деңгейі", "сапа көрсеткіші"],
        "баға": ["құн", "баға деңгейі"],
        "бағасы": ["құны", "баға көрсеткіші"],
        "арзан": ["қолжетімді", "тиімді"],
        "қымбат": ["құнды", "жоғары бағалы"],
        "ұнады": ["көңілімнен шықты", "ұнап қалды", "жарады"],
        "ұнамады": ["көңілім толмады", "ұнаған жоқ"],
        "керемет": ["тамаша", "ғажап", "күшті"],
        "тамаша": ["керемет", "өте жақсы"],
        "сатушы": ["дүкен қызметкері", "сатушысы"],
        "рахмет": ["алғыс білдіремін", "мың алғыс"],
        "әдемі": ["көркем", "сәнді"],
        "ыңғайлы": ["қолайлы", "жайлы"],
        "берік": ["мықты", "төзімді"],
        "жаңа": ["соңғы үлгідегі", "жаңадан шыққан"],
        "ақау": ["кемшілік", "ақаулық"],
        "болды": ["болып шықты", "болған еді"],
        "болады": ["болуы мүмкін", "болып табылады"],
        "маған": ["бізге", "өзіме"],
        "күттім": ["күтіп отырдым", "күткен едім"],
        "келді": ["келіп жетті", "жеткізілді"],
        "алдым": ["сатып алдым", "қабылдап алдым"],
        "тапсырыс": ["заказ", "тапсырысым"],
        "ризамын": ["өте қуаныштымын", "риза болдым"],
    }

    def __init__(
        self,
        device: str = "cpu",
        model_name_fwd: Optional[str] = None,
        model_name_bwd: Optional[str] = None,
        name: str = "back_translation",
        tier: str = "tier4_semantic",
        offline_mode: bool = False,
    ):
        super().__init__(name=name, tier=tier)
        self.device = device
        self.model_name_fwd = model_name_fwd
        self.model_name_bwd = model_name_bwd
        self.offline_mode = offline_mode
        self._fwd_pipeline: Any = None
        self._bwd_pipeline: Any = None

        if not offline_mode and model_name_fwd and model_name_bwd:
            try:
                from transformers import pipeline

                self._fwd_pipeline = pipeline("translation", model=model_name_fwd, device=device)
                self._bwd_pipeline = pipeline("translation", model=model_name_bwd, device=device)
            except Exception:
                self.offline_mode = True

    def perturb(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        if not text or rate <= 0.0:
            return text

        # If live HF models are available and not in offline mode, execute neural round-trip
        if not self.offline_mode and self._fwd_pipeline and self._bwd_pipeline:
            try:
                intermediate = self._fwd_pipeline(text)[0]["translation_text"]
                back_translated = self._bwd_pipeline(intermediate)[0]["translation_text"]
                return back_translated
            except Exception:
                pass  # Fall back to deterministic offline simulation

        # Deterministic offline semantic simulation:
        # Find synonymous substitution matches
        word_regex = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
        words = list(word_regex.finditer(text))
        if not words:
            return text

        matches: List[Tuple[int, int, str, str]] = []
        rng = self.get_rng(seed)

        for m in words:
            w_lower = m.group(0).lower()
            if w_lower in self.OFFLINE_SYNONYMS:
                syn_candidates = self.OFFLINE_SYNONYMS[w_lower]
                chosen_syn = rng.choice(syn_candidates)
                matches.append((m.start(), m.end(), m.group(0), chosen_syn))

        budget = self.calculate_budget(len(matches), rate)
        if budget <= 0 or not matches:
            # Fallback if no specific dictionary matches: apply semantic modal marker
            if rate > 0 and len(words) > 0:
                markers = ["шынында да", "жалпы алғанда", "негізінен", "негізі"]
                marker = rng.choice(markers)
                return f"{preserve_phrase_case(marker, marker.capitalize())}, {text[0].lower()}{text[1:]}"
            return text

        chosen_matches = rng.sample(matches, budget)
        chosen_matches.sort(key=lambda item: item[0], reverse=True)

        for start, end, orig_text, repl in chosen_matches:
            cased_repl = preserve_phrase_case(orig_text, repl)
            text = text[:start] + cased_repl + text[end:]

        return text


class LLMParaphraser(BasePerturbator):
    """
    Simulates or executes generative adversarial LLM rephrasing for Kazakh reviews.
    Reformulates review sentences while preserving ground-truth sentiment and key facts.

    Equipped with an offline deterministic paraphrase simulator for local testing
    and optional HuggingFace pipeline support for GPU inference.
    """

    REPHRASE_TEMPLATES: ClassVar[List[Tuple[re.Pattern, str]]] = [
        (re.compile(r"\bтауар\s+жақсы\b", re.IGNORECASE), "өнімнің сапасы жоғары"),
        (re.compile(r"\bөте\s+жақсы\b", re.IGNORECASE), "керемет сапалы"),
        (re.compile(r"\bмаған\s+ұнады\b", re.IGNORECASE), "көңілімнен толық шықты"),
        (re.compile(r"\bмаған\s+қатты\s+ұнады\b", re.IGNORECASE), "өте риза болдым"),
        (re.compile(r"\bжеткізу\s+тез\s+болды\b", re.IGNORECASE), "жеткізу қызметі жылдам орындалды"),
        (re.compile(r"\bбағасы\s+арзан\b", re.IGNORECASE), "құны өте тиімді екен"),
        (re.compile(r"\bсатушыға\s+рахмет\b", re.IGNORECASE), "сатушыға үлкен алғыс білдіремін"),
        (re.compile(r"\bкеңес\s+беремін\b", re.IGNORECASE), "баршаға шын жүректен ұсынамын"),
        (re.compile(r"\bұсынамын\b", re.IGNORECASE), "сатып алуға кеңес беремін"),
        (re.compile(r"\bсатып\s+алдым\b", re.IGNORECASE), "тапсырыс беріп алған едім"),
    ]

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        name: str = "llm_paraphrase",
        tier: str = "tier4_semantic",
        offline_mode: bool = False,
    ):
        super().__init__(name=name, tier=tier)
        self.model_name = model_name
        self.device = device
        self.offline_mode = offline_mode
        self._pipeline: Any = None

        if not offline_mode and model_name:
            try:
                from transformers import pipeline

                self._pipeline = pipeline("text2text-generation", model=model_name, device=device)
            except Exception:
                self.offline_mode = True

    def perturb(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        if not text or rate <= 0.0:
            return text

        # If live LLM pipeline is loaded and active
        if not self.offline_mode and self._pipeline:
            prompt = f"Қазақ тілінде келесі пікірдің мағынасын сақтай отырып басқа сөздермен қайта жазыңыз: {text}"
            try:
                out = self._pipeline(prompt, max_new_tokens=128, do_sample=True, seed=seed)
                generated = out[0]["generated_text"].strip()
                if generated:
                    return generated
            except Exception:
                pass  # Fall back to deterministic simulator

        # Deterministic offline paraphrase simulation:
        # Check for phrase template matches
        covered: Set[int] = set()
        matches: List[Tuple[int, int, str, str]] = []

        for pattern, repl in self.REPHRASE_TEMPLATES:
            for m in pattern.finditer(text):
                span_set = set(range(m.start(), m.end()))
                if not (span_set & covered):
                    covered |= span_set
                    matches.append((m.start(), m.end(), m.group(0), repl))

        rng = self.get_rng(seed)
        if matches:
            budget = self.calculate_budget(len(matches), rate)
            chosen_matches = rng.sample(matches, budget)
            chosen_matches.sort(key=lambda item: item[0], reverse=True)
            for start, end, orig_text, repl in chosen_matches:
                cased_repl = preserve_phrase_case(orig_text, repl)
                text = text[:start] + cased_repl + text[end:]
            return text

        # Stylistic conversational discourse rephrase
        prefixes = [
            "Жалпы алғанда, ",
            "Атап айтқанда, ",
            "Шынында да, ",
            "Өз тәжірибемнен айтсам, ",
        ]
        prefix = rng.choice(prefixes)
        return f"{prefix}{text[0].lower()}{text[1:]}" if len(text) > 1 else f"{prefix}{text}"
