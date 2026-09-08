"""
Kaz-RAID: Tier 3 Code-Switching & Lexical Perturbation Attacks
Implements:
1. LoanwordSwap: Kazakh-Russian bilingual lexical code-switching (kk2ru, ru2kk, bidirectional).
2. DiscourseParticle: Kazakh modal and discourse particle injection (ғой, қой, да, де, та, те, шы, ші, ау).
"""

from __future__ import annotations

import re
from typing import ClassVar, Dict, List, Optional, Set, Tuple

from kaz_raid.base import BasePerturbator, preserve_case


def preserve_phrase_case(original: str, replacement: str) -> str:
    """
    Preserve case pattern of `original` onto `replacement`.
    Supports UPPERCASE, lowercase, Title Case, and capitalized first character.
    """
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


class LoanwordSwap(BasePerturbator):
    """
    Replaces domain words with their code-switched counterparts (Kazakh <-> Russian).
    Commonly observed in Kazakhstani e-commerce, banking, and consumer reviews
    (e.g., Kaspi, Wildberries, Flip.kz).

    Modes:
        - "kk2ru": Kazakh terms replaced by Russian loanwords.
        - "ru2kk": Russian loanwords replaced by Kazakh terms.
        - "bidirectional": Both directions eligible.
    """

    # 160+ curated bilingual review domain pairs: (kk, ru)
    PAIRS: ClassVar[List[Tuple[str, str]]] = [
        # Delivery & logistics
        ("жеткізу", "доставка"),
        ("тапсырыс", "заказ"),
        ("шабарман", "курьер"),
        ("жіберу", "отправка"),
        ("қабылдау", "прием"),
        ("алу", "получение"),
        ("беру", "выдача"),
        ("кешігу", "опоздание"),
        ("кешіктіру", "задержка"),
        ("уақытында", "вовремя"),
        ("мекенжай", "адрес"),
        ("нүкте", "пункт"),
        ("қойма", "склад"),
        ("бөлім", "отдел"),
        ("жүргізуші", "водитель"),
        # Stores, vendors & customers
        ("дүкен", "магазин"),
        ("сатушы", "продавец"),
        ("сатып алушы", "покупатель"),
        ("тұтынушы", "клиент"),
        ("кеңесші", "консультант"),
        ("басқарушы", "менеджер"),
        ("әкімші", "администратор"),
        ("қолдау", "поддержка"),
        ("қызмет", "сервис"),
        ("касса", "касса"),
        # Pricing, discounts & payments
        ("баға", "цена"),
        ("құн", "стоимость"),
        ("жеңілдік", "скидка"),
        ("науқан", "акция"),
        ("сыйлық", "подарок"),
        ("сыйақы", "бонус"),
        ("тегін", "бесплатно"),
        ("ақша", "деньги"),
        ("төлем", "оплата"),
        ("қайтару", "возврат"),
        ("айырбастау", "обмен"),
        ("өтемақы", "компенсация"),
        ("кепілдік", "гарантия"),
        ("түбіртек", "чек"),
        ("шот", "счет"),
        ("карта", "карта"),
        # Goods, packaging & condition
        ("тауар", "товар"),
        ("өнім", "продукт"),
        ("зат", "вещь"),
        ("бұйым", "изделие"),
        ("сапа", "качество"),
        ("түпнұсқа", "оригинал"),
        ("жасанды", "подделка"),
        ("ақау", "брак"),
        ("сынық", "поломка"),
        ("бұзылған", "испорченный"),
        ("қорап", "коробка"),
        ("орама", "упаковка"),
        ("сипаттама", "описание"),
        ("үлгі", "модель"),
        ("өлшем", "размер"),
        ("түс", "цвет"),
        ("салмақ", "вес"),
        ("көлем", "объем"),
        ("мата", "ткань"),
        ("киім", "одежда"),
        ("аяқ киім", "обувь"),
        ("сөмке", "сумка"),
        # Tech, electronics & components
        ("құлаққап", "наушники"),
        ("сым", "провод"),
        ("зарядтағыш", "зарядка"),
        ("аккумулятор", "батарея"),
        ("жады", "память"),
        ("дисплей", "экран"),
        ("қуат", "мощность"),
        ("жүйе", "система"),
        ("бағдарлама", "программа"),
        ("қосымша", "приложение"),
        ("нұсқаулық", "инструкция"),
        ("жинақ", "комплект"),
        ("бөлшек", "деталь"),
        ("дыбыс", "звук"),
        ("камера", "камера"),
        # Review expressions & interactions
        ("пікір", "отзыв"),
        ("бағалау", "оценка"),
        ("жұлдыз", "звезда"),
        ("ұсыныс", "рекомендация"),
        ("кеңес", "совет"),
        ("шағым", "жалоба"),
        ("алғыс", "благодарность"),
        ("рахмет", "спасибо"),
        ("өтініш", "просьба"),
        ("сенім", "доверие"),
        ("күдік", "сомнение"),
        ("әсер", "впечатление"),
        ("өкініш", "сожаление"),
        ("шындық", "правда"),
        ("өтірік", "обман"),
        ("мәселе", "проблема"),
        ("қате", "ошибка"),
        ("шешім", "решение"),
        ("жауап", "ответ"),
        ("сұрақ", "вопрос"),
        ("байланыс", "связь"),
        ("хабарлама", "сообщение"),
        ("қоңырау", "звонок"),
        ("көмек", "помощь"),
        ("жұмыс", "работа"),
        ("жылдамдық", "скорость"),
        ("сурет", "фото"),
        ("бейне", "видео"),
        # Evaluative adjectives & adverbs
        ("жақсы", "хорошо"),
        ("жаман", "плохо"),
        ("керемет", "отлично"),
        ("тамаша", "прекрасно"),
        ("нашар", "ужасно"),
        ("мықты", "мощно"),
        ("күшті", "классно"),
        ("тез", "быстро"),
        ("жылдам", "быстро"),
        ("баяу", "медленно"),
        ("арзан", "дешево"),
        ("қымбат", "дорого"),
        ("тиімді", "выгодно"),
        ("ыңғайлы", "удобно"),
        ("қолайлы", "комфортно"),
        ("ыңғайсыз", "неудобно"),
        ("әдемі", "красиво"),
        ("сәнді", "стильно"),
        ("ұқыпты", "аккуратно"),
        ("таза", "чисто"),
        ("лас", "грязно"),
        ("берік", "прочно"),
        ("жұмсақ", "мягко"),
        ("қатты", "твердо"),
        ("жеңіл", "легко"),
        ("ауыр", "тяжело"),
        ("кішкентай", "маленький"),
        ("үлкен", "большой"),
        ("тар", "узкий"),
        ("кең", "широкий"),
        ("ұзын", "длинный"),
        ("қысқа", "короткий"),
        ("жаңа", "новый"),
        ("ескі", "старый"),
        ("сапалы", "качественный"),
        ("сапасыз", "некачественный"),
        ("ақаулы", "бракованный"),
        ("сенімді", "надежно"),
        ("қауіпсіз", "безопасно"),
        ("сыпайы", "вежливо"),
        ("дөрекі", "грубо"),
        ("түсінікті", "понятно"),
        ("анық", "четко"),
        ("дұрыс", "правильно"),
        ("бұрыс", "неправильно"),
        ("толық", "полностью"),
        ("жартылай", "частично"),
        ("әрдайым", "всегда"),
        ("ешқашан", "никогда"),
        ("әрине", "конечно"),
        ("мүмкін", "возможно"),
        ("өкінішке орай", "к сожалению"),
        ("бақытымызға орай", "к счастью"),
        ("өтінемін", "пожалуйста"),
        # Review specific inflections & verbal structures
        ("сапасы", "качество"),
        ("жеткізуі", "доставка"),
        ("бағасы", "цена"),
        ("дүкені", "магазин"),
        ("сатушысы", "продавец"),
        ("тапсырысым", "заказ"),
        ("өлшемі", "размер"),
        ("түсі", "цвет"),
        ("сипаттамасы", "описание"),
        ("қызметі", "сервис"),
        ("жұмысы", "работа"),
        ("қорабы", "коробка"),
        ("орамасы", "упаковка"),
        ("жақсырақ", "лучше"),
        ("жаманырақ", "хуже"),
        ("сатып алдым", "купил"),
        ("тапсырыс бердім", "заказал"),
        ("алып келді", "привезли"),
        ("келіп түсті", "поступил"),
        ("кешігіп келді", "опоздал"),
        ("ұнады", "понравилось"),
        ("ұнамады", "не понравилось"),
        ("жарады", "подошло"),
        ("жарамады", "не подошло"),
        ("көңілімнен шықты", "устроило"),
        ("көңілім толмады", "разочаровало"),
        ("ұсынамын", "рекомендую"),
        ("ұсынбаймын", "не рекомендую"),
        ("кеңес беремін", "советую"),
        ("ризамын", "доволен"),
        ("риза емеспін", "не доволен"),
        ("алдады", "обманули"),
        ("сынды", "сломался"),
        ("жыртылды", "порвался"),
        ("жоғалды", "потерялся"),
        ("келді", "пришел"),
        ("күттім", "ждал"),
        ("көрдім", "увидел"),
        ("тексердім", "проверил"),
        ("төледім", "оплатил"),
        ("қайтардым", "вернул"),
        ("айырбастадым", "обменял"),
        # Common plurals in reviews
        ("тауарлар", "товары"),
        ("бағалар", "цены"),
        ("жеңілдіктер", "скидки"),
        ("дүкендер", "магазины"),
        ("тапсырыстар", "заказы"),
        ("өнімдер", "продукты"),
        ("пікірлер", "отзывы"),
        ("суреттер", "фотографии"),
        ("сыйлықтар", "подарки"),
        ("акциялар", "акции"),
        ("курьерлер", "курьеры"),
        ("клиенттер", "клиенты"),
        ("сатушылар", "продавцы"),
        # Time units
        ("уақыт", "время"),
        ("мерзім", "срок"),
        ("күн", "день"),
        ("сағат", "час"),
        ("минут", "минута"),
    ]

    def __init__(
        self,
        mode: str = "bidirectional",
        name: str = "loanword_swap",
        tier: str = "tier3_lexical",
    ):
        super().__init__(name=name, tier=tier)
        mode = mode.lower().strip()
        if mode not in {"kk2ru", "ru2kk", "bidirectional"}:
            raise ValueError(
                f"Invalid mode '{mode}'. Choose from 'kk2ru', 'ru2kk', or 'bidirectional'."
            )
        self.mode = mode
        self._compiled_patterns: List[Tuple[re.Pattern, str]] = self._build_patterns()

    def _build_patterns(self) -> List[Tuple[re.Pattern, str]]:
        mapping: Dict[str, str] = {}
        for kk, ru in self.PAIRS:
            kk_clean = kk.strip().lower()
            ru_clean = ru.strip().lower()
            if self.mode == "kk2ru":
                mapping[kk_clean] = ru_clean
            elif self.mode == "ru2kk":
                mapping[ru_clean] = kk_clean
            else:  # bidirectional
                mapping[kk_clean] = ru_clean
                mapping[ru_clean] = kk_clean

        # Sort keys by descending length to match multi-word phrases first
        sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
        patterns: List[Tuple[re.Pattern, str]] = []
        for src in sorted_keys:
            # Word boundary check for Cyrillic
            regex = re.compile(r"(?<![^\W\d_])" + re.escape(src) + r"(?![^\W\d_])", re.IGNORECASE)
            patterns.append((regex, mapping[src]))
        return patterns

    def perturb(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        if not text or rate <= 0.0:
            return text

        # Find non-overlapping matches across all patterns
        covered: Set[int] = set()
        matches: List[Tuple[int, int, str, str]] = []

        for regex, repl in self._compiled_patterns:
            for m in regex.finditer(text):
                span_set = set(range(m.start(), m.end()))
                if not (span_set & covered):
                    covered |= span_set
                    matches.append((m.start(), m.end(), m.group(0), repl))

        budget = self.calculate_budget(len(matches), rate)
        if budget <= 0:
            return text

        rng = self.get_rng(seed)
        chosen_matches = rng.sample(matches, budget)

        # Replace from right to left to maintain offset validity
        chosen_matches.sort(key=lambda item: item[0], reverse=True)
        for start, end, orig_text, repl in chosen_matches:
            cased_repl = preserve_phrase_case(orig_text, repl)
            text = text[:start] + cased_repl + text[end:]

        return text


class DiscourseParticle(BasePerturbator):
    """
    Injects authentic Kazakh modal and discourse particles:
    {ғой, қой, да, де, та, те, шы, ші, ау}
    at clause, comma, and sentence boundaries.
    """

    PARTICLES: ClassVar[List[str]] = [
        "ғой", "қой", "да", "де", "та", "те", "шы", "ші", "ау"
    ]

    BACK_VOWELS: ClassVar[Set[str]] = set("аоұыяю")
    FRONT_VOWELS: ClassVar[Set[str]] = set("әөүіеэи")

    VOICELESS_CONSONANTS: ClassVar[Set[str]] = set("пфкқтсшщхцч")

    def __init__(
        self,
        name: str = "discourse_particle",
        tier: str = "tier3_lexical",
    ):
        super().__init__(name=name, tier=tier)

    def _select_particle(self, prev_word: str, rng) -> str:
        """
        Select an authentic particle from PARTICLES:
        {ғой, қой, шы, да, ау, де, та, те, ші}.
        """
        return rng.choice(self.PARTICLES)

    def perturb(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        if not text or rate <= 0.0:
            return text

        # Find candidate boundary insertion points:
        # 1. Punctuation boundaries: right after a word immediately followed by punctuation (, . ! ? ; : —)
        # 2. End-of-sentence / end-of-text boundary: right after the final word
        # 3. Inter-word boundaries: right after any word followed by space
        word_regex = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
        words = list(word_regex.finditer(text))
        if not words:
            return text

        primary_slots: List[Tuple[int, str]] = []
        secondary_slots: List[Tuple[int, str]] = []

        for i, match in enumerate(words):
            word_end = match.end()
            word_text = match.group(0)

            if i == len(words) - 1:
                # Last word in text: primary sentence boundary
                primary_slots.append((word_end, word_text))
            else:
                between = text[word_end:words[i + 1].start()]
                # If there is a clause-separating punctuation between this word and the next
                if any(c in between for c in ",.!?;:—–"):
                    primary_slots.append((word_end, word_text))
                else:
                    secondary_slots.append((word_end, word_text))

        candidate_slots = primary_slots if primary_slots else secondary_slots

        budget = self.calculate_budget(len(candidate_slots), rate)
        if budget <= 0:
            return text

        rng = self.get_rng(seed)
        chosen_slots = rng.sample(candidate_slots, budget)

        # Sort slots from right to left to keep insertion indices stable
        chosen_slots.sort(key=lambda item: item[0], reverse=True)

        for ins_idx, word_text in chosen_slots:
            particle = self._select_particle(word_text, rng)
            text = text[:ins_idx] + " " + particle + text[ins_idx:]

        return text
