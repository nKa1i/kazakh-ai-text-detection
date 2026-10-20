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
            if word.isupper():
                return mapped.upper()
            elif word.istitle():
                return mapped.capitalize()
            return mapped
        return word

    def normalize_text(self, text: str) -> str:
        if not text:
            return ""
        tokens = re.split(r'(\s+|[^\w\s])', text)
        result = []
        for token in tokens:
            if re.match(r'^[а-яёәіңғүұқөһА-ЯЁӘІҢҒҮҰҚӨҺ]+$', token):
                result.append(self.normalize_word(token))
            else:
                result.append(token)
        return "".join(result)

    def has_russian_layout_artifacts(self, word: str) -> bool:
        w_lower = word.lower()
        return w_lower in self.word_map
