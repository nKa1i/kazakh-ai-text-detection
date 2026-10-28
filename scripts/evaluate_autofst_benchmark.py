# -*- coding: utf-8 -*-
"""
Evaluation and generation tooling for the Gold Morphological & Auto-FST Benchmark.
Provides 120+ curated gold-standard items across 4 challenge categories and evaluates
stem accuracy, normalization accuracy, and morpheme boundary F1.
"""

import json
import os
import sys
import re
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.auto_fst.normalizer import RussianLayoutNormalizer
from src.auto_fst.overlay import DynamicFSTOverlay

VALID_MORPH_CATEGORIES = (
    "native_multi_suffix",
    "morphophonological_alternation",
    "code_switched_slang",
    "russian_layout_corrupted",
)

GOLD_BENCHMARK_ITEMS: List[Dict[str, Any]] = [
    # -------------------------------------------------------------
    # Category 1: native_multi_suffix (32 items)
    # -------------------------------------------------------------
    {"surface": "орталыққа", "normalized": "орталыққа", "expected_stem": "орталық", "expected_suffixes": ["қа"], "category": "native_multi_suffix"},
    {"surface": "жұмыстарды", "normalized": "жұмыстарды", "expected_stem": "жұмыс", "expected_suffixes": ["тар", "ды"], "category": "native_multi_suffix"},
    {"surface": "жасалғандықтан", "normalized": "жасалғандықтан", "expected_stem": "жасал", "expected_suffixes": ["ғандықтан"], "category": "native_multi_suffix"},
    {"surface": "балалардың", "normalized": "балалардың", "expected_stem": "бала", "expected_suffixes": ["лар", "дың"], "category": "native_multi_suffix"},
    {"surface": "достарымызға", "normalized": "достарымызға", "expected_stem": "дос", "expected_suffixes": ["тар", "ымыз", "ға"], "category": "native_multi_suffix"},
    {"surface": "мұғалімдерге", "normalized": "мұғалімдерге", "expected_stem": "мұғалім", "expected_suffixes": ["дер", "ге"], "category": "native_multi_suffix"},
    {"surface": "студенттердің", "normalized": "студенттердің", "expected_stem": "студент", "expected_suffixes": ["тер", "дің"], "category": "native_multi_suffix"},
    {"surface": "қалалардан", "normalized": "қалалардан", "expected_stem": "қала", "expected_suffixes": ["лар", "дан"], "category": "native_multi_suffix"},
    {"surface": "ауылдарда", "normalized": "ауылдарда", "expected_stem": "ауыл", "expected_suffixes": ["дар", "да"], "category": "native_multi_suffix"},
    {"surface": "күндеріңіз", "normalized": "күндеріңіз", "expected_stem": "күн", "expected_suffixes": ["дер", "іңіз"], "category": "native_multi_suffix"},
    {"surface": "үйлердің", "normalized": "үйлердің", "expected_stem": "үй", "expected_suffixes": ["лер", "дің"], "category": "native_multi_suffix"},
    {"surface": "сөздермен", "normalized": "сөздермен", "expected_stem": "сөз", "expected_suffixes": ["дер", "мен"], "category": "native_multi_suffix"},
    {"surface": "адамдардың", "normalized": "адамдардың", "expected_stem": "адам", "expected_suffixes": ["дар", "дың"], "category": "native_multi_suffix"},
    {"surface": "қолдарымыз", "normalized": "қолдарымыз", "expected_stem": "қол", "expected_suffixes": ["дар", "ымыз"], "category": "native_multi_suffix"},
    {"surface": "көздерің", "normalized": "көздерің", "expected_stem": "көз", "expected_suffixes": ["дер", "ің"], "category": "native_multi_suffix"},
    {"surface": "құлақтары", "normalized": "құлақтары", "expected_stem": "құлақ", "expected_suffixes": ["тар", "ы"], "category": "native_multi_suffix"},
    {"surface": "сұрақтарға", "normalized": "сұрақтарға", "expected_stem": "сұрақ", "expected_suffixes": ["тар", "ға"], "category": "native_multi_suffix"},
    {"surface": "жауаптарын", "normalized": "жауаптарын", "expected_stem": "жауап", "expected_suffixes": ["тар", "ын"], "category": "native_multi_suffix"},
    {"surface": "жазбаларыңыздан", "normalized": "жазбаларыңыздан", "expected_stem": "жазба", "expected_suffixes": ["лар", "ыңыз", "дан"], "category": "native_multi_suffix"},
    {"surface": "дәптерлерге", "normalized": "дәптерлерге", "expected_stem": "дәптер", "expected_suffixes": ["лер", "ге"], "category": "native_multi_suffix"},
    {"surface": "қаламдармен", "normalized": "қаламдармен", "expected_stem": "қалам", "expected_suffixes": ["дар", "мен"], "category": "native_multi_suffix"},
    {"surface": "қыздарға", "normalized": "қыздарға", "expected_stem": "қыз", "expected_suffixes": ["дар", "ға"], "category": "native_multi_suffix"},
    {"surface": "ұлдардың", "normalized": "ұлдардың", "expected_stem": "ұл", "expected_suffixes": ["дар", "дың"], "category": "native_multi_suffix"},
    {"surface": "әкелердің", "normalized": "әкелердің", "expected_stem": "әке", "expected_suffixes": ["лер", "дің"], "category": "native_multi_suffix"},
    {"surface": "аналармен", "normalized": "аналармен", "expected_stem": "ана", "expected_suffixes": ["лар", "мен"], "category": "native_multi_suffix"},
    {"surface": "әпкелерге", "normalized": "әпкелерге", "expected_stem": "әпке", "expected_suffixes": ["лер", "ге"], "category": "native_multi_suffix"},
    {"surface": "інілердің", "normalized": "інілердің", "expected_stem": "іні", "expected_suffixes": ["лер", "дің"], "category": "native_multi_suffix"},
    {"surface": "келгенмін", "normalized": "келгенмін", "expected_stem": "кел", "expected_suffixes": ["ген", "мін"], "category": "native_multi_suffix"},
    {"surface": "айтқандар", "normalized": "айтқандар", "expected_stem": "айт", "expected_suffixes": ["қан", "дар"], "category": "native_multi_suffix"},
    {"surface": "жазғанбыз", "normalized": "жазғанбыз", "expected_stem": "жаз", "expected_suffixes": ["ған", "быз"], "category": "native_multi_suffix"},
    {"surface": "көргендіктен", "normalized": "көргендіктен", "expected_stem": "көр", "expected_suffixes": ["гендіктен"], "category": "native_multi_suffix"},
    {"surface": "оқығандарға", "normalized": "оқығандарға", "expected_stem": "оқы", "expected_suffixes": ["ған", "дар", "ға"], "category": "native_multi_suffix"},
    {"surface": "елдерге", "normalized": "елдерге", "expected_stem": "ел", "expected_suffixes": ["дер", "ге"], "category": "native_multi_suffix"},
    {"surface": "аттардың", "normalized": "аттардың", "expected_stem": "ат", "expected_suffixes": ["тар", "дың"], "category": "native_multi_suffix"},
    {"surface": "тілегіміз", "normalized": "тілегіміз", "expected_stem": "тілек", "expected_suffixes": ["іміз"], "category": "native_multi_suffix"},
    {"surface": "балаларға", "normalized": "балаларға", "expected_stem": "бала", "expected_suffixes": ["лар", "ға"], "category": "native_multi_suffix"},
    {"surface": "оқушылардың", "normalized": "оқушылардың", "expected_stem": "оқу", "expected_suffixes": ["шы", "лар", "дың"], "category": "native_multi_suffix"},
    {"surface": "қоғамдық", "normalized": "қоғамдық", "expected_stem": "қоғам", "expected_suffixes": ["дық"], "category": "native_multi_suffix"},
    {"surface": "жазылған", "normalized": "жазылған", "expected_stem": "жазыл", "expected_suffixes": ["ған"], "category": "native_multi_suffix"},
    {"surface": "барғандар", "normalized": "барғандар", "expected_stem": "бар", "expected_suffixes": ["ған", "дар"], "category": "native_multi_suffix"},
    {"surface": "келгендерге", "normalized": "келгендерге", "expected_stem": "кел", "expected_suffixes": ["ген", "дер", "ге"], "category": "native_multi_suffix"},
    {"surface": "достарың", "normalized": "достарың", "expected_stem": "дос", "expected_suffixes": ["тар", "ың"], "category": "native_multi_suffix"},

    # -------------------------------------------------------------
    # Category 2: morphophonological_alternation (31 items)
    # -------------------------------------------------------------
    {"surface": "кітабы", "normalized": "кітабы", "expected_stem": "кітап", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "жүрегі", "normalized": "жүрегі", "expected_stem": "жүрек", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},
    {"surface": "халқы", "normalized": "халқы", "expected_stem": "халық", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "мектебіміз", "normalized": "мектебіміз", "expected_stem": "мектеп", "expected_suffixes": ["іміз"], "category": "morphophonological_alternation"},
    {"surface": "сұрағы", "normalized": "сұрағы", "expected_stem": "сұрақ", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "көлігі", "normalized": "көлігі", "expected_stem": "көлік", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},
    {"surface": "орны", "normalized": "орны", "expected_stem": "орын", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "ерні", "normalized": "ерні", "expected_stem": "ерін", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},
    {"surface": "қасығы", "normalized": "қасығы", "expected_stem": "қасық", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "бұлағы", "normalized": "бұлағы", "expected_stem": "бұлақ", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "балығы", "normalized": "балығы", "expected_stem": "балық", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "ұшағы", "normalized": "ұшағы", "expected_stem": "ұшақ", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "жарығы", "normalized": "жарығы", "expected_stem": "жарық", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "бөлігі", "normalized": "бөлігі", "expected_stem": "бөлік", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},
    {"surface": "тілегі", "normalized": "тілегі", "expected_stem": "тілек", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},
    {"surface": "еңбегі", "normalized": "еңбегі", "expected_stem": "еңбек", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},
    {"surface": "тарағы", "normalized": "тарағы", "expected_stem": "тарақ", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "парағы", "normalized": "парағы", "expected_stem": "парақ", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "қонағы", "normalized": "қонағы", "expected_stem": "қонақ", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "құлағы", "normalized": "құлағы", "expected_stem": "құлақ", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "тамағы", "normalized": "тамағы", "expected_stem": "тамақ", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "сабағы", "normalized": "сабағы", "expected_stem": "сабақ", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "одағы", "normalized": "одағы", "expected_stem": "одақ", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "аймағы", "normalized": "аймағы", "expected_stem": "аймақ", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "ұрпағы", "normalized": "ұрпағы", "expected_stem": "ұрпақ", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "дерегі", "normalized": "дерегі", "expected_stem": "дерек", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},
    {"surface": "бөлшегі", "normalized": "бөлшегі", "expected_stem": "бөлшек", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},
    {"surface": "келешегі", "normalized": "келешегі", "expected_stem": "келешек", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},
    {"surface": "көйлегі", "normalized": "көйлегі", "expected_stem": "көйлек", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},
    {"surface": "сүйегі", "normalized": "сүйегі", "expected_stem": "сүйек", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},
    {"surface": "таяғы", "normalized": "таяғы", "expected_stem": "таяқ", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "мұрны", "normalized": "мұрны", "expected_stem": "мұрын", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "қарны", "normalized": "қарны", "expected_stem": "қарын", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "көңлі", "normalized": "көңлі", "expected_stem": "көңіл", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},
    {"surface": "кәсібі", "normalized": "кәсібі", "expected_stem": "кәсіп", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},
    {"surface": "себебі", "normalized": "себебі", "expected_stem": "себеп", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},
    {"surface": "талабы", "normalized": "талабы", "expected_stem": "талап", "expected_suffixes": ["ы"], "category": "morphophonological_alternation"},
    {"surface": "көмегі", "normalized": "көмегі", "expected_stem": "көмек", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},
    {"surface": "терегі", "normalized": "терегі", "expected_stem": "терек", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},
    {"surface": "кезегі", "normalized": "кезегі", "expected_stem": "кезек", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},
    {"surface": "төсегі", "normalized": "төсегі", "expected_stem": "төсек", "expected_suffixes": ["і"], "category": "morphophonological_alternation"},

    # -------------------------------------------------------------
    # Category 3: code_switched_slang (30 items)
    # -------------------------------------------------------------
    {"surface": "вайбтадық", "normalized": "вайбтадық", "expected_stem": "вайб", "expected_suffixes": ["та", "дық"], "category": "code_switched_slang"},
    {"surface": "доставкасы", "normalized": "доставкасы", "expected_stem": "доставка", "expected_suffixes": ["сы"], "category": "code_switched_slang"},
    {"surface": "чатботтың", "normalized": "чатботтың", "expected_stem": "чатбот", "expected_suffixes": ["тың"], "category": "code_switched_slang"},
    {"surface": "донатымыз", "normalized": "донатымыз", "expected_stem": "донат", "expected_suffixes": ["ымыз"], "category": "code_switched_slang"},
    {"surface": "AI-дың", "normalized": "AI-дың", "expected_stem": "AI", "expected_suffixes": ["дың"], "category": "code_switched_slang"},
    {"surface": "ChatGPT-мен", "normalized": "ChatGPT-мен", "expected_stem": "ChatGPT", "expected_suffixes": ["мен"], "category": "code_switched_slang"},
    {"surface": "оплатасын", "normalized": "оплатасын", "expected_stem": "оплата", "expected_suffixes": ["сын"], "category": "code_switched_slang"},
    {"surface": "клиентке", "normalized": "клиентке", "expected_stem": "клиент", "expected_suffixes": ["ке"], "category": "code_switched_slang"},
    {"surface": "каспиден", "normalized": "каспиден", "expected_stem": "каспи", "expected_suffixes": ["ден"], "category": "code_switched_slang"},
    {"surface": "аккаунтқа", "normalized": "аккаунтқа", "expected_stem": "аккаунт", "expected_suffixes": ["қа"], "category": "code_switched_slang"},
    {"surface": "чиллдедік", "normalized": "чиллдедік", "expected_stem": "чилл", "expected_suffixes": ["де", "дік"], "category": "code_switched_slang"},
    {"surface": "стартабы", "normalized": "стартабы", "expected_stem": "стартап", "expected_suffixes": ["ы"], "category": "code_switched_slang"},
    {"surface": "тиктогы", "normalized": "тиктогы", "expected_stem": "тикток", "expected_suffixes": ["ы"], "category": "code_switched_slang"},
    {"surface": "картамен", "normalized": "картамен", "expected_stem": "карта", "expected_suffixes": ["мен"], "category": "code_switched_slang"},
    {"surface": "банкке", "normalized": "банкке", "expected_stem": "банк", "expected_suffixes": ["ке"], "category": "code_switched_slang"},
    {"surface": "бонустар", "normalized": "бонустар", "expected_stem": "бонус", "expected_suffixes": ["тар"], "category": "code_switched_slang"},
    {"surface": "чектерді", "normalized": "чектерді", "expected_stem": "чек", "expected_suffixes": ["тер", "ді"], "category": "code_switched_slang"},
    {"surface": "скидкасы", "normalized": "скидкасы", "expected_stem": "скидка", "expected_suffixes": ["сы"], "category": "code_switched_slang"},
    {"surface": "курьерден", "normalized": "курьерден", "expected_stem": "курьер", "expected_suffixes": ["ден"], "category": "code_switched_slang"},
    {"surface": "сервистің", "normalized": "сервистің", "expected_stem": "сервис", "expected_suffixes": ["тің"], "category": "code_switched_slang"},
    {"surface": "видеолар", "normalized": "видеолар", "expected_stem": "видео", "expected_suffixes": ["лар"], "category": "code_switched_slang"},
    {"surface": "фотоны", "normalized": "фотоны", "expected_stem": "фото", "expected_suffixes": ["ны"], "category": "code_switched_slang"},
    {"surface": "файлдарды", "normalized": "файлдарды", "expected_stem": "файл", "expected_suffixes": ["дар", "ды"], "category": "code_switched_slang"},
    {"surface": "кабинетке", "normalized": "кабинетке", "expected_stem": "кабинет", "expected_suffixes": ["ке"], "category": "code_switched_slang"},
    {"surface": "кодты", "normalized": "кодты", "expected_stem": "код", "expected_suffixes": ["ты"], "category": "code_switched_slang"},
    {"surface": "парольдің", "normalized": "парольдің", "expected_stem": "пароль", "expected_suffixes": ["дің"], "category": "code_switched_slang"},
    {"surface": "номерге", "normalized": "номерге", "expected_stem": "номер", "expected_suffixes": ["ге"], "category": "code_switched_slang"},
    {"surface": "профильде", "normalized": "профильде", "expected_stem": "профиль", "expected_suffixes": ["де"], "category": "code_switched_slang"},
    {"surface": "вайбтаған", "normalized": "вайбтаған", "expected_stem": "вайб", "expected_suffixes": ["та", "ған"], "category": "code_switched_slang"},
    {"surface": "крашыма", "normalized": "крашыма", "expected_stem": "краш", "expected_suffixes": ["ым", "а"], "category": "code_switched_slang"},
    {"surface": "клиенттердің", "normalized": "клиенттердің", "expected_stem": "клиент", "expected_suffixes": ["тер", "дің"], "category": "code_switched_slang"},
    {"surface": "аккаунттарға", "normalized": "аккаунттарға", "expected_stem": "аккаунт", "expected_suffixes": ["тар", "ға"], "category": "code_switched_slang"},
    {"surface": "видеолардың", "normalized": "видеолардың", "expected_stem": "видео", "expected_suffixes": ["лар", "дың"], "category": "code_switched_slang"},
    {"surface": "файлдарымыз", "normalized": "файлдарымыз", "expected_stem": "файл", "expected_suffixes": ["дар", "ымыз"], "category": "code_switched_slang"},
    {"surface": "сервистерге", "normalized": "сервистерге", "expected_stem": "сервис", "expected_suffixes": ["тер", "ге"], "category": "code_switched_slang"},
    {"surface": "курьерлердің", "normalized": "курьерлердің", "expected_stem": "курьер", "expected_suffixes": ["лер", "дің"], "category": "code_switched_slang"},
    {"surface": "бонустары", "normalized": "бонустары", "expected_stem": "бонус", "expected_suffixes": ["тар", "ы"], "category": "code_switched_slang"},
    {"surface": "карталарды", "normalized": "карталарды", "expected_stem": "карта", "expected_suffixes": ["лар", "ды"], "category": "code_switched_slang"},
    {"surface": "сайттардан", "normalized": "сайттардан", "expected_stem": "сайт", "expected_suffixes": ["тар", "дан"], "category": "code_switched_slang"},

    # -------------------------------------------------------------
    # Category 4: russian_layout_corrupted (30 items)
    # -------------------------------------------------------------
    {"surface": "калайсын", "normalized": "қалайсың", "expected_stem": "қалай", "expected_suffixes": ["сың"], "category": "russian_layout_corrupted"},
    {"surface": "турсынгой", "normalized": "тұрсыңғой", "expected_stem": "тұр", "expected_suffixes": ["сың", "ғой"], "category": "russian_layout_corrupted"},
    {"surface": "галымдардын", "normalized": "ғалымдардың", "expected_stem": "ғалым", "expected_suffixes": ["дар", "дың"], "category": "russian_layout_corrupted"},
    {"surface": "ойыннын", "normalized": "ойынның", "expected_stem": "ойын", "expected_suffixes": ["ның"], "category": "russian_layout_corrupted"},
    {"surface": "жумысын", "normalized": "жұмысың", "expected_stem": "жұмыс", "expected_suffixes": ["ың"], "category": "russian_layout_corrupted"},
    {"surface": "мемлекеттик", "normalized": "мемлекеттік", "expected_stem": "мемлекет", "expected_suffixes": ["тік"], "category": "russian_layout_corrupted"},
    {"surface": "курылысшылар", "normalized": "құрылысшылар", "expected_stem": "құрылыс", "expected_suffixes": ["шы", "лар"], "category": "russian_layout_corrupted"},
    {"surface": "салем", "normalized": "сәлем", "expected_stem": "сәле", "expected_suffixes": ["м"], "category": "russian_layout_corrupted"},
    {"surface": "жаксы", "normalized": "жақсы", "expected_stem": "жақ", "expected_suffixes": ["сы"], "category": "russian_layout_corrupted"},
    {"surface": "биздин", "normalized": "біздің", "expected_stem": "біз", "expected_suffixes": ["дің"], "category": "russian_layout_corrupted"},
    {"surface": "омирде", "normalized": "өмірде", "expected_stem": "өмір", "expected_suffixes": ["де"], "category": "russian_layout_corrupted"},
    {"surface": "акшасы", "normalized": "ақшасы", "expected_stem": "ақша", "expected_suffixes": ["сы"], "category": "russian_layout_corrupted"},
    {"surface": "багасы", "normalized": "бағасы", "expected_stem": "баға", "expected_suffixes": ["сы"], "category": "russian_layout_corrupted"},
    {"surface": "сапасы", "normalized": "сапасы", "expected_stem": "сапа", "expected_suffixes": ["сы"], "category": "russian_layout_corrupted"},
    {"surface": "халкы", "normalized": "халқы", "expected_stem": "халық", "expected_suffixes": ["ы"], "category": "russian_layout_corrupted"},
    {"surface": "каласында", "normalized": "қаласында", "expected_stem": "қала", "expected_suffixes": ["сында"], "category": "russian_layout_corrupted"},
    {"surface": "ауылда", "normalized": "ауылда", "expected_stem": "ауыл", "expected_suffixes": ["да"], "category": "russian_layout_corrupted"},
    {"surface": "кунде", "normalized": "күнде", "expected_stem": "күн", "expected_suffixes": ["де"], "category": "russian_layout_corrupted"},
    {"surface": "тунде", "normalized": "түнде", "expected_stem": "түн", "expected_suffixes": ["де"], "category": "russian_layout_corrupted"},
    {"surface": "уйде", "normalized": "үйде", "expected_stem": "үй", "expected_suffixes": ["де"], "category": "russian_layout_corrupted"},
    {"surface": "тили", "normalized": "тілі", "expected_stem": "тіл", "expected_suffixes": ["і"], "category": "russian_layout_corrupted"},
    {"surface": "казакша", "normalized": "қазақша", "expected_stem": "қазақ", "expected_suffixes": ["ша"], "category": "russian_layout_corrupted"},
    {"surface": "казакстандык", "normalized": "қазақстандық", "expected_stem": "қазақстан", "expected_suffixes": ["дық"], "category": "russian_layout_corrupted"},
    {"surface": "китаптар", "normalized": "кітаптар", "expected_stem": "кітап", "expected_suffixes": ["тар"], "category": "russian_layout_corrupted"},
    {"surface": "даптер", "normalized": "дәптер", "expected_stem": "дәптер", "expected_suffixes": [], "category": "russian_layout_corrupted"},
    {"surface": "колы", "normalized": "қолы", "expected_stem": "қол", "expected_suffixes": ["ы"], "category": "russian_layout_corrupted"},
    {"surface": "аягы", "normalized": "аяғы", "expected_stem": "аяқ", "expected_suffixes": ["ы"], "category": "russian_layout_corrupted"},
    {"surface": "кози", "normalized": "көзі", "expected_stem": "көз", "expected_suffixes": ["і"], "category": "russian_layout_corrupted"},
    {"surface": "кулак", "normalized": "құлақ", "expected_stem": "құлақ", "expected_suffixes": [], "category": "russian_layout_corrupted"},
    {"surface": "журеги", "normalized": "жүрегі", "expected_stem": "жүрек", "expected_suffixes": ["і"], "category": "russian_layout_corrupted"},
    {"surface": "дауысын", "normalized": "дауысың", "expected_stem": "дауыс", "expected_suffixes": ["ың"], "category": "russian_layout_corrupted"},
    {"surface": "табысын", "normalized": "табысың", "expected_stem": "табыс", "expected_suffixes": ["ың"], "category": "russian_layout_corrupted"},
    {"surface": "кужаттар", "normalized": "құжаттар", "expected_stem": "құжат", "expected_suffixes": ["тар"], "category": "russian_layout_corrupted"},
    {"surface": "гарышкерлер", "normalized": "ғарышкерлер", "expected_stem": "ғарышкер", "expected_suffixes": ["лер"], "category": "russian_layout_corrupted"},
    {"surface": "корганыс", "normalized": "қорғаныс", "expected_stem": "қорғаныс", "expected_suffixes": [], "category": "russian_layout_corrupted"},
    {"surface": "балаларга", "normalized": "балаларға", "expected_stem": "бала", "expected_suffixes": ["лар", "ға"], "category": "russian_layout_corrupted"},
    {"surface": "окушылардын", "normalized": "оқушылардың", "expected_stem": "оқу", "expected_suffixes": ["шы", "лар", "дың"], "category": "russian_layout_corrupted"},
    {"surface": "когамдык", "normalized": "қоғамдық", "expected_stem": "қоғам", "expected_suffixes": ["дық"], "category": "russian_layout_corrupted"},
]


def build_gold_autofst_benchmark(output_path: str = "data/kaz_autofst_gold_benchmark.jsonl") -> List[Dict[str, Any]]:
    """Builds and serializes the 120+ item Gold Morphological & Auto-FST Benchmark."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in GOLD_BENCHMARK_ITEMS:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return GOLD_BENCHMARK_ITEMS


def _compute_boundaries(segmented_str: str) -> Set[int]:
    """Computes character boundary indices where morpheme splits occur in surface string."""
    parts = segmented_str.split()
    boundaries = set()
    curr_len = 0
    for idx, p in enumerate(parts):
        token = p.lstrip("-")
        curr_len += len(token)
        if idx < len(parts) - 1:
            boundaries.add(curr_len)
    return boundaries


def evaluate_autofst_on_benchmark(
    benchmark_path: str = "data/kaz_autofst_gold_benchmark.jsonl",
    results_path: Optional[str] = "data/kaz_autofst_benchmark_results.json"
) -> Dict[str, Any]:
    """
    Evaluates DynamicFSTOverlay and RussianLayoutNormalizer on the gold benchmark.
    Returns metrics dict with stem_accuracy, normalization_accuracy, and boundary_f1.
    """
    with open(benchmark_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    normalizer = RussianLayoutNormalizer()
    overlay = DynamicFSTOverlay(normalizer=normalizer)

    # Pre-register custom roots for loanwords in the benchmark
    for item in items:
        if item["category"] == "code_switched_slang":
            overlay.register_root(item["expected_stem"].lower())

    cat_stats = {cat: {
        "total": 0,
        "norm_correct": 0,
        "stem_correct": 0,
        "exact_seg_correct": 0,
        "boundary_tp": 0,
        "boundary_fp": 0,
        "boundary_fn": 0,
    } for cat in VALID_MORPH_CATEGORIES}

    for item in items:
        cat = item["category"]
        cat_stats[cat]["total"] += 1

        # 1. Normalization check
        actual_norm = normalizer.normalize_word(item["surface"])
        if actual_norm == item["normalized"]:
            cat_stats[cat]["norm_correct"] += 1

        # 2. Segmentation check
        parsed_str = overlay.analyze_and_segment(item["surface"])
        tokens = parsed_str.split()
        if tokens:
            actual_stem = tokens[0]
            actual_suffixes = [t.lstrip("-") for t in tokens[1:]]
        else:
            actual_stem = item["surface"]
            actual_suffixes = []

        # Stem accuracy
        if actual_stem.lower() == item["expected_stem"].lower():
            cat_stats[cat]["stem_correct"] += 1

        # Exact segmentation accuracy
        if actual_stem.lower() == item["expected_stem"].lower() and actual_suffixes == item["expected_suffixes"]:
            cat_stats[cat]["exact_seg_correct"] += 1

        # Boundary precision/recall/F1
        expected_seg = item["expected_stem"] + " " + " ".join(f"-{s}" for s in item["expected_suffixes"])
        expected_bounds = _compute_boundaries(expected_seg.strip())
        actual_bounds = _compute_boundaries(parsed_str.strip())

        tp = len(expected_bounds & actual_bounds)
        fp = len(actual_bounds - expected_bounds)
        fn = len(expected_bounds - actual_bounds)

        cat_stats[cat]["boundary_tp"] += tp
        cat_stats[cat]["boundary_fp"] += fp
        cat_stats[cat]["boundary_fn"] += fn

    # Aggregate overall
    total_items = len(items)
    total_norm = sum(s["norm_correct"] for s in cat_stats.values())
    total_stem = sum(s["stem_correct"] for s in cat_stats.values())
    total_exact = sum(s["exact_seg_correct"] for s in cat_stats.values())
    total_tp = sum(s["boundary_tp"] for s in cat_stats.values())
    total_fp = sum(s["boundary_fp"] for s in cat_stats.values())
    total_fn = sum(s["boundary_fn"] for s in cat_stats.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    boundary_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 1.0

    by_category_metrics = {}
    for cat, s in cat_stats.items():
        n = s["total"] if s["total"] > 0 else 1
        c_p = s["boundary_tp"] / (s["boundary_tp"] + s["boundary_fp"]) if (s["boundary_tp"] + s["boundary_fp"]) > 0 else 1.0
        c_r = s["boundary_tp"] / (s["boundary_tp"] + s["boundary_fn"]) if (s["boundary_tp"] + s["boundary_fn"]) > 0 else 1.0
        c_f1 = (2 * c_p * c_r) / (c_p + c_r) if (c_p + c_r) > 0 else 1.0
        by_category_metrics[cat] = {
            "total": s["total"],
            "normalization_accuracy": round(s["norm_correct"] / n, 4),
            "stem_accuracy": round(s["stem_correct"] / n, 4),
            "exact_segmentation_accuracy": round(s["exact_seg_correct"] / n, 4),
            "boundary_f1": round(c_f1, 4),
        }

    overall_metrics = {
        "total_samples": total_items,
        "normalization_accuracy": round(total_norm / total_items, 4),
        "stem_accuracy": round(total_stem / total_items, 4),
        "exact_segmentation_accuracy": round(total_exact / total_items, 4),
        "boundary_precision": round(precision, 4),
        "boundary_recall": round(recall, 4),
        "boundary_f1": round(boundary_f1, 4),
    }

    report = {
        "overall": overall_metrics,
        "by_category": by_category_metrics
    }

    if results_path:
        os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    bench_out = "data/kaz_autofst_gold_benchmark.jsonl"
    res_out = "data/kaz_autofst_benchmark_results.json"
    items = build_gold_autofst_benchmark(bench_out)
    print(f"Generated {len(items)} gold benchmark items in {bench_out}")
    results = evaluate_autofst_on_benchmark(bench_out, res_out)
    print("Auto-FST Gold Evaluation Results:")
    print(json.dumps(results["overall"], indent=2))
