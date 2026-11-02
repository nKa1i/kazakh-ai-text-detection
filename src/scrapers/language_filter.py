# -*- coding: utf-8 -*-
"""
src/scrapers/language_filter.py: Lightweight Language Identification (LID) filter for Kazakh social comments.

Classifies incoming comments into Kazakh (kk), Russian (ru), Code-mixed (mixed),
or Undetermined (und) to ensure only valid Kazakh texts enter the Auto-FST pipeline.
"""

import re
from typing import Optional, Set, Tuple

# Unique Kazakh Cyrillic characters (not present in standard Russian alphabet)
KAZAKH_SPECIFIC_LETTERS: Set[str] = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")

# Frequent Russian functional stop words (unambiguous indicators of Russian syntax)
RUSSIAN_FUNCTIONAL_WORDS: Set[str] = {
    "что", "как", "это", "этот", "эта", "эти", "этом", "этой", "здесь", "тут",
    "там", "где", "куда", "откуда", "почему", "зачем", "когда", "если", "чтобы",
    "или", "либо", "то", "есть", "был", "была", "было", "были", "будет", "будут",
    "не", "ни", "да", "нет", "на", "в", "во", "из", "изо", "за", "под", "над",
    "перед", "через", "при", "про", "без", "до", "для", "о", "об", "обо", "по",
    "к", "ко", "с", "со", "от", "ото", "у", "я", "меня", "мне", "мной", "ты",
    "тебя", "тебе", "тобой", "он", "его", "ему", "им", "нём", "она", "её", "ей",
    "ею", "ней", "оно", "мы", "нас", "нам", "нами", "вы", "вас", "вам", "вами",
    "они", "их", "ними", "них", "спасибо", "пожалуйста", "здравствуйте", "привет",
    "добрый", "день", "вечер", "утро", "сколько", "стоит", "цена", "можно",
    "нужно", "надо", "нельзя", "очень", "хорошо", "отлично", "супер", "просто",
    "тоже", "также", "еще", "ещё", "уже", "сейчас", "потом", "только", "даже",
    "больше", "меньше", "совсем", "конечно", "согласен", "подскажите", "наличии",
    "заказать", "доставка", "такую", "вещь", "автором", "решение", "актуально",
    "помогите", "ответить", "откуда", "берутся", "люди", "думала", "сразу",
    "блокируйте", "аккаунт", "приятно", "знаю", "просить", "лайкать",
    "профессионализм", "ответственность", "человек", "опыт", "работы",
    "смысле", "данные", "сотрудника", "вопрос", "ответ", "наверное",
}

# Frequent Kazakh words, pronouns, and Russian-keyboard spellings
KAZAKH_LEXICON_WORDS: Set[str] = {
    # Native & Russian-layout pronouns & demonstratives
    "бұл", "бул", "бұлар", "булар", "бұны", "буны", "бұған", "буган", "бұдан", "будан",
    "осы", "осылай", "мынау", "мына", "анау", "ана", "өз", "оз", "өзі", "ози",
    "өзім", "озим", "өзің", "озин", "өзіне", "озине", "өздері", "оздери",
    "біз", "биз", "біздің", "биздин", "бізге", "бизге", "бізді", "бизди", "бізден", "бизден",
    "сіз", "сиз", "сіздің", "сиздин", "сізге", "сизге", "сізді", "сизди", "сіздер", "сиздер",
    "сен", "сенің", "сенин", "саған", "саган", "маған", "маган", "оған", "оган",
    "оның", "онын", "оларды", "оларды", "олардың", "олардын", "барлық", "барлык",
    "барлығы", "барлыгы", "бәрі", "бари", "бәрін", "барин", "осында", "сосын", "енді", "енди",
    # Frequent adverbs, nouns, verbs, greetings
    "рахмет", "сәлем", "салем", "сәлеметсіз", "салеметсиз", "жақсы", "жаксы", "жаман",
    "өте", "оте", "қатты", "катты", "көп", "коп", "әлі", "али", "қазір", "казир",
    "бүгін", "бугин", "ертең", "ертен", "кеше", "жаңа", "жана", "дұрыс", "дурыс",
    "болды", "болады", "болмады", "болсын", "болып", "жатыр", "тұр", "тур",
    "жүр", "жур", "отыр", "келді", "келди", "келемін", "келемин", "кетті", "кетти",
    "алдым", "бердім", "бердим", "көрдім", "кордим", "білемін", "билемин",
    "білмеймін", "билмеймин", "түсіндім", "тусиндим", "тексер", "тексерші", "тексерши",
    "қара", "кара", "айт", "жаз", "сөйле", "сойле", "күтіңіз", "кутиниз", "жіберіңіз", "жибериниз",
    "керек", "қажет", "кажет", "емес", "мүмкін", "мумкин", "әрине", "арине",
    "жоқ", "жок", "бар", "деген", "үшін", "ушин", "бірақ", "бирак", "және", "немесе",
    "туралы", "арқылы", "аркылы", "сияқты", "сиякты", "секілді", "секилди",
    "қайда", "кайда", "қашан", "кашан", "неге", "қалай", "калай", "кім", "ким", "не",
    # Fintech & Social neologisms
    "каспи", "каспиден", "каспиге", "каспидің", "каспидин", "каспимен",
    "ватсап", "ватсапқа", "ватсапка", "ватсаптан", "теңге", "тенге", "ақша", "акша",
    "тапсырыс", "қосымша", "косымша", "қосымшасы", "косымшасы", "керемет", "құттықтаймын", "куттыктаймын",
}

# Common Kazakh inflectional endings (useful for words not in explicit lexicon)
KAZAKH_SUFFIX_REGEX = re.compile(
    r"(?:лар|лер|дар|дер|тар|тер|ның|нің|дың|дің|тың|тің|ға|ге|қа|ке|"
    r"ны|ні|ды|ді|ты|ті|да|де|та|те|дан|ден|тан|тен|мен|бен|пен|"
    r"мын|мін|сың|сің|сыз|сіз|мыз|міз|ңыздар|ңіздер|дылар|ділер)$",
    re.IGNORECASE,
)


def detect_comment_language(text: str) -> Tuple[str, float, str]:
    """
    Identifies the primary language of a social comment.

    Args:
        text: Input comment string.

    Returns:
        Tuple of (lang_code, confidence, reason)
        lang_code is one of:
          - 'kk': Kazakh (native or Russian-layout)
          - 'ru': Russian
          - 'mixed': Bilingual Kazakh-Russian code-mixing
          - 'und': Undetermined / noise / english
    """
    if not text or not text.strip():
        return "und", 0.0, "empty_text"

    clean_text = text.strip()

    # 1. Check for presence of Cyrillic characters
    has_cyrillic = bool(re.search(r"[\u0400-\u04FF]", clean_text))
    if not has_cyrillic:
        return "und", 0.0, "no_cyrillic"

    # 2. Check for unique Kazakh Cyrillic letters
    kazakh_letters_found = [ch for ch in clean_text if ch in KAZAKH_SPECIFIC_LETTERS]

    # Tokenize into clean lowercase words
    words = re.findall(r"[а-яёәғқңөұүһіА-ЯЁӘҒҚҢӨҰҮҺІ]+", clean_text.lower())
    if not words:
        return "und", 0.0, "no_words"

    # Count Russian vs Kazakh signals
    ru_hits = 0
    kk_hits = 0

    for w in words:
        if w in RUSSIAN_FUNCTIONAL_WORDS:
            ru_hits += 1
        elif w in KAZAKH_LEXICON_WORDS:
            kk_hits += 1
        elif KAZAKH_SUFFIX_REGEX.search(w):
            kk_hits += 1

    # 3. Decision Matrix
    # Case A: Contains native Kazakh letters (strong signal)
    if kazakh_letters_found:
        if ru_hits >= 2 and ru_hits > kk_hits:
            return "mixed", 0.75, "kazakh_letters_with_russian_stop_words"
        return "kk", 0.95, f"kazakh_letters_count_{len(kazakh_letters_found)}"

    # Case B: No native Kazakh letters (user on Russian keyboard or Russian text)
    if kk_hits > 0 and kk_hits >= ru_hits:
        return "kk", 0.85, f"kazakh_vocabulary_hits_{kk_hits}_vs_ru_{ru_hits}"

    if ru_hits > 0 and ru_hits > kk_hits:
        return "ru", 0.90, f"russian_vocabulary_hits_{ru_hits}_vs_kk_{kk_hits}"

    # Default heuristic: If Russian-specific hard sign 'ъ' or 'ё' present and no Kazakh indicators
    if any(ch in clean_text.lower() for ch in ("ъ", "ё")):
        return "ru", 0.80, "russian_orthography_letters"

    return "und", 0.40, "insufficient_morphological_signals"


def is_kazakh_comment(text: str, allow_mixed: bool = False) -> bool:
    """
    Returns True if the text is classified as Kazakh.

    Args:
        text: Comment string to test.
        allow_mixed: If True, code-mixed comments ('mixed') are accepted.

    Returns:
        Boolean indicating Kazakh classification.
    """
    lang, _, _ = detect_comment_language(text)
    if lang == "kk":
        return True
    if allow_mixed and lang == "mixed":
        return True
    return False
