"""Shared FST morphological analyzer for Kazakh AI-detection notebooks.

Extracted from `mBERT_tuned.ipynb` and `KazRoBERTa_+_FSR.ipynb` so all training
and evaluation code uses the exact same segmentation. Handles native Kazakh Turkic
agglutinative morphology and code-switched loanwords (e.g. доставкасы -> доставка -сы).
"""

import os
import re
import json

COMMON_LOANWORD_ROOTS = [
    'доставка', 'оплата', 'возврат', 'заказ', 'бонус', 'акция', 'клиент', 'карта',
    'банк', 'номер', 'приложение', 'аккаунт', 'профиль', 'страница', 'пароль', 'код',
    'каспи', 'инстаграм', 'ватсап', 'телеграм', 'сет', 'сеть', 'чек', 'счет',
    'отзыв', 'товар', 'заявка', 'гранд', 'грант', 'кабинет', 'сумма', 'меню', 'плюс',
    'видео', 'фото', 'файл', 'арна', 'оператор', 'комиссия', 'процент', 'качество',
    'сервис', 'курьер', 'бронь', 'скидка'
]


class AdvancedKazakhFSTAnalyzer:
    def __init__(self):
        self.loanword_roots = COMMON_LOANWORD_ROOTS
        self.cases = [
            'ның', 'нің', 'дың', 'дің', 'тың', 'тің',
            'ға', 'ге', 'қа', 'ке', 'на', 'не',
            'ны', 'ні', 'ды', 'ді', 'ты', 'ті',
            'да', 'де', 'та', 'те', 'нда', 'нде',
            'дан', 'ден', 'тан', 'тен', 'нан', 'нен',
            'мен', 'бен', 'пен',
        ]
        self.possessives = [
            'ларымыз', 'леріміз', 'дарымыз', 'деріміз', 'тарымыз', 'теріміз',
            'ларыңыз', 'леріңіз', 'дарыңыз', 'деріңіз', 'тарыңыз', 'теріңіз',
            'мыз', 'міз', 'ңыз', 'ңіз', 'ымыз', 'іміз', 'ыңыз', 'іңіз',
            'лары', 'лері', 'дары', 'дері', 'тары', 'тері',
            'сын', 'сін', 'мын', 'мін',
            'м', 'ң', 'ы', 'і', 'сы', 'сі',
        ]
        self.plurals = ['лар', 'лер', 'дар', 'дер', 'тар', 'тер']

        self.case_re = re.compile(r'(' + '|'.join(self.cases) + r')$')
        self.poss_re = re.compile(r'(' + '|'.join(self.possessives) + r')$')
        self.plur_re = re.compile(r'(' + '|'.join(self.plurals) + r')$')

    def _segment_loanword(self, word):
        w_lower = word.lower()
        for root in self.loanword_roots:
            if w_lower.startswith(root) and len(w_lower) > len(root):
                sfx = w_lower[len(root):]
                # Match suffix recursively if multi-suffix
                suffixes = []
                m = self.case_re.search(sfx)
                if m:
                    suffixes.insert(0, m.group(1))
                    sfx = sfx[:m.start()]
                if sfx:
                    m_p = self.poss_re.search(sfx)
                    if m_p:
                        suffixes.insert(0, m_p.group(1))
                        sfx = sfx[:m_p.start()]
                if sfx:
                    m_pl = self.plur_re.search(sfx)
                    if m_pl:
                        suffixes.insert(0, m_pl.group(1))
                        sfx = sfx[:m_pl.start()]

                if suffixes:
                    # Preserve original root casing
                    orig_root = word[:len(root)]
                    return orig_root + " " + " ".join(f"-{s}" for s in suffixes)
                else:
                    # Fallback single suffix split for loanwords
                    sfx_fallback = word[len(root):]
                    if len(sfx_fallback) >= 1:
                        return word[:len(root)] + f" -{sfx_fallback}"
        return None

    def analyze_and_segment(self, text):
        if not isinstance(text, str):
            return text
        words = text.split()
        processed_words = []
        for word in words:
            if len(word) <= 3:
                processed_words.append(word)
                continue

            # First, check code-switched loanwords
            loanword_seg = self._segment_loanword(word)
            if loanword_seg:
                processed_words.append(loanword_seg)
                continue

            # Native Kazakh Turkic morphological segmentation
            original = word
            suffixes = []

            match = self.case_re.search(word)
            if match:
                suffixes.insert(0, match.group(1))
                word = word[:match.start()]

            if len(word) > 3:
                match = self.poss_re.search(word)
                if match:
                    suffixes.insert(0, match.group(1))
                    word = word[:match.start()]

            if len(word) > 3:
                match = self.plur_re.search(word)
                if match:
                    suffixes.insert(0, match.group(1))
                    word = word[:match.start()]

            if suffixes:
                processed_words.append(word + " " + " ".join(f"-{s}" for s in suffixes))
            else:
                processed_words.append(original)
        return " ".join(processed_words)


fst_analyzer = AdvancedKazakhFSTAnalyzer()


def analyze_and_segment(text):
    return fst_analyzer.analyze_and_segment(text)


if __name__ == "__main__":
    samples = [
        "Бұл жазбаларыңыздан AI арқылы жасалғандықтан, оларды тексеру керек.",
        "Каспиден доставкасы өте тез болды, оплатасын жасадым.",
        "Инстаграмнан бонусқа акция таптым."
    ]
    for s in samples:
        print("Original:", s)
        print("Hybrid FST:", analyze_and_segment(s))
        print("-" * 50)
