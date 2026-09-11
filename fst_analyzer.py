"""Shared FST morphological analyzer for Kazakh AI-detection notebooks.

Extracted from `mBERT_tuned.ipynb` and `KazRoBERTa_+_FSR.ipynb` so all training
and evaluation code uses the exact same segmentation. Handles native Kazakh Turkic
agglutinative morphology (nouns + verbs) and code-switched loanwords (e.g. доставкасы -> доставка -сы).
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
        
        # Noun Case Endings
        self.cases = [
            'нің', 'ның', 'дың', 'дің', 'тың', 'тің',
            'ға', 'ге', 'қа', 'ке', 'на', 'не',
            'ны', 'ні', 'ды', 'ді', 'ты', 'ті',
            'да', 'де', 'та', 'те', 'нда', 'нде',
            'дан', 'ден', 'тан', 'тен', 'нан', 'нен',
            'мен', 'бен', 'пен',
        ]
        
        # Noun Possessive Endings
        self.possessives = [
            'ларымыз', 'леріміз', 'дарымыз', 'деріміз', 'тарымыз', 'теріміз',
            'ларыңыз', 'леріңіз', 'дарыңыз', 'деріңіз', 'тарыңыз', 'теріңіз',
            'мыз', 'міз', 'ңыз', 'ңіз', 'ымыз', 'іміз', 'ыңыз', 'іңіз',
            'лары', 'лері', 'дары', 'дері', 'тары', 'тері',
            'сын', 'сін', 'мын', 'мін',
            'м', 'ң', 'ы', 'і', 'сы', 'сі',
        ]
        
        # Plural Markers
        self.plurals = ['лар', 'лер', 'дар', 'дер', 'тар', 'тер']

        # Verbal Suffixes (Tenses, Participles & Verbal Nouns)
        self.verbal_tenses = [
            'ғандықтан', 'гендіктен', 'қандықтан', 'кендіктен',
            'атын', 'етін', 'йтын', 'йтін',
            'ған', 'ген', 'қан', 'кен',
            'ады', 'еді', 'йды', 'йді',
            'мақ', 'мек', 'бақ', 'бек', 'пақ', 'пек',
            'ды', 'ді', 'ты', 'ті',
        ]

        # Verbal Personal Agreement Endings
        self.verbal_persons = [
            'мын', 'мін', 'бын', 'бін', 'пын', 'пін',
            'мыз', 'міз', 'быз', 'біз', 'пыз', 'піз',
            'сың', 'сің', 'сыздар', 'сіздер'
        ]

        # Verbal Negation (Болымсыздық жұрнақтары)
        self.verbal_negations = ['ба', 'бе', 'па', 'пе', 'ма', 'ме']

        # Verbalizer / Derivational Suffixes (Сөзжасам жұрнақтары)
        self.verbalizers = ['лан', 'лен', 'дан', 'ден', 'тан', 'тен']

        self.case_re = re.compile(r'(' + '|'.join(self.cases) + r')$')
        self.poss_re = re.compile(r'(' + '|'.join(self.possessives) + r')$')
        self.plur_re = re.compile(r'(' + '|'.join(self.plurals) + r')$')
        self.verb_tense_re = re.compile(r'(' + '|'.join(self.verbal_tenses) + r')$')
        self.verb_person_re = re.compile(r'(' + '|'.join(self.verbal_persons) + r')$')

    def _segment_loanword(self, word):
        w_lower = word.lower()
        for root in self.loanword_roots:
            if w_lower.startswith(root) and len(w_lower) > len(root):
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

            # Native Kazakh Morphological Segmentation (Nouns & Verbs)
            original = word
            suffixes = []
            had_verb = False

            # 1. Verbal Personal Agreement Suffixes (e.g. келгенмін -> -мін)
            m_vper = self.verb_person_re.search(word)
            if m_vper and len(word[:m_vper.start()]) >= 3:
                suffixes.insert(0, m_vper.group(1))
                word = word[:m_vper.start()]
                had_verb = True

            # 2. Verbal Tense / Participle Suffixes (e.g. жасалғандықтан -> -ғандықтан / -ған)
            m_vt = self.verb_tense_re.search(word)
            if m_vt and len(word[:m_vt.start()]) >= 3:
                suffixes.insert(0, m_vt.group(1))
                word = word[:m_vt.start()]
                had_verb = True

            # 2b. Verbal Negation Suffixes (e.g. қанағаттанба -> -ба, келме -> -ме)
            if had_verb:
                for neg in self.verbal_negations:
                    if word.lower().endswith(neg) and len(word[:-len(neg)]) >= 3:
                        sfx = word[-len(neg):]
                        word = word[:-len(neg)]
                        suffixes.insert(0, sfx)
                        break

            # 2c. Verbalizer Derivational Suffixes (e.g. қанағаттан -> -тан, пайдалан -> -лан)
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
                suffixes.insert(0, match.group(1))
                word = word[:match.start()]

            # 4. Noun Possessives
            if len(word) > 3:
                match = self.poss_re.search(word)
                if match and len(word[:match.start()]) >= 3:
                    suffixes.insert(0, match.group(1))
                    word = word[:match.start()]

            # 5. Plurals
            if len(word) > 3:
                match = self.plur_re.search(word)
                if match and len(word[:match.start()]) >= 3:
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
        "Кеше кешке орталыққа келгенмін және жұмыстарды көргендіктен қайтарды."
    ]
    for s in samples:
        print("Original  :", s)
        print("Verbal FST:", analyze_and_segment(s))
        print("-" * 50)
