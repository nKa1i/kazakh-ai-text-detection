"""Shared FST morphological analyzer for Kazakh AI-detection notebooks.

Extracted from `mBERT_tuned.ipynb` and `KazRoBERTa_+_FSR.ipynb` so all training
and evaluation code uses the exact same segmentation. If the analyzer changes,
update once here, not in three places.
"""

import re


class AdvancedKazakhFSTAnalyzer:
    def __init__(self):
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
            'м', 'ң', 'ы', 'і',
        ]
        self.plurals = ['лар', 'лер', 'дар', 'дер', 'тар', 'тер']

        self.case_re = re.compile(r'(' + '|'.join(self.cases) + r')$')
        self.poss_re = re.compile(r'(' + '|'.join(self.possessives) + r')$')
        self.plur_re = re.compile(r'(' + '|'.join(self.plurals) + r')$')

    def analyze_and_segment(self, text):
        if not isinstance(text, str):
            return text
        words = text.split()
        processed_words = []
        for word in words:
            if len(word) <= 4:
                processed_words.append(word)
                continue

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


if __name__ == "__main__":
    sample = "Бұл жазбаларыңыздан AI арқылы жасалғандықтан, оларды тексеру керек."
    print("Original:", sample)
    print("FST    :", fst_analyzer.analyze_and_segment(sample))
