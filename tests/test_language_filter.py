# -*- coding: utf-8 -*-
"""
tests/test_language_filter.py: Unit tests for Kazakh Language Identification (LID) filter.
"""

import unittest

from src.scrapers.language_filter import detect_comment_language, is_kazakh_comment


class TestLanguageFilter(unittest.TestCase):
    def test_native_kazakh_with_specific_letters(self):
        samples = [
            "Бүгін ауа райы өте жақсы болды, рахмет!",
            "Қазақстанның болашағы жастардың қолында.",
            "Каспи қосымшасы арқылы төлеңіз.",
            "Сәлеметсіз бе, бұл зат әлі бар ма?",
            "Өте керемет жаңалық, құттықтаймын!",
        ]
        for s in samples:
            lang, conf, reason = detect_comment_language(s)
            self.assertEqual(lang, "kk", f"Failed on: {s}")
            self.assertTrue(is_kazakh_comment(s))

    def test_kazakh_on_russian_layout(self):
        samples = [
            "Оте жаксы жасалган, каспиден алдым, рахмет сизге.",
            "Бугин кешке карай келемин, кутиниз.",
            "Биздин елде барлык жумыс дурыс журип жатыр.",
            "Казир хабарласамын, номер калдырыныз.",
            "Каспиге акша аудардым, тексерши.",
        ]
        for s in samples:
            lang, conf, reason = detect_comment_language(s)
            self.assertEqual(lang, "kk", f"Failed on: {s}")
            self.assertTrue(is_kazakh_comment(s))

    def test_pure_russian_comments(self):
        samples = [
            "Здравствуйте, подскажите пожалуйста, сколько стоит доставка?",
            "Спасибо большое за полезный пост, очень актуально сейчас.",
            "В наличии есть этот товар или уже все разобрали?",
            "Где можно заказать такую же вещь?",
            "Полностью согласен с автором, это правильное решение.",
        ]
        for s in samples:
            lang, conf, reason = detect_comment_language(s)
            self.assertEqual(lang, "ru", f"Failed on: {s}")
            self.assertFalse(is_kazakh_comment(s))

    def test_code_mixed_comments(self):
        text = "Рахмет сізге, очень полезно было прочитать этот пост."
        lang, conf, reason = detect_comment_language(text)
        self.assertIn(lang, ("kk", "mixed"))
        # When allow_mixed is True, is_kazakh_comment returns True
        self.assertTrue(is_kazakh_comment(text, allow_mixed=True))

    def test_undetermined_and_noise(self):
        noise = [
            "12345 67890",
            "??? !!! ...",
            "Hello, this is pure english comment.",
        ]
        for n in noise:
            lang, conf, reason = detect_comment_language(n)
            self.assertEqual(lang, "und")
            self.assertFalse(is_kazakh_comment(n))


if __name__ == "__main__":
    unittest.main()
