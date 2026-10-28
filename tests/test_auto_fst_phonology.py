# -*- coding: utf-8 -*-
"""Unit tests for Kazakh two-level morphophonological stem boundary restoration."""

import unittest
from src.auto_fst.phonology import MorphophonologicalRestorer


class TestMorphophonologicalRestorer(unittest.TestCase):
    def setUp(self):
        self.restorer = MorphophonologicalRestorer()

    def test_native_voicing_alternation_restoration(self):
        # кітап + ы -> кітабы (raw stem 'кітаб' -> 'кітап')
        self.assertEqual(self.restorer.restore_stem("кітаб", "ы"), "кітап")
        # жүрек + і -> жүрегі (raw stem 'жүрег' -> 'жүрек')
        self.assertEqual(self.restorer.restore_stem("жүрег", "і"), "жүрек")
        # тарақ + ы -> тарағы (raw stem 'тарағ' -> 'тарақ')
        self.assertEqual(self.restorer.restore_stem("тарағ", "ы"), "тарақ")
        # мектеп + іміз -> мектебіміз (raw stem 'мектеб' -> 'мектеп')
        self.assertEqual(self.restorer.restore_stem("мектеб", "іміз"), "мектеп")

    def test_loanword_voicing_alternation_restoration(self):
        # тикток + ы -> тиктогы ('тиктог' -> 'тикток')
        self.assertEqual(
            self.restorer.restore_stem("тиктог", "ы", custom_roots={"тикток", "стартап"}),
            "тикток"
        )
        # стартап + ы -> стартабы ('стартаб' -> 'стартап')
        self.assertEqual(
            self.restorer.restore_stem("стартаб", "ы", custom_roots={"тикток", "стартап"}),
            "стартап"
        )

    def test_vowel_syncope_restoration(self):
        # халық + ы -> халқы ('халқ' -> 'халық')
        self.assertEqual(self.restorer.restore_stem("халқ", "ы"), "халық")
        # орын + ы -> орны ('орн' -> 'орын')
        self.assertEqual(self.restorer.restore_stem("орн", "ы"), "орын")
        # ерін + і -> ерні ('ерн' -> 'ерін')
        self.assertEqual(self.restorer.restore_stem("ерн", "і"), "ерін")

    def test_non_alternating_voiced_stems_preserved(self):
        # клуб + ы -> клубы ('клуб' is genuinely ending in 'б', not 'п')
        self.assertEqual(self.restorer.restore_stem("клуб", "ы"), "клуб")
        # вайб + ы -> вайбы ('вайб' must stay 'вайб')
        self.assertEqual(self.restorer.restore_stem("вайб", "ы", custom_roots={"вайб"}), "вайб")


if __name__ == "__main__":
    unittest.main()
