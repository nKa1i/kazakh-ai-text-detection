import unittest


class TestKazRaidTier2(unittest.TestCase):
    def test_suffix_tamperer_stripping(self):
        from kaz_raid.morphological import SuffixTamperer
        p = SuffixTamperer(mode="strip")
        text = "Каспиден заттарды тез алдым"
        res = p.perturb(text, rate=0.30, seed=42)
        self.assertNotEqual(res, text)

    def test_suffix_tamperer_vowel_harmony(self):
        from kaz_raid.morphological import SuffixTamperer
        p = SuffixTamperer(mode="harmony")
        text = "балаларға кітаптарды берді"
        res = p.perturb(text, rate=0.30, seed=42)
        self.assertNotEqual(res, text)

    def test_colloquial_contractor(self):
        from kaz_raid.morphological import ColloquialContractor
        p = ColloquialContractor()
        text = "Мен күтіп жатырмын, тауар келе жатыр"
        res = p.perturb(text, rate=0.50, seed=42)
        self.assertTrue("жатырм" in res or "кеватр" in res or "жатыр" in res)

    def test_determinism_and_edge_cases(self):
        from kaz_raid.morphological import SuffixTamperer, ColloquialContractor
        s = SuffixTamperer()
        c = ColloquialContractor()
        text = "жақсы заттар"
        self.assertEqual(s.perturb(text, rate=0.2, seed=42), s.perturb(text, rate=0.2, seed=42))
        self.assertEqual(c.perturb(text, rate=0.2, seed=42), c.perturb(text, rate=0.2, seed=42))
        self.assertEqual(s.perturb(text, rate=0.0), text)
        self.assertEqual(c.perturb(text, rate=0.0), text)

    def test_suffix_tamperer_mixed_mode(self):
        from kaz_raid.morphological import SuffixTamperer
        p = SuffixTamperer(mode="mixed")
        text = "Каспиден заттарды алып келді"
        res = p.perturb(text, rate=0.50, seed=42)
        self.assertNotEqual(res, text)

    def test_suffix_tamperer_case_preservation(self):
        from kaz_raid.morphological import SuffixTamperer
        p = SuffixTamperer(mode="harmony")
        text = "БАЛАЛАРҒА"
        res = p.perturb(text, rate=0.50, seed=42)
        self.assertTrue(res.isupper())
        self.assertNotEqual(res, text)

    def test_colloquial_contractor_case_preservation(self):
        from kaz_raid.morphological import ColloquialContractor
        p = ColloquialContractor()
        text = "Келемін"
        res = p.perturb(text, rate=0.50, seed=42)
        self.assertEqual(res, "Келем")

        text_upper = "КЕЛЕМІН"
        res_upper = p.perturb(text_upper, rate=0.50, seed=42)
        self.assertEqual(res_upper, "КЕЛЕМ")

        text_multi = "Келе жатыр"
        res_multi = p.perturb(text_multi, rate=0.50, seed=42)
        self.assertEqual(res_multi, "Кеватр")

    def test_colloquial_contractor_patterns_coverage(self):
        from kaz_raid.morphological import ColloquialContractor
        p = ColloquialContractor()
        self.assertGreaterEqual(len(p.PATTERNS), 20)

    def test_empty_and_no_target_text(self):
        from kaz_raid.morphological import SuffixTamperer, ColloquialContractor
        for cls in [SuffixTamperer, ColloquialContractor]:
            p = cls()
            self.assertEqual(p.perturb("", rate=0.20, seed=42), "")
            self.assertEqual(p.perturb("12345 67890", rate=0.20, seed=42), "12345 67890")


if __name__ == "__main__":
    unittest.main()
