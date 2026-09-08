import unittest


class TestKazRaidTiers3And4(unittest.TestCase):
    def test_loanword_swap_bidirectional(self):
        from kaz_raid.code_switch import LoanwordSwap
        p = LoanwordSwap(mode="bidirectional")
        text = "Тауар сапасы жақсы, жеткізу тез болды"
        res = p.perturb(text, rate=0.50, seed=42)
        self.assertNotEqual(res, text)
        self.assertTrue("доставка" in res.lower() or "товар" in res.lower() or "качество" in res.lower())

    def test_loanword_swap_modes(self):
        from kaz_raid.code_switch import LoanwordSwap
        p_kk2ru = LoanwordSwap(mode="kk2ru")
        p_ru2kk = LoanwordSwap(mode="ru2kk")
        text_kk = "жеткізу өте тез"
        text_ru = "доставка өте тез"
        res1 = p_kk2ru.perturb(text_kk, rate=1.0, seed=42)
        self.assertTrue("доставка" in res1.lower())
        res2 = p_ru2kk.perturb(text_ru, rate=1.0, seed=42)
        self.assertTrue("жеткізу" in res2.lower())

    def test_loanword_swap_dictionary_and_casing(self):
        from kaz_raid.code_switch import LoanwordSwap
        p = LoanwordSwap()
        self.assertGreaterEqual(len(p.PAIRS), 150)

        # Casing preservation test
        res_upper = p.perturb("ТАУАР ЖЕТКІЗУ", rate=1.0, seed=42)
        self.assertTrue(res_upper.isupper())

        res_title = p.perturb("Тауар Жеткізу", rate=1.0, seed=42)
        self.assertTrue(res_title.istitle())

        # Invalid mode check
        with self.assertRaises(ValueError):
            LoanwordSwap(mode="invalid_mode")

    def test_discourse_particle_insertion(self):
        from kaz_raid.code_switch import DiscourseParticle
        p = DiscourseParticle()
        text = "Керемет өнім, маған қатты ұнады"
        res = p.perturb(text, rate=0.50, seed=42)
        self.assertTrue(any(pt in res for pt in ["ғой", "қой", "шы", "да", "ау"]))

    def test_discourse_particle_without_punctuation(self):
        from kaz_raid.code_switch import DiscourseParticle
        p = DiscourseParticle()
        text = "Тауар сапасы күшті"
        res = p.perturb(text, rate=0.50, seed=42)
        self.assertNotEqual(res, text)
        self.assertTrue(any(pt in res for pt in p.PARTICLES))

    def test_semantic_interfaces(self):
        from kaz_raid.semantic import RoundTripTranslator, LLMParaphraser
        t = RoundTripTranslator()
        l = LLMParaphraser()
        text = "Тауар жақсы"
        self.assertIsInstance(t.perturb(text, rate=0.1, seed=42), str)
        self.assertIsInstance(l.perturb(text, rate=0.1, seed=42), str)

    def test_determinism_and_zero_rate(self):
        from kaz_raid.code_switch import LoanwordSwap, DiscourseParticle
        from kaz_raid.semantic import RoundTripTranslator, LLMParaphraser
        operators = [
            LoanwordSwap(),
            DiscourseParticle(),
            RoundTripTranslator(),
            LLMParaphraser(),
        ]
        text = "Тауар өте сапалы, дүкенге рахмет"
        for op in operators:
            # Zero rate check
            self.assertEqual(op.perturb(text, rate=0.0, seed=42), text)
            # Empty text check
            self.assertEqual(op.perturb("", rate=0.5, seed=42), "")
            # Determinism check
            res1 = op.perturb(text, rate=0.3, seed=42)
            res2 = op.perturb(text, rate=0.3, seed=42)
            self.assertEqual(res1, res2)

    def test_unified_benchmark_registry(self):
        import kaz_raid
        from kaz_raid.base import BasePerturbator
        operators = kaz_raid.get_all_operators()
        self.assertEqual(len(operators), 9)
        self.assertIn("homoglyph_swap", operators)
        self.assertIn("keyboard_typo", operators)
        self.assertIn("zero_width_injection", operators)
        self.assertIn("suffix_tamperer", operators)
        self.assertIn("colloquial_contractor", operators)
        self.assertIn("loanword_swap", operators)
        self.assertIn("discourse_particle", operators)
        self.assertIn("back_translation", operators)
        self.assertIn("llm_paraphrase", operators)

        # Validate that each operator is an instance of BasePerturbator
        for name, op in operators.items():
            self.assertIsInstance(op, BasePerturbator)
            self.assertIsInstance(op.name, str)
            self.assertIsInstance(op.tier, str)


if __name__ == "__main__":
    unittest.main()
