import unittest

from kaz_raid.base import BasePerturbator, calculate_budget, preserve_case
from kaz_raid.orthographic import HomoglyphSwap, KeyboardTypo, ZeroWidthInjection


class TestKazRaidTier1(unittest.TestCase):
    def test_homoglyph_swap(self):
        p = HomoglyphSwap()
        text = "өте жақсы сапалы тауар"
        res = p.perturb(text, rate=0.20, seed=42)
        self.assertNotEqual(res, text)
        self.assertEqual(len(res), len(text))
        # Determinism check
        self.assertEqual(res, p.perturb(text, rate=0.20, seed=42))

    def test_keyboard_typo(self):
        p = KeyboardTypo()
        text = "қазақша керемет өнім"
        res = p.perturb(text, rate=0.20, seed=42)
        self.assertNotEqual(res, text)
        # Determinism check
        self.assertEqual(res, p.perturb(text, rate=0.20, seed=42))

    def test_zero_width_injection(self):
        p = ZeroWidthInjection()
        text = "доставка"
        res = p.perturb(text, rate=0.20, seed=42)
        self.assertTrue("\u200b" in res or "\u200c" in res)
        # Zero width chars must not be at the very start or very end
        self.assertFalse(res.startswith("\u200b") or res.startswith("\u200c"))
        self.assertFalse(res.endswith("\u200b") or res.endswith("\u200c"))
        # Determinism check
        self.assertEqual(res, p.perturb(text, rate=0.20, seed=42))

    def test_short_text_guaranteed_minimum(self):
        p = HomoglyphSwap()
        text = "зат"  # 3 chars
        res = p.perturb(text, rate=0.05, seed=42)
        self.assertNotEqual(res, text)

    def test_case_preservation(self):
        p = HomoglyphSwap()
        text = "АЛМАТЫ"
        res = p.perturb(text, rate=0.50, seed=42)
        # Should replace Cyrillic 'А' with Latin 'A'
        self.assertTrue("A" in res)

    def test_keyboard_typo_case_preservation(self):
        p = KeyboardTypo()
        text = "ҚАЗАҚСТАН"
        res = p.perturb(text, rate=0.50, seed=42)
        # Should preserve uppercase, e.g., 'Қ' -> 'К'
        self.assertTrue("К" in res or "Ә" in res or "Н" in res)
        self.assertEqual(res, res.upper())

    def test_zero_rate_returns_original(self):
        for cls in [HomoglyphSwap, KeyboardTypo, ZeroWidthInjection]:
            p = cls()
            text = "өте керемет заттар бар"
            self.assertEqual(p.perturb(text, rate=0.0, seed=42), text)

    def test_empty_and_no_target_text(self):
        for cls in [HomoglyphSwap, KeyboardTypo, ZeroWidthInjection]:
            p = cls()
            self.assertEqual(p.perturb("", rate=0.20, seed=42), "")
            self.assertEqual(p.perturb("12345 67890", rate=0.20, seed=42), "12345 67890")

    def test_budget_calculation(self):
        # rate > 0 and n > 0 -> at least 1
        self.assertEqual(calculate_budget(1, 0.05), 1)
        self.assertEqual(calculate_budget(10, 0.20), 2)
        self.assertEqual(calculate_budget(10, 0.25), 3)
        self.assertEqual(calculate_budget(0, 0.20), 0)
        self.assertEqual(calculate_budget(10, 0.0), 0)
        self.assertEqual(BasePerturbator.calculate_budget(5, 0.10), 1)

    def test_preserve_case_helper(self):
        self.assertEqual(preserve_case("A", "b"), "B")
        self.assertEqual(preserve_case("a", "b"), "b")
        self.assertEqual(preserve_case("Hello", "world"), "World")
        self.assertEqual(preserve_case("HELLO", "world"), "WORLD")
        self.assertEqual(preserve_case("", "world"), "world")


if __name__ == "__main__":
    unittest.main()
