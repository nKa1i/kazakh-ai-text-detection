import unittest
from fst_analyzer import fst_analyzer

class TestFSTCodeSwitch(unittest.TestCase):

    def test_native_kazakh_segmentation(self):
        sample = "жазбаларыңыздан"
        result = fst_analyzer.analyze_and_segment(sample)
        self.assertIn("жазба", result)
        self.assertIn("-лар", result)

    def test_loanword_code_switch_segmentation(self):
        cases = [
            ("доставкасы", "доставка -сы"),
            ("оплатасын", "оплата -сын"),
            ("каспиден", "каспи -ден"),
            ("заказды", "заказ -ды"),
            ("инстаграмнан", "инстаграм -нан"),
            ("бонусқа", "бонус -қа"),
            ("клиентке", "клиент -ке")
        ]
        for word, expected in cases:
            res = fst_analyzer.analyze_and_segment(word)
            self.assertIn("-", res, f"Failed for {word}: got {res}")

if __name__ == "__main__":
    unittest.main()
