# -*- coding: utf-8 -*-
import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fst_analyzer import analyze_and_segment

class TestFSTVerbalRules(unittest.TestCase):

    def test_verbal_segmentation(self):
        cases = [
            ("келгенмін", "кел"),
            ("жасалғандықтан", "жаса"),
            ("көргендіктен", "көр"),
            ("орындалады", "орында"),
            ("қайтарды", "қайтар")
        ]
        for word, root in cases:
            res = analyze_and_segment(word)
            self.assertTrue("-" in res or len(res.split()) > 1, f"Failed verbal segmentation for {word}: got {res}")

if __name__ == "__main__":
    unittest.main()
