# -*- coding: utf-8 -*-
"""
Unit tests verifying Auto-FST v2 (DynamicFSTOverlay + RussianLayoutNormalizer)
integration in MorphemeTokenizer and OfflineHeuristicDetector.
"""

import unittest

from api.morpheme_tokenizer import MorphemeTokenizer
from models.heuristic_detector import OfflineHeuristicDetector
from src.auto_fst.overlay import DynamicFSTOverlay


class TestMorphemeTokenizerAutoFSTv2(unittest.TestCase):
    def test_defaults_to_dynamic_fst_overlay_and_expanded_vocab(self):
        tok = MorphemeTokenizer()
        self.assertIsInstance(tok.fst, DynamicFSTOverlay)
        self.assertGreater(len(tok.vocab), 80)
        self.assertLess(len(tok.vocab), 240)
        for expected_affix in ("-лар", "-ден", "-сы", "-лық", "-лік", "-дық", "-дік", "-ғой", "-қой"):
            self.assertIn(expected_affix, tok.vocab)

    def test_russian_layout_noisy_text_canonicalizes_affixes(self):
        tok = MorphemeTokenizer()
        noisy = "Бугин каспиден заказ бердим, доставкасы оте жаксы болдыгой"
        morph_tokens = tok.tokenize_text(noisy)
        self.assertIn("-ден", morph_tokens)
        self.assertIn("-сы", morph_tokens)
        self.assertIn("-ғой", morph_tokens)
        ids = tok.encode(noisy)
        self.assertNotIn(tok.unk_token_id, [tok.vocab["-ден"], tok.vocab["-сы"], tok.vocab["-ғой"]])
        self.assertGreater(len(ids), 6)


class TestOfflineHeuristicDetectorAutoFSTv2(unittest.TestCase):
    def setUp(self):
        self.detector = OfflineHeuristicDetector(calibrated_threshold=0.9980)

    def test_russian_layout_corrupted_ai_text_detected(self):
        noisy_ai = (
            "Корытындылай келе, бугинги танда бул маселе заманауи когам ушин манызды рол аткарады. "
            "Осыган орай жуйели турде тиимдилигин арттыру басым багыттардын бири болып табылады."
        )
        res = self.detector.predict_document(noisy_ai)
        self.assertEqual(res.verdict, "Machine-Generated")
        self.assertGreaterEqual(res.document_ai_probability, 0.9980)

    def test_russian_layout_colloquial_human_review_remains_authentic(self):
        noisy_human = (
            "Бугин каспиден заказ бердим, доставкасы оте жаксы болдыгой рахмет! "
            "Тауардын сапасы мен багасы маган катты унады, курьер тез акелди."
        )
        res = self.detector.predict_document(noisy_human)
        self.assertEqual(res.verdict, "Authentic Human")
        self.assertLess(res.document_ai_probability, 0.25)

    def test_single_copula_human_control_stays_below_calibrated_threshold(self):
        casual_two_sent = "Бұл бірінші қазақша сөйлем. Ал бұл екінші сөйлем болып табылады."
        res = self.detector.predict_document(casual_two_sent)
        self.assertEqual(res.verdict, "Authentic Human")
        self.assertLess(res.document_ai_probability, 0.9980)

    def test_bureaucratic_ai_prose_without_qorytyndylai_kele_detected(self):
        formal_ai = (
            "Аталған бағдарламаның шеңберінде экономикалық тиімділікті арттыру бойынша кешенді іс-шаралар жүзеге асырылуда. "
            "Мемлекеттік органдар мен жеке сектор арасындағы өзара іс-қимыл тұрақты дамуды қамтамасыз етеді."
        )
        res = self.detector.predict_document(formal_ai)
        self.assertEqual(res.verdict, "Machine-Generated")
        self.assertGreaterEqual(res.document_ai_probability, 0.9980)

    def test_dynamic_gate_shifts_monotonically_across_regimes(self):
        clean_formal = (
            "Қазақстан Республикасындағы цифрлық инфрақұрылым қоғамның барлық салаларында "
            "біртіндеп дамып келеді және жаңа жобалар іске қосылды."
        )
        code_switched = (
            "Каспиден доставкасы өте тез болды, курьерге рахметімді айттым ғой, "
            "приложениеде бонустар мен скидкалар күшті екен."
        )
        russian_noisy = (
            "Бугин каспиден заказ бердим, доставкасы оте жаксы болдыгой, "
            "бирак курьер сагат бесте звандаган жок."
        )

        g_formal = self.detector.compute_dynamic_gate(clean_formal)
        g_codeswitched = self.detector.compute_dynamic_gate(code_switched)
        g_noisy = self.detector.compute_dynamic_gate(russian_noisy)

        self.assertGreater(g_formal, 0.58)
        self.assertGreater(g_formal, g_codeswitched)
        self.assertGreater(g_codeswitched, g_noisy)
        self.assertLess(g_noisy, 0.40)


if __name__ == "__main__":
    unittest.main()
