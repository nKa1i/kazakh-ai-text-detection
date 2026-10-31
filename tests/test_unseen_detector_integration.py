# -*- coding: utf-8 -*-
"""
Unit tests verifying Stage 1 Generator-Invariant Unseen Detection and Stage 2
GeneratorProfileResult attachment in OfflineHeuristicDetector and DocumentAnalysisResult.
"""

import unittest

from models.heuristic_detector import OfflineHeuristicDetector


class TestUnseenDetectorIntegration(unittest.TestCase):
    def setUp(self):
        self.detector = OfflineHeuristicDetector(calibrated_threshold=0.9980)

    def test_unseen_calque_generator_without_seen_ai_markers_is_detected_and_flagged_ood(self):
        unseen_calque_doc = (
            "Өз кезегінде аталмыш бағыттағы жұмыстарды үйлестіру мәселелері бірінші кезектегі міндеттердің бірі болып келеді. "
            "Сонымен бірге тәжірибелік тұрғысынан алғанда жаңа стандарттарды қабылдауға үлкен мән беріледі және барлық факторлар назарға алынады. "
            "Осының арқасында қойылған мақсаттарға толық көлемде қол жеткізуге болады деп күтілуде."
        )
        res = self.detector.predict_document(unseen_calque_doc)
        self.assertEqual(res.verdict, "Machine-Generated")
        self.assertGreaterEqual(res.document_ai_probability, 0.9980)
        self.assertIsNotNone(res.generator_profile)
        self.assertTrue(res.generator_profile["is_unseen_generator"])
        self.assertEqual(res.generator_profile["predicted_family"], "Unseen / Out-of-Distribution LLM")

    def test_unseen_formal_synthetic_prose_without_exact_lexical_markers_is_detected(self):
        markerless_synthetic = (
            "Индустриялық кешендерді жаңғырту барысында автоматтандырылған басқару модульдері өндірістік циклге біріктірілді. "
            "Логистикалық тізбектерді цифрлық бақылау арқылы шикізат шығындары төмендетілді және сапа стандарттары сақталды."
        )
        res = self.detector.predict_document(markerless_synthetic)
        self.assertEqual(res.verdict, "Machine-Generated")
        self.assertGreaterEqual(res.document_ai_probability, 0.9980)
        self.assertIsNotNone(res.generator_profile)

    def test_authentic_human_documents_remain_below_threshold_with_human_profile(self):
        human_controls = [
            "Бұл бірінші қазақша сөйлем. Ал бұл екінші сөйлем болып табылады.",
            "Кеше кешке орталыққа достармен бірге барғанбыз, бірақ көрме жабық болып шықты. Каспиден тауардың жеткізілуі өте тез болды, курьерге рахметімді айттым ғой.",
            "Университетте өткен ғылыми конференцияда көптеген пайдалы ақпарат алдық. Досым маған керемет фильм ұсынған еді, кешке соны отбасымызбен бірге тамашаладық.",
        ]
        for text in human_controls:
            res = self.detector.predict_document(text)
            self.assertEqual(res.verdict, "Authentic Human", msg=f"Failed on human text: {text}")
            self.assertLess(res.document_ai_probability, 0.9980)
            self.assertIsNotNone(res.generator_profile)
            self.assertEqual(res.generator_profile["predicted_family"], "Authentic Human")
            self.assertFalse(res.generator_profile["is_unseen_generator"])

    def test_to_dict_serializes_generator_profile(self):
        ai_doc = (
            "Қарастырылып отырған алгоритмдік шешімдер күрделі деректер құрылымын оңтайландыру мақсатында жүйелі түрде талданды. "
            "Мәселені шешудің бірінші кезеңінде барлық шектеулерді ескере отырып, математикалық модель құрастырылды."
        )
        res = self.detector.predict_document(ai_doc)
        payload = res.to_dict()
        self.assertIn("generator_profile", payload)
        self.assertEqual(payload["generator_profile"]["predicted_family"], "DeepSeek-R1")


if __name__ == "__main__":
    unittest.main()
