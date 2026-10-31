# -*- coding: utf-8 -*-
"""
Unit tests for Phase 2 MultiModelGeneratorProfiler:
- 8D morphosyntactic feature extraction via DynamicFSTOverlay
- 5-family centroid attribution across Sovereign, Reasoning, and Multilingual LLMs
- Open-set Out-of-Distribution (OOD) rejection on held-out and calque-heavy generators
- Graceful handling of human, empty, and Russian-layout noisy inputs
"""

import unittest

from models.generator_profiler import (
    FEATURE_NAMES,
    KNOWN_GENERATOR_FAMILIES,
    GeneratorProfileResult,
    MultiModelGeneratorProfiler,
)


class TestMultiModelGeneratorProfiler(unittest.TestCase):
    def setUp(self):
        self.profiler = MultiModelGeneratorProfiler()

    def test_feature_extraction_returns_all_8_named_dimensions(self):
        sample = (
            "Қазақстан Республикасындағы цифрландыру үдерістері қоғамның барлық салаларында қарқынды түрде дамып келеді. "
            "Аталған бағдарламаның шеңберінде экономикалық тиімділікті арттыру бойынша кешенді іс-шаралар жүзеге асырылуда."
        )
        feats = self.profiler.extract_morphosyntactic_profile(sample)
        self.assertEqual(tuple(feats.keys()), FEATURE_NAMES)
        for name, val in feats.items():
            self.assertIsInstance(val, float, msg=f"Feature {name} must be float")
            self.assertGreaterEqual(val, 0.0, msg=f"Feature {name} must be non-negative")
        self.assertGreater(feats["abstract_deriv_density"], 0.05)
        self.assertGreater(feats["affix_depth_per_word"], 0.40)

    def test_attributes_distinct_known_generator_families(self):
        samples = {
            "ISSAI-KazLLM-70B": (
                "Қазақстан Республикасындағы цифрландыру үдерістері қоғамның барлық салаларында қарқынды түрде дамып келеді. "
                "Аталған бағдарламаның шеңберінде экономикалық тиімділікті арттыру бойынша кешенді іс-шаралар жүзеге асырылуда. "
                "Құқықтық реттеу тетіктерін жетілдіру шеңберінде қолданыстағы заңнамалық актілерге тиісті түзетулер енгізілді."
            ),
            "Sherkala-8B": (
                "Табиғи ресурстарды тиімді пайдалану мәселелері заманауи экологиялық саясаттың басым бағыттарының бірі болып табылады. "
                "Ауыл шаруашылығын әртараптандыру бағдарламасы азық-түлік қауіпсіздігін нығайтудың шешуші факторы болып табылады. "
                "Су ресурстарын үнемді пайдалану технологияларын енгізу өңірлік тұрақтылықты қамтамасыз етуге негіз болады."
            ),
            "DeepSeek-R1": (
                "Қарастырылып отырған алгоритмдік шешімдер күрделі деректер құрылымын оңтайландыру мақсатында жүйелі түрде талданды. "
                "Мәселені шешудің бірінші кезеңінде барлық шектеулерді ескере отырып, математикалық модель құрастырылды. "
                "Теориялық негіздеме шеңберінде градиенттік төмендеу әдісінің конвергенция жылдамдығы дәлелденді."
            ),
            "GPT-4o": (
                "Жасанды интеллект жүйелерінің дамуы заманауи ғылым мен білім беру саласында жаңа мүмкіндіктерге жол ашуда. "
                "Көптілді тілдік модельдер қазақ тіліндегі мәтіндерді талдау және сапалы генерациялау бойынша жоғары нәтижелер көрсетуде. "
                "Көпмодальды архитектуралар мәтін, сурет және дыбыстық ақпаратты бір мезетте талдауға мүмкіндік береді."
            ),
            "Qwen-2.5-72B": (
                "Халықаралық ынтымақтастық аясында жүзеге асырылып жатқан жобалар еліміздің инвестициялық тартымдылылығын арттырады. "
                "Халықаралық сауда-экономикалық қатынастарды дамыту стратегиялық әріптестікті тереңдетуге мүмкіндік береді. "
                "Адами капиталды дамыту жөніндегі бастамалар ұлттық экономиканың ұзақ мерзімді тұрақтылығына кепілдік береді."
            ),
        }

        for expected_family, text in samples.items():
            res = self.profiler.profile_document(text, ai_probability=0.999, verdict="Machine-Generated")
            self.assertIsInstance(res, GeneratorProfileResult)
            self.assertFalse(res.is_unseen_generator, msg=f"{expected_family} should not be flagged as OOD")
            self.assertEqual(res.predicted_family, expected_family)
            self.assertGreater(res.confidence, 0.30)
            self.assertEqual(set(res.family_probabilities.keys()), set(KNOWN_GENERATOR_FAMILIES))
            self.assertAlmostEqual(sum(res.family_probabilities.values()), 1.0, places=3)
            self.assertGreaterEqual(len(res.dominant_traits), 2)

    def test_open_set_flags_unseen_calque_generator_as_ood(self):
        unseen_calque = (
            "Өз кезегінде аталмыш бағыттағы жұмыстарды үйлестіру мәселелері бірінші кезектегі міндеттердің бірі болып келеді. "
            "Сонымен бірге тәжірибелік тұрғысынан алғанда жаңа стандарттарды қабылдауға үлкен мән беріледі және барлық факторлар назарға алынады. "
            "Осының арқасында қойылған мақсаттарға толық көлемде қол жеткізуге болады деп күтілуде."
        )
        res = self.profiler.profile_document(unseen_calque, ai_probability=0.999, verdict="Machine-Generated")
        self.assertTrue(res.is_unseen_generator)
        self.assertEqual(res.predicted_family, "Unseen / Out-of-Distribution LLM")
        self.assertGreater(res.ood_distance, res.ood_threshold)

    def test_leave_one_generator_out_active_families_flags_held_out_reasoning_llm(self):
        four_family_profiler = MultiModelGeneratorProfiler(
            active_families=["ISSAI-KazLLM-70B", "Sherkala-8B", "GPT-4o", "Qwen-2.5-72B"]
        )
        deepseek_sample = (
            "Қарастырылып отырған алгоритмдік шешімдер күрделі деректер құрылымын оңтайландыру мақсатында жүйелі түрде талданды. "
            "Дифференциалдық теңдеулер жүйесін шешу барысында шекаралық шарттар қатаң түрде сақталды. "
            "Эксперименттік деректердің дисперсиясын талдау нәтижесінде статистикалық гипотезалар толықтай расталды."
        )
        res = four_family_profiler.profile_document(deepseek_sample, ai_probability=0.999, verdict="Machine-Generated")
        self.assertTrue(res.is_unseen_generator)
        self.assertEqual(res.predicted_family, "Unseen / Out-of-Distribution LLM")

    def test_authentic_human_and_empty_inputs_handled_cleanly(self):
        human_text = "Каспиден тауардың жеткізілуі өте тез болды, курьерге рахметімді айттым ғой."
        res_human = self.profiler.profile_document(human_text, ai_probability=0.04, verdict="Authentic Human")
        self.assertFalse(res_human.is_unseen_generator)
        self.assertEqual(res_human.predicted_family, "Authentic Human")
        self.assertEqual(res_human.archetype_label, "Human Authorship")

        res_empty = self.profiler.profile_document("", ai_probability=0.0, verdict="Authentic Human")
        self.assertEqual(res_empty.predicted_family, "Awaiting Input")
        self.assertFalse(res_empty.is_unseen_generator)


if __name__ == "__main__":
    unittest.main()
