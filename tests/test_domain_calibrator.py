# -*- coding: utf-8 -*-
"""
Unit tests for Phase 4 DomainCalibrator (models/domain_calibrator.py):
- Exact ROC-AUC and FPR@95%TPR calculation
- Threshold stability curve generation over tau in [0.90, 0.999]
- Safety verification at global threshold tau = 0.9980 (FPR <= 0.05)
- Rule-based Kazakh functional genre classification across all 5 domains
- Zero decorative emojis enforcement
"""

import re
import unittest

from models.domain_calibrator import (
    DOMAIN_DISPLAY_LABELS,
    SUPPORTED_DOMAINS,
    DomainCalibrator,
    DomainMetrics,
    ThresholdPoint,
)


EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\ufe0f]")


class TestDomainCalibrator(unittest.TestCase):
    def setUp(self):
        self.calibrator = DomainCalibrator(default_threshold=0.9980, max_safe_fpr=0.05)

    def test_roc_auc_and_fpr_at_95_tpr_computation(self):
        y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        y_scores = [0.02, 0.05, 0.08, 0.12, 0.20, 0.9985, 0.9990, 0.9992, 0.9995, 0.9998]

        auc = self.calibrator.compute_roc_auc(y_true, y_scores)
        fpr95 = self.calibrator.compute_fpr_at_tpr(y_true, y_scores, target_tpr=0.95)
        metrics = self.calibrator.evaluate_domain_predictions("academic", y_true, y_scores)

        self.assertAlmostEqual(auc, 1.0, places=4)
        self.assertAlmostEqual(fpr95, 0.0, places=4)
        self.assertIsInstance(metrics, DomainMetrics)
        self.assertEqual(metrics.domain, "academic")
        self.assertTrue(metrics.is_threshold_safe)
        self.assertEqual(metrics.fpr_at_calibrated_tau, 0.0)
        self.assertAlmostEqual(metrics.f1_score, 1.0, places=4)
        self.assertAlmostEqual(metrics.accuracy, 1.0, places=4)

    def test_threshold_stability_curve_bounds_and_monotonicity(self):
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_scores = [0.05, 0.91, 0.96, 0.995, 0.9982, 0.9988, 0.9992, 0.9996]

        curve = self.calibrator.compute_threshold_stability_curve(y_true, y_scores)
        self.assertGreaterEqual(len(curve), 6)
        self.assertIsInstance(curve[0], ThresholdPoint)
        self.assertAlmostEqual(curve[0].threshold, 0.90, places=3)
        self.assertAlmostEqual(curve[-1].threshold, 0.999, places=3)

        # FPR must be non-increasing as threshold increases
        fprs = [pt.fpr for pt in curve]
        for i in range(1, len(fprs)):
            self.assertLessEqual(fprs[i], fprs[i - 1] + 1e-9)

    def test_classify_domain_across_all_5_genres(self):
        samples = {
            "news": "Астана қаласында халықаралық форум өз жұмысын бастады. Тілшілер мен сарапшылар жиынға қатысуда.",
            "academic": "Мақалада қазақ тілінің морфологиялық агглютинативті құрылымы мен эмпирикалық гипотезасы талданды.",
            "official_legal": "Қазақстан Республикасы Министрлігінің бұйрығымен осы қаулы және заңнамалық кодекс қолданысқа енгізіледі.",
            "fiction_literature": "Күзгі іңірде ауыл шетіндегі кәрі терек сыбдырлап, әжемнің самаурын шайы есіме түсті ғой.",
            "social_casual": "Kaspi-ден заказ берген шаңсорғыш өте керемет екен, курьер тез әкеліп берді, рахмет!",
        }
        for expected_dom, text in samples.items():
            detected = self.calibrator.classify_domain(text)
            self.assertEqual(
                detected,
                expected_dom,
                msg=f"Expected {expected_dom}, got {detected} for text: {text}",
            )

    def test_domain_robustness_badge_and_zero_emojis(self):
        for dom in SUPPORTED_DOMAINS:
            self.assertIn(dom, DOMAIN_DISPLAY_LABELS)
            rec_tau = self.calibrator.recommend_threshold(dom)
            self.assertAlmostEqual(rec_tau, 0.9980, places=4)
            for lang in ("en", "kz"):
                badge = self.calibrator.get_domain_robustness_badge(dom, lang=lang)
                self.assertEqual(badge["domain_key"], dom)
                self.assertIn("domain_label", badge)
                self.assertIn("badge_text", badge)
                self.assertIsNone(EMOJI_RE.search(badge["badge_text"]))


if __name__ == "__main__":
    unittest.main()
