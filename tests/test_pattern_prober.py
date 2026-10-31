# -*- coding: utf-8 -*-
"""
Unit tests for Phase 3 GeneratorPatternProber:
- Extraction of all 9 morphosyntactic and syntactic pattern metrics
- Statistical separation (p < 0.001) on chain_len_entropy and copula_ratio
- Z-score deviation calculation against human baseline and generator references
- Loanword vowel harmony violation and unassimilated Latin token detection
- Bilingual diagnostic pattern explanations and zero-emoji enforcement
"""

import math
import re
import unittest

from models.pattern_prober import (
    FAMILY_REFERENCE_PROBES,
    HUMAN_BASELINE_STATS,
    PROBE_METRIC_NAMES,
    RADAR_AXIS_KEYS,
    GeneratorPatternProber,
    PatternExplanation,
    PatternProbeResult,
)
from scripts.evaluate_multi_generator_profiler import build_multi_generator_eval_corpus


EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\ufe0f]")


def _welch_p_value(xs, ys):
    n1, n2 = len(xs), len(ys)
    m1 = sum(xs) / float(n1)
    m2 = sum(ys) / float(n2)
    v1 = sum((x - m1) ** 2 for x in xs) / float(max(1, n1 - 1))
    v2 = sum((y - m2) ** 2 for y in ys) / float(max(1, n2 - 1))
    t_stat = (m1 - m2) / math.sqrt(v1 / float(n1) + v2 / float(n2) + 1e-12)
    p_val = math.erfc(abs(t_stat) / math.sqrt(2.0))
    return t_stat, p_val


class TestGeneratorPatternProber(unittest.TestCase):
    def setUp(self):
        self.prober = GeneratorPatternProber()
        self.human_sample = (
            "Кеше кешке орталыққа достармен бірге барғанбыз, бірақ көрме жабық болып шықты. "
            "Ауылдағы ата-әжемізге демалыста арнайы барып, біраз көмектесіп қайттық. "
            "Мектептегі сыныптастарымызбен он жылдан кейін қайта жүздесіп, шүйіркелесіп қалдық."
        )
        self.ai_sample = (
            "Табиғи ресурстарды тиімді пайдалану мәселелері заманауи экологиялық саясаттың басым бағыттарының бірі болып табылады. "
            "Осыған байланысты, өңірлік даму саясаты теңгерімді экономикалық өсімді қамтамасыз етудің маңызды құралы болып саналады. "
            "Өз кезегінде қойылған мақсаттарға толық көлемде қол жеткізуге болады деп күтілуде."
        )

    def test_probe_extracts_all_metrics_with_valid_ranges(self):
        res_h = self.prober.probe(self.human_sample)
        res_a = self.prober.probe(self.ai_sample)

        self.assertIsInstance(res_h, PatternProbeResult)
        self.assertEqual(tuple(res_h.to_dict().keys()), PROBE_METRIC_NAMES)

        self.assertGreater(res_h.chain_len_entropy, res_a.chain_len_entropy)
        self.assertGreater(res_h.chain_len_variance, 0.5)
        self.assertEqual(res_h.copula_ratio, 0.0)
        self.assertGreater(res_a.copula_ratio, 5.0)
        self.assertGreater(res_h.synthetic_verb_ratio, res_a.synthetic_verb_ratio)
        self.assertGreater(res_a.discourse_rigidity_score, res_h.discourse_rigidity_score)
        self.assertGreaterEqual(res_h.sov_compliance_score, res_a.sov_compliance_score)
        self.assertGreater(res_h.lexical_ttr, 0.5)
        self.assertGreaterEqual(res_h.burstiness, 0.0)

    def test_statistical_significance_across_multi_generator_corpus(self):
        corpus = build_multi_generator_eval_corpus(docs_per_family=12, random_seed=42)
        human_probes = [self.prober.probe(d["text"]) for d in corpus if d["label"] == 0]
        ai_probes = [self.prober.probe(d["text"]) for d in corpus if d["label"] == 1]

        h_ent = [p.chain_len_entropy for p in human_probes]
        a_ent = [p.chain_len_entropy for p in ai_probes]
        t_ent, p_ent = _welch_p_value(h_ent, a_ent)
        self.assertGreater(sum(h_ent) / len(h_ent), sum(a_ent) / len(a_ent))
        self.assertGreater(t_ent, 3.3)
        self.assertLess(p_ent, 0.001, msg=f"Expected chain_len_entropy p < 0.001, got {p_ent:.3e}")

        h_cop = [p.copula_ratio for p in human_probes]
        a_cop = [p.copula_ratio for p in ai_probes]
        t_cop, p_cop = _welch_p_value(h_cop, a_cop)
        self.assertLess(sum(h_cop) / len(h_cop), sum(a_cop) / len(a_cop))
        self.assertLess(t_cop, -3.3)
        self.assertLess(p_cop, 0.001, msg=f"Expected copula_ratio p < 0.001, got {p_cop:.3e}")

    def test_compute_deviation_z_scores_and_family_reference(self):
        dev_ai = self.prober.compute_deviation(self.ai_sample)
        dev_hu = self.prober.compute_deviation(self.human_sample)

        self.assertEqual(tuple(dev_ai.keys()), PROBE_METRIC_NAMES)
        self.assertGreater(dev_ai["copula_ratio"], 3.0)
        self.assertGreater(dev_ai["discourse_rigidity_score"], 1.5)
        self.assertLess(dev_ai["synthetic_verb_ratio"], -2.0)
        self.assertLess(abs(dev_hu["copula_ratio"]), 1.0)

        for fam in FAMILY_REFERENCE_PROBES:
            fam_dev = self.prober.compute_deviation(self.ai_sample, family=fam)
            self.assertEqual(tuple(fam_dev.keys()), PROBE_METRIC_NAMES)

    def test_loan_assimilation_detects_disharmonic_and_bare_english_tokens(self):
        clean_loans = "Университетте өткен конференцияда курьерге және вокзалдағы маманға алғыс айттық."
        broken_loans = "Біздің серверға жаңа framework пен футболге арналған модельға update орнатылды."

        res_clean = self.prober.probe(clean_loans)
        res_broken = self.prober.probe(broken_loans)

        self.assertEqual(res_clean.loan_assimilation_score, 1.0)
        self.assertLess(res_broken.loan_assimilation_score, 0.4)

    def test_explain_patterns_and_radar_normalization_and_chunk_input(self):
        explanations = self.prober.explain_patterns(self.ai_sample)
        self.assertGreaterEqual(len(explanations), 2)
        for exp in explanations:
            self.assertIsInstance(exp, PatternExplanation)
            self.assertTrue(exp.badge_en)
            self.assertTrue(exp.badge_kz)
            self.assertTrue(exp.message_en)
            self.assertTrue(exp.message_kz)
            self.assertIsNone(EMOJI_RE.search(exp.message_en))
            self.assertIsNone(EMOJI_RE.search(exp.message_kz))

        radar = self.prober.normalize_for_radar(self.prober.probe(self.ai_sample))
        self.assertEqual(tuple(radar.keys()), RADAR_AXIS_KEYS)
        for k, v in radar.items():
            self.assertGreaterEqual(v, 0.0, msg=k)
            self.assertLessEqual(v, 1.0, msg=k)

        # Tokenized/chunked sequence input
        chunks = [
            "Кеше кешке орталыққа достармен бірге барғанбыз.",
            "Ауылдағы ата-әжемізге барып көмектесіп қайттық.",
        ]
        res_chunks = self.prober.probe(chunks)
        self.assertGreater(res_chunks.total_words, 8)

        # Empty input safety
        empty_res = self.prober.probe("   ")
        self.assertEqual(empty_res.total_words, 0)
        self.assertEqual(empty_res.copula_ratio, 0.0)
        self.assertEqual(len(HUMAN_BASELINE_STATS), len(PROBE_METRIC_NAMES))


if __name__ == "__main__":
    unittest.main()
