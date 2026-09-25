# tests/test_verify_and_filter_claims.py
# -*- coding: utf-8 -*-
"""
Tests for claim verification, quality filtering, and Cohen's Kappa evaluation tooling.
"""

import os
import json
import tempfile
import unittest
from scripts.verify_and_filter_claims import (
    filter_and_clean_claims,
    compute_cohens_kappa,
    export_human_annotation_package,
    split_train_dev_test,
    export_dataset_splits,
    VALID_LABELS,
)


class TestVerifyAndFilterClaims(unittest.TestCase):
    def test_filter_and_clean_claims(self):
        raw = [
            {
                "id": "c1",
                "claim": "Абай Құнанбайұлы 1845 жылы туған қазақ ақыны және ойшылы.",
                "label": "SUPPORTS",
                "evidence_sentences": ["Абай 1845 жылы туған."]
            },
            {
                "id": "c2",
                "claim": "Қысқа тұжырым.",  # Too short (< 8 tokens)
                "label": "SUPPORTS",
                "evidence_sentences": ["Дәлел."]
            },
            {
                "id": "c3",
                "claim": "Қазақстанның астанасы Астана қаласы болып 1998 жылы ресми түрде бекітілді.",
                "label": "INVALID_LABEL",  # Invalid label
                "evidence_sentences": ["Дәлел."]
            },
            {
                "id": "c4",
                "claim": "Байқоңыр ғарыш айлағы 1955 жылы салынған әлемдегі ең үлкен айлақ.",
                "label": "REFUTES",
                "evidence_sentences": ["Байқоңыр 1955 жылы салынды."]
            },
            {
                "id": "c5",
                "claim": "Семей полигоны туралы мәлімет халық арасында кеңінен талқыланған болатын.",
                "label": "SUPPORTS",
                "evidence_sentences": []  # Empty evidence for SUPPORTS -> rejected
            },
            {
                "id": "c6",
                "claim": "Бұл мәлімдемеге қатысты тарихи құжаттарда нақты деректер мүлдем кездеспейді.",
                "label": "NOT_ENOUGH_INFO",
                "evidence_sentences": []  # Valid NOT_ENOUGH_INFO with empty evidence
            },
            {
                "id": "c7",
                "claim": "Бұл мәлімдемеге қатысты тарихи деректер анықталмаған және белгісіз жағдайда қалуда.",
                "label": "not_enough_info",  # Lowercase label normalized
                "evidence_sentences": ["Кездейсоқ мәтін"]  # NOT_ENOUGH_INFO must have empty evidence
            }
        ]
        valid, stats = filter_and_clean_claims(raw, min_tokens=8, max_tokens=35)
        self.assertEqual(stats["total_input"], 7)
        self.assertEqual(stats["rejected_token_length"], 1)
        self.assertEqual(stats["rejected_label"], 1)
        self.assertEqual(stats["rejected_empty_evidence"], 1)
        self.assertEqual(stats["valid_count"], 4)
        self.assertEqual(len(valid), 4)

        # Ensure c7 label normalized and evidence_sentences emptied for NOT_ENOUGH_INFO
        c7_cleaned = next(c for c in valid if c["id"] == "c7")
        self.assertEqual(c7_cleaned["label"], "NOT_ENOUGH_INFO")
        self.assertEqual(c7_cleaned["evidence_sentences"], [])

    def test_filter_and_clean_claims_token_limits(self):
        long_claim = " ".join(["сөз"] * 40)
        short_claim = " ".join(["сөз"] * 5)
        ok_claim = " ".join(["сөз"] * 10)
        claims = [
            {"id": "l1", "claim": long_claim, "label": "SUPPORTS", "evidence_sentences": ["s"]},
            {"id": "s1", "claim": short_claim, "label": "SUPPORTS", "evidence_sentences": ["s"]},
            {"id": "ok1", "claim": ok_claim, "label": "SUPPORTS", "evidence_sentences": ["s"]}
        ]
        valid, stats = filter_and_clean_claims(claims, min_tokens=8, max_tokens=35)
        self.assertEqual(stats["rejected_token_length"], 2)
        self.assertEqual(stats["valid_count"], 1)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["id"], "ok1")

    def test_filter_and_clean_claims_empty_input(self):
        valid, stats = filter_and_clean_claims([])
        self.assertEqual(valid, [])
        self.assertEqual(stats["total_input"], 0)
        self.assertEqual(stats["valid_count"], 0)
        self.assertEqual(stats["rejected_token_length"], 0)
        self.assertEqual(stats["rejected_label"], 0)
        self.assertEqual(stats["rejected_empty_evidence"], 0)

    def test_compute_cohens_kappa_perfect(self):
        rater_a = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS"]
        rater_b = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS"]
        res = compute_cohens_kappa(rater_a, rater_b)
        self.assertEqual(res["sample_size"], 4.0)
        self.assertAlmostEqual(res["observed_agreement"], 1.0)
        self.assertAlmostEqual(res["kappa"], 1.0)

    def test_compute_cohens_kappa_imperfect(self):
        rater_a = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS"]
        rater_b = ["SUPPORTS", "SUPPORTS", "NOT_ENOUGH_INFO", "REFUTES"]
        res = compute_cohens_kappa(rater_a, rater_b)
        self.assertEqual(res["sample_size"], 4.0)
        self.assertLess(res["kappa"], 1.0)
        self.assertGreaterEqual(res["kappa"], -1.0)
        self.assertIn("expected_chance_agreement", res)

    def test_compute_cohens_kappa_case_insensitive(self):
        rater_a = ["supports", "refutes", "not_enough_info"]
        rater_b = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        res = compute_cohens_kappa(rater_a, rater_b)
        self.assertAlmostEqual(res["observed_agreement"], 1.0)
        self.assertAlmostEqual(res["kappa"], 1.0)

    def test_compute_cohens_kappa_all_same_class(self):
        rater_a = ["SUPPORTS", "SUPPORTS", "SUPPORTS"]
        rater_b = ["SUPPORTS", "SUPPORTS", "SUPPORTS"]
        res = compute_cohens_kappa(rater_a, rater_b)
        self.assertAlmostEqual(res["observed_agreement"], 1.0)
        self.assertAlmostEqual(res["kappa"], 1.0)

    def test_compute_cohens_kappa_invalid_inputs(self):
        with self.assertRaises(ValueError):
            compute_cohens_kappa([], [])

        with self.assertRaises(ValueError):
            compute_cohens_kappa(["SUPPORTS"], ["SUPPORTS", "REFUTES"])

    def test_export_human_annotation_package_subset(self):
        claims = [
            {"id": f"s_{i}", "claim": f"Тұжырым {i}", "label": "SUPPORTS", "evidence_sentences": ["Д"]}
            for i in range(10)
        ] + [
            {"id": f"r_{i}", "claim": f"Тұжырым {i}", "label": "REFUTES", "evidence_sentences": ["Д"]}
            for i in range(10)
        ] + [
            {"id": f"n_{i}", "claim": f"Тұжырым {i}", "label": "NOT_ENOUGH_INFO", "evidence_sentences": []}
            for i in range(10)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "nested", "sample.jsonl")
            export_human_annotation_package(claims, sample_size=15, output_path=out_file)
            self.assertTrue(os.path.exists(out_file))

            loaded = []
            with open(out_file, "r", encoding="utf-8") as f:
                for line in f:
                    loaded.append(json.loads(line))

            self.assertEqual(len(loaded), 15)
            labels = [r["label"] for r in loaded]
            # Balanced distribution: 5 of each
            self.assertEqual(labels.count("SUPPORTS"), 5)
            self.assertEqual(labels.count("REFUTES"), 5)
            self.assertEqual(labels.count("NOT_ENOUGH_INFO"), 5)

    def test_export_human_annotation_package_fewer_than_sample(self):
        claims = [
            {"id": "1", "claim": "Тұжырым бір", "label": "SUPPORTS", "evidence_sentences": ["Д"]}
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "sample.jsonl")
            export_human_annotation_package(claims, sample_size=10, output_path=out_file)
            self.assertTrue(os.path.exists(out_file))

            with open(out_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            self.assertEqual(len(lines), 1)

    def test_split_train_dev_test(self):
        sample_claims = []
        domains = ["history", "science", "literature", "geography"]
        labels = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        for d in domains:
            for l in labels:
                for i in range(10):
                    sample_claims.append({
                        "id": f"{d}_{l}_{i}",
                        "claim": f"Тұжырым {d} {l} {i} сөздерімен берілген үлгі мәтін",
                        "label": l,
                        "domain": d,
                        "evidence_sentences": ["Дәлел сөйлем"] if l != "NOT_ENOUGH_INFO" else []
                    })
        self.assertEqual(len(sample_claims), 120)

        train, dev, test = split_train_dev_test(sample_claims, 0.70, 0.15, 0.15, seed=42)
        total_split = len(train) + len(dev) + len(test)
        self.assertEqual(total_split, 120)
        self.assertAlmostEqual(len(train) / 120, 0.70, delta=0.08)
        self.assertAlmostEqual(len(dev) / 120, 0.15, delta=0.08)
        self.assertAlmostEqual(len(test) / 120, 0.15, delta=0.08)

        # Check stratification across domains
        train_domains = set(c["domain"] for c in train)
        self.assertEqual(train_domains, set(domains))
        test_domains = set(c["domain"] for c in test)
        self.assertEqual(test_domains, set(domains))

    def test_export_dataset_splits(self):
        sample_claims = [
            {
                "id": f"c_{i}",
                "claim": f"Тұжырым {i} мазмұны бойынша нақты жазылған үлгілік сөйлем",
                "label": "SUPPORTS" if i % 2 == 0 else "NOT_ENOUGH_INFO",
                "domain": "history",
                "evidence_sentences": ["Дәлел"] if i % 2 == 0 else []
            }
            for i in range(20)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_dataset_splits(sample_claims, output_dir=tmpdir, seed=42)
            for key in ["cleaned", "train", "dev", "test"]:
                self.assertIn(key, paths)
                self.assertTrue(os.path.exists(paths[key]))


if __name__ == "__main__":
    unittest.main()
