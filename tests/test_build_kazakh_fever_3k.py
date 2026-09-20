# -*- coding: utf-8 -*-
"""
tests/test_build_kazakh_fever_3k.py: Unit tests for Kazakh-FEVER 3K generation and validation pipeline.
"""

import os
import json
import tempfile
import unittest

from scripts.build_kazakh_fever_3k import (
    generate_candidate_claims_prompt,
    validate_claim_record,
    partition_dataset,
    main,
)


class TestBuildKazakhFever3K(unittest.TestCase):
    """Unit test suite for Kazakh-FEVER 3K dataset generation, validation, and partitioning."""

    def test_prompt_enforces_hard_nei_and_span_grounding(self):
        """Asserts prompt contains Hard NEI, SUPPORTS, REFUTES, NOT_ENOUGH_INFO, and evidence constraints."""
        prompt = generate_candidate_claims_prompt(
            "Абай Құнанбайұлы",
            "Абай 1845 жылы Семей өңірінде дүниеге келген ұлы қазақ ақыны.",
            "wikipedia"
        )
        self.assertIn("Абай Құнанбайұлы", prompt)
        self.assertIn("SUPPORTS", prompt)
        self.assertIn("REFUTES", prompt)
        self.assertIn("NOT_ENOUGH_INFO", prompt)
        self.assertIn("Hard NEI", prompt)
        self.assertIn("evidence_sentence", prompt)

    def test_validate_claim_record_valid(self):
        """Asserts valid records pass schema and semantic validation."""
        rec = {
            "id": "kz_fever_0001",
            "claim": "Абай Құнанбайұлы 1845 жылы Шығыс Қазақстанда дүниеге келген ұлы қазақ ақыны.",
            "evidence_sentences": ["Абай 1845 жылы Семей өңірінде дүниеге келген ұлы қазақ ақыны."],
            "label": "SUPPORTS",
            "domain": "wikipedia"
        }
        self.assertTrue(validate_claim_record(rec))

    def test_validate_claim_record_invalid_label(self):
        """Asserts invalid labels or empty evidence fail validation."""
        rec_invalid_label = {
            "id": "kz_fever_0002",
            "claim": "Абай Құнанбайұлы 1845 жылы Шығыс Қазақстанда дүниеге келген ұлы ақын.",
            "evidence_sentences": ["Абай 1845 жылы дүниеге келген."],
            "label": "INVALID_LABEL",
            "domain": "wikipedia"
        }
        self.assertFalse(validate_claim_record(rec_invalid_label))

        rec_empty_evidence = {
            "id": "kz_fever_0003",
            "claim": "Абай Құнанбайұлы 1845 жылы Шығыс Қазақстанда дүниеге келген ұлы ақын.",
            "evidence_sentences": [],
            "label": "SUPPORTS",
            "domain": "wikipedia"
        }
        self.assertFalse(validate_claim_record(rec_empty_evidence))

        rec_empty_refutes = {
            "id": "kz_fever_0004",
            "claim": "Абай Құнанбайұлы 1845 жылы Шығыс Қазақстанда дүниеге келген ұлы ақын.",
            "evidence_sentences": [],
            "label": "REFUTES",
            "domain": "wikipedia"
        }
        self.assertFalse(validate_claim_record(rec_empty_refutes))

    def test_validate_claim_record_length_bounds(self):
        """Asserts claims < 8 tokens or > 35 tokens fail validation."""
        rec_short = {
            "id": "kz_fever_0005",
            "claim": "Қысқа сөйлем осында тұр.",
            "evidence_sentences": ["Қысқа сөйлем осында тұр дәлел."],
            "label": "SUPPORTS",
            "domain": "wikipedia"
        }
        self.assertFalse(validate_claim_record(rec_short))

        words = ["сөз"] * 36
        rec_long = {
            "id": "kz_fever_0006",
            "claim": " ".join(words),
            "evidence_sentences": ["Дәлел сөйлем бар."],
            "label": "SUPPORTS",
            "domain": "wikipedia"
        }
        self.assertFalse(validate_claim_record(rec_long))

        words_8 = ["сөз"] * 8
        rec_8 = {
            "id": "kz_fever_0007",
            "claim": " ".join(words_8),
            "evidence_sentences": ["Дәлел сөйлем бар."],
            "label": "SUPPORTS",
            "domain": "wikipedia"
        }
        self.assertTrue(validate_claim_record(rec_8))

        words_35 = ["сөз"] * 35
        rec_35 = {
            "id": "kz_fever_0008",
            "claim": " ".join(words_35),
            "evidence_sentences": ["Дәлел сөйлем бар."],
            "label": "SUPPORTS",
            "domain": "wikipedia"
        }
        self.assertTrue(validate_claim_record(rec_35))

    def test_partition_dataset_distribution(self):
        """Asserts stratified partition preserves domain and label ratios across train/dev/test."""
        dummy_data = []
        domains = ["wikipedia", "factcheck_kz"]
        labels = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        for i in range(300):
            dummy_data.append({
                "id": f"rec_{i:04d}",
                "claim": f"Бұл тесттік мәлімдеме мәтіні нөмірі {i} болып табылады.",
                "evidence_sentences": ["Дәлел сөйлем."] if labels[i % 3] != "NOT_ENOUGH_INFO" else [],
                "label": labels[i % 3],
                "domain": domains[i % 2]
            })

        train, dev, test = partition_dataset(dummy_data)
        self.assertEqual(len(train) + len(dev) + len(test), 300)
        self.assertGreater(len(train), len(dev))
        self.assertGreater(len(dev), 0)
        self.assertGreater(len(test), 0)

        self.assertTrue(190 <= len(train) <= 210)
        self.assertTrue(45 <= len(dev) <= 55)
        self.assertTrue(45 <= len(test) <= 55)

        for split in [train, dev, test]:
            split_domains = set(r["domain"] for r in split)
            split_labels = set(r["label"] for r in split)
            self.assertEqual(split_domains, set(domains))
            self.assertEqual(split_labels, set(labels))

    def test_dry_run_generation(self):
        """Asserts --dry-run produces valid candidate JSONL records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "dry_run_candidates.jsonl")
            exit_code = main(["--dry-run", "--output", out_file])
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(out_file))

            with open(out_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line.strip()) for line in f if line.strip()]

            self.assertGreater(len(lines), 0)
            for rec in lines:
                self.assertTrue(validate_claim_record(rec), f"Failed validation: {rec}")


if __name__ == "__main__":
    unittest.main()
