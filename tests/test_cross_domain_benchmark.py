# -*- coding: utf-8 -*-
"""
Unit tests for Phase 4 Cross-Domain Multi-Genre Benchmark Dataset:
- Schema integrity for all 500 records across 5 Kazakh functional genres
- Exact domain balance (100 per domain: 50 authentic human, 50 AI)
- Generator diversity (5 known families + 1 unseen calque family per domain)
- Cryptographic SHA-256 uniqueness and zero overlap with training/seed sets
- Zero decorative emojis across all dataset records and metadata
"""

import hashlib
import json
import os
import re
import unittest

from scripts.build_cross_domain_benchmark import (
    DOMAIN_KEYS,
    KNOWN_GENERATORS,
    UNSEEN_GENERATOR,
    build_cross_domain_benchmark_dataset,
    validate_cross_domain_benchmark,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "data", "cross_domain_benchmark.json")
EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\ufe0f]")


class TestCrossDomainBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(BENCHMARK_PATH):
            build_cross_domain_benchmark_dataset(output_path=BENCHMARK_PATH)
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            cls.records = json.load(f)

    def test_total_record_count_and_schema_contract(self):
        self.assertIsInstance(self.records, list)
        self.assertEqual(len(self.records), 500)

        required_keys = {
            "id",
            "domain",
            "text",
            "label",
            "generator",
            "word_count",
            "sentence_count",
            "sha256",
            "metadata",
        }
        for rec in self.records:
            self.assertEqual(set(rec.keys()), required_keys)
            self.assertIn(rec["domain"], DOMAIN_KEYS)
            self.assertIn(rec["label"], (0, 1))
            self.assertIsInstance(rec["text"], str)
            self.assertGreaterEqual(rec["word_count"], 18)
            self.assertGreaterEqual(rec["sentence_count"], 2)
            expected_sha = hashlib.sha256(rec["text"].strip().encode("utf-8")).hexdigest()
            self.assertEqual(rec["sha256"], expected_sha)
            self.assertIn("source_register", rec["metadata"])
            self.assertIn("has_layout_shift", rec["metadata"])
            self.assertIsInstance(rec["metadata"]["has_layout_shift"], bool)

    def test_domain_and_label_stratification(self):
        self.assertEqual(
            DOMAIN_KEYS,
            ("news", "academic", "official_legal", "fiction_literature", "social_casual"),
        )
        for domain in DOMAIN_KEYS:
            dom_recs = [r for r in self.records if r["domain"] == domain]
            self.assertEqual(len(dom_recs), 100, msg=f"Domain {domain} must have 100 records")
            human_recs = [r for r in dom_recs if r["label"] == 0]
            ai_recs = [r for r in dom_recs if r["label"] == 1]
            self.assertEqual(len(human_recs), 50, msg=f"Domain {domain} must have 50 human records")
            self.assertEqual(len(ai_recs), 50, msg=f"Domain {domain} must have 50 AI records")
            for h in human_recs:
                self.assertEqual(h["generator"], "human")
            ai_gens = {a["generator"] for a in ai_recs}
            for k_gen in KNOWN_GENERATORS:
                self.assertIn(k_gen, ai_gens)
            self.assertIn(UNSEEN_GENERATOR, ai_gens)

    def test_cryptographic_sha256_isolation_and_validator(self):
        seed_path = os.path.join(PROJECT_ROOT, "data", "seed_human_reviews_1k.json")
        overlap_paths = [seed_path] if os.path.isfile(seed_path) else []
        report = validate_cross_domain_benchmark(
            self.records,
            seed_overlap_paths=overlap_paths,
        )
        self.assertTrue(report["is_valid"])
        self.assertEqual(report["total_records"], 500)
        self.assertEqual(report["unique_sha256_count"], 500)
        self.assertEqual(report["seed_overlap_count"], 0)

    def test_zero_decorative_emojis_in_dataset(self):
        raw_json = json.dumps(self.records, ensure_ascii=False)
        self.assertIsNone(
            EMOJI_RE.search(raw_json),
            "Dataset JSON must contain zero decorative emojis",
        )


if __name__ == "__main__":
    unittest.main()
