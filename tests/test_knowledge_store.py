# -*- coding: utf-8 -*-
"""
tests/test_knowledge_store.py: Unit tests for Evidence Data Structures and KnowledgeStore.
"""

import os
import json
import tempfile
import unittest

from verification.evidence import (
    EvidencePassage,
    AtomicClaim,
    ClaimVerificationResult,
    DocumentTrustResult,
)
from verification.knowledge_store import KnowledgeStore


class TestEvidenceDataStructures(unittest.TestCase):
    """Verifies dataclass creation, serialization, and default behaviors."""

    def test_evidence_passage_creation(self):
        p = EvidencePassage(
            passage_id="wiki_kz_001",
            title="Қазақстан",
            text="Қазақстан 1991 жылы тәуелсіздігін жариялады.",
            source_url="https://kk.wikipedia.org/wiki/Kazakhstan",
            similarity_score=0.95,
            matched_stems=["қазақстан", "1991", "жыл", "тәуелсіздік"]
        )
        self.assertEqual(p.passage_id, "wiki_kz_001")
        self.assertEqual(p.title, "Қазақстан")
        self.assertAlmostEqual(p.similarity_score, 0.95)
        self.assertEqual(len(p.matched_stems), 4)
        d = p.to_dict()
        self.assertEqual(d["passage_id"], "wiki_kz_001")
        self.assertEqual(d["title"], "Қазақстан")

    def test_atomic_claim_creation(self):
        c = AtomicClaim(
            claim_id="c_01",
            text="Қазақстан 1991 жылы тәуелсіздік алды.",
            source_sentence="Меніңше, Қазақстан 1991 жылы тәуелсіздік алды.",
            start_char=9,
            end_char=49,
            is_verifiable=True
        )
        self.assertEqual(c.claim_id, "c_01")
        self.assertTrue(c.is_verifiable)
        d = c.to_dict()
        self.assertEqual(d["start_char"], 9)
        self.assertEqual(d["end_char"], 49)

    def test_claim_verification_result(self):
        c = AtomicClaim(claim_id="c_01", text="Астана — бас қала.", source_sentence="Астана — бас қала.")
        p = EvidencePassage(passage_id="p1", title="Астана", text="Астана қаласы — Қазақстанның астанасы.")
        res = ClaimVerificationResult(
            claim=c,
            verdict="SUPPORTED",
            confidence=0.92,
            evidence=[p],
            explanation="Дереккөз бойынша Астана Қазақстанның астанасы болып табылады."
        )
        self.assertEqual(res.verdict, "SUPPORTED")
        self.assertEqual(len(res.evidence), 1)
        d = res.to_dict()
        self.assertEqual(d["verdict"], "SUPPORTED")
        self.assertEqual(d["confidence"], 0.92)

    def test_document_trust_result(self):
        res = DocumentTrustResult(
            doc_text="Бұл құжат.",
            ai_risk=0.10,
            factual_risk=0.05,
            trust_risk=0.075,
            quadrant_verdict="Verified Human Fact",
            claims=[],
            total_claims=0,
            supported_count=0,
            refuted_count=0,
            nei_count=0
        )
        self.assertEqual(res.quadrant_verdict, "Verified Human Fact")
        self.assertAlmostEqual(res.trust_risk, 0.075)


class TestKnowledgeStore(unittest.TestCase):
    """Verifies KnowledgeStore passage management and FST-stemmed inverted index."""

    def setUp(self):
        self.store = KnowledgeStore()

    def test_add_and_retrieve_passage(self):
        p = EvidencePassage(
            passage_id="p_test_1",
            title="Алматы",
            text="Алматы — Қазақстанның оңтүстігіндегі ірі мегаполис."
        )
        self.store.add_passage(p)
        retrieved = self.store.get_passage("p_test_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.title, "Алматы")
        self.assertEqual(len(self.store), 1)

    def test_fst_stemmed_inverted_index(self):
        """Verifies that words with case affixes (Алматының) are indexed by their root stem (алматы)."""
        p = EvidencePassage(
            passage_id="p_alm_1",
            title="Алматы қаласы",
            text="Алматының табиғаты өте көркем. Таулары биік."
        )
        self.store.add_passage(p)
        index = self.store.get_inverted_index()
        # 'алматы' should be in index, not merely 'алматының'
        self.assertIn("алматы", index)
        self.assertIn("p_alm_1", [item[0] for item in index["алматы"]])

    def test_load_from_jsonl(self):
        sample_records = [
            {
                "passage_id": "doc_01",
                "title": "Абай",
                "text": "Абай Құнанбайұлы 1845 жылы дүниеге келген.",
                "source_url": "https://kk.wikipedia.org/wiki/Abai"
            },
            {
                "passage_id": "doc_02",
                "title": "Астана",
                "text": "Астана қаласы 1997 жылы жаңа астана болып белгіленді.",
                "source_url": "https://kk.wikipedia.org/wiki/Astana"
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
            for rec in sample_records:
                tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tmp_path = tmp.name

        try:
            store = KnowledgeStore.from_jsonl(tmp_path)
            self.assertEqual(len(store), 2)
            p1 = store.get_passage("doc_01")
            self.assertEqual(p1.title, "Абай")
            index = store.get_inverted_index()
            self.assertIn("абай", index)
            self.assertIn("астана", index)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
