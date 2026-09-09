import unittest
from kaz_mage.document import DocumentChunk
from kaz_mage.aggregator import DocumentAggregator


class TestDocumentAggregator(unittest.TestCase):
    def setUp(self):
        self.agg = DocumentAggregator(calibrated_threshold=0.9980, top_k_cfg=2)

    def test_pure_human_document(self):
        chunks = [
            DocumentChunk(0, "Текст 1", 0, 10, 50, 3, ai_probability=0.02, gate_value=0.51, is_ai=False),
            DocumentChunk(1, "Текст 2", 11, 21, 60, 4, ai_probability=0.08, gate_value=0.52, is_ai=False),
            DocumentChunk(2, "Текст 3", 22, 32, 55, 3, ai_probability=0.04, gate_value=0.51, is_ai=False),
        ]
        res = self.agg.aggregate(chunks, total_words=165, total_sentences=10)
        self.assertEqual(res.verdict, "Authentic Human")
        self.assertLess(res.document_ai_probability, 0.10)
        self.assertEqual(res.ai_content_ratio, 0.0)
        self.assertEqual(res.total_chunks, 3)

    def test_hybrid_document_isolated_ai_insertion(self):
        # 4 human chunks + 1 AI chunk (20% AI volume)
        chunks = [
            DocumentChunk(0, "Хуман 1", 0, 10, 50, 3, ai_probability=0.03, gate_value=0.51, is_ai=False),
            DocumentChunk(1, "Хуман 2", 11, 21, 50, 3, ai_probability=0.05, gate_value=0.52, is_ai=False),
            DocumentChunk(2, "AI блок", 22, 32, 50, 3, ai_probability=0.9995, gate_value=0.52, is_ai=True),
            DocumentChunk(3, "Хуман 3", 33, 43, 50, 3, ai_probability=0.04, gate_value=0.51, is_ai=False),
            DocumentChunk(4, "Хуман 4", 44, 54, 50, 3, ai_probability=0.02, gate_value=0.51, is_ai=False),
        ]
        res = self.agg.aggregate(chunks, total_words=250, total_sentences=15)
        self.assertEqual(res.verdict, "Partially AI / Hybrid")
        self.assertIsNotNone(res.worst_chunk)
        self.assertEqual(res.worst_chunk.index, 2)
        self.assertAlmostEqual(res.ai_content_ratio, 0.20, places=2)
        self.assertGreaterEqual(res.document_ai_probability, 0.9980)

    def test_machine_generated_document(self):
        chunks = [
            DocumentChunk(0, "AI 1", 0, 10, 50, 3, ai_probability=0.9998, gate_value=0.52, is_ai=True),
            DocumentChunk(1, "AI 2", 11, 21, 50, 3, ai_probability=0.9992, gate_value=0.52, is_ai=True),
            DocumentChunk(2, "AI 3", 22, 32, 50, 3, ai_probability=0.9999, gate_value=0.52, is_ai=True),
        ]
        res = self.agg.aggregate(chunks, total_words=150, total_sentences=9)
        self.assertEqual(res.verdict, "Machine-Generated")
        self.assertEqual(res.ai_content_ratio, 1.0)
        self.assertGreaterEqual(res.document_ai_probability, 0.9990)

    def test_empty_document_safety(self):
        res = self.agg.aggregate([], total_words=0, total_sentences=0)
        self.assertEqual(res.verdict, "Authentic Human")
        self.assertEqual(res.total_chunks, 0)
        self.assertEqual(res.total_words, 0)
        self.assertEqual(res.ai_content_ratio, 0.0)
        self.assertIsNone(res.worst_chunk)

    def test_top_k_worst_chunk_calculation(self):
        # M=1 -> K=1
        c1 = [DocumentChunk(0, "Текст", 0, 10, 10, 1, ai_probability=0.40)]
        res1 = self.agg.aggregate(c1)
        self.assertAlmostEqual(res1.document_ai_probability, 0.40)
        self.assertEqual(res1.total_words, 10)
        self.assertEqual(res1.total_sentences, 1)

        # M=3 human -> K=1 (ceil(0.25 * 3) = 1) -> max chunk
        c3 = [
            DocumentChunk(0, "T1", 0, 10, 10, 1, ai_probability=0.10),
            DocumentChunk(1, "T2", 10, 20, 10, 1, ai_probability=0.30),
            DocumentChunk(2, "T3", 20, 30, 10, 1, ai_probability=0.20),
        ]
        res3 = self.agg.aggregate(c3)
        self.assertAlmostEqual(res3.document_ai_probability, 0.30)

        # M=10 human -> K=2 (ceil(0.25 * 10) = 3 -> min(2, 3) = 2) -> average of top 2
        c10 = [
            DocumentChunk(i, f"T{i}", i * 10, (i + 1) * 10, 10, 1, ai_probability=0.01 * (i + 1))
            for i in range(10)
        ]
        # Top 2 are 0.10 (i=9) and 0.09 (i=8), average = 0.095
        res10 = self.agg.aggregate(c10)
        self.assertAlmostEqual(res10.document_ai_probability, 0.095)

    def test_serialization_compatibility(self):
        chunks = [
            DocumentChunk(0, "Текст", 0, 10, 20, 2, ai_probability=0.9995, gate_value=0.53, is_ai=False)
        ]
        res = self.agg.aggregate(chunks)
        d = res.to_dict()
        self.assertEqual(d["verdict"], "Machine-Generated")
        self.assertTrue(chunks[0].is_ai)
        self.assertEqual(d["worst_chunk"]["index"], 0)
        self.assertEqual(len(d["chunks"]), 1)


if __name__ == "__main__":
    unittest.main()
