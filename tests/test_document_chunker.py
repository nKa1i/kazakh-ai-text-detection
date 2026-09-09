import unittest
from kaz_mage.document import DocumentChunk, DocumentAnalysisResult

class TestDocumentDataStructures(unittest.TestCase):
    def test_document_chunk_serialization(self):
        chunk = DocumentChunk(
            index=0,
            text="Бұл сынақ мәтіні.",
            start_char=0,
            end_char=17,
            word_count=3,
            sentence_count=1,
            ai_probability=0.05,
            gate_value=0.52,
            is_ai=False
        )
        d = chunk.to_dict()
        self.assertEqual(d["index"], 0)
        self.assertEqual(d["word_count"], 3)
        self.assertFalse(d["is_ai"])

    def test_document_analysis_result_serialization(self):
        chunk = DocumentChunk(0, "Мәтін", 0, 5, 1, 1, 0.999, 0.51, True)
        res = DocumentAnalysisResult(
            verdict="Machine-Generated",
            document_ai_probability=0.999,
            ai_content_ratio=1.0,
            calibrated_threshold=0.998,
            total_words=1,
            total_sentences=1,
            total_chunks=1,
            worst_chunk=chunk,
            chunks=[chunk]
        )
        d = res.to_dict()
        self.assertEqual(d["verdict"], "Machine-Generated")
        self.assertEqual(len(d["chunks"]), 1)
        self.assertEqual(d["worst_chunk"]["ai_probability"], 0.999)

if __name__ == "__main__":
    unittest.main()
