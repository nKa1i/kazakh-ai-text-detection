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


class TestSentencePreservingChunker(unittest.TestCase):
    def test_kazakh_abbreviations_protection(self):
        from kaz_mage.chunker import SentencePreservingChunker
        chunker = SentencePreservingChunker()
        text = "Бұл 2024 ж. болған оқиға. Онда т.б. мәселелер мен 15 ғғ. мұралары қаралды."
        sentences = chunker.split_sentences(text)
        # Should be exactly 2 sentences, NOT split at "ж.", "т.б.", or "ғғ."
        self.assertEqual(len(sentences), 2)
        self.assertIn("2024 ж.", sentences[0][0])
        self.assertIn("т.б.", sentences[1][0])
        # Offsets
        for s_text, s_start, s_end in sentences:
            self.assertEqual(text[s_start:s_end], s_text)

    def test_kazakh_quotes_protection(self):
        from kaz_mage.chunker import SentencePreservingChunker
        chunker = SentencePreservingChunker()
        text = "«Бұл өте маңызды жоба!» деді министр. Ол жұмыстың сәтті аяқталғанын айтты."
        sentences = chunker.split_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertTrue(sentences[0][0].startswith("«Бұл"))
        for s_text, s_start, s_end in sentences:
            self.assertEqual(text[s_start:s_end], s_text)

    def test_chunking_word_budget_and_overlap(self):
        from kaz_mage.chunker import SentencePreservingChunker
        chunker = SentencePreservingChunker(max_words=30, overlap_sentences=1)
        sents = [f"Бұл құжаттың {i}-ші сөйлемі болып табылады және мағыналы ақпарат береді." for i in range(5)]
        doc = " ".join(sents)
        chunks = chunker.chunk_document(doc)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c.word_count, 45)
            # Exact offset verification
            self.assertEqual(doc[c.start_char : c.end_char], c.text)

    def test_empty_and_whitespace_inputs(self):
        from kaz_mage.chunker import SentencePreservingChunker
        chunker = SentencePreservingChunker()
        self.assertEqual(chunker.chunk_document(""), [])
        self.assertEqual(chunker.chunk_document("   \n\n\t  "), [])

    def test_kazakh_quotes_without_attribution(self):
        from kaz_mage.chunker import SentencePreservingChunker
        chunker = SentencePreservingChunker()
        text = "«Бұл өте жақсы бастама!» Жаңа жылда жаңа жұмыстар күтілуде."
        sentences = chunker.split_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0][0], "«Бұл өте жақсы бастама!»")
        self.assertEqual(sentences[1][0], "Жаңа жылда жаңа жұмыстар күтілуде.")
        for s_text, s_start, s_end in sentences:
            self.assertEqual(text[s_start:s_end], s_text)

    def test_paragraph_breaks_with_headings_and_abbreviations(self):
        from kaz_mage.chunker import SentencePreservingChunker
        chunker = SentencePreservingChunker()
        text = "1. Кіріспе\n\nБұл алғашқы бөлім.\n\nҚұжаттар т.б.\n\nКелесі бөлім."
        sentences = chunker.split_sentences(text)
        self.assertEqual(len(sentences), 4)
        self.assertEqual(sentences[0][0], "1. Кіріспе")
        self.assertEqual(sentences[1][0], "Бұл алғашқы бөлім.")
        self.assertEqual(sentences[2][0], "Құжаттар т.б.")
        self.assertEqual(sentences[3][0], "Келесі бөлім.")
        for s_text, s_start, s_end in sentences:
            self.assertEqual(text[s_start:s_end], s_text)

    def test_single_sentence_document_chunking(self):
        from kaz_mage.chunker import SentencePreservingChunker
        chunker = SentencePreservingChunker()
        text = "Бұл жалғыз сөйлем."
        chunks = chunker.chunk_document(text)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.index, 0)
        self.assertEqual(chunk.text, "Бұл жалғыз сөйлем.")
        self.assertEqual(chunk.start_char, 0)
        self.assertEqual(chunk.end_char, len(text))
        self.assertEqual(chunk.sentence_count, 1)
        self.assertEqual(chunk.word_count, 3)
        self.assertEqual(text[chunk.start_char:chunk.end_char], chunk.text)


if __name__ == "__main__":
    unittest.main()

