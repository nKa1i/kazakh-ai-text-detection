import unittest
from models.document_detector import DocumentDetector

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


class DummyMorphoModel:
    def __init__(self):
        self.device = "cpu"

    def eval(self):
        pass

    def to(self, dev):
        self.device = dev
        return self

    def __call__(self, input_ids, attention_mask, morpheme_ids=None):
        b = input_ids.shape[0] if hasattr(input_ids, "shape") else len(input_ids)
        if HAS_TORCH and torch is not None:
            logits = torch.zeros((b, 2))
            logits[:, 0] = 3.0
            logits[:, 1] = -3.0
            if b > 1:
                logits[-1, 0] = -5.0
                logits[-1, 1] = 5.0
            gate = torch.full((b, 768), 0.52)
            return {"logits": logits, "gate": gate}
        else:
            logits = [[3.0, -3.0] for _ in range(b)]
            if b > 1:
                logits[-1] = [-5.0, 5.0]
            gate = [[0.52] * 768 for _ in range(b)]
            return {"logits": logits, "gate": gate}


class TestDocumentDetector(unittest.TestCase):
    def test_predict_document_integration(self):
        detector = DocumentDetector(
            model=DummyMorphoModel(),
            raw_tokenizer=None,
            morpheme_tokenizer=None,
            calibrated_threshold=0.9980,
        )
        long_text = "Қазақстанның цифрлық дамуы жоғары қарқынмен жүріп жатыр. " * 30
        res = detector.predict_document(long_text, batch_size=2)
        self.assertIn(res.verdict, ["Authentic Human", "Partially AI / Hybrid", "Machine-Generated"])
        self.assertGreater(res.total_chunks, 1)
        self.assertIsNotNone(res.worst_chunk)
        self.assertEqual(len(res.chunks), res.total_chunks)
        self.assertGreater(res.total_words, 50)

    def test_empty_and_short_document(self):
        detector = DocumentDetector(
            model=DummyMorphoModel(),
            raw_tokenizer=None,
            morpheme_tokenizer=None,
            calibrated_threshold=0.9980,
        )
        res_empty = detector.predict_document("")
        self.assertEqual(res_empty.verdict, "Authentic Human")
        self.assertEqual(res_empty.total_chunks, 0)

        res_short = detector.predict_document("Бұл қысқа пікір.")
        self.assertEqual(res_short.total_chunks, 1)
        self.assertEqual(res_short.verdict, "Authentic Human")

    def test_hybrid_document_synthesis(self):
        class TextAwareMockModel:
            def __init__(self, target_marker):
                self.target_marker = target_marker
                self.current_texts = []

            def eval(self):
                pass

            def to(self, dev):
                return self

            def set_current_texts(self, texts):
                self.current_texts = texts

            def __call__(self, input_ids, attention_mask, morpheme_ids=None):
                b = input_ids.shape[0] if hasattr(input_ids, "shape") else len(input_ids)
                if HAS_TORCH and torch is not None:
                    logits = torch.zeros((b, 2))
                    for idx in range(b):
                        txt = self.current_texts[idx] if idx < len(self.current_texts) else ""
                        if self.target_marker in txt:
                            logits[idx, 0] = -5.0
                            logits[idx, 1] = 5.0
                        else:
                            logits[idx, 0] = 4.0
                            logits[idx, 1] = -4.0
                    return {"logits": logits, "gate": torch.full((b, 768), 0.52)}
                else:
                    logits = []
                    for idx in range(b):
                        txt = self.current_texts[idx] if idx < len(self.current_texts) else ""
                        if self.target_marker in txt:
                            logits.append([-5.0, 5.0])
                        else:
                            logits.append([4.0, -4.0])
                    gate = [[0.52] * 768 for _ in range(b)]
                    return {"logits": logits, "gate": gate}

        marker = "ЖАСАНДЫ_ИНТЕЛЛЕКТ_ИНЖЕКЦИЯСЫ"
        mock_model = TextAwareMockModel(marker)
        # Hook to capture chunk texts as they pass through detector
        detector = DocumentDetector(
            model=mock_model,
            raw_tokenizer=None,
            morpheme_tokenizer=None,
            calibrated_threshold=0.9980,
            max_words=20,
            overlap_sentences=1
        )

        orig_predict = detector.predict_document
        def text_aware_predict(text, top_k=2, batch_size=16):
            chunks = detector.chunker.chunk_document(text)
            mock_model.set_current_texts([c.text for c in chunks])
            return orig_predict(text, top_k=top_k, batch_size=batch_size)

        para1 = "Астана қаласында халықаралық форум өтті. Оған көптеген сарапшылар мен ғалымдар қатысты. Басты тақырып экономикалық ынтымақтастық болды.\n\n"
        para2 = "Елімізде жаңа технологиялық паркер мен зертханалар ашылуда. Бұл жастарға үлкен мүмкіндік береді. Ғылым мен өндіріс байланысы күшеюде.\n\n"
        para3_injected = f"Бұл абзац {marker} арқылы жасалған. Онда генеративті модель құрастырған арнайы синтетикалық мәліметтер бар.\n\n"
        para4 = "Қорытындылай келе, алдағы жылдары цифрландыру саласында тың серпіліс болады деп күтілуде. Халықаралық рейтингтер де осыны растайды."

        hybrid_doc = para1 + para2 + para3_injected + para4
        res = text_aware_predict(hybrid_doc, batch_size=10)

        self.assertEqual(res.verdict, "Partially AI / Hybrid")
        self.assertIsNotNone(res.worst_chunk)
        self.assertIn(marker, res.worst_chunk.text)
        self.assertTrue(res.worst_chunk.is_ai)
        self.assertGreater(res.worst_chunk.ai_probability, 0.9980)
        self.assertEqual(hybrid_doc[res.worst_chunk.start_char : res.worst_chunk.end_char], res.worst_chunk.text)
        self.assertGreater(res.ai_content_ratio, 0.0)
        self.assertLess(res.ai_content_ratio, 0.70)

    def test_unbatched_1d_logits_guard(self):
        class Model1D:
            def eval(self):
                pass
            def to(self, d):
                return self
            def __call__(self, input_ids, attention_mask, morpheme_ids=None):
                if HAS_TORCH and torch is not None:
                    return {"logits": torch.tensor([2.0, -2.0]), "gate": torch.full((1, 768), 0.52)}
                else:
                    return {"logits": [2.0, -2.0], "gate": [0.52] * 768}

        detector = DocumentDetector(
            model=Model1D(),
            raw_tokenizer=None,
            morpheme_tokenizer=None,
            calibrated_threshold=0.9980
        )
        res = detector.predict_document("Қысқа мәтін сынағы.")
        self.assertEqual(res.total_chunks, 1)
        self.assertEqual(res.verdict, "Authentic Human")


if __name__ == "__main__":
    unittest.main()
