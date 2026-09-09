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
            # Return mock logits: default human, if b > 1 make last chunk AI
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
        long_text = "\u049a\u0430\u0437\u0430\u049b\u0441\u0442\u0430\u043d\u043d\u044b\u04a3 \u0446\u0438\u0444\u0440\u043b\u044b\u049b \u0434\u0430\u043c\u0443\u044b \u0436\u043e\u0493\u0430\u0440\u044b \u049b\u0430\u0440\u049b\u044b\u043d\u043c\u0435\u043d \u0436\u04af\u0440\u0456\u043f \u0436\u0430\u0442\u044b\u0440. " * 30
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

        res_short = detector.predict_document("\u0411\u04b1\u043b \u049b\u044b\u0441\u049b\u0430 \u043f\u0456\u043a\u0456\u0440.")
        self.assertEqual(res_short.total_chunks, 1)
        self.assertEqual(res_short.verdict, "Authentic Human")


if __name__ == "__main__":
    unittest.main()
