# -*- coding: utf-8 -*-
"""
tests/test_onnx_dual_stream_detector.py: Unit tests for ONNXDualStreamDetector runner.
"""

import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

from kaz_mage.document import DocumentAnalysisResult
from models.export_onnx import export_to_onnx
from models.morpho_contrastive_detector import MorphoContrastiveDetector, MorphemeEncoder
from models.quantize_onnx import quantize_onnx_model


class _TinyBackbone(nn.Module):
    def __init__(self, vocab_size: int = 100, embed_dim: int = 64):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_ids, attention_mask=None):
        emb = self.embeddings(input_ids)
        if attention_mask is not None:
            emb = emb * attention_mask.unsqueeze(-1).to(emb.dtype)
        h = self.linear(emb)
        class _Out:
            def __init__(self, state):
                self.last_hidden_state = state
        return _Out(h)


class TestONNXDualStreamDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.fp32_path = os.path.join(cls.tmpdir.name, "model.onnx")
        cls.int8_path = os.path.join(cls.tmpdir.name, "model_quantized.onnx")

        # Create lightweight detector and export both FP32 and INT8 ONNX
        cls.embed_dim = 64
        cls.vocab_size = 100
        cls.morpheme_vocab_size = 50
        cls.seq_len = 32

        detector = MorphoContrastiveDetector(
            roberta_model=_TinyBackbone(vocab_size=cls.vocab_size, embed_dim=cls.embed_dim),
            morpheme_vocab_size=cls.morpheme_vocab_size,
            embed_dim=cls.embed_dim,
            proj_dim=32,
        )
        detector.morph_encoder = MorphemeEncoder(
            vocab_size=cls.morpheme_vocab_size,
            embed_dim=cls.embed_dim,
            num_layers=1,
            nhead=2,
            dim_feedforward=128,
            max_pos=64,
        )
        detector.eval()

        export_to_onnx(detector, cls.fp32_path, seq_len=cls.seq_len, opset_version=18)
        quantize_onnx_model(cls.fp32_path, cls.int8_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_fallback_when_model_missing(self):
        from models.document_detector import ONNXDualStreamDetector

        det = ONNXDualStreamDetector(model_path="non_existent_model_file.onnx")
        self.assertFalse(det.is_onnx_active)

        text = "Бұл қазақ тіліндегі қарапайым адам жазған мәтін."
        res = det.predict_document(text)
        self.assertIsInstance(res, DocumentAnalysisResult)
        self.assertIn(res.verdict, ["Authentic Human", "Suspicious / Hybrid", "Machine-Generated"])
        self.assertIsNotNone(res.generator_profile)

    def test_onnx_inference_with_fp32_model(self):
        from models.document_detector import ONNXDualStreamDetector

        det = ONNXDualStreamDetector(
            model_path=self.fp32_path,
            calibrated_threshold=0.9980,
            vocab_size=self.vocab_size,
            morpheme_vocab_size=self.morpheme_vocab_size,
            max_seq_len=self.seq_len,
        )
        self.assertTrue(det.is_onnx_active)

        text = (
            "Қазақстанның экологиялық мәселелері бүгінгі таңда ерекше маңызға ие болып отыр. "
            "Арал теңізінің тартылуы мен Семей полигонының зардаптары ұзақ жылдар бойы талқылануда."
        )
        res = det.predict_document(text)
        self.assertIsInstance(res, DocumentAnalysisResult)
        self.assertGreater(len(res.chunks), 0)
        for chunk in res.chunks:
            self.assertTrue(0.0 <= chunk.ai_probability <= 1.0)
            self.assertTrue(0.0 <= chunk.gate_value <= 1.0)

        # Dictionary backward compatibility
        d = det.predict(text)
        self.assertIn("ai_probability", d)
        self.assertIn("verdict", d)
        self.assertIn("ai_content_ratio", d)
        self.assertIn("generator_profile", d)

    def test_onnx_inference_with_int8_quantized_model(self):
        from models.document_detector import ONNXDualStreamDetector

        det = ONNXDualStreamDetector(
            model_path=self.int8_path,
            calibrated_threshold=0.9980,
            vocab_size=self.vocab_size,
            morpheme_vocab_size=self.morpheme_vocab_size,
            max_seq_len=self.seq_len,
        )
        self.assertTrue(det.is_onnx_active)

        text = "Жасанды интеллект модельдері мәтінді сапалы өңдеуге мүмкіндік береді."
        res = det.predict_document(text)
        self.assertIsInstance(res, DocumentAnalysisResult)
        self.assertTrue(0.0 <= res.document_ai_probability <= 1.0)

    def test_empty_text_handling(self):
        from models.document_detector import ONNXDualStreamDetector

        det = ONNXDualStreamDetector(
            model_path=self.int8_path,
            vocab_size=self.vocab_size,
            morpheme_vocab_size=self.morpheme_vocab_size,
            max_seq_len=self.seq_len,
        )
        res = det.predict_document("")
        self.assertEqual(res.document_ai_probability, 0.0)
        self.assertEqual(res.total_chunks, 0)
        self.assertEqual(res.verdict, "Authentic Human")


if __name__ == "__main__":
    unittest.main()
