# -*- coding: utf-8 -*-
"""
tests/test_quantize_onnx.py: Unit tests for dynamic INT8 ONNX quantization.
"""

import os
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from models.export_onnx import export_to_onnx
from models.morpho_contrastive_detector import MorphoContrastiveDetector, MorphemeEncoder


class _TinyBackbone(nn.Module):
    def __init__(self, vocab_size: int = 50, embed_dim: int = 64):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_ids, attention_mask=None):
        emb = self.embeddings(input_ids)
        if attention_mask is not None:
            emb = emb * attention_mask.unsqueeze(-1).to(emb.dtype)
        h = self.linear2(torch.relu(self.linear1(emb)))
        class _Out:
            def __init__(self, state):
                self.last_hidden_state = state
        return _Out(h)


class TestQuantizeONNX(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 64
        self.vocab_size = 50
        self.morpheme_vocab_size = 30
        self.seq_len = 32

        tiny_roberta = _TinyBackbone(vocab_size=self.vocab_size, embed_dim=self.embed_dim)
        self.detector = MorphoContrastiveDetector(
            roberta_model=tiny_roberta,
            morpheme_vocab_size=self.morpheme_vocab_size,
            embed_dim=self.embed_dim,
            proj_dim=32,
        )
        self.detector.morph_encoder = MorphemeEncoder(
            vocab_size=self.morpheme_vocab_size,
            embed_dim=self.embed_dim,
            num_layers=1,
            nhead=2,
            dim_feedforward=128,
            max_pos=64,
        )
        self.detector.eval()

    def test_quantize_onnx_model_creates_valid_int8_graph(self):
        from models.quantize_onnx import quantize_onnx_model

        with tempfile.TemporaryDirectory() as tmpdir:
            fp32_path = os.path.join(tmpdir, "model_fp32.onnx")
            int8_path = os.path.join(tmpdir, "model_int8.onnx")

            export_to_onnx(
                self.detector,
                fp32_path,
                seq_len=self.seq_len,
                opset_version=18,
            )
            self.assertTrue(os.path.isfile(fp32_path))

            result_path = quantize_onnx_model(fp32_path, int8_path)
            self.assertEqual(os.path.abspath(int8_path), os.path.abspath(result_path))
            self.assertTrue(os.path.isfile(int8_path))

            # Check that quantized model is valid ONNX
            onnx_model = onnx.load(int8_path)
            onnx.checker.check_model(onnx_model)

            # Check inference with ONNX Runtime on quantized model
            session = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
            batch_size = 2
            input_ids_np = np.random.randint(0, self.vocab_size, size=(batch_size, self.seq_len), dtype=np.int64)
            mask_np = np.ones((batch_size, self.seq_len), dtype=np.int64)
            morph_np = np.random.randint(0, self.morpheme_vocab_size, size=(batch_size, self.seq_len), dtype=np.int64)

            ort_inputs = {
                "input_ids": input_ids_np,
                "attention_mask": mask_np,
                "morph_ids": morph_np,
            }
            outputs = session.run(None, ort_inputs)
            self.assertEqual(len(outputs), 2)
            ai_prob, gate = outputs
            self.assertEqual(ai_prob.shape, (batch_size, 1))
            self.assertEqual(gate.shape, (batch_size, 1))
            self.assertTrue(np.all(ai_prob >= 0.0) and np.all(ai_prob <= 1.0))
            self.assertTrue(np.all(gate >= 0.0) and np.all(gate <= 1.0))

    def test_quantize_missing_file_raises_filenotfound(self):
        from models.quantize_onnx import quantize_onnx_model

        with self.assertRaises(FileNotFoundError):
            quantize_onnx_model("non_existent_file.onnx", "output.onnx")


if __name__ == "__main__":
    unittest.main()
