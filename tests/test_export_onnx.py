# -*- coding: utf-8 -*-
"""
tests/test_export_onnx.py: Unit tests for UnifiedDualStreamExportWrapper and ONNX export.
"""

import os
import tempfile
import unittest

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

from models.morpho_contrastive_detector import MorphoContrastiveDetector, MorphemeEncoder


class _TinyBackbone(nn.Module):
    """Lightweight deterministic backbone for fast offline ONNX testing."""
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


class TestExportONNX(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 64
        self.vocab_size = 100
        self.morpheme_vocab_size = 50
        self.seq_len = 32

        # Create tiny detector with mock backbone
        tiny_roberta = _TinyBackbone(vocab_size=self.vocab_size, embed_dim=self.embed_dim)
        self.detector = MorphoContrastiveDetector(
            roberta_model=tiny_roberta,
            morpheme_vocab_size=self.morpheme_vocab_size,
            embed_dim=self.embed_dim,
            proj_dim=32,
        )
        # Resize morpheme encoder for tiny dimensions
        self.detector.morph_encoder = MorphemeEncoder(
            vocab_size=self.morpheme_vocab_size,
            embed_dim=self.embed_dim,
            num_layers=1,
            nhead=2,
            dim_feedforward=128,
            max_pos=64,
        )
        self.detector.gate_fc = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.detector.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, 2)
        )
        self.detector.eval()

    def test_export_wrapper_forward_pass_shapes_and_ranges(self):
        from models.export_onnx import UnifiedDualStreamExportWrapper

        wrapper = UnifiedDualStreamExportWrapper(self.detector)
        wrapper.eval()

        batch_size = 3
        input_ids = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, self.seq_len), dtype=torch.long)
        morph_ids = torch.randint(0, self.morpheme_vocab_size, (batch_size, self.seq_len), dtype=torch.long)

        with torch.no_grad():
            ai_prob, gate_weight = wrapper(input_ids, attention_mask, morph_ids)

        self.assertEqual(ai_prob.shape, (batch_size, 1))
        self.assertEqual(gate_weight.shape, (batch_size, 1))
        self.assertTrue(torch.all(ai_prob >= 0.0) and torch.all(ai_prob <= 1.0))
        self.assertTrue(torch.all(gate_weight >= 0.0) and torch.all(gate_weight <= 1.0))

    def test_export_to_onnx_creates_valid_graph_with_onnxruntime_parity(self):
        from models.export_onnx import export_to_onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            out_path = export_to_onnx(
                self.detector,
                onnx_path,
                seq_len=self.seq_len,
                opset_version=14,
            )
            self.assertTrue(os.path.isfile(out_path))
            self.assertGreater(os.path.getsize(out_path), 5_000)

            # Validate ONNX schema
            onnx_model = onnx.load(out_path)
            onnx.checker.check_model(onnx_model)

            # Test inference with ONNX Runtime
            ort_session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
            batch_size = 2
            input_ids_np = torch.randint(0, self.vocab_size, (batch_size, self.seq_len)).numpy()
            attention_mask_np = torch.ones((batch_size, self.seq_len), dtype=torch.int64).numpy()
            morph_ids_np = torch.randint(0, self.morpheme_vocab_size, (batch_size, self.seq_len)).numpy()

            ort_inputs = {
                "input_ids": input_ids_np,
                "attention_mask": attention_mask_np,
                "morph_ids": morph_ids_np,
            }
            ort_outs = ort_session.run(None, ort_inputs)
            self.assertEqual(len(ort_outs), 2)
            ai_prob_ort, gate_ort = ort_outs

            self.assertEqual(ai_prob_ort.shape, (batch_size, 1))
            self.assertEqual(gate_ort.shape, (batch_size, 1))

            # Verify numerical parity with PyTorch wrapper
            from models.export_onnx import UnifiedDualStreamExportWrapper
            wrapper = UnifiedDualStreamExportWrapper(self.detector)
            with torch.no_grad():
                ai_prob_pt, gate_pt = wrapper(
                    torch.from_numpy(input_ids_np),
                    torch.from_numpy(attention_mask_np),
                    torch.from_numpy(morph_ids_np),
                )

            max_prob_diff = float(torch.max(torch.abs(ai_prob_pt - torch.from_numpy(ai_prob_ort))))
            self.assertLess(max_prob_diff, 1e-4)


if __name__ == "__main__":
    unittest.main()
