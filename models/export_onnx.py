# -*- coding: utf-8 -*-
"""
models/export_onnx.py: Unified dual-stream ONNX export wrapper and exporter for MorphoContrastiveDetector.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedDualStreamExportWrapper(nn.Module):
    """
    Unified PyTorch export wrapper encapsulating MorphoContrastiveDetector for ONNX tracing.

    Accepts:
        input_ids (Tensor[B, L_text], long)
        attention_mask (Tensor[B, L_text], long)
        morph_ids (Tensor[B, L_morph], long)

    Outputs:
        ai_probability (Tensor[B, 1], float32 in [0, 1])
        gate_weight (Tensor[B, 1], float32 in [0, 1])
    """

    def __init__(self, detector: nn.Module):
        super().__init__()
        self.detector = detector

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        morph_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.detector(
            input_ids=input_ids,
            attention_mask=attention_mask,
            morpheme_ids=morph_ids,
        )
        logits = out["logits"]
        probs = F.softmax(logits, dim=-1)
        ai_probability = probs[:, 1:2]

        gate = out["gate"]
        gate_weight = torch.mean(gate, dim=-1, keepdim=True)
        return ai_probability, gate_weight


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    seq_len: int = 256,
    opset_version: int = 18,
    device: str = "cpu",
    dynamo: bool = False,
) -> str:
    """
    Exports a MorphoContrastiveDetector instance to a single unified ONNX graph.

    Args:
        model: Trained or initialized MorphoContrastiveDetector instance.
        output_path: Destination path for the .onnx model file.
        seq_len: Default sequence length for dummy trace inputs.
        opset_version: ONNX operator set version (default 18).
        device: Device to place dummy tensors on ('cpu').
        dynamo: Whether to use Dynamo exporter (default False for stable dynamic shapes).

    Returns:
        Absolute path to the exported ONNX model.
    """
    wrapper = UnifiedDualStreamExportWrapper(model)
    wrapper.eval()
    wrapper.to(device)

    dummy_input_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    dummy_morph_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    export_kwargs = {
        "export_params": True,
        "opset_version": opset_version,
        "do_constant_folding": True,
        "input_names": ["input_ids", "attention_mask", "morph_ids"],
        "output_names": ["ai_probability", "gate_weight"],
        "dynamic_axes": {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "morph_ids": {0: "batch_size", 1: "seq_len"},
            "ai_probability": {0: "batch_size"},
            "gate_weight": {0: "batch_size"},
        },
    }
    if dynamo is not None:
        export_kwargs["dynamo"] = dynamo

    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask, dummy_morph_ids),
        output_path,
        **export_kwargs,
    )

    # Validate exported ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    return os.path.abspath(output_path)
