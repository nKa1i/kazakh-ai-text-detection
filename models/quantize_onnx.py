# -*- coding: utf-8 -*-
"""
models/quantize_onnx.py: Dynamic INT8 quantization pipeline for ONNX neural models.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    HAS_ORT_QUANT = True
except ImportError:
    HAS_ORT_QUANT = False
    QuantType = None
    quantize_dynamic = None


def quantize_onnx_model(
    input_path: str,
    output_path: str,
    weight_type: Optional[object] = None,
    per_channel: bool = False,
    reduce_range: bool = False,
) -> str:
    """
    Applies dynamic INT8 post-training quantization to an ONNX model file.

    Args:
        input_path: Path to the FP32 ONNX model file.
        output_path: Path where the quantized INT8 model should be saved.
        weight_type: QuantType enum (default QuantType.QInt8 if None).
        per_channel: Whether to quantize weights per-channel.
        reduce_range: Quantize weights with 7-bits instead of 8-bits.

    Returns:
        Absolute path to the generated INT8 model.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ImportError: If onnxruntime.quantization is not available.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input ONNX file not found: {input_path}")

    if not HAS_ORT_QUANT or quantize_dynamic is None:
        raise ImportError("onnxruntime.quantization is required for dynamic quantization")

    if weight_type is None:
        weight_type = QuantType.QInt8

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        per_channel=per_channel,
        reduce_range=reduce_range,
        weight_type=weight_type,
    )

    return os.path.abspath(output_path)
