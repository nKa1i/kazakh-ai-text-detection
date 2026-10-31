# Phase 1: Production Quantized ONNX Neural Engine Design Spec

## Executive Summary
This specification defines the architecture, export pipeline, dynamic INT8 quantization, and CPU runtime for the unified dual-stream Kazakh AI text detector (`MorphoContrastiveDetector`). 
The goal is to move beyond offline heuristic fallbacks and execute live neural inference on standard CPU instances (including Hugging Face Spaces free-tier and local machines) with sub-50ms chunk latency and full parity with PyTorch weights.

---

## 1. System Architecture

```
                          Raw Kazakh Text
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
       Text Tokenizer                   Auto-FST v2 Overlay
(kz-roberta-conversational)             (DynamicFSTOverlay)
               │                                 │
   input_ids, attention_mask                morph_ids
       (Batch, 256)                        (Batch, 256)
               │                                 │
               └────────────────┬────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │     Unified Dual-Stream ONNX Graph           │
         │  (model_quantized.onnx - INT8 Dynamic)       │
         │                                              │
         │  1. RoBERTa Text Backbone                    │
         │     -> H_text (Batch, 256, 768)              │
         │                                              │
         │  2. MorphemeEncoder (Pos + FST Embeddings)   │
         │     -> H_morph (Batch, 256, 768)             │
         │                                              │
         │  3. Gated Cross-Attention Fusion             │
         │     A = Softmax(Q_text * K_morph^T / sqrt(d))│
         │     H_cross = A * V_morph                    │
         │     z_fused = DynamicGate(H_text, H_cross)   │
         │                                              │
         │  4. Classification Head                      │
         │     logits -> ai_probability (Float)         │
         │     dynamic_gate -> gate_weight (Float)      │
         └──────────────────────┬───────────────────────┘
                                │
                                ▼
                DocumentAggregator (Top-K Pooling)
                                │
                                ▼
              DocumentAnalysisResult (Verdict & Ratio)
```

---

## 2. Component Specifications

### 2.1. Unified ONNX Export Wrapper (`models/export_onnx.py`)
- **Wrapper Class**: `UnifiedDualStreamExportWrapper(nn.Module)`
- **Responsibilities**:
  - Encapsulates `MorphoContrastiveDetector` to accept 3 standard tensor inputs:
    1. `input_ids`: `torch.LongTensor` of shape `[batch_size, seq_len]`
    2. `attention_mask`: `torch.LongTensor` of shape `[batch_size, seq_len]`
    3. `morph_ids`: `torch.LongTensor` of shape `[batch_size, seq_len]`
  - Emits 2 outputs:
    1. `ai_probability`: `torch.FloatTensor` of shape `[batch_size, 1]` (Sigmoid probability)
    2. `gate_weight`: `torch.FloatTensor` of shape `[batch_size, 1]` ($g \in [0, 1]$)
- **Dynamic Axes**:
  - `batch_size`: dynamic dimension index 0 for all inputs and outputs.
  - `seq_len`: fixed at 256 (standard chunk budget) or dynamic `[seq_len]` up to 256.

### 2.2. Dynamic INT8 Quantizer (`models/quantize_onnx.py`)
- **Engine**: `onnxruntime.quantization.quantize_dynamic`
- **Target Weights**: Linear layers, attention projection matrices, and embedding matrices.
- **Quantization Type**: `QuantType.QInt8` for weights, `QuantType.QUInt8` for activations where supported.
- **Target File Size**: Reduced from ~480MB (FP32) to ~110MB (INT8).
- **Target CPU Latency**: $\le 45\text{ms}$ per 256-token chunk on standard multi-core x86_64 CPU.

### 2.3. Production ONNX Inference Runner (`ONNXDualStreamDetector`)
- **Location**: [`models/document_detector.py`](file:///C:/Users/Roza/Desktop/projects/kazakh-ai-text-detection/models/document_detector.py)
- **Capabilities**:
  - Cold-start verification: checks for presence of `model_quantized.onnx` or `model.onnx`.
  - Fallback mechanism: if ONNX files or `onnxruntime` are missing, automatically falls back to `OfflineHeuristicDetector` with a logged warning, preserving zero-crash stability.
  - Sliding-window document chunking via `SentencePreservingChunker(max_words=120)`.
  - Top-K worst-chunk document aggregation via `DocumentAggregator(calibrated_threshold=0.9980)`.
  - Full API compatibility with `OfflineHeuristicDetector.predict_document(text)`.

### 2.4. Performance & Parity Verification (`scripts/benchmark_onnx_inference.py`)
- **Parity Check**: Asserts max absolute difference between PyTorch FP32 output and ONNX INT8 output is within numerical tolerance ($|\Delta P| \le 0.03$).
- **Latency Benchmark**: Measures P50, P90, and P99 latency over 50 test chunks; asserts P50 $< 50\text{ms}$.
- **Memory Benchmark**: Asserts peak RAM during inference $\le 350\text{MB}$.

---

## 3. Interfaces & Data Contracts

### 3.1. Input Signature
```python
{
    "input_ids": np.ndarray(dtype=np.int64, shape=(batch_size, 256)),
    "attention_mask": np.ndarray(dtype=np.int64, shape=(batch_size, 256)),
    "morph_ids": np.ndarray(dtype=np.int64, shape=(batch_size, 256)),
}
```

### 3.2. Output Signature
```python
{
    "ai_probability": np.ndarray(dtype=np.float32, shape=(batch_size, 1)),
    "gate_weight": np.ndarray(dtype=np.float32, shape=(batch_size, 1)),
}
```

---

## 4. Error Handling & Safety
1. **Missing Model File**: Graceful fallback to `OfflineHeuristicDetector` ensuring Web UI and Space never fail cold-start.
2. **Empty / Whitespace Input**: Returns standard `DocumentAnalysisResult(verdict="Authentic Human", document_ai_probability=0.0)`.
3. **Mismatched Lengths**: Automatic padding or truncation to 256 tokens in preprocessing.
