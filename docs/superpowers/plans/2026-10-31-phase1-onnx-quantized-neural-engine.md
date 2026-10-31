# Phase 1: Production Quantized ONNX Neural Engine Implementation Plan

> **Goal**: Build a unified ONNX export, dynamic INT8 quantization pipeline, and production-grade CPU inference runner (`ONNXDualStreamDetector`) achieving sub-50ms CPU chunk inference for the dual-stream Kazakh AI text detector, with graceful fallback to `OfflineHeuristicDetector`.

---

## Technical Context & Constraints
- **Python**: 3.12 at `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe`.
- **Primary Dependencies**: `torch`, `transformers`, `onnx`, `onnxruntime` (ensure installed via `pip install onnx onnxruntime`).
- **Encoding**: `$env:PYTHONIOENCODING="utf-8"`.
- **Rule**: Strictly zero decorative emojis across all code, tests, docstrings, and commit messages.
- **TDD Workflow**: Write tests first, verify test failure, implement feature, verify tests pass, commit.

---

## Proposed Tasks Breakdown

### Task 1: Unified ONNX Export Wrapper & Verification
**Objective**: Build `UnifiedDualStreamExportWrapper` in `models/export_onnx.py` that wraps `MorphoContrastiveDetector` into a clean 3-input (`input_ids`, `attention_mask`, `morph_ids`) and 2-output (`ai_probability`, `gate_weight`) PyTorch module, and exports it to ONNX format.

**Files**:
- Create: `models/export_onnx.py`
- Create: `tests/test_export_onnx.py`

**Steps**:
1.1. Create `tests/test_export_onnx.py`:
   - Test wrapper instantiation and forward pass with dummy batch `[2, 256]`.
   - Test export to a temporary `.onnx` file using `torch.onnx.export`.
   - Validate exported ONNX model structure using `onnx.checker.check_model`.
   - Assert output shapes match `[batch_size, 1]`.
1.2. Run test to verify failure (`python -m unittest tests/test_export_onnx.py`).
1.3. Implement `models/export_onnx.py`:
   - Define `UnifiedDualStreamExportWrapper(nn.Module)`.
   - Implement `export_to_onnx(model, output_path, seq_len=256, opset_version=14)`.
   - Include dynamic batch axes `{'input_ids': {0: 'batch_size'}, 'output': ...}`.
1.4. Run tests and verify all pass:
   `python -m unittest tests/test_export_onnx.py`
1.5. Commit:
   `git add models/export_onnx.py tests/test_export_onnx.py`
   `git commit -m "feat(onnx): add unified dual-stream ONNX export wrapper and verification"`

---

### Task 2: Dynamic INT8 Quantization Pipeline
**Objective**: Build `quantize_onnx_model()` in `models/quantize_onnx.py` using `onnxruntime.quantization.quantize_dynamic` to quantize FP32 ONNX graph to INT8 weights, reducing size from ~450MB to <120MB and improving CPU latency.

**Files**:
- Create: `models/quantize_onnx.py`
- Create: `tests/test_quantize_onnx.py`

**Steps**:
2.1. Create `tests/test_quantize_onnx.py`:
   - Export a lightweight dummy ONNX model.
   - Run `quantize_onnx_model(input_onnx_path, output_quant_path)`.
   - Verify output file exists, is smaller in file size than FP32 source.
   - Run inference using `onnxruntime.InferenceSession` on quantized model and assert numerical output is valid float $\in [0, 1]$.
2.2. Run test to verify failure (`python -m unittest tests/test_quantize_onnx.py`).
2.3. Implement `models/quantize_onnx.py`:
   - Implement `quantize_onnx_model(input_path, output_path, per_channel=False, weight_type=QuantType.QInt8)`.
   - Include error handling for missing files and validation checks.
2.4. Run tests and verify all pass:
   `python -m unittest tests/test_quantize_onnx.py`
2.5. Commit:
   `git add models/quantize_onnx.py tests/test_quantize_onnx.py`
   `git commit -m "feat(onnx): implement dynamic INT8 quantization pipeline for CPU inference"`

---

### Task 3: Production `ONNXDualStreamDetector` Runner
**Objective**: Build `ONNXDualStreamDetector` in `models/document_detector.py` providing seamless full-document analysis using the quantized ONNX engine, integrating `SentencePreservingChunker`, `DocumentAggregator`, and automatic fallback to `OfflineHeuristicDetector`.

**Files**:
- Modify: `models/document_detector.py`
- Create: `tests/test_onnx_dual_stream_detector.py`

**Steps**:
3.1. Create `tests/test_onnx_dual_stream_detector.py`:
   - Test instantiation with missing ONNX file falls back gracefully to `OfflineHeuristicDetector` without throwing an unhandled exception.
   - Test inference with a mock ONNX session returns valid `DocumentAnalysisResult`.
   - Test chunking and Top-K aggregation integration.
   - Test dynamic gate extraction and sliding window parity.
3.2. Run test to verify failure (`python -m unittest tests/test_onnx_dual_stream_detector.py`).
3.3. Implement `ONNXDualStreamDetector` in `models/document_detector.py`:
   - Check for `model_quantized.onnx` or `model.onnx` at initialization.
   - Setup `onnxruntime.InferenceSession` with optimized session options (`intra_op_num_threads=4`, `GraphOptimizationLevel.ORT_ENABLE_ALL`).
   - Tokenize text stream with `AutoTokenizer` and morphological stream with `DynamicFSTOverlay` + `MorphemeTokenizer`.
   - Implement `_score_chunk(text) -> Tuple[float, float]`.
   - Implement `predict_document(text) -> DocumentAnalysisResult`.
3.4. Run tests and verify all pass:
   `python -m unittest tests/test_onnx_dual_stream_detector.py`
3.5. Commit:
   `git add models/document_detector.py tests/test_onnx_dual_stream_detector.py`
   `git commit -m "feat(detector): add production ONNXDualStreamDetector with heuristic fallback"`

---

### Task 4: Performance, Latency & Parity Benchmark
**Objective**: Build `scripts/benchmark_onnx_inference.py` to benchmark CPU latency (P50, P90, P99), RAM footprint, and numerical parity against PyTorch baseline across 50 real Kazakh evaluation chunks.

**Files**:
- Create: `scripts/benchmark_onnx_inference.py`
- Create: `tests/test_benchmark_onnx_inference.py`

**Steps**:
4.1. Create `tests/test_benchmark_onnx_inference.py`:
   - Assert benchmarking script runs cleanly with `--dry-run` flag.
   - Verify generated benchmark JSON artifact contains latency statistics (`p50_ms`, `p90_ms`, `p99_ms`, `mean_ms`).
   - Assert P50 latency $\le 50.0\text{ms}$ on multi-core CPU.
4.2. Run test to verify failure (`python -m unittest tests/test_benchmark_onnx_inference.py`).
4.3. Implement `scripts/benchmark_onnx_inference.py`:
   - Warm-up loop (5 iterations).
   - Timing loop (50 iterations) with high-resolution `time.perf_counter_ns()`.
   - Compare raw outputs between PyTorch and ONNX INT8 to calculate Mean Absolute Error (MAE).
   - Write output to `data/onnx_cpu_benchmark_results.json`.
4.4. Run tests and execute benchmark:
   `python -m unittest tests/test_benchmark_onnx_inference.py`
   `python scripts/benchmark_onnx_inference.py --dry-run`
4.5. Commit:
   `git add scripts/benchmark_onnx_inference.py tests/test_benchmark_onnx_inference.py`
   `git commit -m "feat(benchmark): add CPU latency and parity verification suite for ONNX runtime"`

---

### Task 5: UI & Hugging Face Space Integration
**Objective**: Wire `ONNXDualStreamDetector` into `ui/app.py` and `hf_space/app.py`, update `requirements.txt` to include `onnxruntime`, and verify zero-downtime fallback.

**Files**:
- Modify: `ui/app.py`
- Modify: `hf_space/ui/app.py`
- Modify: `hf_space/requirements.txt`
- Modify: `tests/test_hf_space_bundle.py`

**Steps**:
5.1. Update `tests/test_hf_space_bundle.py`:
   - Assert `onnxruntime` is listed in `hf_space/requirements.txt`.
   - Assert `hf_space/models/document_detector.py` exports `ONNXDualStreamDetector`.
5.2. Update `ui/app.py` and `hf_space/ui/app.py`:
   - Prefer `ONNXDualStreamDetector` when model weights exist, falling back seamlessly to `OfflineHeuristicDetector`.
5.3. Synchronize `models/document_detector.py` to `hf_space/models/document_detector.py`.
5.4. Run tests to verify bundle and UI parity:
   `python -m unittest tests/test_hf_space_bundle.py tests/test_ui_integration.py`
5.5. Commit:
   `git add ui/app.py hf_space/ tests/test_hf_space_bundle.py`
   `git commit -m "feat(deploy): wire ONNXDualStreamDetector into Gradio UI and Hugging Face Space bundle"`

---

## Verification & Completion Checklist
- [ ] All 5 new test files pass: `test_export_onnx.py`, `test_quantize_onnx.py`, `test_onnx_dual_stream_detector.py`, `test_benchmark_onnx_inference.py`, `test_hf_space_bundle.py`.
- [ ] Full regression discovery passes: `python -m unittest discover -s tests`.
- [ ] Generated `data/onnx_cpu_benchmark_results.json` confirms P50 CPU latency $< 50\text{ms}$.
- [ ] Working tree is clean and committed.
