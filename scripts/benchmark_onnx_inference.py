# -*- coding: utf-8 -*-
"""
scripts/benchmark_onnx_inference.py: Benchmark CPU latency, RAM/model size, and numerical parity.
Compares PyTorch, ONNX FP32, and dynamic INT8 ONNX inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure project root is on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    ort = None
    HAS_ORT = False

from models.export_onnx import export_to_onnx, UnifiedDualStreamExportWrapper
from models.morpho_contrastive_detector import MorphoContrastiveDetector, MorphemeEncoder
from models.quantize_onnx import quantize_onnx_model


SAMPLE_KAZAKH_CHUNKS = [
    "Қазақстанның экологиялық мәселелері бүгінгі таңда ерекше маңызға ие болып отыр. Арал теңізінің тартылуы талқылануда.",
    "Бұл мәтін жасанды интеллект моделі арқылы автоматты түрде дайындалған. Осыған байланысты нақты шешімдер қажет.",
    "Абай Құнанбайұлының қара сөздері мен философиялық толғаулары қазақ әдебиетінің алтын қорына енген мәдени мұра болып табылады.",
    "Қорытындылай келе, цифрлық трансформация барлық мемлекеттік қызметтердің тиімділігін арттыруға мүмкіндік береді.",
    "Ғылым мен білім саласындағы соңғы жаңалықтар жастардың дамуына зор серпін береді және халықаралық ынтымақтастықты нығайтады.",
]


class _BenchBackbone(nn.Module):
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


def _compute_percentiles(timings_ms: List[float]) -> Dict[str, float]:
    arr = np.array(timings_ms, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "std_ms": round(float(np.std(arr)), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p90_ms": round(float(np.percentile(arr, 90)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
    }


def run_benchmark(
    dry_run: bool = False,
    output_path: str = "data/onnx_cpu_benchmark_results.json",
    num_iterations: int = 50,
    warmup: int = 5,
    num_threads: int = 4,
) -> Dict[str, Any]:
    if not HAS_TORCH or not HAS_ORT:
        raise RuntimeError("torch and onnxruntime are required for running benchmarks")

    seq_len = 32 if dry_run else 128
    vocab_size = 100
    morpheme_vocab_size = 50
    embed_dim = 64 if dry_run else 128
    iters = 2 if dry_run else num_iterations
    warmup_iters = 1 if dry_run else warmup

    # 1. Build PyTorch model
    torch_model = MorphoContrastiveDetector(
        roberta_model=_BenchBackbone(vocab_size=vocab_size, embed_dim=embed_dim),
        morpheme_vocab_size=morpheme_vocab_size,
        embed_dim=embed_dim,
        proj_dim=32,
    )
    torch_model.morph_encoder = MorphemeEncoder(
        vocab_size=morpheme_vocab_size,
        embed_dim=embed_dim,
        num_layers=1,
        nhead=2,
        dim_feedforward=128,
        max_pos=256,
    )
    torch_model.eval()
    pt_wrapper = UnifiedDualStreamExportWrapper(torch_model)
    pt_wrapper.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        fp32_onnx_path = os.path.join(tmpdir, "model_fp32.onnx")
        int8_onnx_path = os.path.join(tmpdir, "model_int8.onnx")

        # 2. Export FP32 ONNX
        export_to_onnx(torch_model, fp32_onnx_path, seq_len=seq_len, opset_version=18)
        # 3. Quantize to INT8 ONNX
        quantize_onnx_model(fp32_onnx_path, int8_onnx_path)

        fp32_size_bytes = os.path.getsize(fp32_onnx_path)
        int8_size_bytes = os.path.getsize(int8_onnx_path)

        # 4. Setup ONNX sessions
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = num_threads
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        sess_fp32 = ort.InferenceSession(fp32_onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
        sess_int8 = ort.InferenceSession(int8_onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

        # Prepare dummy batch (B=1 single chunk inference)
        input_ids_np = np.random.randint(0, vocab_size, size=(1, seq_len), dtype=np.int64)
        mask_np = np.ones((1, seq_len), dtype=np.int64)
        morph_np = np.random.randint(0, morpheme_vocab_size, size=(1, seq_len), dtype=np.int64)

        ort_inputs = {
            "input_ids": input_ids_np,
            "attention_mask": mask_np,
            "morph_ids": morph_np,
        }
        pt_input_ids = torch.from_numpy(input_ids_np)
        pt_mask = torch.from_numpy(mask_np)
        pt_morph = torch.from_numpy(morph_np)

        # Warmup
        for _ in range(warmup_iters):
            with torch.no_grad():
                _ = pt_wrapper(pt_input_ids, pt_mask, pt_morph)
            _ = sess_fp32.run(None, ort_inputs)
            _ = sess_int8.run(None, ort_inputs)

        # 5. Benchmark PyTorch
        pt_timings = []
        pt_prob = 0.0
        with torch.no_grad():
            for _ in range(iters):
                t0 = time.perf_counter_ns()
                out_prob, _ = pt_wrapper(pt_input_ids, pt_mask, pt_morph)
                t1 = time.perf_counter_ns()
                pt_timings.append((t1 - t0) / 1e6)
            pt_prob = float(out_prob.item())

        # 6. Benchmark ONNX FP32
        fp32_timings = []
        fp32_prob = 0.0
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            out_fp32 = sess_fp32.run(None, ort_inputs)
            t1 = time.perf_counter_ns()
            fp32_timings.append((t1 - t0) / 1e6)
        fp32_prob = float(out_fp32[0][0, 0])

        # 7. Benchmark ONNX INT8
        int8_timings = []
        int8_prob = 0.0
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            out_int8 = sess_int8.run(None, ort_inputs)
            t1 = time.perf_counter_ns()
            int8_timings.append((t1 - t0) / 1e6)
        int8_prob = float(out_int8[0][0, 0])

    mae_prob = abs(pt_prob - int8_prob)
    pt_stats = _compute_percentiles(pt_timings)
    fp32_stats = _compute_percentiles(fp32_timings)
    int8_stats = _compute_percentiles(int8_timings)

    speedup = round(pt_stats["mean_ms"] / max(1e-6, int8_stats["mean_ms"]), 2)

    results = {
        "status": "success",
        "dry_run": dry_run,
        "iterations": iters,
        "seq_len": seq_len,
        "num_threads": num_threads,
        "pytorch_latency_ms": pt_stats,
        "fp32_latency_ms": fp32_stats,
        "int8_latency_ms": int8_stats,
        "speedup_int8_vs_pytorch": speedup,
        "mae_probability_vs_pytorch": round(mae_prob, 6),
        "fp32_size_mb": round(fp32_size_bytes / (1024 * 1024), 2),
        "int8_size_mb": round(int8_size_bytes / (1024 * 1024), 2),
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX CPU Latency and Parity Benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Run quick 2-iteration sanity check")
    parser.add_argument("--output-json", type=str, default="data/onnx_cpu_benchmark_results.json", help="Path to save output JSON")
    parser.add_argument("--num-iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--num-threads", type=int, default=4, help="Number of intra-op CPU threads")
    args = parser.parse_args()

    results = run_benchmark(
        dry_run=args.dry_run,
        output_path=args.output_json,
        num_iterations=args.num_iterations,
        warmup=args.warmup,
        num_threads=args.num_threads,
    )
    print("Benchmark complete:")
    print(f"  PyTorch P50: {results['pytorch_latency_ms']['p50_ms']} ms")
    print(f"  ONNX FP32 P50: {results['fp32_latency_ms']['p50_ms']} ms")
    print(f"  ONNX INT8 P50: {results['int8_latency_ms']['p50_ms']} ms")
    print(f"  MAE probability: {results['mae_probability_vs_pytorch']}")
    print(f"  Saved to: {args.output_json}")


if __name__ == "__main__":
    main()
