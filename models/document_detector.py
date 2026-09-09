"""
Kazakh Document Detector Facade.

Orchestrates multi-paragraph sliding window segmentation, batched dual-stream
morphological and semantic inference with defensive micro-batching, and
Top-K worst-chunk risk score aggregation.
"""

import math
from contextlib import nullcontext
from typing import Any, List, Optional, Union

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    HAS_TORCH = False

try:
    from kaz_mage.document import DocumentChunk, DocumentAnalysisResult
    from kaz_mage.chunker import SentencePreservingChunker
    from kaz_mage.aggregator import DocumentAggregator
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from kaz_mage.document import DocumentChunk, DocumentAnalysisResult
    from kaz_mage.chunker import SentencePreservingChunker
    from kaz_mage.aggregator import DocumentAggregator


class DocumentDetector:
    """
    Production-grade document-level AI detector facade for Kazakh texts.

    Coordinates:
    1. Sentence-preserving window chunking via SentencePreservingChunker.
    2. Batched dual-stream inference with defensive micro-batching.
    3. Top-K worst-chunk risk score aggregation and 3-tier verdict classification.
    """

    def __init__(
        self,
        model: Any,
        raw_tokenizer: Optional[Any] = None,
        morpheme_tokenizer: Optional[Any] = None,
        calibrated_threshold: float = 0.9980,
        device: Optional[str] = None,
        max_words: int = 200,
        overlap_sentences: int = 1,
    ) -> None:
        """
        Initialize the DocumentDetector.

        Args:
            model: PyTorch model or callable detector (e.g. MorphoContrastiveDetector).
            raw_tokenizer: Contextual text tokenizer (HuggingFace AutoTokenizer or None).
            morpheme_tokenizer: FST morpheme tokenizer (MorphemeTokenizer or None).
            calibrated_threshold: Decision threshold for chunk-level classification (default: 0.9980).
            device: Computing device ('cpu', 'cuda', or None for auto-detection).
            max_words: Target word budget per sliding window chunk.
            overlap_sentences: Number of overlapping boundary sentences between chunks.
        """
        self.raw_tokenizer = raw_tokenizer
        self.morpheme_tokenizer = morpheme_tokenizer
        self.calibrated_threshold = float(calibrated_threshold)
        self.max_words = int(max_words)
        self.overlap_sentences = int(overlap_sentences)
        self.chunker = SentencePreservingChunker(
            max_words=self.max_words,
            overlap_sentences=self.overlap_sentences,
        )

        # Resolve device
        if device is not None:
            self.device = str(device)
        elif hasattr(model, "device"):
            self.device = str(getattr(model, "device"))
        elif HAS_TORCH and torch is not None and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = model
        if hasattr(self.model, "to") and callable(self.model.to):
            try:
                res = self.model.to(self.device)
                if res is not None:
                    self.model = res
            except Exception:
                pass

        if hasattr(self.model, "eval") and callable(self.model.eval):
            try:
                self.model.eval()
            except Exception:
                pass

    def predict_document(
        self,
        text: str,
        top_k: int = 2,
        batch_size: int = 16,
    ) -> DocumentAnalysisResult:
        """
        Perform document-level inference across sliding window chunks.

        Args:
            text: Arbitrary length input document in Kazakh.
            top_k: Top-K worst chunk count configuration for pooling.
            batch_size: Maximum micro-batch size for model forward passes.

        Returns:
            DocumentAnalysisResult containing verdict, probabilities, and chunks breakdown.
        """
        aggregator = DocumentAggregator(
            calibrated_threshold=self.calibrated_threshold,
            top_k_cfg=top_k,
        )

        if not text or not text.strip():
            return aggregator.aggregate([], total_words=0, total_sentences=0)

        # Segment sentences & compute global totals
        sentences = self.chunker.split_sentences(text)
        total_sentences = len(sentences)
        total_words = len(text.split())

        chunks = self.chunker.chunk_document(text)
        if not chunks:
            return aggregator.aggregate([], total_words=total_words, total_sentences=total_sentences)

        effective_batch_size = max(1, int(batch_size))

        # Ensure model is in eval mode
        if hasattr(self.model, "eval") and callable(self.model.eval):
            try:
                self.model.eval()
            except Exception:
                pass

        no_grad_ctx = torch.no_grad() if (HAS_TORCH and torch is not None) else nullcontext()

        with no_grad_ctx:
            # Process in micro-batches
            for start_idx in range(0, len(chunks), effective_batch_size):
                batch_chunks = chunks[start_idx : start_idx + effective_batch_size]
                batch_texts = [c.text for c in batch_chunks]
                b = len(batch_texts)

                # 1. Prepare raw token ids & attention mask
                input_ids = None
                attention_mask = None

                if self.raw_tokenizer is not None:
                    try:
                        tokenized = self.raw_tokenizer(
                            batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors="pt" if (HAS_TORCH and torch is not None) else None,
                        )
                        if isinstance(tokenized, dict) or hasattr(tokenized, "get"):
                            input_ids = tokenized.get("input_ids")
                            attention_mask = tokenized.get("attention_mask")
                        elif hasattr(tokenized, "input_ids"):
                            input_ids = tokenized.input_ids
                            attention_mask = getattr(tokenized, "attention_mask", None)
                        else:
                            input_ids = tokenized
                    except Exception:
                        tokenized = self.raw_tokenizer(batch_texts)
                        if isinstance(tokenized, dict) or hasattr(tokenized, "get"):
                            input_ids = tokenized.get("input_ids")
                            attention_mask = tokenized.get("attention_mask")
                        elif hasattr(tokenized, "input_ids"):
                            input_ids = tokenized.input_ids
                            attention_mask = getattr(tokenized, "attention_mask", None)
                        else:
                            input_ids = tokenized

                if input_ids is None:
                    if HAS_TORCH and torch is not None:
                        input_ids = torch.ones((b, 16), dtype=torch.long)
                        attention_mask = torch.ones((b, 16), dtype=torch.long)
                    else:
                        input_ids = [[1] * 16 for _ in range(b)]
                        attention_mask = [[1] * 16 for _ in range(b)]

                # Convert to tensors and place on device if torch is available
                if HAS_TORCH and torch is not None:
                    if not isinstance(input_ids, torch.Tensor):
                        try:
                            input_ids = torch.tensor(input_ids, dtype=torch.long)
                        except Exception:
                            pass
                    if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
                        try:
                            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
                        except Exception:
                            pass
                    if attention_mask is None and isinstance(input_ids, torch.Tensor):
                        attention_mask = (input_ids != 0).long()

                    if self.device is not None:
                        if hasattr(input_ids, "to"):
                            input_ids = input_ids.to(self.device)
                        if hasattr(attention_mask, "to") and attention_mask is not None:
                            attention_mask = attention_mask.to(self.device)

                # 2. Prepare morpheme token ids
                morpheme_ids = None
                if self.morpheme_tokenizer is not None:
                    if hasattr(self.morpheme_tokenizer, "batch_encode"):
                        morpheme_ids = self.morpheme_tokenizer.batch_encode(batch_texts)
                    elif callable(self.morpheme_tokenizer):
                        morpheme_ids = self.morpheme_tokenizer(batch_texts)

                    if HAS_TORCH and torch is not None and not isinstance(morpheme_ids, torch.Tensor):
                        try:
                            morpheme_ids = torch.tensor(morpheme_ids, dtype=torch.long)
                        except Exception:
                            pass
                    if self.device is not None and hasattr(morpheme_ids, "to"):
                        morpheme_ids = morpheme_ids.to(self.device)
                elif hasattr(self.model, "morph_encoder"):
                    # Defensive fallback for dual-stream models expecting morpheme_ids
                    if HAS_TORCH and torch is not None:
                        seq_l = input_ids.shape[1] if hasattr(input_ids, "shape") and len(input_ids.shape) > 1 else 16
                        morpheme_ids = torch.zeros(
                            (b, seq_l),
                            dtype=torch.long,
                            device=self.device if self.device else None,
                        )
                    else:
                        morpheme_ids = [[0] * 16 for _ in range(b)]

                # 3. Model forward pass
                out = None
                try:
                    if morpheme_ids is not None:
                        out = self.model(input_ids, attention_mask=attention_mask, morpheme_ids=morpheme_ids)
                    else:
                        out = self.model(input_ids, attention_mask=attention_mask)
                except TypeError:
                    try:
                        if morpheme_ids is not None:
                            out = self.model(input_ids=input_ids, attention_mask=attention_mask, morpheme_ids=morpheme_ids)
                        else:
                            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    except TypeError:
                        try:
                            out = self.model(input_ids, attention_mask, morpheme_ids)
                        except TypeError:
                            out = self.model(input_ids, attention_mask)

                # 4. Extract logits & gate
                if isinstance(out, dict):
                    logits = out.get("logits")
                    gate = out.get("gate")
                elif hasattr(out, "logits"):
                    logits = out.logits
                    gate = getattr(out, "gate", None)
                elif isinstance(out, (tuple, list)):
                    logits = out[0]
                    gate = out[1] if len(out) > 1 else None
                else:
                    logits = out
                    gate = None

                # 5. Compute ai_probability via softmax
                if HAS_TORCH and torch is not None and isinstance(logits, torch.Tensor):
                    probs = torch.softmax(logits, dim=-1)
                    if probs.shape[-1] >= 2:
                        ai_probs = probs[:, 1].detach().cpu().tolist()
                    else:
                        ai_probs = torch.sigmoid(logits).squeeze(-1).detach().cpu().tolist()
                    if isinstance(ai_probs, (float, int)):
                        ai_probs = [float(ai_probs)]
                else:
                    if hasattr(logits, "tolist"):
                        logits_list = logits.tolist()
                    elif isinstance(logits, (list, tuple)):
                        logits_list = list(logits)
                    else:
                        logits_list = [[0.0, 0.0] for _ in range(b)]

                    ai_probs = []
                    for row in logits_list:
                        if isinstance(row, (int, float)):
                            p = 1.0 / (1.0 + math.exp(-float(row)))
                            ai_probs.append(p)
                        elif isinstance(row, (list, tuple)) and len(row) >= 2:
                            m = max(float(row[0]), float(row[1]))
                            e0 = math.exp(float(row[0]) - m)
                            e1 = math.exp(float(row[1]) - m)
                            p = e1 / (e0 + e1) if (e0 + e1) > 0 else 0.5
                            ai_probs.append(p)
                        else:
                            ai_probs.append(0.0)

                # 6. Assign ai_probability and gate_value to chunks
                for j, chunk in enumerate(batch_chunks):
                    p_ai = float(ai_probs[j]) if j < len(ai_probs) else 0.0

                    g_val = 0.5
                    if gate is not None:
                        try:
                            if HAS_TORCH and torch is not None and isinstance(gate, torch.Tensor):
                                if j < gate.shape[0]:
                                    chunk_gate = gate[j]
                                    if chunk_gate.numel() > 1:
                                        g_val = float(chunk_gate.float().mean().item())
                                    else:
                                        g_val = float(chunk_gate.float().item())
                            elif isinstance(gate, (list, tuple)) and j < len(gate):
                                gj = gate[j]
                                if isinstance(gj, (list, tuple)) and len(gj) > 0:
                                    g_val = float(sum(gj) / len(gj))
                                elif isinstance(gj, (int, float)):
                                    g_val = float(gj)
                            elif isinstance(gate, (int, float)):
                                g_val = float(gate)
                        except Exception:
                            g_val = 0.5

                    chunk.ai_probability = p_ai
                    chunk.gate_value = g_val
                    chunk.is_ai = bool(p_ai >= self.calibrated_threshold)

        return aggregator.aggregate(
            chunks=chunks,
            total_words=total_words,
            total_sentences=total_sentences,
        )
