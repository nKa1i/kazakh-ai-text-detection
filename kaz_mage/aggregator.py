import math
from typing import List, Optional
from .document import DocumentChunk, DocumentAnalysisResult


class DocumentAggregator:
    """Aggregates chunk-level predictions into a document-level verdict.

    Uses Top-K worst-chunk pooling and volume-weighted AI content ratio
    to provide robust three-tier classification:
    - "Authentic Human": document peak risk score is below the calibrated threshold.
    - "Partially AI / Hybrid": peak risk score exceeds threshold and AI content ratio < 70%.
    - "Machine-Generated": peak risk score exceeds threshold and AI content ratio >= 70%.
    """

    def __init__(self, calibrated_threshold: float = 0.9980, top_k_cfg: int = 2) -> None:
        self.calibrated_threshold = float(calibrated_threshold)
        self.top_k_cfg = int(top_k_cfg)

    def aggregate(
        self,
        chunks: List[DocumentChunk],
        total_words: int = 0,
        total_sentences: int = 0,
    ) -> DocumentAnalysisResult:
        """Aggregate a sequence of scored document chunks.

        Args:
            chunks: List of DocumentChunk instances with ai_probability populated.
            total_words: Total words in the source document (falls back to chunk sum).
            total_sentences: Total sentences in the source document (falls back to chunk sum).

        Returns:
            DocumentAnalysisResult containing verdict, probabilities, and breakdown.
        """
        if not chunks:
            return DocumentAnalysisResult(
                verdict="Authentic Human",
                document_ai_probability=0.0,
                ai_content_ratio=0.0,
                calibrated_threshold=self.calibrated_threshold,
                total_words=total_words,
                total_sentences=total_sentences,
                total_chunks=0,
                worst_chunk=None,
                chunks=[],
            )

        # Update is_ai flag for each chunk based on calibrated threshold
        for c in chunks:
            c.is_ai = bool(c.ai_probability >= self.calibrated_threshold)

        worst_chunk = max(chunks, key=lambda c: c.ai_probability)
        sorted_probs = sorted([c.ai_probability for c in chunks], reverse=True)

        # Dynamic Top-K pooling: K = max(1, min(k_cfg, ceil(0.25 * M)))
        k_val = max(1, min(self.top_k_cfg, math.ceil(0.25 * len(chunks))))

        # If there are fewer detected AI chunks than K (e.g., an isolated AI insertion),
        # bound K by the number of AI chunks so that a real AI injection is not diluted by human chunks.
        ai_chunks_count = sum(1 for c in chunks if c.ai_probability >= self.calibrated_threshold)
        if 0 < ai_chunks_count < k_val:
            k_val = ai_chunks_count

        p_worst = sum(sorted_probs[:k_val]) / float(k_val)

        # Volume-weighted AI content ratio
        total_chunk_words = sum(c.word_count for c in chunks)
        if total_chunk_words > 0:
            ai_chunk_words = sum(
                c.word_count for c in chunks if c.ai_probability >= self.calibrated_threshold
            )
            ai_ratio = float(ai_chunk_words) / float(total_chunk_words)
        else:
            ai_ratio = 0.0

        # Three-tier verdict classification
        if p_worst < self.calibrated_threshold:
            verdict = "Authentic Human"
        elif ai_ratio < 0.70:
            verdict = "Partially AI / Hybrid"
        else:
            verdict = "Machine-Generated"

        resolved_total_words = total_words if total_words > 0 else total_chunk_words
        resolved_total_sentences = (
            total_sentences if total_sentences > 0 else sum(c.sentence_count for c in chunks)
        )

        return DocumentAnalysisResult(
            verdict=verdict,
            document_ai_probability=float(p_worst),
            ai_content_ratio=float(ai_ratio),
            calibrated_threshold=self.calibrated_threshold,
            total_words=resolved_total_words,
            total_sentences=resolved_total_sentences,
            total_chunks=len(chunks),
            worst_chunk=worst_chunk,
            chunks=chunks,
        )
