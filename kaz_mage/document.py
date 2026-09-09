from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DocumentChunk:
    """Represents a single window/chunk of a segmented document."""
    index: int
    text: str
    start_char: int
    end_char: int
    word_count: int
    sentence_count: int
    ai_probability: float = 0.0
    gate_value: float = 0.5
    is_ai: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk to a dictionary."""
        return {
            "index": self.index,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "ai_probability": self.ai_probability,
            "gate_value": self.gate_value,
            "is_ai": self.is_ai,
        }


@dataclass
class DocumentAnalysisResult:
    """Aggregated analysis result for an entire document."""
    verdict: str
    document_ai_probability: float
    ai_content_ratio: float
    calibrated_threshold: float = 0.9980
    total_words: int = 0
    total_sentences: int = 0
    total_chunks: int = 0
    worst_chunk: Optional[DocumentChunk] = None
    chunks: List[DocumentChunk] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize document analysis result to a dictionary."""
        worst_chunk_dict = None
        if self.worst_chunk is not None:
            worst_chunk_dict = (
                self.worst_chunk.to_dict()
                if hasattr(self.worst_chunk, "to_dict")
                else self.worst_chunk
            )

        return {
            "verdict": self.verdict,
            "document_ai_probability": self.document_ai_probability,
            "ai_content_ratio": self.ai_content_ratio,
            "calibrated_threshold": self.calibrated_threshold,
            "total_words": self.total_words,
            "total_sentences": self.total_sentences,
            "total_chunks": self.total_chunks,
            "worst_chunk": worst_chunk_dict,
            "chunks": [
                c.to_dict() if hasattr(c, "to_dict") else c
                for c in self.chunks
            ],
        }
