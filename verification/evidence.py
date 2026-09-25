# -*- coding: utf-8 -*-
"""
verification/evidence.py: Core Data Structures for Kazakh Factual Evidence Verification.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class EvidencePassage:
    """Represents a factual passage retrieved from the knowledge corpus."""
    passage_id: str
    title: str
    text: str
    source_url: str = ""
    similarity_score: float = 0.0
    matched_stems: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AtomicClaim:
    """Represents an isolated, verifiable proposition extracted from a document."""
    claim_id: str
    text: str
    source_sentence: str = ""
    start_char: int = 0
    end_char: int = 0
    is_verifiable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimVerificationResult:
    """Outcome of NLI fact-checking on a single atomic claim."""
    claim: AtomicClaim
    verdict: str  # "SUPPORTED" | "REFUTED" | "NOT ENOUGH INFO"
    confidence: float = 0.0
    evidence: List[EvidencePassage] = field(default_factory=list)
    explanation: str = ""
    morphological_features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "evidence": [e.to_dict() for e in self.evidence],
            "explanation": self.explanation,
            "morphological_features": self.morphological_features
        }


@dataclass
class DocumentTrustResult:
    """Comprehensive trust assessment uniting AI generation risk and factual risk."""
    doc_text: str
    ai_risk: float  # [0.0, 1.0] from neural detector (Topic 1 & 2)
    factual_risk: float  # [0.0, 1.0] from NLI contradiction penalties
    trust_risk: float  # Fused risk score
    quadrant_verdict: str  # "Verified Human Fact" | "Human Misinformation" | "Accurate AI Synthesis" | "Hallucinatory AI Disinformation"
    claims: List[ClaimVerificationResult] = field(default_factory=list)
    total_claims: int = 0
    supported_count: int = 0
    refuted_count: int = 0
    nei_count: int = 0
    t_fact: float = 0.0
    t_gen: float = 0.0
    composite_trust: float = 0.0
    quadrant_code: str = "Q1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_text": self.doc_text,
            "ai_risk": round(self.ai_risk, 4),
            "factual_risk": round(self.factual_risk, 4),
            "trust_risk": round(self.trust_risk, 4),
            "quadrant_verdict": self.quadrant_verdict,
            "claims": [c.to_dict() for c in self.claims],
            "total_claims": self.total_claims,
            "supported_count": self.supported_count,
            "refuted_count": self.refuted_count,
            "nei_count": self.nei_count,
            "t_fact": round(self.t_fact, 4),
            "t_gen": round(self.t_gen, 4),
            "composite_trust": round(self.composite_trust, 4),
            "quadrant_code": self.quadrant_code
        }
