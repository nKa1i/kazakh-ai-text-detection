# -*- coding: utf-8 -*-
"""
verification/verifier.py: Unified High-Level Facade for Trustworthy Document Verification.
Coordinates Document AI Detection, Atomic Claim Extraction, Evidence Retrieval,
NLI Consistency Verification, and Dual-Risk Trust Scoring.
"""

import os
from typing import Optional, Any, List

from verification.evidence import DocumentTrustResult, ClaimVerificationResult
from verification.knowledge_store import KnowledgeStore
from verification.retriever import HybridEvidenceRetriever
from verification.claim_extractor import KazakhClaimExtractor
from verification.nli_verifier import NLIClaimVerifier
from verification.trust_scorer import DualRiskTrustScorer


class TrustworthyDocumentVerifier:
    """
    High-level facade orchestrating stylistic AI text detection with
    encyclopedic evidence retrieval and factual verification.
    """

    def __init__(
        self,
        ai_detector: Optional[Any] = None,
        knowledge_store: Optional[KnowledgeStore] = None,
        corpus_path: Optional[str] = None,
        alpha: float = 0.50
    ):
        self.ai_detector = ai_detector
        
        # Initialize knowledge store
        if knowledge_store is not None:
            self.store = knowledge_store
        else:
            default_corpus = corpus_path or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "kazakh_knowledge_corpus.jsonl"
            )
            if os.path.exists(default_corpus):
                self.store = KnowledgeStore.from_jsonl(default_corpus)
            else:
                self.store = KnowledgeStore()

        self.retriever = HybridEvidenceRetriever(self.store)
        self.extractor = KazakhClaimExtractor()
        self.nli = NLIClaimVerifier()
        self.scorer = DualRiskTrustScorer(alpha=alpha)

    def _get_ai_risk(self, text: str) -> float:
        """Runs the underlying AI detector or fallback heuristic to compute AI probability."""
        if self.ai_detector is not None:
            try:
                if hasattr(self.ai_detector, "predict_document"):
                    res = self.ai_detector.predict_document(text)
                    return getattr(res, "document_ai_probability", 0.0)
                elif hasattr(self.ai_detector, "predict"):
                    res = self.ai_detector.predict(text)
                    if isinstance(res, dict):
                        return float(res.get("ai_probability", 0.0))
            except Exception:
                pass

        # Fallback to local heuristic detector if present
        try:
            from models.heuristic_detector import OfflineHeuristicDetector
            hd = OfflineHeuristicDetector()
            res = hd.predict(text)
            return float(res.get("ai_probability", 0.0))
        except Exception:
            return 0.0

    def verify(self, text: str) -> DocumentTrustResult:
        """
        Executes end-to-end factual and stylistic verification on a Kazakh document.
        """
        t_clean = (text or "").strip()
        if not t_clean:
            return DocumentTrustResult(
                doc_text=text,
                ai_risk=0.0,
                factual_risk=0.0,
                trust_risk=0.0,
                quadrant_verdict="Verified Human Fact",
                claims=[],
                total_claims=0,
                supported_count=0,
                refuted_count=0,
                nei_count=0
            )

        # 1. AI Generation Risk
        ai_risk = self._get_ai_risk(t_clean)

        # 2. Extract Atomic Factual Claims
        atomic_claims = self.extractor.extract_claims(t_clean)

        # 3. Retrieve Evidence and Verify Claims
        verification_results: List[ClaimVerificationResult] = []
        for claim in atomic_claims:
            evidence = self.retriever.retrieve(claim.text, top_k=3)
            v_res = self.nli.verify_claim(claim, evidence)
            verification_results.append(v_res)

        # 4. Dual-Risk Trust Scoring
        return self.scorer.score(
            doc_text=t_clean,
            ai_risk=ai_risk,
            claims=verification_results
        )
