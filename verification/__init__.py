# -*- coding: utf-8 -*-
"""
verification: Evidence-Grounded Factual Verification Engine for Kazakh.
"""

from verification.evidence import (
    EvidencePassage,
    AtomicClaim,
    ClaimVerificationResult,
    DocumentTrustResult,
)
from verification.knowledge_store import KnowledgeStore
from verification.retriever import HybridEvidenceRetriever
from verification.claim_extractor import KazakhClaimExtractor
from verification.nli_verifier import NLIClaimVerifier
from verification.trust_scorer import DualRiskTrustScorer
from verification.verifier import TrustworthyDocumentVerifier

__all__ = [
    "EvidencePassage",
    "AtomicClaim",
    "ClaimVerificationResult",
    "DocumentTrustResult",
    "KnowledgeStore",
    "HybridEvidenceRetriever",
    "KazakhClaimExtractor",
    "NLIClaimVerifier",
    "DualRiskTrustScorer",
    "TrustworthyDocumentVerifier",
]
