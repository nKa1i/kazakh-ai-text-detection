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

__all__ = [
    "EvidencePassage",
    "AtomicClaim",
    "ClaimVerificationResult",
    "DocumentTrustResult",
    "KnowledgeStore",
]
