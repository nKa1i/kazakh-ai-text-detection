# -*- coding: utf-8 -*-
"""
verification/nli_verifier.py: 3-Way Natural Language Inference (NLI) Claim Verifier.
Evaluates claim-evidence consistency across SUPPORTED, REFUTED, and NOT ENOUGH INFO,
detecting numerical/temporal conflicts and morphological negation mismatches.
"""

import re
from typing import List, Tuple, Optional, Set

from verification.evidence import AtomicClaim, EvidencePassage, ClaimVerificationResult


NEGATION_PATTERNS = [
    r'\b(?:емес|жоқ|болмаған|болмады|болмайды)\b',
    r'(?:баған|беген|паған|пеген|маған|меген)[.?!]?$',
    r'(?:бады|беді|пады|педі|мады|меді)[.?!]?$',
    r'(?:байды|бейді|пайды|пейді|майды|мейді)[.?!]?$',
    r'(?:бас|бес|пас|пес|мас|мес)[.?!]?$',
]


class NLIClaimVerifier:
    """
    Evaluates evidence against atomic Kazakh claims to assign:
    - SUPPORTED (Factual confirmation)
    - REFUTED (Factual contradiction, temporal mismatch, or negation conflict)
    - NOT ENOUGH INFO (Insufficient or unrelated evidence)
    """

    def __init__(self, min_similarity_threshold: float = 0.20):
        self.min_sim = min_similarity_threshold

    def extract_years(self, text: str) -> List[str]:
        """Extracts 4-digit historical calendar years from text."""
        if not text:
            return []
        return re.findall(r'\b(1\d{3}|20\d{2})\b', text)

    def extract_numbers(self, text: str) -> List[str]:
        """Extracts all digit sequences from text."""
        if not text:
            return []
        return re.findall(r'\b\d+\b', text)

    def has_negation(self, text: str) -> bool:
        """Detects whether a sentence contains Kazakh verbal or copular negation."""
        if not text:
            return False
        t_low = text.lower()
        for pat in NEGATION_PATTERNS:
            if re.search(pat, t_low, flags=re.UNICODE):
                return True
        return False

    def verify_claim(
        self,
        claim: AtomicClaim,
        evidence_passages: List[EvidencePassage]
    ) -> ClaimVerificationResult:
        """
        Verifies a single AtomicClaim against a list of candidate EvidencePassage objects.
        """
        if not evidence_passages or not claim or not claim.text.strip():
            return ClaimVerificationResult(
                claim=claim,
                verdict="NOT ENOUGH INFO",
                confidence=0.50,
                evidence=[],
                explanation="Бұл мәлімдемені растайтын немесе теріске шығаратын дереккөз табылмады."
            )

        # Filter out passages below minimum similarity threshold
        valid_evidence = [p for p in evidence_passages if p.similarity_score >= self.min_sim]
        if not valid_evidence:
            return ClaimVerificationResult(
                claim=claim,
                verdict="NOT ENOUGH INFO",
                confidence=0.50,
                evidence=evidence_passages[:1],
                explanation="Табылған дереккөздердің сәйкестік деңгейі тым төмен."
            )

        top_evidence = valid_evidence[0]
        claim_text = claim.text
        ev_text = top_evidence.text

        # 1. Temporal & Numerical Contradiction Check
        claim_years = self.extract_years(claim_text)
        ev_years = self.extract_years(ev_text)

        if claim_years and ev_years:
            overlap_years = set(claim_years) & set(ev_years)
            if not overlap_years:
                # Factual temporal conflict on the same topic!
                c_str = ", ".join(claim_years)
                e_str = ", ".join(ev_years)
                return ClaimVerificationResult(
                    claim=claim,
                    verdict="REFUTED",
                    confidence=0.90,
                    evidence=valid_evidence,
                    explanation=f"Мәлімдемедегі уақыт ({c_str}) ресми деректегі уақытпен ({e_str}) сәйкес келмейді."
                )

        # 2. Polar Negation Mismatch Check
        c_neg = self.has_negation(claim_text)
        e_neg = self.has_negation(ev_text)

        if c_neg != e_neg:
            # If high topical overlap but opposite polarity: direct contradiction
            if top_evidence.similarity_score >= 0.35 or len(top_evidence.matched_stems) >= 2:
                return ClaimVerificationResult(
                    claim=claim,
                    verdict="REFUTED",
                    confidence=0.88,
                    evidence=valid_evidence,
                    explanation="Мәлімдемедегі терістеу мағынасы ресми дереккөзбен қайшы келеді."
                )

        # 3. Entailment / Supported Check
        # High similarity score or multiple shared factual stems
        if top_evidence.similarity_score >= 0.35 or len(top_evidence.matched_stems) >= 2:
            conf = max(0.70, min(0.98, top_evidence.similarity_score))
            matched_str = ", ".join(top_evidence.matched_stems[:4])
            return ClaimVerificationResult(
                claim=claim,
                verdict="SUPPORTED",
                confidence=round(conf, 4),
                evidence=valid_evidence,
                explanation=f"Дереккөз бойынша ақпарат расталды ({top_evidence.title}): {matched_str} сәйкестігі анықталды."
            )

        # 4. Fallback: Not Enough Info
        return ClaimVerificationResult(
            claim=claim,
            verdict="NOT ENOUGH INFO",
            confidence=0.50,
            evidence=valid_evidence,
            explanation="Дереккөз бойынша ақпарат толық расталмады."
        )
