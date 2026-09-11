# -*- coding: utf-8 -*-
"""
verification/trust_scorer.py: Dual-Risk Trust Scorer & Four-Quadrant Matrix.
Fuses stylistic AI generation probability with factual contradiction penalties.
"""

from typing import List, Optional

from verification.evidence import (
    ClaimVerificationResult,
    DocumentTrustResult,
)


class DualRiskTrustScorer:
    """
    Combines AI Generation Risk and Factual Contradiction Risk:
    Risk_Trust = α · Risk_AI + (1 - α) · Risk_Fact
    and assigns a quadrant in the Four-Quadrant Trust Matrix.
    """

    PENALTIES = {
        "SUPPORTED": 0.0,
        "NOT ENOUGH INFO": 0.25,
        "REFUTED": 1.0,
    }

    def __init__(self, alpha: float = 0.50):
        self.alpha = max(0.0, min(1.0, alpha))

    def compute_factual_risk(self, claims: List[ClaimVerificationResult]) -> float:
        """Computes average factual contradiction penalty across all verified claims."""
        if not claims:
            return 0.0

        total_penalty = 0.0
        for c in claims:
            total_penalty += self.PENALTIES.get(c.verdict, 0.25)

        return total_penalty / len(claims)

    def determine_quadrant(self, ai_risk: float, factual_risk: float) -> str:
        """
        Assigns one of four quadrants:
        - Verified Human Fact (Low AI, Low Fact Risk)
        - Human Misinformation (Low AI, High Fact Risk)
        - Accurate AI Synthesis (High AI, Low Fact Risk)
        - Hallucinatory AI Disinformation (High AI, High Fact Risk)
        """
        is_ai = ai_risk >= 0.50
        is_factual_error = factual_risk >= 0.40

        if not is_ai and not is_factual_error:
            return "Verified Human Fact"
        elif not is_ai and is_factual_error:
            return "Human Misinformation"
        elif is_ai and not is_factual_error:
            return "Accurate AI Synthesis"
        else:
            return "Hallucinatory AI Disinformation"

    def score(
        self,
        doc_text: str,
        ai_risk: float,
        claims: List[ClaimVerificationResult]
    ) -> DocumentTrustResult:
        """Calculates combined trust risk and returns structured DocumentTrustResult."""
        factual_risk = self.compute_factual_risk(claims)
        ai_r_clamped = max(0.0, min(1.0, float(ai_risk)))
        fact_r_clamped = max(0.0, min(1.0, float(factual_risk)))

        trust_risk = self.alpha * ai_r_clamped + (1.0 - self.alpha) * fact_r_clamped
        quadrant = self.determine_quadrant(ai_r_clamped, fact_r_clamped)

        sup_count = sum(1 for c in claims if c.verdict == "SUPPORTED")
        ref_count = sum(1 for c in claims if c.verdict == "REFUTED")
        nei_count = sum(1 for c in claims if c.verdict == "NOT ENOUGH INFO")

        return DocumentTrustResult(
            doc_text=doc_text,
            ai_risk=round(ai_r_clamped, 4),
            factual_risk=round(fact_r_clamped, 4),
            trust_risk=round(trust_risk, 4),
            quadrant_verdict=quadrant,
            claims=claims,
            total_claims=len(claims),
            supported_count=sup_count,
            refuted_count=ref_count,
            nei_count=nei_count
        )
