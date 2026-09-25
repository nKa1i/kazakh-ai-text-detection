# -*- coding: utf-8 -*-
"""
verification/verifier.py: Unified High-Level Facade for Trustworthy Document Verification.
Coordinates Document AI Detection, Atomic Claim Extraction, Evidence Retrieval,
Dual-Mode NLI Consistency Verification (Morphological vs Baseline), and 2D Continuous Trust Matrix Scoring.
"""

import os
import re
import math
from typing import Optional, Any, List, Dict, Tuple

from verification.evidence import DocumentTrustResult, ClaimVerificationResult, EvidencePassage
from verification.knowledge_store import KnowledgeStore
from verification.retriever import HybridEvidenceRetriever
from verification.claim_extractor import KazakhClaimExtractor
from verification.nli_verifier import NLIClaimVerifier
from verification.trust_scorer import DualRiskTrustScorer
from models.morpho_nli_verifier import MorphologicalAffixExtractor, OfflineHeuristicNLIVerifier


class TrustworthyDocumentVerifier:
    """
    High-level facade orchestrating stylistic AI text detection with
    encyclopedic evidence retrieval and factual verification.
    Supports dual verification modes ('morpho' vs 'baseline') and continuous
    2D Trust Matrix Cartesian coordinates.
    """

    def __init__(
        self,
        ai_detector: Optional[Any] = None,
        knowledge_store: Optional[KnowledgeStore] = None,
        corpus_path: Optional[str] = None,
        alpha: float = 0.50,
        verifier_mode: str = "morpho"
    ):
        self.ai_detector = ai_detector
        self.verifier_mode = (verifier_mode or "morpho").lower().strip()

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
        self.morpho_extractor = MorphologicalAffixExtractor()
        self.morpho_nli = OfflineHeuristicNLIVerifier(extractor=self.morpho_extractor)
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

    def _select_best_evidence_and_predict(
        self,
        claim_text: str,
        evidence: List[EvidencePassage]
    ) -> Tuple[str, Dict[str, Any], List[float]]:
        """
        Selects the most informative evidence passage/sentence and computes
        morphological feature alignment and NLI prediction.
        """
        if not evidence:
            raw_feats = self.morpho_extractor.extract_features(claim_text, "")
            pred = self.morpho_nli.predict_pair(claim_text, "")
            return "", pred, raw_feats

        candidates: List[str] = []
        for p in evidence:
            p_text = (p.text or "").strip()
            if p_text:
                candidates.append(p_text)
                # Split passage into constituent sentences
                sentences = re.split(r'(?<=[.!?])\s+', p_text)
                for s in sentences:
                    s_clean = s.strip()
                    if s_clean and len(s_clean) >= 10 and s_clean not in candidates:
                        candidates.append(s_clean)

        if not candidates:
            first_text = (evidence[0].text or "").strip() if evidence else ""
            candidates = [first_text]

        best_cand = candidates[0]
        best_pred = self.morpho_nli.predict_pair(claim_text, best_cand)
        best_feats = self.morpho_extractor.extract_features(claim_text, best_cand)
        best_score = -1.0

        for cand in candidates:
            p_res = self.morpho_nli.predict_pair(claim_text, cand)
            raw_lbl = str(p_res.get("label", "")).upper()
            conf = float(p_res.get("confidence", 0.0))
            feats = p_res.get("features", [])
            stem_overlap = feats[9] if len(feats) > 9 else 0.0

            # Contradictions are prioritized to penalize factual hallucinations
            if raw_lbl in ("REFUTES", "REFUTED"):
                score = 30.0 + conf * 10.0 + stem_overlap
            elif raw_lbl in ("SUPPORTED", "SUPPORTS"):
                score = 20.0 + conf * 10.0 + stem_overlap
            else:
                score = 10.0 + stem_overlap * 10.0 + conf

            if score > best_score:
                best_score = score
                best_cand = cand
                best_pred = p_res
                best_feats = feats if feats else self.morpho_extractor.extract_features(claim_text, cand)

        return best_cand, best_pred, best_feats

    def _build_morpho_dict(self, raw_feats: List[float]) -> Dict[str, Any]:
        """Converts raw 16-dimensional float vector into labeled diagnostic dictionary."""
        feats = list(raw_feats) + [0.0] * max(0, 16 - len(raw_feats))
        morpho_dict = {
            "claim_negation": float(feats[0]),
            "evidence_negation": float(feats[1]),
            "directional_negation_mismatch": float(feats[2]),
            "calendar_year_conflict": float(feats[3]),
            "number_mismatch": float(feats[4]),
            "has_claim_evidential": bool(feats[5]),
            "has_evidence_evidential": bool(feats[6]),
            "has_modal_necessity": bool(feats[7]),
            "has_modal_possibility": bool(feats[8]),
            "fst_root_jaccard": float(feats[9]),
            "surface_token_overlap": float(feats[10]),
            "lexical_fst_divergence": float(feats[11]),
            "subject_proper_noun_match": float(feats[12]),
            "case_suffix_alignment": float(feats[13]),
            "claim_length_norm": float(feats[14]),
            "evidence_length_norm": float(feats[15]),
            "fst_jaccard": float(feats[9]),
        }
        return morpho_dict

    def verify(self, text: str, verifier_mode: Optional[str] = None) -> DocumentTrustResult:
        """
        Executes end-to-end factual and stylistic verification on a Kazakh document.
        Supports dual modes ('morpho' vs 'baseline') and returns full 2D continuous trust coordinates.
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
                nei_count=0,
                t_fact=0.0,
                t_gen=1.0,
                composite_trust=round(math.sqrt(0.5), 4),
                quadrant_code="Q1"
            )

        mode = (verifier_mode or self.verifier_mode or "morpho").lower().strip()

        # 1. AI Generation Risk & Generative Authenticity Axis T_gen
        ai_risk = self._get_ai_risk(t_clean)
        ai_r_clamped = max(0.0, min(1.0, float(ai_risk)))
        t_gen = max(0.0, min(1.0, 1.0 - ai_r_clamped))

        # 2. Extract Atomic Factual Claims
        atomic_claims = self.extractor.extract_claims(t_clean)

        # 3. Retrieve Evidence and Verify Claims (Dual Modes)
        verification_results: List[ClaimVerificationResult] = []
        for claim in atomic_claims:
            evidence = self.retriever.retrieve(claim.text, top_k=3)

            if mode == "baseline":
                v_res = self.nli.verify_claim(claim, evidence)
                verification_results.append(v_res)
            else:
                best_cand, pred, raw_feats = self._select_best_evidence_and_predict(claim.text, evidence)
                morpho_dict = self._build_morpho_dict(raw_feats)

                raw_label = str(pred.get("label", "NOT_ENOUGH_INFO")).upper().strip()
                if raw_label in ("REFUTES", "REFUTED"):
                    verdict = "REFUTED"
                elif raw_label in ("SUPPORTED", "SUPPORTS"):
                    verdict = "SUPPORTED"
                else:
                    verdict = "NOT ENOUGH INFO"

                conf = float(pred.get("confidence", 0.50))
                explanation = str(pred.get("explanation", ""))

                v_res = ClaimVerificationResult(
                    claim=claim,
                    verdict=verdict,
                    confidence=round(conf, 4),
                    evidence=evidence,
                    explanation=explanation,
                    morphological_features=morpho_dict
                )
                verification_results.append(v_res)

        # 4. Compute Continuous 2D Trust Matrix Metrics & Quadrants
        sup_count = sum(1 for c in verification_results if c.verdict == "SUPPORTED")
        ref_count = sum(1 for c in verification_results if c.verdict == "REFUTED")
        nei_count = sum(1 for c in verification_results if c.verdict == "NOT ENOUGH INFO")

        factual_risk = self.scorer.compute_factual_risk(verification_results)
        fact_r_clamped = max(0.0, min(1.0, float(factual_risk)))
        trust_risk = self.scorer.alpha * ai_r_clamped + (1.0 - self.scorer.alpha) * fact_r_clamped

        # Factual Veracity Score T_fact in [-1.0, 1.0]
        if not verification_results:
            t_fact = 0.0
        else:
            total_veracity = 0.0
            for c in verification_results:
                v = (c.verdict or "").upper().strip()
                if v in ("SUPPORTED", "SUPPORTS"):
                    sign_y = 1.0
                elif v in ("REFUTED", "REFUTES"):
                    sign_y = -1.0
                else:
                    sign_y = 0.0
                total_veracity += sign_y * float(c.confidence)
            t_fact = total_veracity / len(verification_results)

        t_fact = max(-1.0, min(1.0, float(t_fact)))

        # Continuous Composite Trust T(x) = sqrt(0.5 * (max(0, T_fact)^2 + T_gen^2))
        composite_trust = math.sqrt(0.5 * (max(0.0, t_fact) ** 2 + t_gen ** 2))
        composite_trust = max(0.0, min(1.0, float(composite_trust)))

        # Four-Quadrant Classification:
        # Q1: T_fact > 0 and T_gen >= 0.5 ("Verified Human Fact")
        # Q2: T_fact <= 0 and T_gen >= 0.5 ("Human Misinformation")
        # Q3: T_fact > 0 and T_gen < 0.5 ("Accurate AI Synthesis")
        # Q4: T_fact <= 0 and T_gen < 0.5 ("Hallucinatory AI Disinformation")
        if not verification_results:
            if t_gen >= 0.5:
                quadrant_code = "Q1"
                quadrant_verdict = "Verified Human Fact"
            else:
                quadrant_code = "Q3"
                quadrant_verdict = "Accurate AI Synthesis"
        else:
            if t_fact > 0.0:
                if t_gen >= 0.5:
                    quadrant_code = "Q1"
                    quadrant_verdict = "Verified Human Fact"
                else:
                    quadrant_code = "Q3"
                    quadrant_verdict = "Accurate AI Synthesis"
            else:
                if t_gen >= 0.5:
                    quadrant_code = "Q2"
                    quadrant_verdict = "Human Misinformation"
                else:
                    quadrant_code = "Q4"
                    quadrant_verdict = "Hallucinatory AI Disinformation"

        return DocumentTrustResult(
            doc_text=t_clean,
            ai_risk=round(ai_r_clamped, 4),
            factual_risk=round(fact_r_clamped, 4),
            trust_risk=round(trust_risk, 4),
            quadrant_verdict=quadrant_verdict,
            claims=verification_results,
            total_claims=len(verification_results),
            supported_count=sup_count,
            refuted_count=ref_count,
            nei_count=nei_count,
            t_fact=round(t_fact, 4),
            t_gen=round(t_gen, 4),
            composite_trust=round(composite_trust, 4),
            quadrant_code=quadrant_code
        )
