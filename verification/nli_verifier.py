# -*- coding: utf-8 -*-
"""
verification/nli_verifier.py: 3-Way Natural Language Inference (NLI) Claim Verifier.
Evaluates claim-evidence consistency across SUPPORTED, REFUTED, and NOT ENOUGH INFO,
detecting numerical/temporal conflicts and morphological negation mismatches.
"""

import re
from typing import List, Tuple, Optional, Set

from verification.evidence import AtomicClaim, EvidencePassage, ClaimVerificationResult
from verification.retriever import KAZAKH_STOPWORDS
from fst_analyzer import AdvancedKazakhFSTAnalyzer


NEGATION_PATTERNS = [
    r'\b(?:емес|жоқ|болмаған|болмады|болмайды|табылмайды)\b',
    r'(?:баған|беген|паған|пеген|маған|меген)[.?!]?$',
    r'(?:бады|беді|пады|педі|мады|меді)[.?!]?$',
    r'(?:байды|бейді|пайды|пейді|майды|мейді)[.?!]?$',
    r'(?:бас|бес|пас|пес|мас|мес)[.?!]?$',
]

GENERIC_HEAD_NOUNS = {
    "қала", "қаласы", "ауыл", "ауылы", "облыс", "облысы",
    "республика", "республикасы", "ел", "елі", "мемлекет", "мемлекеті",
    "көл", "көлі", "өзен", "өзені", "тау", "тауы"
}


class NLIClaimVerifier:
    """
    Evaluates evidence against atomic Kazakh claims to assign:
    - SUPPORTED (Factual confirmation with high directional predicate coverage)
    - REFUTED (Factual contradiction, temporal mismatch, entity substitution, or negation conflict)
    - NOT ENOUGH INFO (Insufficient coverage or unrelated evidence)
    """

    def __init__(
        self,
        min_similarity_threshold: float = 0.20,
        fst_analyzer: Optional[AdvancedKazakhFSTAnalyzer] = None
    ):
        self.min_sim = min_similarity_threshold
        self.fst = fst_analyzer or AdvancedKazakhFSTAnalyzer()

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

    def extract_content_stems(self, text: str) -> List[str]:
        """Extracts normalized, non-stopword morphological root stems."""
        if not text:
            return []
        tokens = re.findall(r'[\w\d]+(?:-[\w\d]+)*', text.lower(), flags=re.UNICODE)
        stems = []
        for t in tokens:
            if t in KAZAKH_STOPWORDS or len(t) < 2:
                continue
            # In Kazakh, -стан is a toponymic root suffix ending in nasal 'н'.
            # Ablative after 'н' is always -нан/-нен, never -тан/-тен.
            if t.endswith("стан") and len(t) >= 5:
                stems.append(t)
                continue
            # Strip case suffix
            match = self.fst.case_re.search(t)
            if match and len(t[:match.start()]) >= 3:
                base = t[:match.start()]
                stems.append(base)
            else:
                stems.append(t)
        return stems

    def extract_subject_proper_nouns(self, text: str) -> Set[str]:
        """Extracts distinguishing proper nouns and named entities from claim subject."""
        if not text:
            return set()
        
        # Check nominal equative copula ('—' or '–')
        if " — " in text or " – " in text:
            sep = " — " if " — " in text else " – "
            subject_part = text.split(sep)[0].strip()
        else:
            # First 1-2 words if capitalized
            words = text.split()
            cap_words = [w for w in words[:2] if w and w[0].isupper()]
            subject_part = " ".join(cap_words) if cap_words else ""

        if not subject_part:
            return set()

        raw_stems = set(self.extract_content_stems(subject_part))
        # Exclude generic administrative head nouns
        proper_stems = {s for s in raw_stems if s not in GENERIC_HEAD_NOUNS}
        return proper_stems

    def verify_claim(
        self,
        claim: AtomicClaim,
        evidence_passages: List[EvidencePassage]
    ) -> ClaimVerificationResult:
        """
        Verifies a single AtomicClaim against candidate EvidencePassage objects.
        """
        if not evidence_passages or not claim or not claim.text.strip():
            return ClaimVerificationResult(
                claim=claim,
                verdict="NOT ENOUGH INFO",
                confidence=0.50,
                evidence=[],
                explanation="Бұл мәлімдемені растайтын немесе теріске шығаратын дереккөз табылмады."
            )

        # Filter passages below minimum similarity threshold
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
        ev_full = f"{top_evidence.title} {top_evidence.text}"

        # Directional content stem coverage calculation
        claim_content_stems = self.extract_content_stems(claim_text)
        evidence_stems = set(self.extract_content_stems(ev_full))
        matched_stems = set(claim_content_stems) & evidence_stems
        coverage = len(matched_stems) / len(set(claim_content_stems)) if claim_content_stems else 0.0

        # 1. Subject Entity Substitution Mismatch Check
        claim_proper = self.extract_subject_proper_nouns(claim_text)
        title_proper = self.extract_subject_proper_nouns(top_evidence.title)

        if claim_proper and title_proper:
            c_diff = {s for s in claim_proper if s not in evidence_stems and not any(s in e for e in evidence_stems)}
            t_diff = {s for s in title_proper if s not in claim_content_stems and not any(s in c for c in claim_content_stems)}

            if (c_diff or not (claim_proper & title_proper)) and coverage >= 0.35:
                if c_diff or t_diff:
                    c_subj_str = ", ".join(sorted(list(claim_proper)))
                    t_subj_str = ", ".join(sorted(list(title_proper)))
                    return ClaimVerificationResult(
                        claim=claim,
                        verdict="REFUTED",
                        confidence=0.92,
                        evidence=valid_evidence,
                        explanation=f"Субъект сәйкес келмейді: мәлімдемедегі «{c_subj_str}» ресми дереккөздегі «{t_subj_str}» нысанымен ауыстырылған."
                    )

        # 2. Temporal & Numerical Contradiction Check
        claim_years = self.extract_years(claim_text)
        ev_years = self.extract_years(ev_text)

        if claim_years and ev_years:
            overlap_years = set(claim_years) & set(ev_years)
            if not overlap_years:
                # Year mismatch is only a contradiction if the predicate event stems have high coverage
                if coverage >= 0.40:
                    c_str = ", ".join(claim_years)
                    e_str = ", ".join(ev_years)
                    return ClaimVerificationResult(
                        claim=claim,
                        verdict="REFUTED",
                        confidence=0.92,
                        evidence=valid_evidence,
                        explanation=f"Мәлімдемедегі уақыт ({c_str}) ресми деректегі уақытпен ({e_str}) сәйкес келмейді."
                    )

        # 3. Polar Negation Mismatch Check
        c_neg = self.has_negation(claim_text)
        e_neg = self.has_negation(ev_text)

        if c_neg != e_neg:
            # If high topical overlap but opposite polarity: direct contradiction
            if coverage >= 0.35 and top_evidence.similarity_score >= 0.30:
                return ClaimVerificationResult(
                    claim=claim,
                    verdict="REFUTED",
                    confidence=0.90,
                    evidence=valid_evidence,
                    explanation="Мәлімдемедегі терістеу мағынасы ресми дереккөзбен қайшы келеді."
                )

        # 4. Entailment / Supported Check
        # Requires high directional coverage of claim content tokens AND valid similarity score
        if coverage >= 0.55 and top_evidence.similarity_score >= 0.35:
            conf = max(0.70, min(0.98, top_evidence.similarity_score))
            matched_str = ", ".join(sorted(list(matched_stems))[:4])
            return ClaimVerificationResult(
                claim=claim,
                verdict="SUPPORTED",
                confidence=round(conf, 4),
                evidence=valid_evidence,
                explanation=f"Дереккөз бойынша ақпарат расталды ({top_evidence.title}): {matched_str} сәйкестігі анықталды."
            )

        # 5. Fallback: Not Enough Info (Unrelated claims or hallucinated predicates on known entities)
        return ClaimVerificationResult(
            claim=claim,
            verdict="NOT ENOUGH INFO",
            confidence=0.50,
            evidence=valid_evidence,
            explanation="Дереккөз бойынша ақпарат толық расталмады немесе мәлімдеме мазмұны жеткіліксіз."
        )
