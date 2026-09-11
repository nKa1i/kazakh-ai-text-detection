# -*- coding: utf-8 -*-
"""
verification/claim_extractor.py: Kazakh Atomic Claim Extractor.
Decomposes long Kazakh text into atomic, checkable factual claims,
stripping epistemic hedges and splitting coordinate clauses.
"""

import re
from typing import List, Optional

from kaz_mage.chunker import SentencePreservingChunker
from fst_analyzer import AdvancedKazakhFSTAnalyzer
from verification.evidence import AtomicClaim


EPISTEMIC_HEDGES = [
    r'^меніңше[,\s]*',
    r'^менің ойымша[,\s]*',
    r'^өкінішке орай[,\s]*',
    r'^айта кету керек[,\s]*',
    r'^байқағанымдай[,\s]*',
    r'^шындығында[,\s]*',
    r'^шынында[,\s]*',
    r'^әрине[,\s]*',
    r'^сөзсіз[,\s]*',
    r'^білуімше[,\s]*',
    r'^анығында[,\s]*',
    r'^жалпы алғанда[,\s]*',
]

SUBJECTIVE_SENTIMENT_PATTERNS = [
    r'\bұна(?:ды|мады|йды|майды)\b',
    r'\bкөңілімнен\s+шық(?:ты|пады)\b',
    r'\bөте\s+керемет\b',
    r'\bкеремет\s+туынды\b',
    r'\bкүшті\s+екен\b',
    r'\bжаман\s+емес\b',
    r'\bұсынбаймын\b',
    r'\bсапасы\s+(?:керемет|нашар|жақсы)\b',
]


class KazakhClaimExtractor:
    """
    Extracts atomic, verifiable propositions from Kazakh prose while
    stripping subjective discourse hedges and splitting coordinate clauses.
    """

    def __init__(
        self,
        chunker: Optional[SentencePreservingChunker] = None,
        fst_analyzer: Optional[AdvancedKazakhFSTAnalyzer] = None
    ):
        self.chunker = chunker or SentencePreservingChunker()
        self.fst = fst_analyzer or AdvancedKazakhFSTAnalyzer()

    def strip_hedges(self, sentence: str) -> str:
        """Removes introductory subjective and epistemic hedges and restores capitalization."""
        s = (sentence or "").strip()
        for pat in EPISTEMIC_HEDGES:
            match = re.search(pat, s, flags=re.IGNORECASE | re.UNICODE)
            if match:
                s = s[match.end():].strip()
                break
        if s:
            s = s[0].upper() + s[1:]
        return s

    def is_subjective_opinion(self, text: str) -> bool:
        """Determines whether a sentence expresses pure personal sentiment without factual grounding."""
        t_low = text.lower()
        # If it has a specific 4-digit year, it often contains a factual claim
        has_year = bool(re.search(r'\b\d{4}\b', text))
        if has_year:
            return False

        for pat in SUBJECTIVE_SENTIMENT_PATTERNS:
            if re.search(pat, t_low, flags=re.IGNORECASE | re.UNICODE):
                return True
        return False

    def is_verifiable_proposition(self, clause: str) -> bool:
        """
        Validates whether a proposition contains an encyclopedic checkable structure:
        nominal copula ('—', 'болып табылады'), factual verb, or specific date/entity.
        """
        c = clause.strip()
        words = c.split()
        if len(words) < 3:
            return False

        # 1. Contains a date, year, or specific number
        if re.search(r'\b\d+(?:-[а-яА-ЯәіңғүұқөһӘІҢҒҮҰҚӨҺ]+)?\b', c):
            return True

        # 2. Contains nominal equative dash copula
        if " — " in c or " – " in c:
            return True

        # 3. Contains copula phrase or copular auxiliary
        c_low = c.lower()
        if any(cop in c_low for cop in ["болып табылады", "екен", "болды", "болған", "деп аталады"]):
            return True

        # 4. Contains known factual predicate suffixes
        if re.search(r'(?:ылды|ілді|улды|үлді|алды|елді|ылған|ілген|ған|ген|қан|кен)[.?!]?$', c_low):
            return True

        return True

    def split_compound_clauses(self, sentence: str) -> List[str]:
        """
        Splits coordinate clauses joined by 'және', 'әрі', 'бірақ', 'ал'
        into independent atomic propositions, propagating head subjects when ellipted.
        """
        s = sentence.strip()
        # Extract potential head subject before dash copula or first capitalized words
        head_subject = ""
        copula_symbol = " — "
        if " — " in s:
            head_subject = s.split(" — ")[0].strip()
            copula_symbol = " — "
        elif " – " in s:
            head_subject = s.split(" – ")[0].strip()
            copula_symbol = " – "
        elif " - " in s:
            head_subject = s.split(" - ")[0].strip()
            copula_symbol = " - "

        # Look for coordinating conjunctions with surrounding whitespace
        conj_pattern = r'\s+(?:және|әрі|бірақ|ал|дегенмен)\s+'
        parts = re.split(conj_pattern, s, flags=re.UNICODE)
        if len(parts) <= 1:
            return [s]

        clauses = []
        for i, part in enumerate(parts):
            p_clean = part.strip()
            if not p_clean:
                continue

            # If this is a subsequent coordinate clause and it lacks the head subject
            if i > 0 and head_subject and head_subject not in p_clean:
                # Prepend the head subject with the copula
                p_clean = f"{head_subject}{copula_symbol}{p_clean[0].lower() + p_clean[1:]}"
            else:
                p_clean = p_clean[0].upper() + p_clean[1:]

            if not p_clean.endswith(('.', '!', '?')):
                p_clean += '.'
            if len(p_clean.split()) >= 3:
                clauses.append(p_clean)

        return clauses if len(clauses) > 1 else [s]

    def extract_claims(self, text: str) -> List[AtomicClaim]:
        """
        Decomposes input Kazakh text into a list of AtomicClaim objects.
        """
        if not text or not text.strip():
            return []

        raw_sentences = self.chunker.split_sentences(text)
        claims: List[AtomicClaim] = []
        claim_counter = 1

        for item in raw_sentences:
            if isinstance(item, tuple):
                sent_text, s_start, s_end = item
            else:
                sent_text = str(item)
                s_start = 0
                s_end = len(sent_text)

            s_clean = sent_text.strip()
            if not s_clean or len(s_clean.split()) < 2:
                continue

            # Check subjective opinion filter
            if self.is_subjective_opinion(s_clean):
                continue

            # Strip hedges
            dehedged = self.strip_hedges(s_clean)
            if not dehedged or len(dehedged.split()) < 2:
                continue

            # Split compound clauses
            sub_clauses = self.split_compound_clauses(dehedged)

            for clause in sub_clauses:
                verifiable = self.is_verifiable_proposition(clause)
                if not verifiable:
                    continue

                # Locate character offset in original text if possible
                start_char = text.find(clause[:20]) if len(clause) >= 20 else text.find(clause)
                start_char = max(0, start_char)
                end_char = start_char + len(clause)

                claims.append(
                    AtomicClaim(
                        claim_id=f"claim_{claim_counter:03d}",
                        text=clause,
                        source_sentence=s_clean,
                        start_char=start_char,
                        end_char=end_char,
                        is_verifiable=True
                    )
                )
                claim_counter += 1

        return claims
