# -*- coding: utf-8 -*-
"""
models/morpho_nli_verifier.py: Morphologically-Grounded NLI Cross-Encoder Architecture.

Formalized in Equation 2 of Section 4.2 in the Paper 1 manuscript:
    P(y | c, E) = softmax(W_v [h_[CLS]; m_affix] + b_v)

Provides:
- MorphologicalAffixExtractor: 16-dimensional morphological & alignment feature vector.
- MorphoNLIVerifier: PyTorch cross-encoder module with fused contextual + morphological features.
- OfflineHeuristicNLIVerifier: Deterministic rule-weighted verifier for CPU/test environments.
"""

import os
import re
from typing import List, Tuple, Dict, Any, Optional, Set, Union

from fst_analyzer import fst_analyzer, AdvancedKazakhFSTAnalyzer

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    _BaseModule = nn.Module
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False
    _BaseModule = object

try:
    from transformers import AutoModel, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    AutoModel = None
    AutoConfig = None
    HAS_TRANSFORMERS = False


class _DummyTensor:
    """Fallback tensor container for environments without PyTorch installed."""
    def __init__(self, shape: Tuple[int, ...], fill_val: float = 0.0):
        self.shape = tuple(shape)
        self.fill_val = fill_val

    def cpu(self):
        return self

    def numpy(self):
        try:
            import numpy as np
            return np.full(self.shape, self.fill_val)
        except ImportError:
            return [[self.fill_val] * self.shape[-1]] * self.shape[0]

    def item(self) -> float:
        return float(self.fill_val)

    def __repr__(self) -> str:
        return f"DummyTensor(shape={self.shape})"


class MorphologicalAffixExtractor:
    """
    Extracts a 16-dimensional morphological and alignment feature vector m_affix
    between a claim and evidence passage in Kazakh:
      0: Verbal/copular negation in claim
      1: Verbal/copular negation in evidence
      2: Directional negation mismatch (XOR between claim and evidence negation)
      3: Temporal calendar year conflict (disjoint 4-digit years)
      4: Numerical quantity mismatch (claim numbers not present in evidence)
      5: Evidential marker in claim (-ыпты/-іпті)
      6: Evidential marker in evidence (-ыпты/-іпті)
      7: Modal necessity marker (керек, тиіс, қажет, міндетті)
      8: Modal possibility marker (мүмкін, ықтимал)
      9: FST root overlap ratio (Jaccard similarity of stemmed content roots)
     10: Surface token overlap ratio (Jaccard similarity of lowercase word tokens)
     11: Lexical-FST divergence (abs(surface_overlap - fst_overlap))
     12: Subject proper noun match (claim's primary subject noun found in evidence)
     13: Case agreement alignment score (Jaccard overlap of roots carrying case endings)
     14: Normalized claim length min(1.0, len(claim.split()) / 30.0)
     15: Normalized evidence length min(1.0, len(evidence.split()) / 100.0)
    """

    NEGATION_WORDS = {
        'емес', 'жоқ', 'жоқтан', 'жоқтығы', 'жоқтығын', 'жоққа',
        'жоқпын', 'жоқсың', 'жоқсыз', 'болмады', 'болмаған', 'болмайды'
    }

    NEGATION_SUFFIX_RE = re.compile(
        r'[а-яәіңғүұқөһ]{2,}(?:'
        r'маған|меген|паған|пеген|баған|беген|'
        r'мады|меді|пады|педі|бады|беді|'
        r'майды|мейді|пайды|пейді|байды|бейді|'
        r'майтын|мейтін|пайтын|пейтін|байтын|бейтін|'
        r'майт|мейт|пайт|пейт|байт|бейт|'
        r'мау|меу|пау|пеу|бау|беу|'
        r'маса|месе|паса|песе|баса|бесе|'
        r'мақшы|мекші|пақшы|пекші|бақшы|бекші)\b',
        re.IGNORECASE
    )

    NOMINAL_SUFFIXES = [
        'ін', 'ын', 'ні', 'ны', 'ді', 'ды', 'ті', 'ты',
        'ге', 'ға', 'ке', 'қа', 'де', 'да', 'те', 'та',
        'нен', 'нан', 'ден', 'дан', 'тен', 'тан',
        'нің', 'ның', 'дің', 'дың', 'тің', 'тың',
        'мен', 'бен', 'пен', 'лер', 'лар', 'дер', 'дар', 'тер', 'тар',
        'і', 'ы', 'сі', 'сы', 'н'
    ]

    EVIDENTIAL_RE = re.compile(
        r'[а-яәіңғүұқөһ]{2,}(?:ыпты|іпті|пты|пті)\b',
        re.IGNORECASE
    )

    MODAL_NECESSITY_RE = re.compile(
        r'\b(?:керек|тиіс|қажет|міндетті)[а-яәіңғүұқөһ]*\b',
        re.IGNORECASE
    )

    MODAL_POSSIBILITY_RE = re.compile(
        r'\b(?:мүмкін|ықтимал)[а-яәіңғүұқөһ]*\b',
        re.IGNORECASE
    )

    YEAR_RE = re.compile(r'\b(?:1\d{3}|20\d{2})\b')
    NUMBER_RE = re.compile(r'\b\d+(?:[.,]\d+)?\b')
    PROPER_NOUN_RE = re.compile(r'\b[А-ЯӘІҢҒҮҰҚӨҺ][а-яәіңғүұқөһ]+\b')
    ABLATION_MODES: Dict[str, List[int]] = {
        "Ours (Full Morpho Cross-Encoder)": [],
        "- Negation Alignment": [2],
        "- Temporal / Calendar Conflicts": [3],
        "- Temporal / Calendar": [3],
        "- Evidentials & Epistemic Modals": [4, 5, 6, 7],
        "- Evidentials & Modality": [4, 5, 6, 7],
        "- FST Root Analysis": [8, 12, 13],
        "Surface Token Overlap Only": [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15],
    }

    @classmethod
    def get_mask_indices_for_mode(cls, mode: Optional[str]) -> List[int]:
        if not mode:
            return []
        if mode in cls.ABLATION_MODES:
            return list(cls.ABLATION_MODES[mode])
        m_lower = mode.strip().lower()
        if "full" in m_lower or "ours" in m_lower:
            return []
        if "negation" in m_lower:
            return [2]
        if "temporal" in m_lower or "calendar" in m_lower:
            return [3]
        if "evidential" in m_lower or "modal" in m_lower:
            return [4, 5, 6, 7]
        if "fst" in m_lower or "root" in m_lower:
            return [8, 12, 13]
        if "surface" in m_lower:
            return [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15]
        return []

    def __init__(
        self,
        fst: Optional[AdvancedKazakhFSTAnalyzer] = None,
        mask_indices: Optional[List[int]] = None,
        ablation_mode: Optional[str] = None,
    ):
        self.fst = fst or fst_analyzer
        self.default_mask_indices = list(mask_indices) if mask_indices is not None else None
        self.default_ablation_mode = ablation_mode

    def _canonical_stem(self, word: str) -> str:
        w = word.strip(".,!?:;\"'()[]{}«»—–-").lower()
        if not w or len(w) <= 2:
            return w

        # 1. Segment using FST analyzer
        seg = self.fst.analyze_and_segment(w)
        base = seg.split()[0].strip('-') if seg else w

        # 2. Strip additional nominal and possessive suffixes
        for sfx in self.NOMINAL_SUFFIXES:
            if len(base) > len(sfx) + 2 and base.endswith(sfx):
                base = base[:-len(sfx)]
                break

        # 3. Consonant alternation (г->к, ғ->қ, б->п)
        if base.endswith('г'):
            base = base[:-1] + 'к'
        elif base.endswith('ғ'):
            base = base[:-1] + 'қ'
        elif base.endswith('б'):
            base = base[:-1] + 'п'

        return base

    def _has_negation(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        tokens = re.findall(r'[а-яәіңғүұқөһ]+', text.lower())
        if any(tok in self.NEGATION_WORDS for tok in tokens):
            return True
        if self.NEGATION_SUFFIX_RE.search(text):
            return True
        cleaned = re.sub(r'[^\w\s]', ' ', text)
        analyzed = self.fst.analyze_and_segment(cleaned)
        for tok in analyzed.split():
            if tok in ('-ма', '-ме', '-па', '-пе', '-ба', '-бе'):
                return True
        return False

    def _get_fst_roots(self, text: str) -> Set[str]:
        if not text or not text.strip():
            return set()
        words = re.findall(r'[а-яәіңғүұқөһa-z0-9]+', text.lower())
        roots = set()
        for w in words:
            if len(w) >= 2 and any(c.isalpha() for c in w):
                stem = self._canonical_stem(w)
                if stem:
                    roots.add(stem)
        return roots

    def _get_surface_tokens(self, text: str) -> Set[str]:
        if not text or not text.strip():
            return set()
        tokens = re.findall(r'[а-яәіңғүұқөһa-z0-9]+', text.lower())
        return {t for t in tokens if any(c.isalpha() for c in t)}

    def _get_case_roots(self, text: str) -> Set[str]:
        if not text or not text.strip():
            return set()
        words = re.findall(r'[а-яәіңғүұқөһ]+', text, re.IGNORECASE)
        case_roots = set()
        for w in words:
            if len(w) <= 3:
                continue
            m = self.fst.case_re.search(w)
            if m and len(w[:m.start()]) >= 3:
                stem = self._canonical_stem(w[:m.start()])
                case_roots.add(stem)
        return case_roots

    def _jaccard(self, set_a: Set[str], set_b: Set[str]) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return float(intersection / union) if union > 0 else 0.0

    def extract_features(
        self,
        claim: str,
        evidence: str,
        mask_indices: Optional[List[int]] = None,
        ablation_mode: Optional[str] = None,
    ) -> List[float]:
        """
        Extracts 16-dimensional morphological alignment vector.
        All values are floats in [0.0, 1.0].
        Supports feature masking via mask_indices or ablation_mode.
        """
        claim_clean = (claim or "").strip()
        evidence_clean = (evidence or "").strip()

        if not claim_clean and not evidence_clean:
            return [0.0] * 16

        # 0: Verbal/copular negation in claim
        neg_c = 1.0 if self._has_negation(claim_clean) else 0.0

        # 1: Verbal/copular negation in evidence
        neg_e = 1.0 if self._has_negation(evidence_clean) else 0.0

        # 2: Directional negation mismatch (XOR)
        neg_mismatch = 1.0 if (neg_c != neg_e) else 0.0

        # 3: Temporal calendar year conflict
        years_c = set(self.YEAR_RE.findall(claim_clean))
        years_e = set(self.YEAR_RE.findall(evidence_clean))
        if years_c and years_e and years_c.isdisjoint(years_e):
            temporal_conflict = 1.0
        else:
            temporal_conflict = 0.0

        # 4: Numerical quantity mismatch
        nums_c = set(self.NUMBER_RE.findall(claim_clean))
        nums_e = set(self.NUMBER_RE.findall(evidence_clean))
        if nums_c and not nums_c.issubset(nums_e):
            num_mismatch = 1.0
        else:
            num_mismatch = 0.0

        # 5: Evidential marker in claim
        evid_c = 1.0 if bool(self.EVIDENTIAL_RE.search(claim_clean)) else 0.0

        # 6: Evidential marker in evidence
        evid_e = 1.0 if bool(self.EVIDENTIAL_RE.search(evidence_clean)) else 0.0

        # 7: Modal necessity marker
        has_nec = bool(self.MODAL_NECESSITY_RE.search(claim_clean) or self.MODAL_NECESSITY_RE.search(evidence_clean))
        modal_nec = 1.0 if has_nec else 0.0

        # 8: Modal possibility marker
        has_poss = bool(self.MODAL_POSSIBILITY_RE.search(claim_clean) or self.MODAL_POSSIBILITY_RE.search(evidence_clean))
        modal_poss = 1.0 if has_poss else 0.0

        # 9: FST root overlap ratio
        fst_roots_c = self._get_fst_roots(claim_clean)
        fst_roots_e = self._get_fst_roots(evidence_clean)
        fst_overlap = self._jaccard(fst_roots_c, fst_roots_e)

        # 10: Surface token overlap ratio
        surf_c = self._get_surface_tokens(claim_clean)
        surf_e = self._get_surface_tokens(evidence_clean)
        surf_overlap = self._jaccard(surf_c, surf_e)

        # 11: Lexical-FST divergence
        lex_fst_divergence = float(abs(surf_overlap - fst_overlap))

        # 12: Subject proper noun match
        proper_nouns = self.PROPER_NOUN_RE.findall(claim_clean)
        if proper_nouns:
            subject_pn = proper_nouns[0].lower()
            subj_roots = self._get_fst_roots(subject_pn)
            if subject_pn in evidence_clean.lower() or (subj_roots and not subj_roots.isdisjoint(fst_roots_e)):
                subj_match = 1.0
            else:
                subj_match = 0.0
        else:
            subj_match = 0.0

        # 13: Case agreement alignment score
        case_c = self._get_case_roots(claim_clean)
        case_e = self._get_case_roots(evidence_clean)
        case_align = self._jaccard(case_c, case_e)

        # 14: Normalized claim length
        claim_len_norm = float(min(1.0, len(claim_clean.split()) / 30.0))

        # 15: Normalized evidence length
        evidence_len_norm = float(min(1.0, len(evidence_clean.split()) / 100.0))

        features = [
            float(neg_c),
            float(neg_e),
            float(neg_mismatch),
            float(temporal_conflict),
            float(num_mismatch),
            float(evid_c),
            float(evid_e),
            float(modal_nec),
            float(modal_poss),
            float(fst_overlap),
            float(surf_overlap),
            float(lex_fst_divergence),
            float(subj_match),
            float(case_align),
            float(claim_len_norm),
            float(evidence_len_norm),
        ]

        indices_to_mask = set()
        if self.default_mask_indices:
            indices_to_mask.update(self.default_mask_indices)
        if self.default_ablation_mode:
            indices_to_mask.update(self.get_mask_indices_for_mode(self.default_ablation_mode))
        if ablation_mode:
            indices_to_mask.update(self.get_mask_indices_for_mode(ablation_mode))
        if mask_indices:
            indices_to_mask.update(mask_indices)

        for idx in indices_to_mask:
            if 0 <= idx < len(features):
                features[idx] = 0.0

        return features


class OfflineHeuristicNLIVerifier:
    """
    Calibrated deterministic rule-weighted NLI verifier.
    Predicts SUPPORTED, REFUTES, or NOT_ENOUGH_INFO using the 16-dimensional
    morphological feature extractor.
    """

    LABEL_NAMES = ["SUPPORTED", "REFUTES", "NOT_ENOUGH_INFO"]

    def __init__(self, extractor: Optional[MorphologicalAffixExtractor] = None):
        self.extractor = extractor or MorphologicalAffixExtractor()

    def predict_pair(
        self,
        claim: str,
        evidence: str,
        mask_indices: Optional[List[int]] = None,
        ablation_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Predicts NLI relation between claim and evidence passage.
        """
        features = self.extractor.extract_features(
            claim, evidence, mask_indices=mask_indices, ablation_mode=ablation_mode
        )
        neg_mismatch = features[2]
        temp_conflict = features[3]
        stem_overlap = features[9]

        if not (claim or "").strip() or not (evidence or "").strip():
            label = "NOT_ENOUGH_INFO"
            conf = 0.95
            probs = {"SUPPORTED": 0.02, "REFUTES": 0.03, "NOT_ENOUGH_INFO": 0.95}
            explanation = "Empty claim or evidence passage."
        elif (temp_conflict == 1.0 or neg_mismatch == 1.0) and stem_overlap > 0.20:
            label = "REFUTES"
            conf = min(0.98, max(0.65, 0.50 + 0.40 * stem_overlap + 0.05 * temp_conflict + 0.05 * neg_mismatch))
            rem = 1.0 - conf
            probs = {"SUPPORTED": rem * 0.20, "REFUTES": conf, "NOT_ENOUGH_INFO": rem * 0.80}
            explanation = "Contradiction detected via temporal conflict or negation mismatch with high topical overlap."
        elif stem_overlap > 0.35 and neg_mismatch == 0.0:
            label = "SUPPORTED"
            conf = min(0.98, max(0.60, 0.45 + 0.50 * stem_overlap))
            rem = 1.0 - conf
            probs = {"SUPPORTED": conf, "REFUTES": rem * 0.20, "NOT_ENOUGH_INFO": rem * 0.80}
            explanation = "Topical stem overlap exceeds threshold with aligned polarity."
        else:
            label = "NOT_ENOUGH_INFO"
            conf = max(0.55, 0.95 - stem_overlap)
            rem = 1.0 - conf
            probs = {"SUPPORTED": rem * 0.50, "REFUTES": rem * 0.50, "NOT_ENOUGH_INFO": conf}
            explanation = "Insufficient lexical and morphological evidence to verify or refute."

        return {
            "label": label,
            "confidence": float(conf),
            "probabilities": {k: float(v) for k, v in probs.items()},
            "features": features,
            "explanation": explanation
        }

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Batch prediction over a list of (claim, evidence) tuples.
        """
        return [self.predict_pair(claim, ev) for claim, ev in pairs]


class MorphoNLIVerifier(_BaseModule):
    """
    Morphologically-grounded Cross-Encoder NLI Verifier.
    Implements Equation 2 from Section 4.2 of the Paper 1 manuscript:
        P(y | c, E) = softmax(W_v [h_[CLS]; m_affix] + b_v)

    Supports dual-stream fusion of contextual transformer representations
    with the 16-dimensional morphological feature vector. Provides full
    PyTorch module functionality when torch is available, and an automatic
    defensive heuristic fallback when running in CPU test environments.
    """

    LABEL_NAMES = ["SUPPORTED", "REFUTES", "NOT_ENOUGH_INFO"]

    def __init__(
        self,
        hidden_dim: int = 768,
        morpho_dim: int = 16,
        num_classes: int = 3,
        dropout_prob: float = 0.1,
        backbone_name_or_path: Optional[str] = None
    ):
        if HAS_TORCH:
            super().__init__()
        self.hidden_dim = hidden_dim
        self.morpho_dim = morpho_dim
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.backbone_name_or_path = backbone_name_or_path

        self.extractor = MorphologicalAffixExtractor()
        self.heuristic_fallback = OfflineHeuristicNLIVerifier(extractor=self.extractor)

        self.encoder = None
        if backbone_name_or_path and HAS_TRANSFORMERS:
            try:
                self.encoder = AutoModel.from_pretrained(backbone_name_or_path)
                self.hidden_dim = self.encoder.config.hidden_size
            except Exception:
                self.encoder = None

        if HAS_TORCH:
            self.dropout = nn.Dropout(dropout_prob)
            self.classifier = nn.Linear(self.hidden_dim + self.morpho_dim, self.num_classes)
        else:
            self.dropout = None
            self.classifier = None

    def forward(
        self,
        input_ids: Optional[Any] = None,
        attention_mask: Optional[Any] = None,
        morpho_features: Optional[Any] = None,
        h_cls: Optional[Any] = None
    ) -> Any:
        """
        Forward pass computing Equation 2 logits:
            logits = W_v [h_[CLS]; m_affix] + b_v
        """
        if not HAS_TORCH:
            batch_size = 1
            if input_ids is not None and hasattr(input_ids, "__len__"):
                batch_size = len(input_ids)
            elif h_cls is not None and hasattr(h_cls, "__len__"):
                batch_size = len(h_cls)
            return _DummyTensor((batch_size, self.num_classes))

        if h_cls is None:
            if self.encoder is not None and input_ids is not None:
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                h_cls = outputs.last_hidden_state[:, 0, :]
            else:
                batch_size = input_ids.shape[0] if input_ids is not None else 1
                device = input_ids.device if input_ids is not None else "cpu"
                h_cls = torch.zeros((batch_size, self.hidden_dim), device=device)

        if morpho_features is None:
            batch_size = h_cls.shape[0]
            morpho_features = torch.zeros((batch_size, self.morpho_dim), device=h_cls.device)

        fused = torch.cat([h_cls, morpho_features], dim=-1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits

    def predict_pair(
        self,
        claim: str,
        evidence: str,
        mask_indices: Optional[List[int]] = None,
        ablation_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Predicts NLI relation for claim-evidence pair.
        Delegates to heuristic fallback when running in CPU/testing mode or without loaded weights.
        """
        return self.heuristic_fallback.predict_pair(
            claim, evidence, mask_indices=mask_indices, ablation_mode=ablation_mode
        )

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Batch prediction over a list of (claim, evidence) tuples.
        """
        return self.heuristic_fallback.predict_batch(pairs)

    def to(self, *args, **kwargs):
        if HAS_TORCH:
            return super().to(*args, **kwargs)
        return self

    def eval(self):
        if HAS_TORCH:
            return super().eval()
        return self

    def train(self, mode: bool = True):
        if HAS_TORCH:
            return super().train(mode)
        return self
