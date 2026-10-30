# -*- coding: utf-8 -*-
"""Closed-vocabulary morphological sequence tokenizer powered by Auto-FST v2 (DynamicFSTOverlay)."""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

try:
    from src.auto_fst.overlay import DynamicFSTOverlay, EMPHATIC_PARTICLES
except ImportError:
    try:
        from auto_fst.overlay import DynamicFSTOverlay, EMPHATIC_PARTICLES
    except ImportError:
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        try:
            from src.auto_fst.overlay import DynamicFSTOverlay, EMPHATIC_PARTICLES
        except ImportError:
            DynamicFSTOverlay = None
            EMPHATIC_PARTICLES = ["ғой", "қой", "мыс", "міс", "ау"]

try:
    from api.fst_analyzer import AdvancedKazakhFSTAnalyzer
except ImportError:
    try:
        from fst_analyzer import AdvancedKazakhFSTAnalyzer
    except ImportError:
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        from fst_analyzer import AdvancedKazakhFSTAnalyzer

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<ROOT>", "<LOAN>", "<NOMINAL>", "<VERBAL>", "<EOS>"]


class _DummyScalar(int):
    def item(self):
        return int(self)


class _DummyRow(list):
    def __getitem__(self, idx):
        val = super().__getitem__(idx)
        if isinstance(val, int):
            return _DummyScalar(val)
        return val


class _DummyTensor:
    def __init__(self, data):
        self.data = data
        self.shape = (len(data), len(data[0]) if data else 0)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            r, c = idx
            return _DummyScalar(self.data[r][c])
        row = self.data[idx]
        return _DummyRow(row)

    def __len__(self):
        return len(self.data)

    def tolist(self):
        return self.data

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def __repr__(self):
        return f"tensor({self.data})"


class MorphemeTokenizer:
    """Closed-vocabulary morphological sequence tokenizer for Kazakh FST affixes and roots."""

    def __init__(self, fst_analyzer=None):
        if fst_analyzer is not None:
            self.fst = fst_analyzer
        elif DynamicFSTOverlay is not None:
            self.fst = DynamicFSTOverlay()
        else:
            self.fst = AdvancedKazakhFSTAnalyzer()

        self.vocab: dict[str, int] = {}
        self._build_vocab()
        self.id2token = {idx: tok for tok, idx in self.vocab.items()}
        self.pad_token_id = self.vocab["<PAD>"]
        self.unk_token_id = self.vocab["<UNK>"]
        self.root_token_id = self.vocab["<ROOT>"]
        self.loan_token_id = self.vocab["<LOAN>"]
        self.nominal_token_id = self.vocab["<NOMINAL>"]
        self.verbal_token_id = self.vocab["<VERBAL>"]
        self.eos_token_id = self.vocab["<EOS>"]

    def _build_vocab(self):
        idx = 0
        for tok in SPECIAL_TOKENS:
            self.vocab[tok] = idx
            idx += 1

        all_affixes = set()
        for c in getattr(self.fst, "cases", []):
            all_affixes.add(f"-{c}")
        for p in getattr(self.fst, "possessives", []):
            all_affixes.add(f"-{p}")
        for pl in getattr(self.fst, "plurals", []):
            all_affixes.add(f"-{pl}")
        for vt in getattr(self.fst, "verbal_tenses", []):
            all_affixes.add(f"-{vt}")
        for vp in getattr(self.fst, "verbal_persons", []):
            all_affixes.add(f"-{vp}")
        for vn in getattr(self.fst, "verbal_negations", []):
            all_affixes.add(f"-{vn}")
        for vb in getattr(self.fst, "verbalizers", []):
            all_affixes.add(f"-{vb}")
        for da in getattr(self.fst, "deriv_adjectives", []):
            all_affixes.add(f"-{da}")
        for ag in getattr(self.fst, "agentive_suffixes", []):
            all_affixes.add(f"-{ag}")
        for pt in EMPHATIC_PARTICLES:
            all_affixes.add(f"-{pt}")

        for sfx in sorted(all_affixes):
            self.vocab[sfx] = idx
            idx += 1

    def tokenize_text(self, text: str) -> list[str]:
        """Runs Auto-FST v2 segmentation and extracts canonical morpheme tokens."""
        if not text or not isinstance(text, str):
            return []
        segmented = self.fst.analyze_and_segment(text)
        tokens = []
        for raw_tok in segmented.split():
            cleaned = raw_tok.strip(".,!?;:\"'()[]«»—–")
            if not cleaned:
                continue
            if cleaned.startswith("-") and len(cleaned) > 1:
                tokens.append(cleaned.lower())
            else:
                tokens.append("<ROOT>")
        return tokens

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize_text(text)
        return [self.vocab.get(t, self.unk_token_id) for t in tokens]

    def batch_encode(self, texts: list[str], max_length: int = 256):
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False

        batch_ids = []
        for t in texts:
            ids = self.encode(t)[:max_length]
            if len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
            batch_ids.append(ids)

        if has_torch:
            return torch.tensor(batch_ids, dtype=torch.long)
        return _DummyTensor(batch_ids)
