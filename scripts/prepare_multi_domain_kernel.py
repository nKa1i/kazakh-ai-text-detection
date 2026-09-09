"""
Prepares the self-contained Kaggle runner script: kaggle_runner/diagnostic_kernel.py
Implements Task 4 of Kaz-MultiDomain Tri-Domain Contrastive Training:
Packages the complete self-contained Kaggle runner embedding:
- data/kaz_multi_domain_train_6k.json (6,000 tri-domain training samples) via zlib+base64
- data/kaz_mage_eval_6k.json (6,000 held-out MAGE evaluation samples) via zlib+base64
- AdvancedKazakhFSTAnalyzer and MorphemeTokenizer (vocab ~250, max_length=256)
- MorphemeEncoder (pos_embedding=512) and MorphoContrastiveDetector (dual-stream gated cross-attention)
- DomainStratifiedBatchSampler and MultiDomainCollator
- SupConLoss and InvarianceLoss
- Full evaluation suite (compute_mage_metrics, calculate_domain_degradation, evaluate_quadrant_matrix, generate_mage_markdown_report)
- Training pipeline for Model 5-MultiDomain and Review-Only baseline on Dual Tesla T4 GPUs (fp16, AdamW, lr_backbone=2e-5, lr_morph=1e-4)
- Outputs output/kaz_multi_domain_benchmark_results.json and output/kaz_multi_domain_paper_report.md
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import zlib
from typing import Optional


def build_multi_domain_kernel(
    output_path: Optional[str] = None,
    train_data_path: Optional[str] = None,
    eval_data_path: Optional[str] = None,
    dry_run: bool = False
) -> str:
    """
    Builds the self-contained Kaggle runner script for Kaz-MultiDomain benchmark.
    Embeds both training (6,000 samples) and evaluation (6,000 samples) datasets via zlib+base64.

    Args:
        output_path: Target destination for generated kernel script. Defaults to kaggle_runner/diagnostic_kernel.py.
        train_data_path: Path to kaz_multi_domain_train_6k.json.
        eval_data_path: Path to kaz_mage_eval_6k.json.
        dry_run: If True, compresses a small slice for quick local build verification.

    Returns:
        Absolute path to the generated diagnostic_kernel.py.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if output_path is None:
        output_path = os.path.join(repo_root, "kaggle_runner", "diagnostic_kernel.py")
    if train_data_path is None:
        train_data_path = os.path.join(repo_root, "data", "kaz_multi_domain_train_6k.json")
    if eval_data_path is None:
        eval_data_path = os.path.join(repo_root, "data", "kaz_mage_eval_6k.json")

    # 1. Prepare Training Data
    if dry_run:
        if os.path.exists(train_data_path):
            with open(train_data_path, "r", encoding="utf-8") as f:
                all_train = json.load(f)
                train_data = all_train[:30]
        else:
            train_data = [
                {
                    "id": f"train_{i}",
                    "domain": "consumer_reviews" if i % 3 == 0 else ("news" if i % 3 == 1 else "wikipedia"),
                    "generator": "human" if i % 2 == 0 else "Sherkala-7B",
                    "is_unseen_domain": False,
                    "is_unseen_generator": False,
                    "quadrant": "train",
                    "prefix": "Бүгінгі таңда",
                    "text": f"Бұл оқыту мәтіні {i}, қазақ тіліндегі мысал үлгісі.",
                    "label": 0 if i % 2 == 0 else 1,
                    "char_length": 50,
                    "word_count": 8
                }
                for i in range(30)
            ]
        print(f"[dry_run] Using {len(train_data)} train records for kernel verification.")
    else:
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(f"Training dataset not found at {train_data_path}")
        with open(train_data_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        print(f"Loaded {len(train_data)} full training samples from {train_data_path}.")

    # 2. Prepare Evaluation Data
    if dry_run:
        if os.path.exists(eval_data_path):
            with open(eval_data_path, "r", encoding="utf-8") as f:
                all_eval = json.load(f)
                eval_data = all_eval[:20]
        else:
            eval_data = [
                {
                    "id": f"eval_{i}",
                    "domain": "consumer_reviews" if i % 3 == 0 else ("news" if i % 3 == 1 else "wikipedia"),
                    "generator": "human" if i % 2 == 0 else "Sherkala-7B",
                    "is_unseen_domain": i % 3 != 0,
                    "is_unseen_generator": False,
                    "quadrant": "Q1" if (i % 2 == 1 and i % 3 == 0) else "human_consumer_reviews",
                    "prefix": "Бүгінгі таңда",
                    "text": f"Бұл бағалау мәтіні {i}, қазақ тіліндегі сынақ үлгісі.",
                    "label": 0 if i % 2 == 0 else 1,
                    "char_length": 55,
                    "word_count": 8
                }
                for i in range(20)
            ]
        print(f"[dry_run] Using {len(eval_data)} eval records for kernel verification.")
    else:
        if not os.path.exists(eval_data_path):
            raise FileNotFoundError(f"Evaluation dataset not found at {eval_data_path}")
        with open(eval_data_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        print(f"Loaded {len(eval_data)} full evaluation samples from {eval_data_path}.")

    # Compress datasets using zlib level 9
    raw_train_bytes = json.dumps(train_data, ensure_ascii=False).encode("utf-8")
    comp_train_bytes = zlib.compress(raw_train_bytes, level=9)
    b64_train_str = base64.b64encode(comp_train_bytes).decode("ascii")

    raw_eval_bytes = json.dumps(eval_data, ensure_ascii=False).encode("utf-8")
    comp_eval_bytes = zlib.compress(raw_eval_bytes, level=9)
    b64_eval_str = base64.b64encode(comp_eval_bytes).decode("ascii")

    print(f"Compressed train data: {len(raw_train_bytes):,} bytes -> {len(comp_train_bytes):,} bytes ({len(b64_train_str):,} b64 chars)")
    print(f"Compressed eval data: {len(raw_eval_bytes):,} bytes -> {len(comp_eval_bytes):,} bytes ({len(b64_eval_str):,} b64 chars)")

    # Replace placeholders in kernel template
    kernel_code = KERNEL_TEMPLATE.replace("__MULTI_DOMAIN_TRAIN_B64__", b64_train_str)
    kernel_code = kernel_code.replace("__MAGE_EVAL_DATA_B64__", b64_eval_str)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(kernel_code)

    print(f"Successfully generated {output_path} ({len(kernel_code.encode('utf-8')):,} bytes)")
    return output_path


KERNEL_TEMPLATE = r'''"""
Kaggle Dual Tesla T4 Remote Runner: Kaz-MultiDomain Tri-Domain Contrastive Training & Benchmark
Resolving Cross-Domain Blindspots in Morphological AI Text Detection (Reviews, News, Wikipedia)

Evaluates:
- Baseline: Model 5 (Review-Only Morpho-Contrastive Baseline)
- Treatment: Model 5-MultiDomain (Tri-Domain Stratified Batching + Contrastive Alignment)
- Head-to-Head Comparison on Held-Out 6,000-sample MAGE Benchmark (Q1, Q2, Q3, Q4)
- Dynamic Gate Routing Dynamics across Reviews, News, and Wikipedia

Hardware: Dual Tesla T4 GPUs (fp16, AdamW, lr_backbone=2e-5, lr_morph=1e-4)
Artifacts:
- output/kaz_multi_domain_benchmark_results.json
- output/kaz_multi_domain_paper_report.md
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import os
import random
import re
import sys
import time
import zlib
from itertools import groupby
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# -----------------------------------------------------------------------------
# Hardware & Environment Setup
# -----------------------------------------------------------------------------
print("=" * 80)
print("KAZ-MULTIDOMAIN: TRI-DOMAIN CONTRASTIVE TRAINING & EVALUATION RUNNER")
print("Target: Kaggle Dual Tesla T4 GPUs | Tri-Domain Balanced Alignment")
print("=" * 80)

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Version: {torch.__version__}")
print(f"Primary Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.2f} GB VRAM)")


# -----------------------------------------------------------------------------
# 1. Embedded FST Morphological Analyzer
# -----------------------------------------------------------------------------
COMMON_LOANWORD_ROOTS = [
    'доставка', 'оплата', 'возврат', 'заказ', 'бонус', 'акция', 'клиент', 'карта',
    'банк', 'номер', 'приложение', 'аккаунт', 'профиль', 'страница', 'пароль', 'код',
    'каспи', 'инстаграм', 'ватсап', 'телеграм', 'сет', 'сеть', 'чек', 'счет',
    'отзыв', 'товар', 'заявка', 'гранд', 'грант', 'кабинет', 'сумма', 'меню', 'плюс',
    'видео', 'фото', 'файл', 'арна', 'оператор', 'комиссия', 'процент', 'качество',
    'сервис', 'курьер', 'бронь', 'скидка'
]

class AdvancedKazakhFSTAnalyzer:
    def __init__(self):
        self.loanword_roots = COMMON_LOANWORD_ROOTS
        self.cases = [
            'нің', 'ның', 'дың', 'дің', 'тың', 'тің',
            'ға', 'ге', 'қа', 'ке', 'на', 'не',
            'ны', 'ні', 'ды', 'ді', 'ты', 'ті',
            'да', 'де', 'та', 'те', 'нда', 'нде',
            'дан', 'ден', 'тан', 'тен', 'нан', 'нен',
            'мен', 'бен', 'пен',
        ]
        self.possessives = [
            'ларымыз', 'леріміз', 'дарымыз', 'деріміз', 'тарымыз', 'теріміз',
            'ларыңыз', 'леріңіз', 'дарыңыз', 'деріңіз', 'тарыңыз', 'теріңіз',
            'мыз', 'міз', 'ңыз', 'ңіз', 'ымыз', 'іміз', 'ыңыз', 'іңіз',
            'лары', 'лері', 'дары', 'дері', 'тары', 'тері',
            'сын', 'сін', 'мын', 'мін',
            'м', 'ң', 'ы', 'і', 'сы', 'сі',
        ]
        self.plurals = ['лар', 'лер', 'дар', 'дер', 'тар', 'тер']
        self.verbal_tenses = [
            'ғандықтан', 'гендіктен', 'қандықтан', 'кендіктен',
            'атын', 'етін', 'йтын', 'йтін',
            'ған', 'ген', 'қан', 'кен',
            'ады', 'еді', 'йды', 'йді',
            'мақ', 'мек', 'бақ', 'бек', 'пақ', 'пек',
        ]
        self.verbal_persons = [
            'мын', 'мін', 'бын', 'бін', 'пын', 'пін',
            'мыз', 'міз', 'быз', 'біз', 'пыз', 'піз',
            'сың', 'сің', 'сыздар', 'сіздер'
        ]
        self.case_re = re.compile(r'(' + '|'.join(self.cases) + r')$')
        self.poss_re = re.compile(r'(' + '|'.join(self.possessives) + r')$')
        self.plur_re = re.compile(r'(' + '|'.join(self.plurals) + r')$')
        self.verb_tense_re = re.compile(r'(' + '|'.join(self.verbal_tenses) + r')$')
        self.verb_person_re = re.compile(r'(' + '|'.join(self.verbal_persons) + r')$')

    def _segment_loanword(self, word: str) -> Optional[str]:
        w_lower = word.lower()
        for root in self.loanword_roots:
            if w_lower.startswith(root) and len(w_lower) > len(root):
                sfx_fallback = word[len(root):]
                if len(sfx_fallback) >= 1:
                    return word[:len(root)] + f" -{sfx_fallback}"
        return None

    def analyze_and_segment(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        words = text.split()
        processed_words = []
        for word in words:
            if len(word) <= 3:
                processed_words.append(word)
                continue

            loanword_seg = self._segment_loanword(word)
            if loanword_seg:
                processed_words.append(loanword_seg)
                continue

            original = word
            suffixes = []

            m_vper = self.verb_person_re.search(word)
            if m_vper and len(word[:m_vper.start()]) >= 3:
                suffixes.insert(0, m_vper.group(1))
                word = word[:m_vper.start()]

            m_vt = self.verb_tense_re.search(word)
            if m_vt and len(word[:m_vt.start()]) >= 3:
                suffixes.insert(0, m_vt.group(1))
                word = word[:m_vt.start()]

            match = self.case_re.search(word)
            if match and len(word[:match.start()]) >= 3:
                suffixes.insert(0, match.group(1))
                word = word[:match.start()]

            if len(word) > 3:
                match = self.poss_re.search(word)
                if match and len(word[:match.start()]) >= 3:
                    suffixes.insert(0, match.group(1))
                    word = word[:match.start()]

            if len(word) > 3:
                match = self.plur_re.search(word)
                if match and len(word[:match.start()]) >= 3:
                    suffixes.insert(0, match.group(1))
                    word = word[:match.start()]

            if suffixes:
                processed_words.append(word + " " + " ".join(f"-{s}" for s in suffixes))
            else:
                processed_words.append(original)
        return " ".join(processed_words)


# -----------------------------------------------------------------------------
# 2. Embedded Morpheme Sequence Tokenizer
# -----------------------------------------------------------------------------
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<ROOT>", "<LOAN>", "<NOMINAL>", "<VERBAL>", "<EOS>"]

class MorphemeTokenizer:
    def __init__(self, fst_analyzer: Optional[AdvancedKazakhFSTAnalyzer] = None):
        self.fst = fst_analyzer or AdvancedKazakhFSTAnalyzer()
        self.vocab: Dict[str, int] = {}
        self._build_vocab()
        self.id2token = {idx: tok for tok, idx in self.vocab.items()}
        self.pad_token_id = self.vocab["<PAD>"]
        self.unk_token_id = self.vocab["<UNK>"]
        self.root_token_id = self.vocab["<ROOT>"]
        self.loan_token_id = self.vocab["<LOAN>"]
        self.nominal_token_id = self.vocab["<NOMINAL>"]
        self.verbal_token_id = self.vocab["<VERBAL>"]
        self.eos_token_id = self.vocab["<EOS>"]

    def _build_vocab(self) -> None:
        idx = 0
        for tok in SPECIAL_TOKENS:
            self.vocab[tok] = idx
            idx += 1
        all_affixes = set()
        for c in self.fst.cases:
            all_affixes.add(f"-{c}")
        for p in self.fst.possessives:
            all_affixes.add(f"-{p}")
        for pl in self.fst.plurals:
            all_affixes.add(f"-{pl}")
        for vt in self.fst.verbal_tenses:
            all_affixes.add(f"-{vt}")
        for vp in self.fst.verbal_persons:
            all_affixes.add(f"-{vp}")
        for sfx in sorted(all_affixes):
            self.vocab[sfx] = idx
            idx += 1

    def tokenize_text(self, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []
        segmented = self.fst.analyze_and_segment(text)
        tokens = []
        for word in segmented.split():
            if word.startswith("-") and len(word) > 1:
                tokens.append(word)
            else:
                tokens.append("<ROOT>")
        return tokens

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize_text(text)
        return [self.vocab.get(t, self.unk_token_id) for t in tokens]

    def batch_encode(self, texts: List[str], max_length: int = 256) -> torch.Tensor:
        batch_ids = []
        for t in texts:
            ids = self.encode(t)[:max_length]
            if len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
            batch_ids.append(ids)
        return torch.tensor(batch_ids, dtype=torch.long)


# -----------------------------------------------------------------------------
# 3. Embedded Loss Functions
# -----------------------------------------------------------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, contrast_mode: str = "all", base_temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)
        self.contrast_mode = contrast_mode
        self.base_temperature = float(base_temperature)

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = features.device
        if len(features.shape) < 2:
            raise ValueError("`features` needs to be [bsz, n_views, ...]")
        if len(features.shape) == 2:
            features = features.unsqueeze(1)
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = features[:, 0] if self.contrast_mode == "one" else contrast_feature
        anchor_count = 1 if self.contrast_mode == "one" else contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mask_pos_pairs = mask.sum(1)
        has_pos = (mask_pos_pairs > 0).float()
        safe_mask_pos = torch.where(mask_pos_pairs > 0, mask_pos_pairs, torch.ones_like(mask_pos_pairs))
        mean_log_prob_pos = (mask * log_prob).sum(1) / safe_mask_pos

        if has_pos.sum() > 0:
            loss = - (self.temperature / self.base_temperature) * ((mean_log_prob_pos * has_pos).sum() / has_pos.sum())
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        return loss


class InvarianceLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, z_clean: torch.Tensor, z_adv: torch.Tensor) -> torch.Tensor:
        cos_sim = F.cosine_similarity(z_clean, z_adv, dim=-1)
        loss = 1.0 - cos_sim
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# -----------------------------------------------------------------------------
# 4. Embedded Model Architectures
# -----------------------------------------------------------------------------
class MorphemeEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 250,
        embed_dim: int = 768,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        max_pos: int = 512
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_pos = max_pos
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_pos, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, morpheme_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = morpheme_ids.shape[1]
        pos = torch.arange(seq_len, device=morpheme_ids.device).unsqueeze(0).clamp(max=self.max_pos - 1)
        emb = self.embedding(morpheme_ids) + self.pos_embedding(pos)
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
            out = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)
            mask_expanded = attention_mask.unsqueeze(-1).float()
            h_morph = torch.sum(out * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-8)
        else:
            out = self.transformer(emb)
            h_morph = torch.mean(out, dim=1)
        return h_morph


class _DummyBackbone(nn.Module):
    """Fallback backbone for environments without transformer checkpoint access."""
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: Any = None, attention_mask: Any = None, **kwargs: Any) -> Any:
        b = input_ids.shape[0] if input_ids is not None and hasattr(input_ids, "shape") else 1
        s = input_ids.shape[1] if input_ids is not None and hasattr(input_ids, "shape") and len(input_ids.shape) > 1 else 1
        device = self.linear.weight.device
        lhs = self.linear(torch.zeros(b, s, 768, device=device))

        class _BackboneOut:
            def __init__(self, state: Any):
                self.last_hidden_state = state

        return _BackboneOut(lhs)


class MorphoContrastiveDetector(nn.Module):
    """
    Dual-Stream Gated Network with Dynamic Fusion.
    Stream 1: Pretrained RoBERTa backbone (contextual semantic representation).
    Stream 2: MorphemeEncoder (explicit morphological representation).
    Dynamic Fusion: gate = sigmoid(W_gate [h_sem; h_morph])
    Head: Linear projection + Classifier with joint Cross-Entropy + SupCon Loss.
    """
    def __init__(
        self,
        roberta_model_name: str = "kz-transformers/kaz-roberta-conversational",
        morpheme_vocab_size: int = 250,
        embed_dim: int = 768,
        proj_dim: int = 128,
        lambda_supcon: float = 0.5,
        lambda_inv: float = 0.0,
        temperature: float = 0.07,
        dropout_rate: float = 0.2,
        roberta_model: Optional[nn.Module] = None
    ):
        super().__init__()
        if roberta_model is not None:
            self.roberta = roberta_model
        else:
            try:
                self.roberta = AutoModel.from_pretrained(roberta_model_name)
            except Exception as e:
                print(f"Notice: Loading {roberta_model_name} failed ({e}), using fallback backbone.")
                self.roberta = _DummyBackbone(embed_dim)

        self.morph_encoder = MorphemeEncoder(vocab_size=morpheme_vocab_size, embed_dim=embed_dim, max_pos=512)
        self.gate_fc = nn.Linear(embed_dim * 2, embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, 2)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )
        self.lambda_supcon = float(lambda_supcon)
        self.lambda_inv = float(lambda_inv)
        self.supcon_loss_fn = SupConLoss(temperature=temperature)
        self.inv_loss_fn = InvarianceLoss()
        self.ce_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        morpheme_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        adv_input_ids: Optional[torch.Tensor] = None,
        adv_attention_mask: Optional[torch.Tensor] = None,
        adv_morpheme_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        if attention_mask is None and input_ids is not None:
            attention_mask = (input_ids != 0).long()

        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(roberta_out, "last_hidden_state"):
            h_sem = roberta_out.last_hidden_state[:, 0, :]
        elif isinstance(roberta_out, (tuple, list)):
            h_sem = roberta_out[0][:, 0, :]
        else:
            h_sem = roberta_out[:, 0, :]

        morph_mask = (morpheme_ids != 0).long() if morpheme_ids is not None else None
        h_morph = self.morph_encoder(morpheme_ids, attention_mask=morph_mask)

        combined = torch.cat([h_sem, h_morph], dim=-1)
        gate = torch.sigmoid(self.gate_fc(combined))
        h_fused = gate * h_sem + (1.0 - gate) * h_morph

        logits = self.classifier(h_fused)
        proj = F.normalize(self.projection_head(h_fused), p=2, dim=-1)

        result: Dict[str, Any] = {
            "logits": logits,
            "proj": proj,
            "gate": gate,
            "h_fused": h_fused
        }

        total_loss = None
        if labels is not None:
            ce_loss = self.ce_loss_fn(logits, labels)
            supcon_loss = self.supcon_loss_fn(proj, labels)
            total_loss = ce_loss + self.lambda_supcon * supcon_loss
            result["ce_loss"] = ce_loss
            result["supcon_loss"] = supcon_loss

        if adv_input_ids is not None and adv_morpheme_ids is not None:
            if adv_attention_mask is None:
                adv_attention_mask = (adv_input_ids != 0).long()
            adv_roberta_out = self.roberta(input_ids=adv_input_ids, attention_mask=adv_attention_mask)
            h_sem_adv = adv_roberta_out.last_hidden_state[:, 0, :]

            adv_morph_mask = (adv_morpheme_ids != 0).long()
            h_morph_adv = self.morph_encoder(adv_morpheme_ids, attention_mask=adv_morph_mask)

            combined_adv = torch.cat([h_sem_adv, h_morph_adv], dim=-1)
            gate_adv = torch.sigmoid(self.gate_fc(combined_adv))
            h_fused_adv = gate_adv * h_sem_adv + (1.0 - gate_adv) * h_morph_adv
            proj_adv = F.normalize(self.projection_head(h_fused_adv), p=2, dim=-1)

            inv_loss = self.inv_loss_fn(proj, proj_adv)
            gate_diff = gate_adv - gate

            result["proj_adv"] = proj_adv
            result["gate_adv"] = gate_adv
            result["gate_diff"] = gate_diff
            result["inv_loss"] = inv_loss

            if total_loss is not None:
                total_loss = total_loss + self.lambda_inv * inv_loss

        if total_loss is not None:
            result["loss"] = total_loss

        return result


# -----------------------------------------------------------------------------
# 5. Embedded Domain-Stratified Batch Sampler
# -----------------------------------------------------------------------------
class DomainStratifiedBatchSampler(Sampler):
    """
    Mini-batch sampler ensuring balanced domain representation per batch.
    Partitions dataset indices by (domain, label) pairs and yields balanced mini-batches.
    """
    def __init__(
        self,
        domains: Sequence[Any],
        labels: Sequence[Any],
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        if len(domains) != len(labels):
            raise ValueError("domains and labels must have the same length.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.domains = list(domains)
        self.labels = list(labels)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self.unique_domains = sorted(list(set(self.domains)))
        self.num_domains = len(self.unique_domains)

        self._group_indices: Dict[Tuple[Any, Any], List[int]] = {}
        self._domain_labels: Dict[Any, List[Any]] = {}

        for d in self.unique_domains:
            d_labels = sorted(list(set(
                lbl for dom, lbl in zip(self.domains, self.labels) if dom == d
            )))
            self._domain_labels[d] = d_labels
            for lbl in d_labels:
                self._group_indices[(d, lbl)] = [
                    idx for idx, (dom, l) in enumerate(zip(self.domains, self.labels))
                    if dom == d and l == lbl
                ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _count_batches(self) -> int:
        if not self.domains or self.num_domains == 0:
            return 0
        domain_counts = {
            d: sum(len(self._group_indices[(d, lbl)]) for lbl in self._domain_labels[d])
            for d in self.unique_domains
        }
        batch_count = 0
        while True:
            base_quota = self.batch_size // self.num_domains
            rem = self.batch_size % self.num_domains
            quotas = [base_quota] * self.num_domains
            if rem > 0:
                for r in range(rem):
                    target_d = (batch_count * rem + r) % self.num_domains
                    quotas[target_d] += 1

            can_fulfill = all(
                domain_counts[self.unique_domains[i]] >= quotas[i]
                for i in range(self.num_domains)
            )
            if not can_fulfill:
                if not self.drop_last and sum(domain_counts.values()) > 0:
                    batch_count += 1
                break

            for i in range(self.num_domains):
                domain_counts[self.unique_domains[i]] -= quotas[i]
            batch_count += 1

        return batch_count

    def __len__(self) -> int:
        return self._count_batches()

    def __iter__(self) -> Iterator[List[int]]:
        if not self.domains or self.num_domains == 0:
            return

        rng = random.Random(self.seed + self.epoch)
        pools: Dict[Tuple[Any, Any], List[int]] = {}
        for (d, lbl), idx_list in self._group_indices.items():
            shuffled = list(idx_list)
            if self.shuffle:
                rng.shuffle(shuffled)
            pools[(d, lbl)] = shuffled

        pointers: Dict[Tuple[Any, Any], int] = {k: 0 for k in pools}

        def domain_remaining(dom: Any) -> int:
            return sum(
                len(pools[(dom, lbl)]) - pointers[(dom, lbl)]
                for lbl in self._domain_labels[dom]
            )

        batch_idx = 0
        while True:
            base_quota = self.batch_size // self.num_domains
            rem = self.batch_size % self.num_domains
            quotas = [base_quota] * self.num_domains
            if rem > 0:
                for r in range(rem):
                    target_d = (batch_idx * rem + r) % self.num_domains
                    quotas[target_d] += 1

            can_fulfill = all(
                domain_remaining(self.unique_domains[i]) >= quotas[i]
                for i in range(self.num_domains)
            )

            if not can_fulfill:
                if not self.drop_last:
                    leftovers: List[int] = []
                    for dom in self.unique_domains:
                        for lbl in self._domain_labels[dom]:
                            ptr = pointers[(dom, lbl)]
                            leftovers.extend(pools[(dom, lbl)][ptr:])
                            pointers[(dom, lbl)] = len(pools[(dom, lbl)])
                    if leftovers:
                        if self.shuffle:
                            rng.shuffle(leftovers)
                        yield leftovers
                break

            current_batch: List[int] = []
            for d_idx, dom in enumerate(self.unique_domains):
                q = quotas[d_idx]
                labels_in_d = self._domain_labels[dom]
                num_l = len(labels_in_d)
                if num_l == 0 or q == 0:
                    continue

                base_l_quota = q // num_l
                rem_l = q % num_l
                l_quotas = {lbl: base_l_quota for lbl in labels_in_d}
                if rem_l > 0:
                    for rl in range(rem_l):
                        target_l = labels_in_d[(batch_idx + d_idx + rl) % num_l]
                        l_quotas[target_l] += 1

                dom_samples: List[int] = []
                for lbl in labels_in_d:
                    needed = l_quotas[lbl]
                    avail = len(pools[(dom, lbl)]) - pointers[(dom, lbl)]
                    take = min(needed, avail)
                    ptr = pointers[(dom, lbl)]
                    dom_samples.extend(pools[(dom, lbl)][ptr : ptr + take])
                    pointers[(dom, lbl)] += take

                if len(dom_samples) < q:
                    shortfall = q - len(dom_samples)
                    for lbl in labels_in_d:
                        if shortfall <= 0:
                            break
                        avail = len(pools[(dom, lbl)]) - pointers[(dom, lbl)]
                        take = min(shortfall, avail)
                        ptr = pointers[(dom, lbl)]
                        dom_samples.extend(pools[(dom, lbl)][ptr : ptr + take])
                        pointers[(dom, lbl)] += take
                        shortfall -= take

                current_batch.extend(dom_samples)

            if self.shuffle:
                rng.shuffle(current_batch)

            yield current_batch
            batch_idx += 1


# -----------------------------------------------------------------------------
# 6. Multi-Domain Dataset & Collator
# -----------------------------------------------------------------------------
class MultiDomainDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        morpheme_tokenizer: MorphemeTokenizer,
        max_morph_length: int = 256
    ) -> None:
        self.records = list(records)
        self.morpheme_tokenizer = morpheme_tokenizer
        self.max_morph_length = int(max_morph_length)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        text = rec.get("text", "")
        label = int(rec.get("label", 0))
        domain = str(rec.get("domain", "unknown"))
        sample_id = str(rec.get("id", f"s_{idx}"))
        generator = str(rec.get("generator", "human" if label == 0 else "Sherkala-7B"))

        morph_ids = self.morpheme_tokenizer.encode(text)[:self.max_morph_length]
        if not morph_ids:
            morph_ids = [self.morpheme_tokenizer.root_token_id]

        return {
            "id": sample_id,
            "text": text,
            "label": label,
            "domain": domain,
            "generator": generator,
            "morpheme_ids": morph_ids,
            "quadrant": rec.get("quadrant", "")
        }


class MultiDomainCollator:
    def __init__(
        self,
        tokenizer: Any,
        pad_morph_id: int = 0,
        max_length: int = 256,
        max_morph_length: int = 256
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_morph_id = int(pad_morph_id)
        self.max_length = int(max_length)
        self.max_morph_length = int(max_morph_length)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [b["text"] for b in batch]
        labels = [b["label"] for b in batch]
        domains = [b["domain"] for b in batch]

        if hasattr(self.tokenizer, "__call__"):
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", (input_ids != 0).long())
        else:
            # Fallback simple tensor
            input_ids = torch.ones((len(texts), 16), dtype=torch.long)
            attention_mask = torch.ones((len(texts), 16), dtype=torch.long)

        morph_lists = [b["morpheme_ids"] for b in batch]
        max_m_len = min(self.max_morph_length, max((len(m) for m in morph_lists), default=1))
        padded_morphs = []
        for m in morph_lists:
            cur = list(m)[:max_m_len]
            if len(cur) < max_m_len:
                cur = cur + [self.pad_morph_id] * (max_m_len - len(cur))
            padded_morphs.append(cur)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "morpheme_ids": torch.tensor(padded_morphs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "domains": domains,
            "texts": texts,
            "batch_records": batch
        }


# -----------------------------------------------------------------------------
# 7. Embedded Evaluation Suite
# -----------------------------------------------------------------------------
def _to_list(arr: Any) -> Any:
    if arr is None:
        return None
    if hasattr(arr, "tolist"):
        return arr.tolist()
    if isinstance(arr, (list, tuple)):
        return list(arr)
    return list(arr)

def _compute_roc_and_youden_pure(y_true: Sequence[int], y_prob: Sequence[float]) -> Tuple[float, float, float]:
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.5, 0.5

    pairs = sorted(zip(y_prob, y_true), key=lambda x: x[0], reverse=True)
    distinct_groups = []
    for p, group in groupby(pairs, key=lambda x: x[0]):
        g = list(group)
        pos_in_g = sum(1 for _, y in g if y == 1)
        neg_in_g = len(g) - pos_in_g
        distinct_groups.append((p, pos_in_g, neg_in_g))

    thresholds = [distinct_groups[0][0] + 1.0]
    fprs, tprs = [0.0], [0.0]
    tp, fp = 0, 0
    for p, pos_g, neg_g in distinct_groups:
        tp += pos_g
        fp += neg_g
        thresholds.append(p)
        fprs.append(fp / n_neg)
        tprs.append(tp / n_pos)

    auc = 0.0
    for i in range(1, len(fprs)):
        auc += (fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1]) / 2.0

    j_scores = [t - f for t, f in zip(tprs, fprs)]
    best_idx = max(range(len(j_scores)), key=lambda i: j_scores[i])
    optimal_threshold = min(1.0, thresholds[best_idx])
    eer_idx = min(range(len(fprs)), key=lambda i: abs(fprs[i] - (1.0 - tprs[i])))
    eer = fprs[eer_idx]
    return auc, optimal_threshold, eer

def _calc_classification_stats(y_true: Sequence[int], y_prob: Sequence[float], thresh: float) -> Dict[str, Any]:
    y_pred = [1 if p >= thresh else 0 for p in y_prob]
    total = len(y_true)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    acc = (tp + tn) / total if total > 0 else 0.0

    prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = (2 * prec_1 * rec_1) / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0

    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = (2 * prec_0 * rec_0) / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0

    macro_f1 = (f1_0 + f1_1) / 2.0
    return {
        "accuracy": round(float(acc), 4),
        "f1": round(float(macro_f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "precision": round(float(prec_1), 4),
        "recall": round(float(rec_1), 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

def compute_bootstrap_auc_ci(y_true: Sequence[int], y_prob: Sequence[float], n_bootstrap: int = 1000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    y_true_list = [int(y) for y in y_true]
    y_prob_list = [float(p) for p in y_prob]
    n = len(y_true_list)
    if n < 2 or len(set(y_true_list)) < 2:
        return 0.5, 0.5
    rng = random.Random(seed)
    auc_scores = []
    for _ in range(n_bootstrap):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        b_t = [y_true_list[i] for i in idx]
        if len(set(b_t)) < 2:
            continue
        b_p = [y_prob_list[i] for i in idx]
        auc, _, _ = _compute_roc_and_youden_pure(b_t, b_p)
        auc_scores.append(auc)
    if not auc_scores:
        base_auc, _, _ = _compute_roc_and_youden_pure(y_true_list, y_prob_list)
        return round(float(base_auc), 4), round(float(base_auc), 4)
    auc_scores.sort()
    lower_idx = int(math.floor(len(auc_scores) * (alpha / 2.0)))
    upper_idx = min(int(math.ceil(len(auc_scores) * (1.0 - alpha / 2.0))) - 1, len(auc_scores) - 1)
    return round(float(auc_scores[lower_idx]), 4), round(float(auc_scores[upper_idx]), 4)

def compute_mage_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: Optional[float] = None,
    bootstrap_ci: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    y_true_list = [int(y) for y in _to_list(y_true) or []]
    y_prob_list = [float(p) for p in _to_list(y_prob) or []]
    n_samples = len(y_true_list)
    if n_samples == 0:
        return {"roc_auc": 0.5, "optimal_threshold": 0.5, "accuracy": 0.0, "f1": 0.0, "ci_lower": 0.5, "ci_upper": 0.5}

    try:
        if len(set(y_true_list)) > 1:
            auc_val = float(roc_auc_score(y_true_list, y_prob_list))
            fpr_c, tpr_c, thresh_c = roc_curve(y_true_list, y_prob_list)
            j_sc = tpr_c - fpr_c
            best_idx = int(np.argmax(j_sc))
            opt_thresh = float(thresh_c[best_idx])
            if math.isinf(opt_thresh) or opt_thresh > 1.0:
                opt_thresh = 1.0
        else:
            auc_val, opt_thresh, _ = _compute_roc_and_youden_pure(y_true_list, y_prob_list)
    except Exception:
        auc_val, opt_thresh, _ = _compute_roc_and_youden_pure(y_true_list, y_prob_list)

    eff_thresh = float(opt_thresh) if threshold is None else float(threshold)
    stats_eff = _calc_classification_stats(y_true_list, y_prob_list, eff_thresh)

    if bootstrap_ci and n_samples > 1 and len(set(y_true_list)) > 1:
        ci_lower, ci_upper = compute_bootstrap_auc_ci(y_true_list, y_prob_list, n_bootstrap=n_bootstrap, seed=seed)
    else:
        ci_lower, ci_upper = round(float(auc_val), 4), round(float(auc_val), 4)

    return {
        "roc_auc": round(float(auc_val), 4),
        "optimal_threshold": round(float(opt_thresh), 4),
        "accuracy": stats_eff["accuracy"],
        "f1": stats_eff["f1"],
        "macro_f1": stats_eff["macro_f1"],
        "precision": stats_eff["precision"],
        "recall": stats_eff["recall"],
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_samples": n_samples
    }

def calculate_domain_degradation(quadrant_results: Dict[str, Any]) -> Dict[str, float]:
    def _extract_auc(val: Any) -> float:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict):
            return float(val.get("roc_auc", 0.0))
        return 0.0

    auc_q1 = _extract_auc(quadrant_results.get("Q1", 0.0))
    auc_q2 = _extract_auc(quadrant_results.get("Q2", 0.0))
    auc_q3 = _extract_auc(quadrant_results.get("Q3", 0.0))
    auc_q4 = _extract_auc(quadrant_results.get("Q4", 0.0))

    return {
        "delta_auc_generator": round(float(auc_q1 - auc_q2), 4),
        "delta_auc_domain": round(float(auc_q1 - auc_q3), 4),
        "delta_auc_wild": round(float(auc_q1 - auc_q4), 4),
        "q1_auc": round(float(auc_q1), 4),
        "q2_auc": round(float(auc_q2), 4),
        "q3_auc": round(float(auc_q3), 4),
        "q4_auc": round(float(auc_q4), 4)
    }

def _get_field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)

def evaluate_quadrant_matrix(records: List[Dict[str, Any]], bootstrap_ci: bool = True, n_bootstrap: int = 1000, seed: int = 42) -> Dict[str, Any]:
    quadrant_buckets: Dict[str, List[Dict[str, Any]]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    domain_buckets: Dict[str, List[Dict[str, Any]]] = {}
    domain_gates: Dict[str, List[float]] = {}
    quadrant_gates: Dict[str, List[float]] = {}
    human_controls: List[Dict[str, Any]] = []

    for r in records:
        q = _get_field(r, "quadrant", "")
        dom = str(_get_field(r, "domain", ""))
        gate = _get_field(r, "gate", _get_field(r, "gate_value", None))
        label = int(_get_field(r, "y_true", _get_field(r, "label", 0)))

        if label == 0:
            human_controls.append(r)

        if q in quadrant_buckets:
            quadrant_buckets[q].append(r)

        if dom:
            if dom not in domain_buckets:
                domain_buckets[dom] = []
            domain_buckets[dom].append(r)
            if gate is not None:
                if dom not in domain_gates:
                    domain_gates[dom] = []
                domain_gates[dom].append(float(gate))

        if q and gate is not None:
            if q not in quadrant_gates:
                quadrant_gates[q] = []
            quadrant_gates[q].append(float(gate))

    # Pair negative controls from matching domains if quadrant only has AI samples
    for q_name in ["Q1", "Q2"]:
        q_recs = quadrant_buckets[q_name]
        if q_recs and not any(int(_get_field(r, "label", _get_field(r, "y_true", 0))) == 0 for r in q_recs):
            dom_humans = [h for h in human_controls if str(_get_field(h, "domain", "")) in ("consumer_reviews", "reviews")]
            quadrant_buckets[q_name].extend(dom_humans)

    for q_name in ["Q3", "Q4"]:
        q_recs = quadrant_buckets[q_name]
        if q_recs and not any(int(_get_field(r, "label", _get_field(r, "y_true", 0))) == 0 for r in q_recs):
            dom_humans = [h for h in human_controls if str(_get_field(h, "domain", "")) in ("news", "wikipedia", "wiki")]
            quadrant_buckets[q_name].extend(dom_humans)

    # 1. Evaluate Quadrants
    quadrant_results = {}
    for q_name in ["Q1", "Q2", "Q3", "Q4"]:
        q_recs = quadrant_buckets[q_name]
        if q_recs:
            yt = [int(_get_field(r, "label", _get_field(r, "y_true", 0))) for r in q_recs]
            yp = [float(_get_field(r, "y_prob", _get_field(r, "prob", 0.5))) for r in q_recs]
            quadrant_results[q_name] = compute_mage_metrics(yt, yp, bootstrap_ci=bootstrap_ci, n_bootstrap=n_bootstrap, seed=seed)
        else:
            quadrant_results[q_name] = {"roc_auc": 0.0, "accuracy": 0.0, "f1": 0.0, "optimal_threshold": 0.5}

    # 2. Degradation
    degradation = calculate_domain_degradation(quadrant_results)

    # 3. Dynamic Gate Routing
    mean_gate_by_domain = {dom: round(float(np.mean(g)), 4) for dom, g in domain_gates.items() if g}
    mean_gate_by_quadrant = {q: round(float(np.mean(g)), 4) for q, g in quadrant_gates.items() if g}
    ref_gate = mean_gate_by_domain.get("consumer_reviews", mean_gate_by_domain.get("reviews", None))

    delta_gate_news = round(float(mean_gate_by_domain["news"] - ref_gate), 4) if (ref_gate is not None and "news" in mean_gate_by_domain) else None
    wiki_k = "wikipedia" if "wikipedia" in mean_gate_by_domain else ("wiki" if "wiki" in mean_gate_by_domain else None)
    delta_gate_wiki = round(float(mean_gate_by_domain[wiki_k] - ref_gate), 4) if (ref_gate is not None and wiki_k) else None

    gate_dynamics = {
        "mean_gate_by_domain": mean_gate_by_domain,
        "mean_gate_by_quadrant": mean_gate_by_quadrant,
        "delta_gate_news": delta_gate_news,
        "delta_gate_wiki": delta_gate_wiki,
        "delta_gate_wikipedia": delta_gate_wiki,
        "sample_counts_by_domain": {dom: len(recs) for dom, recs in domain_buckets.items()}
    }

    # 4. Domains
    domain_results = {}
    for dom, dom_recs in domain_buckets.items():
        if dom_recs:
            yt = [int(_get_field(r, "label", _get_field(r, "y_true", 0))) for r in dom_recs]
            yp = [float(_get_field(r, "y_prob", _get_field(r, "prob", 0.5))) for r in dom_recs]
            domain_results[dom] = compute_mage_metrics(yt, yp, bootstrap_ci=bootstrap_ci, n_bootstrap=n_bootstrap, seed=seed)

    return {
        "quadrants": quadrant_results,
        "degradation": degradation,
        "gate_dynamics": gate_dynamics,
        "domains": domain_results,
        "total_samples": len(records)
    }

def generate_mage_markdown_report(results: Dict[str, Any], model_name: str = "Kazakh AI Detector") -> str:
    quads = results.get("quadrants", {})
    deg = results.get("degradation", {})
    gate_dyn = results.get("gate_dynamics", {})
    mean_gate = gate_dyn.get("mean_gate_by_domain", {})

    lines = [
        f"### {model_name}",
        "",
        "| Quadrant | Domain Type | Generator Type | Domain | Generator | ROC-AUC (95% CI) | Accuracy (%) | Macro F1 (%) | Optimal $\\tau^*$ |",
        "|---|---|---|---|---|---|---|---|---|"
    ]
    quad_meta = [
        ("Q1", "Seen", "Seen", "Reviews", "Sherkala-7B"),
        ("Q2", "Seen", "Unseen", "Reviews", "Qwen-2.5-7B"),
        ("Q3", "Unseen", "Seen", "News/Wiki", "Sherkala-7B"),
        ("Q4", "Unseen", "Unseen", "News/Wiki", "Qwen-2.5-7B"),
    ]
    for q_id, dom_type, gen_type, dom, gen in quad_meta:
        qd = quads.get(q_id, {})
        auc = qd.get("roc_auc", 0.0)
        ci_l = qd.get("ci_lower", auc)
        ci_u = qd.get("ci_upper", auc)
        acc = qd.get("accuracy", 0.0) * 100.0 if qd.get("accuracy", 0.0) <= 1.0 else qd.get("accuracy", 0.0)
        f1 = qd.get("f1", 0.0) * 100.0 if qd.get("f1", 0.0) <= 1.0 else qd.get("f1", 0.0)
        opt_t = qd.get("optimal_threshold", 0.5)
        lines.append(
            f"| **{q_id}** | {dom_type} | {gen_type} | {dom} | {gen} | "
            f"{auc:.4f} [{ci_l:.4f}, {ci_u:.4f}] | {acc:.1f}% | {f1:.1f}% | {opt_t:.4f} |"
        )
    lines.extend([
        "",
        f"- **Generator Degradation (Q1 -> Q2)**: `delta_AUC = {deg.get('delta_auc_generator', 0.0):+.4f}`",
        f"- **Domain Degradation (Q1 -> Q3)**: `delta_AUC = {deg.get('delta_auc_domain', 0.0):+.4f}`",
        f"- **Wild Degradation (Q1 -> Q4)**: `delta_AUC = {deg.get('delta_auc_wild', 0.0):+.4f}`",
    ])
    if mean_gate:
        lines.append(f"- **Dynamic Gate Routing**: Reviews `{mean_gate.get('consumer_reviews', 0.0):.3f}`, News `{mean_gate.get('news', 0.0):.3f}`, Wiki `{mean_gate.get('wikipedia', 0.0):.3f}`")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 8. Decompress Embedded Datasets
# -----------------------------------------------------------------------------
print("\n" + "-" * 60)
print("Decompressing Embedded Datasets...")
b64_train = """__MULTI_DOMAIN_TRAIN_B64__"""
train_bytes = zlib.decompress(base64.b64decode(b64_train))
train_records: List[Dict[str, Any]] = json.loads(train_bytes.decode("utf-8"))
print(f"Loaded {len(train_records)} training samples.")

b64_eval = """__MAGE_EVAL_DATA_B64__"""
eval_bytes = zlib.decompress(base64.b64decode(b64_eval))
eval_records: List[Dict[str, Any]] = json.loads(eval_bytes.decode("utf-8"))
print(f"Loaded {len(eval_records)} held-out evaluation samples.")

# Domain breakdown check
train_dom_counts: Dict[str, int] = {}
for r in train_records:
    train_dom_counts[r["domain"]] = train_dom_counts.get(r["domain"], 0) + 1
print(f"Train domain balance: {train_dom_counts}")

eval_dom_counts: Dict[str, int] = {}
for r in eval_records:
    eval_dom_counts[r["domain"]] = eval_dom_counts.get(r["domain"], 0) + 1
print(f"Eval domain balance: {eval_dom_counts}")


# -----------------------------------------------------------------------------
# 9. Tokenizer & Data Loading Initialization
# -----------------------------------------------------------------------------
fst_analyzer = AdvancedKazakhFSTAnalyzer()
morpheme_tokenizer = MorphemeTokenizer(fst_analyzer=fst_analyzer)
print(f"MorphemeTokenizer vocab size: {len(morpheme_tokenizer.vocab)}")

try:
    raw_tokenizer = AutoTokenizer.from_pretrained("kz-transformers/kaz-roberta-conversational")
    print("Loaded tokenizer: kz-transformers/kaz-roberta-conversational")
except Exception as e:
    print(f"Notice: Failed to load online tokenizer ({e}). Using basic character fallback.")
    class _FallbackTokenizer:
        def __call__(self, texts, max_length=256, padding=True, truncation=True, return_tensors="pt"):
            batch_tokens = []
            for t in texts:
                chars = [ord(c) % 50000 + 1 for c in t][:max_length]
                batch_tokens.append(chars)
            max_len = max(len(bt) for bt in batch_tokens) if batch_tokens else 1
            padded = [bt + [0] * (max_len - len(bt)) for bt in batch_tokens]
            t_ids = torch.tensor(padded, dtype=torch.long)
            return {"input_ids": t_ids, "attention_mask": (t_ids != 0).long()}
    raw_tokenizer = _FallbackTokenizer()


# -----------------------------------------------------------------------------
# 10. Training & Inference Routines
# -----------------------------------------------------------------------------
def train_model_pipeline(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    num_epochs: int = 3,
    grad_accum_steps: int = 1,
    tag: str = "Model"
) -> nn.Module:
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        model.train()
        total_loss, total_ce, total_supcon = 0.0, 0.0, 0.0
        num_batches = 0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            inp_ids = batch["input_ids"].to(device)
            att_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            m_ids = batch.get("morpheme_ids")
            if m_ids is not None:
                m_ids = m_ids.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                outputs = model(input_ids=inp_ids, attention_mask=att_mask, morpheme_ids=m_ids, labels=labels)
                loss = outputs["loss"] / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            total_loss += outputs["loss"].item()
            total_ce += outputs.get("ce_loss", outputs["loss"]).item()
            total_supcon += outputs.get("supcon_loss", torch.tensor(0.0)).item()
            num_batches += 1

        elapsed = time.time() - t0
        avg_loss = total_loss / max(1, num_batches)
        avg_ce = total_ce / max(1, num_batches)
        avg_supcon = total_supcon / max(1, num_batches)
        print(f"[{tag}] Epoch {epoch}/{num_epochs} ({elapsed:.1f}s) | Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, SupCon: {avg_supcon:.4f})")

    return model


def predict_eval_dataset(
    model: nn.Module,
    records: List[Dict[str, Any]],
    batch_size: int = 64
) -> List[Dict[str, Any]]:
    model.eval()
    model.to(device)
    records_out = []
    text_list = [r["text"] for r in records]

    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            b_texts = text_list[i:i + batch_size]
            b_recs = records[i:i + batch_size]

            if hasattr(raw_tokenizer, "__call__"):
                enc = raw_tokenizer(b_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
                inp_ids = enc["input_ids"].to(device)
                att_mask = enc.get("attention_mask", (inp_ids != 0).long()).to(device)
            else:
                inp_ids = torch.ones((len(b_texts), 16), dtype=torch.long, device=device)
                att_mask = torch.ones((len(b_texts), 16), dtype=torch.long, device=device)

            m_ids = morpheme_tokenizer.batch_encode(b_texts, max_length=256).to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                out = model(input_ids=inp_ids, attention_mask=att_mask, morpheme_ids=m_ids)

            logits = out["logits"] if isinstance(out, dict) else out.logits
            probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy().tolist()

            gates = None
            if isinstance(out, dict) and "gate" in out and out["gate"] is not None:
                gates = out["gate"].mean(dim=-1).float().cpu().numpy().tolist()

            for idx, r in enumerate(b_recs):
                rec_dict = {
                    "id": r.get("id", f"eval_{i+idx}"),
                    "domain": r.get("domain", ""),
                    "generator": r.get("generator", ""),
                    "quadrant": r.get("quadrant", ""),
                    "label": int(r.get("label", 0)),
                    "y_true": int(r.get("label", 0)),
                    "y_prob": float(probs[idx]),
                    "gate": float(gates[idx]) if gates is not None else 0.5
                }
                records_out.append(rec_dict)

    return records_out


# -----------------------------------------------------------------------------
# 11. Benchmark Execution: Baseline vs. Model 5-MultiDomain
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Kaz-MultiDomain Training & Benchmark")
parser.add_argument("--dry_run", action="store_true", help="Quick verification mode with reduced epochs")
parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
parser.add_argument("--lr_backbone", type=float, default=2e-5, help="Learning rate for backbone (default: 2e-5)")
parser.add_argument("--lr_morph", type=float, default=1e-4, help="Learning rate for morph stream (default: 1e-4)")
parser.add_argument("--lambda_supcon", type=float, default=0.5, help="SupCon loss weight (default: 0.5)")

# Safe argument parsing for Kaggle notebook environments
args, _ = parser.parse_known_args()

is_dry_run = args.dry_run or (len(train_records) < 100)
epochs = 1 if is_dry_run else args.epochs
batch_size = 4 if is_dry_run else args.batch_size
lr_backbone = args.lr_backbone
lr_morph = args.lr_morph
lambda_supcon = args.lambda_supcon

print(f"\nExecution Settings: dry_run={is_dry_run}, epochs={epochs}, batch_size={batch_size}, lr_backbone={lr_backbone}, lr_morph={lr_morph}")

all_model_results = {}
all_model_reports = {}

# -----------------------------------------------------------------------------
# STAGE 1: Model 5 (Review-Only Morpho-Contrastive Baseline)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STAGE 1 / 2: MODEL 5 (Review-Only Morpho-Contrastive Baseline)")
print("Trained exclusively on consumer_reviews to evaluate cross-domain vulnerability")
print("=" * 80)

review_train_records = [r for r in train_records if r.get("domain") in ("consumer_reviews", "reviews")]
if not review_train_records:
    print("Warning: No review records found in training set; using slice.")
    review_train_records = train_records[:max(1, len(train_records) // 3)]

print(f"Review-Only Training Samples: {len(review_train_records)}")

dataset_reviews = MultiDomainDataset(review_train_records, morpheme_tokenizer=morpheme_tokenizer)
collator = MultiDomainCollator(raw_tokenizer, pad_morph_id=morpheme_tokenizer.pad_token_id)
loader_reviews = DataLoader(dataset_reviews, batch_size=batch_size, shuffle=True, collate_fn=collator)

set_seed(42)
model_review_only = MorphoContrastiveDetector(
    morpheme_vocab_size=len(morpheme_tokenizer.vocab) + 20,
    lambda_supcon=lambda_supcon
)

roberta_params = [p for n, p in model_review_only.named_parameters() if "roberta" in n and p.requires_grad]
morph_params = [p for n, p in model_review_only.named_parameters() if "roberta" not in n and p.requires_grad]
optimizer_rev = torch.optim.AdamW([
    {"params": roberta_params, "lr": lr_backbone, "weight_decay": 0.01},
    {"params": morph_params, "lr": lr_morph, "weight_decay": 0.01},
])
total_steps_rev = len(loader_reviews) * epochs
scheduler_rev = get_linear_schedule_with_warmup(optimizer_rev, num_warmup_steps=int(0.1 * total_steps_rev), num_training_steps=max(1, total_steps_rev))

print("Starting training of Model 5 (Review-Only Baseline)...")
model_review_only = train_model_pipeline(
    model_review_only, loader_reviews, optimizer_rev, scheduler_rev,
    num_epochs=epochs, tag="Review-Only Baseline"
)

print("\nEvaluating Model 5 (Review-Only Baseline) on held-out Kaz-MAGE 6,000-sample benchmark...")
recs_eval_rev = predict_eval_dataset(model_review_only, eval_records)
res_review_only = evaluate_quadrant_matrix(recs_eval_rev)
all_model_results["Model 5 (Review-Only Baseline)"] = res_review_only
all_model_reports["Model 5 (Review-Only Baseline)"] = generate_mage_markdown_report(res_review_only, "Model 5 (Review-Only Baseline)")

q_rev = res_review_only["quadrants"]
deg_rev = res_review_only["degradation"]
gate_rev = res_review_only["gate_dynamics"]["mean_gate_by_domain"]
print(f"Review-Only -> Q1: {q_rev['Q1']['roc_auc']:.4f} | Q2: {q_rev['Q2']['roc_auc']:.4f} | Q3: {q_rev['Q3']['roc_auc']:.4f} | Q4: {q_rev['Q4']['roc_auc']:.4f}")
print(f"Review-Only -> Domain Degradation (Q1-Q3): {deg_rev['delta_auc_domain']:+.4f} | Wild Degradation (Q1-Q4): {deg_rev['delta_auc_wild']:+.4f}")
print(f"Review-Only -> Gate Routing: {gate_rev}")

del model_review_only, optimizer_rev, scheduler_rev
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# STAGE 2: Model 5-MultiDomain (Tri-Domain Contrastive Alignment)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STAGE 2 / 2: MODEL 5-MULTIDOMAIN (Tri-Domain Contrastive Alignment)")
print("Trained on Balanced Tri-Domain Corpus (Reviews, News, Wikipedia) via DomainStratifiedBatchSampler")
print("=" * 80)

dataset_multi = MultiDomainDataset(train_records, morpheme_tokenizer=morpheme_tokenizer)
domains_list = [r["domain"] for r in train_records]
labels_list = [r["label"] for r in train_records]

batch_sampler = DomainStratifiedBatchSampler(
    domains=domains_list,
    labels=labels_list,
    batch_size=batch_size,
    shuffle=True,
    seed=42
)
loader_multi = DataLoader(dataset_multi, batch_sampler=batch_sampler, collate_fn=collator)

set_seed(42)
model_multidomain = MorphoContrastiveDetector(
    morpheme_vocab_size=len(morpheme_tokenizer.vocab) + 20,
    lambda_supcon=lambda_supcon
)

roberta_params_md = [p for n, p in model_multidomain.named_parameters() if "roberta" in n and p.requires_grad]
morph_params_md = [p for n, p in model_multidomain.named_parameters() if "roberta" not in n and p.requires_grad]
optimizer_md = torch.optim.AdamW([
    {"params": roberta_params_md, "lr": lr_backbone, "weight_decay": 0.01},
    {"params": morph_params_md, "lr": lr_morph, "weight_decay": 0.01},
])
total_steps_md = len(loader_multi) * epochs
scheduler_md = get_linear_schedule_with_warmup(optimizer_md, num_warmup_steps=int(0.1 * total_steps_md), num_training_steps=max(1, total_steps_md))

print(f"Starting training of Model 5-MultiDomain ({len(loader_multi)} batches per epoch)...")
model_multidomain = train_model_pipeline(
    model_multidomain, loader_multi, optimizer_md, scheduler_md,
    num_epochs=epochs, tag="Model 5-MultiDomain"
)

print("\nEvaluating Model 5-MultiDomain on held-out Kaz-MAGE 6,000-sample benchmark...")
recs_eval_md = predict_eval_dataset(model_multidomain, eval_records)
res_multidomain = evaluate_quadrant_matrix(recs_eval_md)
all_model_results["Model 5-MultiDomain"] = res_multidomain
all_model_reports["Model 5-MultiDomain"] = generate_mage_markdown_report(res_multidomain, "Model 5-MultiDomain")

q_md = res_multidomain["quadrants"]
deg_md = res_multidomain["degradation"]
gate_md = res_multidomain["gate_dynamics"]["mean_gate_by_domain"]
print(f"MultiDomain -> Q1: {q_md['Q1']['roc_auc']:.4f} | Q2: {q_md['Q2']['roc_auc']:.4f} | Q3: {q_md['Q3']['roc_auc']:.4f} | Q4: {q_md['Q4']['roc_auc']:.4f}")
print(f"MultiDomain -> Domain Degradation (Q1-Q3): {deg_md['delta_auc_domain']:+.4f} | Wild Degradation (Q1-Q4): {deg_md['delta_auc_wild']:+.4f}")
print(f"MultiDomain -> Gate Routing: {gate_md}")

del model_multidomain, optimizer_md, scheduler_md
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# 12. Cross-Model Synthesis & Publication Report Generation
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SYNTHESIZING KAZ-MULTIDOMAIN PUBLICATION BENCHMARK REPORT")
print("=" * 80)

full_benchmark_results = {
    "experiment": "Kaz-MultiDomain: Tri-Domain Contrastive Training Benchmark",
    "protocol": "ACL 2024 MAGE Protocol (Seen vs Unseen Domain x Seen vs Unseen Generator)",
    "target_hardware": "Kaggle Dual Tesla T4 GPUs",
    "train_dataset_size": len(train_records),
    "eval_dataset_size": len(eval_records),
    "models": {}
}

for name, res in all_model_results.items():
    clean_res = {k: v for k, v in res.items() if k != "records"}
    full_benchmark_results["models"][name] = clean_res

# Comparative Markdown Report
report_lines = [
    "# Kaz-MultiDomain: Tri-Domain Contrastive Training Benchmark Paper Report",
    "",
    "## 1. Executive Summary: Resolving the Domain Blindspot",
    "",
    "Comprehensive head-to-head comparison of **Model 5 (Review-Only Baseline)** versus **Model 5-MultiDomain (Tri-Domain Contrastive Alignment)** across the 4 ACL 2024 MAGE quadrants on 6,000 held-out Kazakh texts:",
    "- **Q1**: Seen Domain x Seen Generator (Reviews x Sherkala-7B)",
    "- **Q2**: Seen Domain x Unseen Generator (Reviews x Qwen-2.5-7B)",
    "- **Q3**: Unseen Domain x Seen Generator (News/Wiki x Sherkala-7B) — *Primary Evaluation of Domain Blindspot Resolution*",
    "- **Q4 (Wild)**: Unseen Domain x Unseen Generator (News/Wiki x Qwen-2.5-7B)",
    "",
    "| Model | Q1 (Seen x Seen) | Q2 (Unseen Gen) | Q3 (Unseen Dom) | Q4 Wild (Unseen Both) | $\\Delta \\text{AUC}_{\\text{gen}}$ | $\\Delta \\text{AUC}_{\\text{dom}}$ | $\\Delta \\text{AUC}_{\\text{wild}}$ |",
    "|---|---|---|---|---|---|---|---|"
]

for name, res in all_model_results.items():
    q = res["quadrants"]
    d = res["degradation"]
    report_lines.append(
        f"| **{name}** | {q['Q1']['roc_auc']:.4f} ({q['Q1']['f1']*100:.1f}%) | "
        f"{q['Q2']['roc_auc']:.4f} ({q['Q2']['f1']*100:.1f}%) | "
        f"{q['Q3']['roc_auc']:.4f} ({q['Q3']['f1']*100:.1f}%) | "
        f"**{q['Q4']['roc_auc']:.4f} ({q['Q4']['f1']*100:.1f}%)** | "
        f"`{d['delta_auc_generator']:+.4f}` | `{d['delta_auc_domain']:+.4f}` | `**{d['delta_auc_wild']:+.4f}**` |"
    )

report_lines.extend([
    "",
    "## 2. Dynamic Gate Routing Dynamics across Domains",
    "",
    "Empirical routing weight $\\mathbf{g}$ distribution across informal consumer reviews vs. formal long-form prose (News and Wikipedia):",
    "",
    "| Model | Reviews $\\bar{g}$ | News $\\bar{g}$ | Wikipedia $\\bar{g}$ | $\\Delta g_{\\text{news}}$ | $\\Delta g_{\\text{wiki}}$ | Routing Adaptation |",
    "|---|---|---|---|---|---|---|"
])

for name, res in all_model_results.items():
    gd = res["gate_dynamics"]
    mg = gd["mean_gate_by_domain"]
    r_g = mg.get("consumer_reviews", 0.0)
    n_g = mg.get("news", 0.0)
    w_g = mg.get("wikipedia", 0.0)
    d_n = gd.get("delta_gate_news", n_g - r_g)
    d_w = gd.get("delta_gate_wiki", w_g - r_g)
    adaptation = "Semantic adaptation on formal prose" if (d_n is not None and d_n > 0) else "Balanced routing"
    report_lines.append(
        f"| **{name}** | `{r_g:.3f}` | `{n_g:.3f}` | `{w_g:.3f}` | `{d_n:+.3f}` | `{d_w:+.3f}` | {adaptation} |"
    )

report_lines.extend([
    "",
    "## 3. Individual Model Diagnostic Breakdowns",
    ""
])

for name, r_md in all_model_reports.items():
    report_lines.append(r_md)

final_report_md = "\n".join(report_lines)

# Save Outputs to output/ and ./
os.makedirs("output", exist_ok=True)
for p in ["output/kaz_multi_domain_benchmark_results.json", "kaz_multi_domain_benchmark_results.json"]:
    with open(p, "w", encoding="utf-8") as f:
        json.dump(full_benchmark_results, f, ensure_ascii=False, indent=2)
print("Saved output/kaz_multi_domain_benchmark_results.json")

for p in ["output/kaz_multi_domain_paper_report.md", "kaz_multi_domain_paper_report.md"]:
    with open(p, "w", encoding="utf-8") as f:
        f.write(final_report_md)
print("Saved output/kaz_multi_domain_paper_report.md")

print("\n" + "=" * 80)
print("KAZ-MULTIDOMAIN REMOTE BENCHMARK EVALUATION COMPLETE")
print("=" * 80)
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build self-contained Kaggle multi-domain kernel")
    parser.add_argument("--output", type=str, default=None, help="Target output kernel path")
    parser.add_argument("--dry_run", action="store_true", help="Build fast verification kernel with mini slice")
    args = parser.parse_args()

    build_multi_domain_kernel(output_path=args.output, dry_run=args.dry_run)
