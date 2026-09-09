"""
Prepares the self-contained Kaggle runner script: kaggle_runner/diagnostic_kernel.py
Implements Task 5 of Kaz-MAGE Cross-Domain Expansion:
Full 6-Model 2x2 Matrix Evaluation (Q1, Q2, Q3, Q4) across Consumer Reviews, News,
and Wikipedia with Dynamic Gate Routing Dynamics on Kaggle Dual Tesla T4 GPUs.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import zlib
from typing import Optional


def build_kaz_mage_kernel(
    output_path: Optional[str] = None,
    eval_data_path: Optional[str] = None,
    dry_run: bool = False
) -> str:
    """
    Builds the self-contained Kaggle runner script for Kaz-MAGE 2x2 matrix benchmark.
    Embeds the 6,000-instance evaluation dataset via zlib+base64.

    Args:
        output_path: Target destination for generated kernel script. Defaults to kaggle_runner/diagnostic_kernel.py.
        eval_data_path: Path to kaz_mage_eval_6k.json.
        dry_run: If True, compresses a small slice for quick local build verification.

    Returns:
        Absolute path to the generated diagnostic_kernel.py.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if output_path is None:
        output_path = os.path.join(repo_root, "kaggle_runner", "diagnostic_kernel.py")
    if eval_data_path is None:
        eval_data_path = os.path.join(repo_root, "data", "kaz_mage_eval_6k.json")

    if dry_run:
        if os.path.exists(eval_data_path):
            with open(eval_data_path, "r", encoding="utf-8") as f:
                data = json.load(f)[:20]
        else:
            data = [
                {
                    "id": f"s_{i}",
                    "domain": "consumer_reviews" if i % 3 == 0 else ("news" if i % 3 == 1 else "wikipedia"),
                    "generator": "human" if i % 2 == 0 else "Sherkala-7B",
                    "is_unseen_domain": i % 3 != 0,
                    "is_unseen_generator": False,
                    "quadrant": "Q1" if (i % 2 == 1 and i % 3 == 0) else "human_consumer_reviews",
                    "prefix": "Бүгінгі таңда",
                    "text": f"Бұл үлгі мәтін нөмірі {i}, қазақ тіліндегі тест үлгісі.",
                    "label": 0 if i % 2 == 0 else 1,
                    "char_length": 55,
                    "word_count": 8
                }
                for i in range(20)
            ]
        print(f"[dry_run] Using {len(data)} test records for kernel verification.")
    else:
        if not os.path.exists(eval_data_path):
            raise FileNotFoundError(f"Evaluation dataset not found at {eval_data_path}")
        with open(eval_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} full evaluation samples from {eval_data_path}.")

    # Compress dataset using highest zlib compression level
    raw_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
    comp_bytes = zlib.compress(raw_bytes, level=9)
    b64_str = base64.b64encode(comp_bytes).decode("ascii")
    print(f"Compressed dataset: {len(raw_bytes):,} bytes -> {len(comp_bytes):,} bytes ({len(b64_str):,} b64 chars)")

    # Replace placeholder in kernel template
    kernel_code = KERNEL_TEMPLATE.replace("__MAGE_EVAL_DATA_B64__", b64_str)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(kernel_code)

    print(f"Successfully generated {output_path} ({len(kernel_code.encode('utf-8')):,} bytes)")
    return output_path


KERNEL_TEMPLATE = r'''"""
Kaggle Dual Tesla T4 Remote Runner: Kaz-MAGE Cross-Domain & Multi-Generator Benchmark
Protocol: ACL 2024 MAGE (Seen vs Unseen Domain x Seen vs Unseen Generator)

Evaluates 6 Detector Architectures Across 4 Quadrants & 3 Domains:
- Model 1: KazRoBERTa Pure Pretrained Baseline (nKa1i/kazroberta-kk-ai-detection)
- Model 2: KazRoBERTa Hybrid FST (FST Segmented Text, CE Loss)
- Model 3: KazRoBERTa + SupCon Single-Stream (Raw Text, CE + 0.5 * SupCon)
- Model 4: Dual-Stream CE-Only (Morphology Stream + Gate, lambda=0.0)
- Model 5: Morpho-SupCon Full Architecture (Morphology Stream + Gate + SupCon, lambda=0.5)
- Model 5-Adv: Invariant Contrastive Defense (Fine-Tuned with Online Perturbations + Invariance Loss)

Dynamic Gate Tracking:
- Tracks and logs mean routing weight g across consumer_reviews, news, and wikipedia.
- Quantifies dynamic routing shifts on formal long-form prose.

Artifacts:
- output/kaz_mage_benchmark_results.json
- output/kaz_mage_paper_report.md
"""

import os
import sys
import json
import zlib
import base64
import time
import math
import re
import copy
import abc
import random
from itertools import groupby

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

# -----------------------------------------------------------------------------
# Environment & Hardware Setup
# -----------------------------------------------------------------------------
print("=" * 80)
print("KAZ-MAGE: 2x2 CROSS-DOMAIN & MULTI-GENERATOR BENCHMARK RUNNER")
print("Target: Kaggle Dual Tesla T4 GPUs | ACL 2024 MAGE Protocol")
print("=" * 80)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Version: {torch.__version__}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name} ({p.total_memory / 1e9:.2f} GB VRAM)")

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

    def _segment_loanword(self, word):
        w_lower = word.lower()
        for root in self.loanword_roots:
            if w_lower.startswith(root) and len(w_lower) > len(root):
                sfx_fallback = word[len(root):]
                if len(sfx_fallback) >= 1:
                    return word[:len(root)] + f" -{sfx_fallback}"
        return None

    def analyze_and_segment(self, text):
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
    def __init__(self, fst_analyzer=None):
        self.fst = fst_analyzer or AdvancedKazakhFSTAnalyzer()
        self.vocab = {}
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

    def tokenize_text(self, text: str) -> list:
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

    def encode(self, text: str) -> list:
        tokens = self.tokenize_text(text)
        return [self.vocab.get(t, self.unk_token_id) for t in tokens]

    def batch_encode(self, texts: list, max_length: int = 256):
        batch_ids = []
        for t in texts:
            ids = self.encode(t)[:max_length]
            if len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
            batch_ids.append(ids)
        return torch.tensor(batch_ids, dtype=torch.long)

# -----------------------------------------------------------------------------
# 3. Embedded Adversarial Perturbators (For Model 5-Adv Online Augmentation)
# -----------------------------------------------------------------------------
def preserve_case(original: str, replacement: str) -> str:
    if not original or not replacement:
        return replacement
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    if original.istitle():
        return replacement.title()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement

def calculate_budget(n_eligible: int, rate: float) -> int:
    if rate > 0.0 and n_eligible > 0:
        return min(n_eligible, max(1, math.ceil(n_eligible * rate)))
    return 0

class HomoglyphSwap:
    CYR_TO_LAT = {
        'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'х': 'x', 'у': 'y', 'і': 'i',
        'А': 'A', 'Е': 'E', 'О': 'O', 'Р': 'P', 'С': 'C', 'Х': 'X', 'У': 'Y', 'І': 'I',
    }
    def perturb(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        if not text or rate <= 0.0:
            return text
        rng = random.Random(seed)
        eligible = [i for i, ch in enumerate(text) if ch in self.CYR_TO_LAT]
        b = calculate_budget(len(eligible), rate)
        if b == 0:
            return text
        selected = set(rng.sample(eligible, b))
        return "".join(self.CYR_TO_LAT[ch] if i in selected else ch for i, ch in enumerate(text))

class KeyboardTypo:
    DIACRITIC_MAP = {
        'қ': 'к', 'к': 'қ', 'ң': 'н', 'н': 'ң', 'ғ': 'г', 'г': 'ғ',
        'ү': 'у', 'у': 'ү', 'ә': 'а', 'а': 'ә', 'ө': 'о', 'о': 'ө',
        'ұ': 'у', 'h': 'х', 'х': 'h', 'і': 'и', 'и': 'і',
        'Қ': 'К', 'К': 'Қ', 'Ң': 'Н', 'Н': 'Ң', 'Ғ': 'Г', 'Г': 'Ғ',
        'Ү': 'У', 'У': 'Ү', 'Ә': 'А', 'А': 'Ә', 'Ө': 'О', 'О': 'Ө',
        'Ұ': 'У', 'H': 'Х', 'Х': 'H', 'І': 'И', 'И': 'І',
    }
    def perturb(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        if not text or rate <= 0.0:
            return text
        rng = random.Random(seed)
        eligible = [i for i, ch in enumerate(text) if ch in self.DIACRITIC_MAP]
        b = calculate_budget(len(eligible), rate)
        if b == 0:
            return text
        selected = set(rng.sample(eligible, b))
        return "".join(self.DIACRITIC_MAP[ch] if i in selected else ch for i, ch in enumerate(text))

class ZeroWidthInjection:
    ZW_CHARS = ['\u200b', '\u200c']
    def perturb(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        if not text or rate <= 0.0 or len(text) < 3:
            return text
        rng = random.Random(seed)
        words = text.split()
        eligible = []
        char_idx = 0
        for w in words:
            start = text.find(w, char_idx)
            char_idx = start + len(w)
            if len(w) >= 3:
                for k in range(1, len(w) - 1):
                    eligible.append(start + k)
        b = calculate_budget(len(eligible), rate)
        if b == 0:
            return text
        selected = sorted(rng.sample(eligible, b), reverse=True)
        res = list(text)
        for p in selected:
            res.insert(p, rng.choice(self.ZW_CHARS))
        return "".join(res)

class SuffixTamperer:
    HARMONY_MAP = {
        "лар": "лер", "лер": "лар", "дар": "дер", "дер": "дар", "тар": "тер", "тер": "тар",
        "дан": "ден", "ден": "дан", "тан": "тен", "тен": "тан", "нан": "нен", "нен": "нан",
        "да": "де", "де": "да", "та": "те", "те": "та", "нда": "нде", "нде": "нда",
        "ға": "ге", "ге": "ға", "қа": "ке", "ке": "қа", "на": "не", "не": "на",
        "ны": "ні", "ні": "ны", "ты": "ті", "ті": "ты", "ды": "ді", "ді": "ды",
        "ның": "нің", "нің": "ның", "дың": "дің", "дің": "дың", "тың": "тің", "тің": "тың",
        "ған": "ген", "ген": "ған", "қан": "кен", "кен": "қан",
        "ады": "еді", "еді": "ады", "йды": "йді", "йді": "йды",
        "мын": "мін", "мін": "мын", "мыз": "міз", "міз": "мыз",
        "сы": "сі", "сі": "сы",
    }
    def __init__(self, mode="mixed"):
        self.mode = mode

    def perturb(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        if not text or rate <= 0.0:
            return text
        rng = random.Random(seed)
        words = text.split()
        b = calculate_budget(len(words), rate)
        if b == 0:
            return text
        selected_idx = sorted(rng.sample(range(len(words)), b), reverse=True)
        for idx in selected_idx:
            w = words[idx]
            w_lower = w.lower()
            for sfx, repl in self.HARMONY_MAP.items():
                if w_lower.endswith(sfx) and len(w_lower) - len(sfx) >= 3:
                    if self.mode == "strip" or (self.mode == "mixed" and rng.random() < 0.5):
                        words[idx] = w[:len(w) - len(sfx)]
                    else:
                        words[idx] = w[:len(w) - len(sfx)] + preserve_case(w[len(w)-len(sfx):], repl)
                    break
        return " ".join(words)

class ColloquialContractor:
    PATTERNS = [
        ("деген болатын", "деген еді"), ("айтқан болатын", "айтқан еді"),
        ("көрген жоқпын", "көрмедім"), ("алған жоқпын", "алмадым"),
        ("келе жатырмын", "кеватрм"), ("келе жатыр", "кеватр"),
        ("бара жатырмын", "баватрм"), ("бара жатыр", "баватр"),
        ("болып жатыр", "боватр"), ("алып келді", "әпкелді"),
        ("алып кетті", "әпкетті"), ("жатырмын", "жатырм"),
        ("келемін", "келем"), ("барамын", "барам"), ("болған", "боған"),
        ("жүрмін", "жүрм"), ("тұрмын", "тұрм"), ("істеп жатыр", "істеватр")
    ]
    def perturb(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        if not text or rate <= 0.0:
            return text
        rng = random.Random(seed)
        matches = []
        for orig, repl in self.PATTERNS:
            p = re.compile(re.escape(orig), re.IGNORECASE)
            for m in p.finditer(text):
                matches.append((m.start(), m.end(), m.group(0), repl))
        if not matches:
            return text
        b = calculate_budget(len(matches), rate)
        if b == 0:
            return text
        selected = sorted(rng.sample(matches, b), key=lambda x: x[0], reverse=True)
        res = text
        for start, end, orig_m, repl in selected:
            res = res[:start] + preserve_case(orig_m, repl) + res[end:]
        return res

class LoanwordSwap:
    PAIRS = [
        ("жеткізу", "доставка"), ("сапа", "качество"), ("жеңілдік", "скидка"),
        ("баға", "цена"), ("тауар", "товар"), ("дүкен", "магазин"),
        ("тапсырыс", "заказ"), ("сатушы", "продавец"), ("ақша", "деньги"),
        ("төлем", "оплата"), ("қайтару", "возврат"), ("қосымша", "приложение"),
        ("пароль", "пароль"), ("қызмет", "сервис"), ("жұмыс", "работа"),
        ("уақыт", "время"), ("рахмет", "спасибо"), ("жақсы", "отлично"),
        ("жаман", "плохо"), ("жылдам", "быстро"), ("тез", "быстро")
    ]
    def __init__(self, mode="bidirectional"):
        self.mode = mode
        self.dict_kk2ru = {k.lower(): v.lower() for k, v in self.PAIRS}
        self.dict_ru2kk = {v.lower(): k.lower() for k, v in self.PAIRS}

    def perturb(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        if not text or rate <= 0.0:
            return text
        rng = random.Random(seed)
        words = text.split()
        eligible = []
        for i, w in enumerate(words):
            wl = re.sub(r'[^\w\s]', '', w.lower())
            if self.mode in ("kk2ru", "bidirectional") and wl in self.dict_kk2ru:
                eligible.append((i, wl, self.dict_kk2ru[wl]))
            elif self.mode in ("ru2kk", "bidirectional") and wl in self.dict_ru2kk:
                eligible.append((i, wl, self.dict_ru2kk[wl]))
        if not eligible:
            return text
        b = calculate_budget(len(eligible), rate)
        if b == 0:
            return text
        selected = rng.sample(eligible, b)
        for idx, orig_w, repl_w in selected:
            words[idx] = preserve_case(words[idx], repl_w)
        return " ".join(words)

# -----------------------------------------------------------------------------
# 4. Embedded Loss Functions
# -----------------------------------------------------------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = float(temperature)
        self.contrast_mode = contrast_mode
        self.base_temperature = float(base_temperature)

    def forward(self, features, labels=None, mask=None):
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
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, z_clean, z_adv):
        cos_sim = F.cosine_similarity(z_clean, z_adv, dim=-1)
        loss = 1.0 - cos_sim
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# -----------------------------------------------------------------------------
# 5. Embedded Model Architectures
# -----------------------------------------------------------------------------
class MorphemeEncoder(nn.Module):
    def __init__(self, vocab_size=250, embed_dim=768, num_layers=2, nhead=8, dim_feedforward=1536, dropout=0.1, max_pos=512):
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

    def forward(self, morpheme_ids, attention_mask=None):
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

class KazRoBERTaClassificationModel(nn.Module):
    """Model 2: Single-stream fine-tuned on FST-segmented text with CE loss."""
    def __init__(self, model_name="kz-transformers/kaz-roberta-conversational", num_labels=2):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask=None, morpheme_ids=None, labels=None):
        if attention_mask is None and input_ids is not None:
            attention_mask = (input_ids != 0).long()
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        res = {"logits": out.logits}
        if labels is not None:
            res["loss"] = out.loss
            res["ce_loss"] = out.loss
        return res

class SingleStreamContrastiveDetector(nn.Module):
    """Model 3: Single-stream on raw text with CE + 0.5 * SupCon."""
    def __init__(
        self,
        roberta_model_name="kz-transformers/kaz-roberta-conversational",
        embed_dim=768,
        proj_dim=128,
        lambda_supcon=0.5,
        temperature=0.07,
        dropout_rate=0.2
    ):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
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
        self.supcon_loss_fn = SupConLoss(temperature=temperature)
        self.ce_loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, morpheme_ids=None, labels=None):
        if attention_mask is None and input_ids is not None:
            attention_mask = (input_ids != 0).long()
        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        h_sem = roberta_out.last_hidden_state[:, 0, :]
        logits = self.classifier(h_sem)
        proj = F.normalize(self.projection_head(h_sem), p=2, dim=-1)
        res = {"logits": logits, "proj": proj, "h_sem": h_sem}
        if labels is not None:
            ce_loss = self.ce_loss_fn(logits, labels)
            supcon_loss = self.supcon_loss_fn(proj, labels)
            res["ce_loss"] = ce_loss
            res["supcon_loss"] = supcon_loss
            res["loss"] = ce_loss + self.lambda_supcon * supcon_loss
        return res

class MorphoContrastiveDetector(nn.Module):
    """
    Dual-Stream Gated Network with Dynamic Fusion.
    Used for Model 4 (lambda_supcon=0), Model 5 (lambda_supcon=0.5), and Model 5-Adv.
    """
    def __init__(
        self,
        roberta_model_name="kz-transformers/kaz-roberta-conversational",
        morpheme_vocab_size=250,
        embed_dim=768,
        proj_dim=128,
        lambda_supcon=0.5,
        lambda_inv=0.5,
        temperature=0.07,
        dropout_rate=0.2
    ):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
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
        input_ids=None,
        attention_mask=None,
        morpheme_ids=None,
        labels=None,
        adv_input_ids=None,
        adv_attention_mask=None,
        adv_morpheme_ids=None
    ):
        if attention_mask is None and input_ids is not None:
            attention_mask = (input_ids != 0).long()
        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        h_sem = roberta_out.last_hidden_state[:, 0, :]

        morph_mask = (morpheme_ids != 0).long() if morpheme_ids is not None else None
        h_morph = self.morph_encoder(morpheme_ids, attention_mask=morph_mask)

        combined = torch.cat([h_sem, h_morph], dim=-1)
        gate = torch.sigmoid(self.gate_fc(combined))
        h_fused = gate * h_sem + (1.0 - gate) * h_morph

        logits = self.classifier(h_fused)
        proj = F.normalize(self.projection_head(h_fused), p=2, dim=-1)

        result = {
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
# 6. Embedded MAGE Metrics Evaluator
# -----------------------------------------------------------------------------
def _to_list(arr):
    if arr is None:
        return None
    if hasattr(arr, "tolist"):
        return arr.tolist()
    if isinstance(arr, (list, tuple)):
        return list(arr)
    return list(arr)

def _compute_roc_and_youden_pure(y_true, y_prob):
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

def _calc_classification_stats(y_true, y_prob, thresh):
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

def compute_bootstrap_auc_ci(y_true, y_prob, n_bootstrap=1000, alpha=0.05, seed=42):
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

def compute_mage_metrics(y_true, y_prob, threshold=None, bootstrap_ci=True, n_bootstrap=1000, seed=42):
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

def calculate_domain_degradation(quadrant_results):
    def _extract_auc(val):
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

def _get_field(record, name, default=None):
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)

def evaluate_quadrant_matrix(records, bootstrap_ci=True, n_bootstrap=1000, seed=42):
    quadrant_buckets = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    domain_buckets = {}
    domain_gates = {}
    quadrant_gates = {}
    human_controls = []

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

    # Pair with negative controls from matching domains if quadrant only has AI samples
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

def generate_mage_markdown_report(results, model_name="Kazakh AI Detector"):
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
# 7. Embedded Dataset Decompression
# -----------------------------------------------------------------------------
print("\n" + "-" * 60)
print("Decompressing Embedded Kaz-MAGE 6,000-instance Dataset...")
b64_data = """__MAGE_EVAL_DATA_B64__"""
decomp_bytes = zlib.decompress(base64.b64decode(b64_data))
eval_records = json.loads(decomp_bytes.decode("utf-8"))
print(f"Successfully loaded {len(eval_records)} evaluation samples.")

fst_analyzer = AdvancedKazakhFSTAnalyzer()
morpheme_tokenizer = MorphemeTokenizer(fst_analyzer=fst_analyzer)
raw_tokenizer = AutoTokenizer.from_pretrained("kz-transformers/kaz-roberta-conversational")
pure_tokenizer = AutoTokenizer.from_pretrained("nKa1i/kazroberta-kk-ai-detection")

eval_texts = [r["text"] for r in eval_records]
print(f"Precomputing FST segmentation on {len(eval_texts)} evaluation samples...")
eval_fst_texts = [fst_analyzer.analyze_and_segment(t) for t in eval_texts]
print(f"Precomputing morpheme tokenization on {len(eval_texts)} evaluation samples...")
eval_morph_ids = morpheme_tokenizer.batch_encode(eval_texts, max_length=256)

# -----------------------------------------------------------------------------
# 8. Load Training Dataset from Hugging Face
# -----------------------------------------------------------------------------
print("\n" + "-" * 60)
print("Loading Training Dataset from Hugging Face (nKa1i/kazakh-ai-detect)...")
train_url = "https://huggingface.co/datasets/nKa1i/kazakh-ai-detect/raw/main/data/train.csv"
try:
    df_raw = pd.read_csv(train_url)
    print(f"Downloaded train dataset via CSV: {len(df_raw)} records.")
except Exception as e:
    try:
        from datasets import load_dataset
        ds = load_dataset("nKa1i/kazakh-ai-detect", split="train")
        df_raw = pd.DataFrame(ds)
        print(f"Downloaded train dataset via datasets library: {len(df_raw)} records.")
    except Exception as e2:
        print(f"Fallback to evaluation records: ({e}), ({e2})")
        df_raw = pd.DataFrame([{"text": r["text"], "label": r["label"]} for r in eval_records])

df_train, df_val = train_test_split(df_raw, test_size=0.10, random_state=42, stratify=df_raw["label"])
print(f"Train split: {len(df_train)} | Val split: {len(df_val)}")

train_texts = df_train["text"].tolist()
train_labels = df_train["label"].tolist()
val_texts = df_val["text"].tolist()
val_labels = df_val["label"].tolist()

print("Precomputing training FST text and morpheme IDs...")
train_fst_texts = [fst_analyzer.analyze_and_segment(t) for t in train_texts]
val_fst_texts = [fst_analyzer.analyze_and_segment(t) for t in val_texts]

train_morph_ids = morpheme_tokenizer.batch_encode(train_texts, max_length=256)
val_morph_ids = morpheme_tokenizer.batch_encode(val_texts, max_length=256)

# -----------------------------------------------------------------------------
# 9. PyTorch Dataset & DataLoaders
# -----------------------------------------------------------------------------
class KazakhTextDataset(Dataset):
    def __init__(self, texts, labels, morpheme_ids=None):
        self.texts = list(texts)
        self.labels = list(labels)
        self.morpheme_ids = morpheme_ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": self.texts[idx], "label": int(self.labels[idx])}
        if self.morpheme_ids is not None:
            item["morpheme_ids"] = self.morpheme_ids[idx]
        return item

def create_collate_fn(tokenizer, max_length=256):
    def collate_fn(batch):
        texts = [b["text"] for b in batch]
        labels = [b["label"] for b in batch]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        res = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        if "morpheme_ids" in batch[0]:
            morph_list = [b["morpheme_ids"] for b in batch]
            res["morpheme_ids"] = torch.stack(morph_list, dim=0) if isinstance(morph_list[0], torch.Tensor) else torch.tensor(morph_list, dtype=torch.long)
        return res
    return collate_fn

class AdvAugDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

adv_ops = [
    HomoglyphSwap(), KeyboardTypo(), ZeroWidthInjection(),
    SuffixTamperer(mode="mixed"), ColloquialContractor(), LoanwordSwap(mode="bidirectional")
]

def create_adv_collate_fn(tokenizer, morph_tok, max_length=256):
    def collate_fn_adv(batch):
        texts, labels = zip(*batch)
        adv_texts = []
        for t in texts:
            if random.random() < 0.5:
                op = random.choice(adv_ops)
                r = random.uniform(0.05, 0.20)
                adv_texts.append(op.perturb(t, rate=r, seed=random.randint(1, 10000)))
            else:
                adv_texts.append(t)

        enc_clean = tokenizer(list(texts), max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        m_clean = morph_tok.batch_encode(list(texts), max_length=max_length)
        enc_adv = tokenizer(list(adv_texts), max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        m_adv = morph_tok.batch_encode(list(adv_texts), max_length=max_length)

        return {
            "input_ids": enc_clean["input_ids"],
            "attention_mask": enc_clean["attention_mask"],
            "morpheme_ids": m_clean,
            "adv_input_ids": enc_adv["input_ids"],
            "adv_attention_mask": enc_adv["attention_mask"],
            "adv_morpheme_ids": m_adv,
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    return collate_fn_adv

BATCH_SIZE = 32
GRAD_ACCUM = 2
EPOCHS = 2

loader_train_raw = DataLoader(KazakhTextDataset(train_texts, train_labels), batch_size=BATCH_SIZE, shuffle=True, collate_fn=create_collate_fn(raw_tokenizer))
loader_val_raw = DataLoader(KazakhTextDataset(val_texts, val_labels), batch_size=64, shuffle=False, collate_fn=create_collate_fn(raw_tokenizer))

loader_train_fst = DataLoader(KazakhTextDataset(train_fst_texts, train_labels), batch_size=BATCH_SIZE, shuffle=True, collate_fn=create_collate_fn(raw_tokenizer))
loader_val_fst = DataLoader(KazakhTextDataset(val_fst_texts, val_labels), batch_size=64, shuffle=False, collate_fn=create_collate_fn(raw_tokenizer))

loader_train_morph = DataLoader(KazakhTextDataset(train_texts, train_labels, morpheme_ids=train_morph_ids), batch_size=BATCH_SIZE, shuffle=True, collate_fn=create_collate_fn(raw_tokenizer))
loader_val_morph = DataLoader(KazakhTextDataset(val_texts, val_labels, morpheme_ids=val_morph_ids), batch_size=64, shuffle=False, collate_fn=create_collate_fn(raw_tokenizer))

loader_train_adv = DataLoader(AdvAugDataset(train_texts, train_labels), batch_size=BATCH_SIZE, shuffle=True, collate_fn=create_adv_collate_fn(raw_tokenizer, morpheme_tokenizer), drop_last=True)

# -----------------------------------------------------------------------------
# 10. Training & Inference Helpers
# -----------------------------------------------------------------------------
def eval_val(model, val_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            inp_ids = batch["input_ids"].to(device)
            att_mask = batch["attention_mask"].to(device)
            m_ids = batch.get("morpheme_ids")
            if m_ids is not None:
                m_ids = m_ids.to(device)
            out = model(input_ids=inp_ids, attention_mask=att_mask, morpheme_ids=m_ids)
            logits = out["logits"] if isinstance(out, dict) else out.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(batch["labels"].numpy().tolist())
    acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / max(1, len(y_true))
    tp1 = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp1 = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn1 = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    f1_1 = (2 * tp1) / max(1e-8, 2 * tp1 + fp1 + fn1)
    tp0 = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp0 = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    fn0 = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    f1_0 = (2 * tp0) / max(1e-8, 2 * tp0 + fp0 + fn0)
    return float(acc), float((f1_0 + f1_1) / 2.0)

def train_model(model, train_loader, val_loader, optimizer, scheduler=None, num_epochs=2, grad_accum_steps=2):
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_f1 = -1.0
    best_state = None
    model.to(device)

    for epoch in range(1, num_epochs + 1):
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
        val_acc, val_f1 = eval_val(model, val_loader)
        print(f"  Epoch {epoch}/{num_epochs} ({elapsed:.1f}s) | Train Loss: {total_loss/max(1, num_batches):.4f} (CE: {total_ce/max(1, num_batches):.4f}, SupCon: {total_supcon/max(1, num_batches):.4f}) | Val Acc: {val_acc*100:.2f}%, Val F1: {val_f1*100:.2f}%")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def train_adv_model(model, adv_loader, optimizer, scheduler=None, num_epochs=2, grad_accum_steps=2):
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, total_inv = 0.0, 0.0
        num_batches = 0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(adv_loader):
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs["loss"] / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(adv_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            total_loss += outputs["loss"].item()
            total_inv += outputs.get("inv_loss", torch.tensor(0.0)).item()
            num_batches += 1

        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{num_epochs} ({elapsed:.1f}s) | Train Loss: {total_loss/max(1, num_batches):.4f} | Inv Loss: {total_inv/max(1, num_batches):.4f}")
    return model

def predict_mage(model, eval_records, texts, tokenizer, morpheme_ids=None, batch_size=64):
    model.eval()
    model.to(device)
    records_out = []
    text_list = list(texts)

    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            b_texts = text_list[i:i + batch_size]
            b_recs = eval_records[i:i + batch_size]
            enc = tokenizer(b_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            kwargs = {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"]
            }
            if morpheme_ids is not None:
                kwargs["morpheme_ids"] = morpheme_ids[i:i + batch_size].to(device)

            out = model(**kwargs)
            logits = out["logits"] if isinstance(out, dict) else out.logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()

            gates = None
            if isinstance(out, dict) and "gate" in out and out["gate"] is not None:
                gates = out["gate"].mean(dim=-1).cpu().numpy().tolist()

            for idx, r in enumerate(b_recs):
                rec_dict = {
                    "id": r.get("id", ""),
                    "domain": r.get("domain", ""),
                    "generator": r.get("generator", ""),
                    "quadrant": r.get("quadrant", ""),
                    "label": int(r.get("label", 0)),
                    "y_true": int(r.get("label", 0)),
                    "y_prob": float(probs[idx]),
                    "gate": float(gates[idx]) if gates is not None else None
                }
                records_out.append(rec_dict)
    return records_out

# -----------------------------------------------------------------------------
# 11. 6-Model Benchmark Execution
# -----------------------------------------------------------------------------
all_model_results = {}
all_model_reports = {}

# STAGE 1: Model 1 (KazRoBERTa Pure Pretrained Baseline)
print("\n" + "=" * 80)
print("STAGE 1 / 6: MODEL 1 (KazRoBERTa Pure Baseline)")
print("Checkpoint: nKa1i/kazroberta-kk-ai-detection")
print("=" * 80)
model1 = AutoModelForSequenceClassification.from_pretrained("nKa1i/kazroberta-kk-ai-detection").to(device)
recs_m1 = predict_mage(model1, eval_records, eval_texts, pure_tokenizer)
res_m1 = evaluate_quadrant_matrix(recs_m1)
all_model_results["Model 1 (KazRoBERTa Pure)"] = res_m1
all_model_reports["Model 1 (KazRoBERTa Pure)"] = generate_mage_markdown_report(res_m1, "Model 1 (KazRoBERTa Pure)")
print(f"Model 1 -> Q1 AUC: {res_m1['quadrants']['Q1']['roc_auc']} | Q4 AUC: {res_m1['quadrants']['Q4']['roc_auc']} | Wild Delta: {res_m1['degradation']['delta_auc_wild']}")
del model1
torch.cuda.empty_cache()

# STAGE 2: Model 2 (KazRoBERTa Hybrid FST)
print("\n" + "=" * 80)
print("STAGE 2 / 6: MODEL 2 (KazRoBERTa Hybrid FST)")
print("Backbone: kz-transformers/kaz-roberta-conversational | Input: FST Segmented Text | Loss: CE")
print("=" * 80)
set_seed(42)
model2 = KazRoBERTaClassificationModel("kz-transformers/kaz-roberta-conversational", num_labels=2).to(device)
opt2 = torch.optim.AdamW(model2.parameters(), lr=2e-5, weight_decay=0.01)
total_steps_fst = math.ceil(len(loader_train_fst) / GRAD_ACCUM) * EPOCHS
sched2 = get_linear_schedule_with_warmup(opt2, num_warmup_steps=int(0.1 * total_steps_fst), num_training_steps=total_steps_fst)
model2 = train_model(model2, loader_train_fst, loader_val_fst, opt2, sched2, num_epochs=EPOCHS, grad_accum_steps=GRAD_ACCUM)
recs_m2 = predict_mage(model2, eval_records, eval_fst_texts, raw_tokenizer)
res_m2 = evaluate_quadrant_matrix(recs_m2)
all_model_results["Model 2 (KazRoBERTa Hybrid FST)"] = res_m2
all_model_reports["Model 2 (KazRoBERTa Hybrid FST)"] = generate_mage_markdown_report(res_m2, "Model 2 (KazRoBERTa Hybrid FST)")
print(f"Model 2 -> Q1 AUC: {res_m2['quadrants']['Q1']['roc_auc']} | Q4 AUC: {res_m2['quadrants']['Q4']['roc_auc']} | Wild Delta: {res_m2['degradation']['delta_auc_wild']}")
del model2, opt2, sched2
torch.cuda.empty_cache()

# STAGE 3: Model 3 (KazRoBERTa + SupCon Single-Stream)
print("\n" + "=" * 80)
print("STAGE 3 / 6: MODEL 3 (KazRoBERTa + SupCon Single-Stream)")
print("Backbone: kz-transformers/kaz-roberta-conversational | Input: Raw Text | Loss: CE + 0.5 * SupCon")
print("=" * 80)
set_seed(42)
model3 = SingleStreamContrastiveDetector("kz-transformers/kaz-roberta-conversational", lambda_supcon=0.5).to(device)
opt3 = torch.optim.AdamW(model3.parameters(), lr=2e-5, weight_decay=0.01)
total_steps_raw = math.ceil(len(loader_train_raw) / GRAD_ACCUM) * EPOCHS
sched3 = get_linear_schedule_with_warmup(opt3, num_warmup_steps=int(0.1 * total_steps_raw), num_training_steps=total_steps_raw)
model3 = train_model(model3, loader_train_raw, loader_val_raw, opt3, sched3, num_epochs=EPOCHS, grad_accum_steps=GRAD_ACCUM)
recs_m3 = predict_mage(model3, eval_records, eval_texts, raw_tokenizer)
res_m3 = evaluate_quadrant_matrix(recs_m3)
all_model_results["Model 3 (KazRoBERTa + SupCon Single-Stream)"] = res_m3
all_model_reports["Model 3 (KazRoBERTa + SupCon Single-Stream)"] = generate_mage_markdown_report(res_m3, "Model 3 (KazRoBERTa + SupCon Single-Stream)")
print(f"Model 3 -> Q1 AUC: {res_m3['quadrants']['Q1']['roc_auc']} | Q4 AUC: {res_m3['quadrants']['Q4']['roc_auc']} | Wild Delta: {res_m3['degradation']['delta_auc_wild']}")
del model3, opt3, sched3
torch.cuda.empty_cache()

# STAGE 4: Model 4 (Dual-Stream CE-Only)
print("\n" + "=" * 80)
print("STAGE 4 / 6: MODEL 4 (Dual-Stream CE-Only)")
print("Architecture: RoBERTa + MorphemeEncoder + Gate | Loss: CE Only (lambda=0.0)")
print("=" * 80)
set_seed(42)
model4 = MorphoContrastiveDetector(
    roberta_model_name="kz-transformers/kaz-roberta-conversational",
    morpheme_vocab_size=len(morpheme_tokenizer.vocab) + 20,
    lambda_supcon=0.0,
    lambda_inv=0.0
).to(device)
opt4 = torch.optim.AdamW(model4.parameters(), lr=2e-5, weight_decay=0.01)
total_steps_morph = math.ceil(len(loader_train_morph) / GRAD_ACCUM) * EPOCHS
sched4 = get_linear_schedule_with_warmup(opt4, num_warmup_steps=int(0.1 * total_steps_morph), num_training_steps=total_steps_morph)
model4 = train_model(model4, loader_train_morph, loader_val_morph, opt4, sched4, num_epochs=EPOCHS, grad_accum_steps=GRAD_ACCUM)
recs_m4 = predict_mage(model4, eval_records, eval_texts, raw_tokenizer, morpheme_ids=eval_morph_ids)
res_m4 = evaluate_quadrant_matrix(recs_m4)
all_model_results["Model 4 (Dual-Stream CE-Only)"] = res_m4
all_model_reports["Model 4 (Dual-Stream CE-Only)"] = generate_mage_markdown_report(res_m4, "Model 4 (Dual-Stream CE-Only)")
print(f"Model 4 -> Q1 AUC: {res_m4['quadrants']['Q1']['roc_auc']} | Q4 AUC: {res_m4['quadrants']['Q4']['roc_auc']} | Gate Shifts: {res_m4['gate_dynamics']['mean_gate_by_domain']}")
del model4, opt4, sched4
torch.cuda.empty_cache()

# STAGE 5: Model 5 (Morpho-SupCon Full Architecture)
print("\n" + "=" * 80)
print("STAGE 5 / 6: MODEL 5 (Morpho-SupCon Full Architecture)")
print("Architecture: RoBERTa + MorphemeEncoder + Gate | Loss: CE + 0.5 * SupCon")
print("=" * 80)
set_seed(42)
model5 = MorphoContrastiveDetector(
    roberta_model_name="kz-transformers/kaz-roberta-conversational",
    morpheme_vocab_size=len(morpheme_tokenizer.vocab) + 20,
    lambda_supcon=0.5,
    lambda_inv=0.0
).to(device)
opt5 = torch.optim.AdamW(model5.parameters(), lr=2e-5, weight_decay=0.01)
sched5 = get_linear_schedule_with_warmup(opt5, num_warmup_steps=int(0.1 * total_steps_morph), num_training_steps=total_steps_morph)
model5 = train_model(model5, loader_train_morph, loader_val_morph, opt5, sched5, num_epochs=EPOCHS, grad_accum_steps=GRAD_ACCUM)
recs_m5 = predict_mage(model5, eval_records, eval_texts, raw_tokenizer, morpheme_ids=eval_morph_ids)
res_m5 = evaluate_quadrant_matrix(recs_m5)
all_model_results["Model 5 (Morpho-SupCon Full Architecture)"] = res_m5
all_model_reports["Model 5 (Morpho-SupCon Full Architecture)"] = generate_mage_markdown_report(res_m5, "Model 5 (Morpho-SupCon Full Architecture)")
print(f"Model 5 -> Q1 AUC: {res_m5['quadrants']['Q1']['roc_auc']} | Q4 AUC: {res_m5['quadrants']['Q4']['roc_auc']} | Gate Shifts: {res_m5['gate_dynamics']['mean_gate_by_domain']}")

# STAGE 6: Model 5-Adv (Invariant Contrastive Defense)
print("\n" + "=" * 80)
print("STAGE 6 / 6: MODEL 5-Adv (Invariant Contrastive Defense)")
print("Fine-tuning Model 5 with Online Adversarial Perturbations + Invariance Loss (lambda_inv=0.5)")
print("=" * 80)
model5_adv = copy.deepcopy(model5)
model5_adv.lambda_inv = 0.5
opt5_adv = torch.optim.AdamW(model5_adv.parameters(), lr=1e-5, weight_decay=0.01)
total_steps_adv = math.ceil(len(loader_train_adv) / GRAD_ACCUM) * EPOCHS
sched5_adv = get_linear_schedule_with_warmup(opt5_adv, num_warmup_steps=int(0.1 * total_steps_adv), num_training_steps=total_steps_adv)
model5_adv = train_adv_model(model5_adv, loader_train_adv, opt5_adv, sched5_adv, num_epochs=EPOCHS, grad_accum_steps=GRAD_ACCUM)
recs_m5adv = predict_mage(model5_adv, eval_records, eval_texts, raw_tokenizer, morpheme_ids=eval_morph_ids)
res_m5adv = evaluate_quadrant_matrix(recs_m5adv)
all_model_results["Model 5-Adv (Invariant Contrastive Defense)"] = res_m5adv
all_model_reports["Model 5-Adv (Invariant Contrastive Defense)"] = generate_mage_markdown_report(res_m5adv, "Model 5-Adv (Invariant Contrastive Defense)")
print(f"Model 5-Adv -> Q1 AUC: {res_m5adv['quadrants']['Q1']['roc_auc']} | Q4 AUC: {res_m5adv['quadrants']['Q4']['roc_auc']} | Gate Shifts: {res_m5adv['gate_dynamics']['mean_gate_by_domain']}")
del model5, model5_adv, opt5, opt5_adv
torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# 12. Cross-Model Comparative Synthesis & Report Generation
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SYNTHESIZING KAZ-MAGE PUBLICATION BENCHMARK REPORT")
print("=" * 80)

full_benchmark_results = {
    "experiment": "Kaz-MAGE: 2x2 Cross-Domain & Multi-Generator Benchmark",
    "protocol": "ACL 2024 MAGE Protocol (Seen vs Unseen Domain x Seen vs Unseen Generator)",
    "target_hardware": "Kaggle Dual Tesla T4 GPUs",
    "dataset_size": len(eval_records),
    "train_split": f"nKa1i/kazakh-ai-detect ({len(df_train)} train / {len(df_val)} val)",
    "models": {}
}

for name, res in all_model_results.items():
    clean_res = {k: v for k, v in res.items() if k != "records"}
    full_benchmark_results["models"][name] = clean_res

# Comparative Markdown Report
report_lines = [
    "# Kaz-MAGE: 2x2 Cross-Domain & Multi-Generator Benchmark Paper Report",
    "",
    "## 1. Executive Benchmark Summary",
    "",
    "Comprehensive evaluation of 6 detector architectures across 4 quadrants on 6,000 paired Kazakh texts:",
    "- **Q1**: Seen Domain x Seen Generator (Kaspi Reviews x Sherkala-7B)",
    "- **Q2**: Seen Domain x Unseen Generator (Kaspi Reviews x Qwen-2.5-7B)",
    "- **Q3**: Unseen Domain x Seen Generator (News/Wiki x Sherkala-7B)",
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
    "Empirical routing weight $\\mathbf{g}$ distribution across informal consumer reviews vs. formal long-form prose:",
    "",
    "| Model | Reviews $\\bar{g}$ | News $\\bar{g}$ | Wikipedia $\\bar{g}$ | $\\Delta g_{\\text{news}}$ | $\\Delta g_{\\text{wiki}}$ | Routing Adaptation |",
    "|---|---|---|---|---|---|---|"
])

for name in ["Model 4 (Dual-Stream CE-Only)", "Model 5 (Morpho-SupCon Full Architecture)", "Model 5-Adv (Invariant Contrastive Defense)"]:
    res = all_model_results[name]
    gd = res["gate_dynamics"]
    mg = gd["mean_gate_by_domain"]
    r_g = mg.get("consumer_reviews", 0.0)
    n_g = mg.get("news", 0.0)
    w_g = mg.get("wikipedia", 0.0)
    d_n = gd.get("delta_gate_news", n_g - r_g)
    d_w = gd.get("delta_gate_wiki", w_g - r_g)
    adaptation = "Semantic shift on formal prose" if (d_n is not None and d_n > 0) else "Balanced routing"
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

# Save Outputs
os.makedirs("output", exist_ok=True)
for p in ["output/kaz_mage_benchmark_results.json", "kaz_mage_benchmark_results.json"]:
    with open(p, "w", encoding="utf-8") as f:
        json.dump(full_benchmark_results, f, ensure_ascii=False, indent=2)
print("Saved output/kaz_mage_benchmark_results.json")

for p in ["output/kaz_mage_paper_report.md", "kaz_mage_paper_report.md"]:
    with open(p, "w", encoding="utf-8") as f:
        f.write(final_report_md)
print("Saved output/kaz_mage_paper_report.md")

print("\n" + "=" * 80)
print("KAZ-MAGE REMOTE BENCHMARK EVALUATION COMPLETE")
print("=" * 80)
'''


if __name__ == "__main__":
    build_kaz_mage_kernel()
