# Morphology-Aware Robust Contrastive Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, evaluate, and benchmark the **Morphology-Aware Robust Contrastive Detector (`Morpho-SupCon KazRoBERTa`)** to eliminate the 59% cross-generator generalization drop (recovering from 36.75% to $\ge 85\%$) and maintain $\le 1\%$ false positive rate on short texts for our Master's Thesis and ACL/EMNLP submission.

**Architecture:** A Dual-Stream neural network combining (1) a KazRoBERTa contextual semantic encoder with (2) a lightweight 2-layer Transformer morpheme encoder over structured FST affixes ($V_M \approx 250$), fused via a dynamic gating mechanism $\mathbf{g} = \sigma(\mathbf{W}_g [\mathbf{h}_{\text{sem}}; \mathbf{h}_{\text{morph}}] + \mathbf{b}_g)$ and optimized with a multi-task Supervised Contrastive Loss ($\mathcal{L}_{\text{SupCon}}$) + Cross-Entropy ($\mathcal{L}_{\text{CE}}$) objective.

**Tech Stack:** Python 3.10+, PyTorch 2.x, Hugging Face Transformers (`kz-transformers/kaz-roberta-conversational`), Datasets, Scikit-learn, Scipy, Apertium/FST morphology engine (`fst_analyzer.py`), Kaggle GPU runner.

## Global Constraints
- Target venues: ACL / EMNLP / CCF-A, MSc Thesis (Chapters 3 & 4).
- Spec: `specs/2026-09-08-morphology-aware-contrastive-detector-design.md`.
- Save all plans to `plans/` and specs to `specs/`. Exclude agentic directories (`agent_engine/`, `docs/`, `.superpowers/`).
- Code must be tested with TDD (unit test, verify fail, implement, verify pass).
- Must run cleanly both locally and in Kaggle GPU Dual T4 environment (`dauletanekesh/kazakh-gpu-runner-nb`).

---

### Task 1: Morpheme Vocabulary & Sequence Tokenizer

**Files:**
- Create: `api/morpheme_tokenizer.py`
- Test: `tests/test_morpheme_tokenizer.py`

**Interfaces:**
- Consumes: `AdvancedKazakhFSTAnalyzer` from `fst_analyzer.py`.
- Produces: `MorphemeTokenizer` class with:
  - `encode(text: str) -> list[int]`
  - `batch_encode(texts: list[str], max_length: int = 64) -> torch.Tensor`
  - `vocab: dict[str, int]` ($V_M \approx 250$ tokens)
  - `pad_token_id: int`, `unk_token_id: int`

- [ ] **Step 1: Write the failing unit test for MorphemeTokenizer**

Create `tests/test_morpheme_tokenizer.py`:
```python
import unittest

class TestMorphemeTokenizer(unittest.TestCase):
    def test_morpheme_vocab_and_tokenization(self):
        from api.morpheme_tokenizer import MorphemeTokenizer
        tok = MorphemeTokenizer()
        self.assertGreater(len(tok.vocab), 50)
        self.assertIn("<PAD>", tok.vocab)
        self.assertIn("<UNK>", tok.vocab)
        self.assertIn("-лар", tok.vocab)
        self.assertIn("-ден", tok.vocab)
        self.assertIn("-сы", tok.vocab)

        sample = "Каспиден доставкасы өте тез болды"
        ids = tok.encode(sample)
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)

        # Batch encoding with padding
        batch = [sample, "жақсы"]
        tensor = tok.batch_encode(batch, max_length=16)
        self.assertEqual(tensor.shape, (2, 16))
        self.assertEqual(tensor[1, -1].item(), tok.pad_token_id)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_morpheme_tokenizer.py`  
Expected: `ModuleNotFoundError: No module named 'api.morpheme_tokenizer'`

- [ ] **Step 3: Implement MorphemeTokenizer**

Create `api/morpheme_tokenizer.py`:
```python
import re
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fst_analyzer import AdvancedKazakhFSTAnalyzer

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<ROOT>", "<LOAN>", "<NOMINAL>", "<VERBAL>", "<EOS>"]

class MorphemeTokenizer:
    def __init__(self):
        self.fst = AdvancedKazakhFSTAnalyzer()
        self.vocab = {}
        self._build_vocab()
        self.pad_token_id = self.vocab["<PAD>"]
        self.unk_token_id = self.vocab["<UNK>"]

    def _build_vocab(self):
        idx = 0
        for tok in SPECIAL_TOKENS:
            self.vocab[tok] = idx
            idx += 1

        # Collect all explicit affixes from the FST analyzer
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

    def tokenize_text(self, text: str) -> list[str]:
        """Runs FST segmentation and extracts morpheme tokens."""
        segmented = self.fst.analyze_and_segment(text)
        tokens = []
        for word in segmented.split():
            if word.startswith("-"):
                tokens.append(word)
            else:
                tokens.append("<ROOT>")
        return tokens

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize_text(text)
        return [self.vocab.get(t, self.unk_token_id) for t in tokens]

    def batch_encode(self, texts: list[str], max_length: int = 64):
        import torch
        batch_ids = []
        for t in texts:
            ids = self.encode(t)[:max_length]
            if len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
            batch_ids.append(ids)
        return torch.tensor(batch_ids, dtype=torch.long)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_morpheme_tokenizer.py`  
Expected: `Ran 1 test in ... OK`

- [ ] **Step 5: Commit Task 1**

```bash
git add api/morpheme_tokenizer.py tests/test_morpheme_tokenizer.py
git commit -m "feat: implement closed-vocabulary MorphemeTokenizer for FST affixes"
```

---

### Task 2: Multi-Task Supervised Contrastive Loss

**Files:**
- Create: `models/losses.py`
- Test: `tests/test_supcon_loss.py`

**Interfaces:**
- Consumes: Latent feature vectors $\mathbf{z} \in \mathbb{R}^{B \times d}$, ground-truth labels $\mathbf{y} \in \{0, 1\}^B$.
- Produces: `SupConLoss(temperature=0.07)` class computing the normalized contrastive loss with positive mask pooling.

- [x] **Step 1: Write the failing unit test for SupConLoss**

Create `tests/test_supcon_loss.py`:
```python
import unittest
import torch

class TestSupConLoss(unittest.TestCase):
    def test_loss_computation_and_gradients(self):
        from models.losses import SupConLoss
        criterion = SupConLoss(temperature=0.07)
        features = torch.randn(8, 128, requires_grad=True)
        # Normalize features
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])

        loss = criterion(features, labels)
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0.0)

        # Verify gradient backprop
        loss.backward()
        self.assertIsNotNone(features.grad)

    def test_identical_features_minimize_loss(self):
        from models.losses import SupConLoss
        criterion = SupConLoss(temperature=0.07)
        # Perfect clustering
        f1 = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        f2 = torch.tensor([[-1.0, 0.0], [-1.0, 0.0]])
        features = torch.cat([f1, f2], dim=0)
        labels = torch.tensor([0, 0, 1, 1])
        loss_perfect = criterion(features, labels)

        # Disordered clustering
        features_random = torch.randn(4, 2)
        features_random = torch.nn.functional.normalize(features_random, p=2, dim=1)
        loss_random = criterion(features_random, labels)
        self.assertLess(loss_perfect.item(), loss_random.item())

if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_supcon_loss.py`  
Expected: `ModuleNotFoundError: No module named 'models.losses'`

- [x] **Step 3: Implement SupConLoss**

Create `models/losses.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning loss (Khosla et al., NeurIPS 2020).
    Pulls samples with identical class labels together, pushes opposing classes apart.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute cosine similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # Numerical stability via max subtraction
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log-probs
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss
```

- [x] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_supcon_loss.py`  
Expected: `Ran 2 tests in ... OK`

- [x] **Step 5: Commit Task 2**

```bash
git add models/losses.py tests/test_supcon_loss.py
git commit -m "feat: implement Supervised Contrastive Loss (SupConLoss)"
```

---

### Task 3: Dual-Stream Network with Gated Cross-Attention Fusion

**Files:**
- Create: `models/morpho_contrastive_detector.py`
- Test: `tests/test_morpho_detector.py`

**Interfaces:**
- Consumes:
  - `input_ids`, `attention_mask` (from KazRoBERTa tokenizer)
  - `morpheme_ids` (from `MorphemeTokenizer`)
- Produces: `MorphoContrastiveDetector(nn.Module)` with:
  - Forward output: `{"logits": Tensor(B, 2), "proj": Tensor(B, 128), "gate": Tensor(B, 768), "loss": Optional[Tensor]}`
  - `predict(text: str) -> dict`

- [ ] **Step 1: Write the failing unit test for MorphoContrastiveDetector**

Create `tests/test_morpho_detector.py`:
```python
import unittest
import torch

class TestMorphoDetector(unittest.TestCase):
    def test_forward_pass_and_shapes(self):
        from models.morpho_contrastive_detector import MorphoContrastiveDetector
        # Instantiate with dummy small config for testing
        model = MorphoContrastiveDetector(
            roberta_model_name="kz-transformers/kaz-roberta-conversational",
            morpheme_vocab_size=100,
            embed_dim=768,
            proj_dim=128
        )
        batch_size = 4
        input_ids = torch.randint(0, 1000, (batch_size, 16))
        attention_mask = torch.ones((batch_size, 16))
        morpheme_ids = torch.randint(0, 100, (batch_size, 16))
        labels = torch.tensor([0, 1, 0, 1])

        out = model(input_ids, attention_mask, morpheme_ids, labels=labels)
        self.assertIn("logits", out)
        self.assertIn("proj", out)
        self.assertIn("gate", out)
        self.assertIn("loss", out)

        self.assertEqual(out["logits"].shape, (batch_size, 2))
        self.assertEqual(out["proj"].shape, (batch_size, 128))
        self.assertEqual(out["gate"].shape, (batch_size, 768))
        self.assertGreater(out["loss"].item(), 0.0)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_morpho_detector.py`  
Expected: `ModuleNotFoundError: No module named 'models.morpho_contrastive_detector'`

- [ ] **Step 3: Implement MorphoContrastiveDetector**

Create `models/morpho_contrastive_detector.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from models.losses import SupConLoss

class MorphemeEncoder(nn.Module):
    """Lightweight 2-layer Bidirectional Transformer encoding morpheme sequences."""
    def __init__(self, vocab_size: int, embed_dim: int = 768, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, morpheme_ids: torch.Tensor) -> torch.Tensor:
        # morpheme_ids: (B, L)
        emb = self.embedding(morpheme_ids)
        out = self.transformer(emb)
        # Mean-pooling over sequence
        h_morph = torch.mean(out, dim=1)
        return h_morph

class MorphoContrastiveDetector(nn.Module):
    def __init__(
        self,
        roberta_model_name: str = "kz-transformers/kaz-roberta-conversational",
        morpheme_vocab_size: int = 250,
        embed_dim: int = 768,
        proj_dim: int = 128,
        lambda_supcon: float = 0.5,
        temperature: float = 0.07
    ):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        self.morph_encoder = MorphemeEncoder(morpheme_vocab_size, embed_dim=embed_dim)

        # Gated fusion network
        self.gate_fc = nn.Linear(embed_dim * 2, embed_dim)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 2)
        )

        # Contrastive Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )

        self.supcon_loss_fn = SupConLoss(temperature=temperature)
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.lambda_supcon = lambda_supcon

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        morpheme_ids: torch.Tensor,
        labels: torch.Tensor = None
    ) -> dict:
        # Stream 1: Semantic
        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use mean pooling or [CLS]
        h_sem = roberta_out.last_hidden_state[:, 0, :]

        # Stream 2: Morpheme
        h_morph = self.morph_encoder(morpheme_ids)

        # Dynamic Gated Fusion
        combined = torch.cat([h_sem, h_morph], dim=-1)
        gate = torch.sigmoid(self.gate_fc(combined))
        h_fused = gate * h_sem + (1.0 - gate) * h_morph

        # Heads
        logits = self.classifier(h_fused)
        proj = F.normalize(self.projection_head(h_fused), p=2, dim=-1)

        result = {
            "logits": logits,
            "proj": proj,
            "gate": gate,
            "h_fused": h_fused
        }

        if labels is not None:
            ce_loss = self.ce_loss_fn(logits, labels)
            supcon_loss = self.supcon_loss_fn(proj, labels)
            total_loss = ce_loss + self.lambda_supcon * supcon_loss
            result["loss"] = total_loss
            result["ce_loss"] = ce_loss
            result["supcon_loss"] = supcon_loss

        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_morpho_detector.py`  
Expected: `Ran 1 test in ... OK`

- [ ] **Step 5: Commit Task 3**

```bash
git add models/morpho_contrastive_detector.py tests/test_morpho_detector.py
git commit -m "feat: implement MorphoContrastiveDetector dual-stream architecture with gated fusion"
```

---

### Task 4: Evaluation & Threshold Calibration Suite

**Files:**
- Create: `scripts/metrics_evaluator.py`
- Test: `tests/test_metrics_evaluator.py`

**Interfaces:**
- Consumes: Ground-truth `y_true`, predictions `y_pred`, probabilities `y_prob`, lengths `lengths`.
- Produces: Complete dictionary containing ROC-AUC, EER, Youden's J optimal threshold, Accuracy, F1, Length-Stratified FPR/FNR, and McNemar test.

- [x] **Step 1: Write the failing unit test for metrics_evaluator**

Create `tests/test_metrics_evaluator.py`:
```python
import unittest
import numpy as np

class TestMetricsEvaluator(unittest.TestCase):
    def test_metrics_and_roc_auc_calibration(self):
        from scripts.metrics_evaluator import compute_comprehensive_metrics
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 0.95])
        lengths = np.array([40, 50, 70, 90, 45, 55, 75, 100])

        res = compute_comprehensive_metrics(y_true, y_prob, lengths)
        self.assertIn("roc_auc", res)
        self.assertIn("optimal_threshold", res)
        self.assertIn("eer", res)
        self.assertEqual(res["roc_auc"], 1.0)
        self.assertIn("short", res["length_stratification"])
        self.assertEqual(res["length_stratification"]["short"]["fp"], 0)

if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_metrics_evaluator.py`  
Expected: `ModuleNotFoundError: No module named 'scripts.metrics_evaluator'`

- [x] **Step 3: Implement metrics_evaluator**

Create `scripts/metrics_evaluator.py`:
```python
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def compute_comprehensive_metrics(y_true: np.ndarray, y_prob: np.ndarray, lengths: np.ndarray = None, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    
    # 1. Threshold-independent metrics
    try:
        roc_auc = round(float(roc_auc_score(y_true, y_prob)), 4)
        fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_prob)
        # Youden's J
        j_scores = tpr_curve - fpr_curve
        best_idx = np.argmax(j_scores)
        optimal_threshold = round(float(thresholds[best_idx]), 4)
        # EER
        fnr_curve = 1 - tpr_curve
        eer_idx = np.nanargmin(np.abs(fpr_curve - fnr_curve))
        eer = round(float(fpr_curve[eer_idx]), 4)
    except Exception:
        roc_auc, optimal_threshold, eer = 0.5, threshold, 0.5

    # 2. Standard metrics at fixed threshold
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)

    summary = {
        "accuracy": round(float(acc) * 100, 2),
        "f1": round(float(f1) * 100, 2),
        "precision": round(float(prec) * 100, 2),
        "recall": round(float(rec) * 100, 2),
        "roc_auc": roc_auc,
        "optimal_threshold": optimal_threshold,
        "eer": eer,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": round(float(fpr) * 100, 2),
        "fnr": round(float(fnr) * 100, 2)
    }

    # 3. Length stratification
    if lengths is not None:
        strat = {}
        brackets = {
            "short": lengths <= 60,
            "medium": (lengths > 60) & (lengths <= 85),
            "long": lengths > 85
        }
        for b_name, mask in brackets.items():
            if mask.sum() > 0:
                y_t = y_true[mask]
                y_p = y_pred[mask]
                b_cm = confusion_matrix(y_t, y_p, labels=[0, 1])
                b_tn, b_fp, b_fn, b_tp = b_cm.ravel() if b_cm.shape == (2, 2) else (0, 0, 0, 0)
                strat[b_name] = {
                    "total": int(mask.sum()),
                    "accuracy": round(float(accuracy_score(y_t, y_p)) * 100, 2),
                    "fp": int(b_fp),
                    "fn": int(b_fn),
                    "fpr": round(float(b_fp / max(1, b_fp + b_tn)) * 100, 2),
                    "fnr": round(float(b_fn / max(1, b_fn + b_tp)) * 100, 2)
                }
        summary["length_stratification"] = strat

    return summary
```

- [x] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_metrics_evaluator.py`  
Expected: `Ran 1 test in ... OK`

- [x] **Step 5: Commit Task 4**

```bash
git add scripts/metrics_evaluator.py tests/test_metrics_evaluator.py
git commit -m "feat: implement ROC-AUC and threshold calibration metrics evaluator"
```

---

### Task 5: End-to-End Training & LOGO Benchmark Pipeline

**Files:**
- Create: `scripts/train_morpho_contrastive.py`
- Test: `tests/test_training_pipeline.py`

**Interfaces:**
- Consumes: `df_train` (KazSAnDRA + Sherkala), `df_test` (KazSAnDRA + Qwen-2.5-7B).
- Produces: Trained model weights, predictions CSV, and comparative benchmark report.

- [ ] **Step 1: Write integration test for dataset formatting and dual-view batching**

Create `tests/test_training_pipeline.py`:
```python
import unittest
import torch

class TestTrainingPipeline(unittest.TestCase):
    def test_dual_view_batch_construction(self):
        from api.morpheme_tokenizer import MorphemeTokenizer
        tok = MorphemeTokenizer()
        texts = ["Бұл жақсы өнім", "Сапасы төмен"]
        morphemes = tok.batch_encode(texts, max_length=16)
        self.assertEqual(morphemes.shape, (2, 16))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m unittest tests/test_training_pipeline.py`  
Expected: `Ran 1 test in ... OK`

- [ ] **Step 3: Implement training script**

Create `scripts/train_morpho_contrastive.py`:
- Integrates data loading (`nKa1i/kazakh-ai-detect` train split).
- Implements `train_epoch` with multi-task loss ($\mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{SupCon}}$).
- Implements `evaluate_model` with `compute_comprehensive_metrics`.
- Saves all artifacts and metrics to `data/`.

- [ ] **Step 4: Commit Task 5**

```bash
git add scripts/train_morpho_contrastive.py tests/test_training_pipeline.py
git commit -m "feat: implement multi-task contrastive training pipeline"
```

---

### Task 6: Kaggle GPU Remote Runner Bundle & Benchmark Execution

**Files:**
- Modify: `kaggle_runner/diagnostic_kernel.py`
- Script: `scripts/prepare_morpho_kernel.py`
- Run: `kaggle kernels push -p kaggle_runner`

**Interfaces:**
- Consumes: Dual Tesla T4 GPUs on Kaggle (`gpu_t4_x2`).
- Produces: Complete 5-model ablation matrix, ROC-AUC curves, and publication report in `data/morpho_contrastive_paper_report.md`.

- [ ] **Step 1: Write kernel generator script** `scripts/prepare_morpho_kernel.py` embedding `MorphoContrastiveDetector`, `MorphemeTokenizer`, and `kazakh_aigc_paired_2k.json`.
- [ ] **Step 2: Compile-check generated kernel** with `python -m py_compile kaggle_runner/diagnostic_kernel.py`.
- [ ] **Step 3: Push to Kaggle and monitor execution** (`dauletanekesh/kazakh-gpu-runner-nb`).
- [ ] **Step 4: Download results and verify ROC-AUC / OOD recovery**.
- [ ] **Step 5: Commit Task 6 artifacts and paper report**.

```bash
git add kaggle_runner/ specs/ data/
git commit -m "feat: complete Morpho-SupCon benchmark execution and ablation report"
```
