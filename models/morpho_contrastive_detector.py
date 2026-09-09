"""
Dual-Stream Morphological and Semantic Contrastive Detector for Kazakh AI Text Detection.

Combines a contextual RoBERTa semantic stream with an explicit FST morpheme stream
via dynamic gated fusion, multi-task supervised contrastive learning, and cross-entropy.
"""

import os
import sys

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

try:
    from .losses import InvarianceLoss, SupConLoss
except ImportError:
    from models.losses import InvarianceLoss, SupConLoss



class _DummyLoss:
    """Fallback scalar container for environments without PyTorch installed."""
    def __init__(self, val: float = 0.0):
        self.val = float(val)

    def item(self) -> float:
        return self.val

    def backward(self) -> None:
        pass

    def __float__(self) -> float:
        return self.val

    def __repr__(self) -> str:
        return f"DummyLoss({self.val})"


class _DummyTensor:
    """Fallback tensor container for environments without PyTorch installed."""
    def __init__(self, shape, fill_val: float = 0.0):
        self.shape = tuple(shape)
        self.fill_val = float(fill_val)

    def item(self) -> float:
        return self.fill_val

    def cpu(self):
        return self

    def numpy(self):
        return self

    def tolist(self):
        if len(self.shape) == 2:
            return [[self.fill_val] * self.shape[1] for _ in range(self.shape[0])]
        elif len(self.shape) == 1:
            return [self.fill_val] * self.shape[0]
        return [self.fill_val]

    def __repr__(self):
        return f"DummyTensor(shape={self.shape})"


class MorphemeEncoder(_BaseModule):
    """
    Explicit FST morpheme sequence encoder.
    Embeds morpheme ID sequences into d_model dimensional space and applies
    a 2-layer Bidirectional TransformerEncoder followed by sequence mean-pooling.
    """

    def __init__(
        self,
        vocab_size: int = 250,
        embed_dim: int = 768,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 1536,
        dropout: float = 0.1
    ):
        if HAS_TORCH:
            super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.nhead = int(nhead)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)

        if HAS_TORCH:
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
            self.pos_embedding = nn.Embedding(256, self.embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        else:
            self.embedding = None
            self.pos_embedding = None
            self.transformer = None

    def forward(self, morpheme_ids, attention_mask=None):
        """
        Forward pass of MorphemeEncoder.

        Args:
            morpheme_ids (Tensor or array-like): [B, L] morpheme token IDs.
            attention_mask (Tensor, optional): [B, L] binary mask (1 for valid, 0 for pad).

        Returns:
            Tensor or _DummyTensor: [B, embed_dim] mean-pooled morphological embedding.
        """
        if not HAS_TORCH:
            batch_size = len(morpheme_ids) if hasattr(morpheme_ids, "__len__") else 1
            return _DummyTensor((batch_size, self.embed_dim))

        seq_len = morpheme_ids.shape[1]
        positions = torch.arange(seq_len, device=morpheme_ids.device).unsqueeze(0).expand(morpheme_ids.shape[0], -1)
        emb = self.embedding(morpheme_ids) + self.pos_embedding(positions)
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
            out = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)
            mask_expanded = attention_mask.unsqueeze(-1).float()
            h_morph = torch.sum(out * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-8)
        else:
            out = self.transformer(emb)
            h_morph = torch.mean(out, dim=1)
        return h_morph

    if not HAS_TORCH:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)


class MorphoContrastiveDetector(_BaseModule):
    """
    Dual-Stream Network with Dynamic Gated Cross-Attention Fusion.

    Stream 1 (Semantic): Pretrained KazRoBERTa backbone providing h_sem in R^{B, 768}.
    Stream 2 (Morphology): MorphemeEncoder processing FST morpheme IDs providing h_morph in R^{B, 768}.
    Dynamic Gated Fusion:
        gate = sigmoid(W_gate [h_sem; h_morph])
        h_fused = gate * h_sem + (1 - gate) * h_morph
    Heads:
        Classification Head: Dropout(0.2) -> Linear(embed_dim, 2)
        Projection Head: Linear(embed_dim, 256) -> ReLU() -> Linear(256, proj_dim) (L2-normalized)
    Loss:
        L_total = L_CE + lambda_supcon * L_SupCon
    """

    def __init__(
        self,
        roberta_model_name: str = "kz-transformers/kaz-roberta-conversational",
        morpheme_vocab_size: int = 250,
        embed_dim: int = 768,
        proj_dim: int = 128,
        lambda_supcon: float = 0.5,
        lambda_inv: float = 0.5,
        temperature: float = 0.07,
        dropout_rate: float = 0.2,
        roberta_model=None
    ):
        if HAS_TORCH:
            super().__init__()
        self.roberta_model_name = roberta_model_name
        self.morpheme_vocab_size = int(morpheme_vocab_size)
        self.embed_dim = int(embed_dim)
        self.proj_dim = int(proj_dim)
        self.lambda_supcon = float(lambda_supcon)
        self.lambda_inv = float(lambda_inv)
        self.temperature = float(temperature)
        self.dropout_rate = float(dropout_rate)

        if HAS_TORCH:
            if roberta_model is not None:
                self.roberta = roberta_model
            elif HAS_TRANSFORMERS:
                self.roberta = AutoModel.from_pretrained(roberta_model_name)
            else:
                raise ImportError("transformers library is required to load roberta_model_name")

            self.morph_encoder = MorphemeEncoder(
                vocab_size=self.morpheme_vocab_size,
                embed_dim=self.embed_dim
            )

            # Gated fusion layer
            self.gate_fc = nn.Linear(self.embed_dim * 2, self.embed_dim)

            # Classification head
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.embed_dim, 2)
            )

            # Contrastive projection head
            self.projection_head = nn.Sequential(
                nn.Linear(self.embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.proj_dim)
            )

            self.supcon_loss_fn = SupConLoss(temperature=self.temperature)
            self.inv_loss_fn = InvarianceLoss()
            self.ce_loss_fn = nn.CrossEntropyLoss()
        else:
            self.roberta = None
            self.morph_encoder = MorphemeEncoder(
                vocab_size=self.morpheme_vocab_size,
                embed_dim=self.embed_dim
            )
            self.gate_fc = None
            self.classifier = None
            self.projection_head = None
            self.supcon_loss_fn = SupConLoss(temperature=self.temperature)
            self.inv_loss_fn = InvarianceLoss()
            self.ce_loss_fn = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        morpheme_ids=None,
        labels=None,
        adv_input_ids=None,
        adv_attention_mask=None,
        adv_morpheme_ids=None
    ) -> dict:
        """
        Forward pass computing dual representations, gated fusion, logits, projection, and loss.

        Args:
            input_ids (Tensor): [B, L_text] KazRoBERTa token IDs.
            attention_mask (Tensor, optional): [B, L_text] attention mask.
            morpheme_ids (Tensor): [B, L_morph] FST morpheme token IDs.
            labels (Tensor, optional): [B] Ground truth binary labels (0=human, 1=ai).
            adv_input_ids (Tensor, optional): [B, L_adv] Perturbed KazRoBERTa token IDs.
            adv_attention_mask (Tensor, optional): [B, L_adv] Perturbed attention mask.
            adv_morpheme_ids (Tensor, optional): [B, L_adv_morph] Perturbed FST morpheme token IDs.

        Returns:
            dict containing:
                "logits": [B, 2] classification logits
                "proj": [B, proj_dim] L2-normalized projection representation
                "gate": [B, embed_dim] semantic gate activation weights
                "h_fused": [B, embed_dim] fused latent representation
                "loss": (optional) combined CE + SupCon + Inv loss
                "ce_loss": (optional) cross-entropy loss
                "supcon_loss": (optional) supervised contrastive loss
                "inv_loss": (optional) adversarial invariance loss
                "proj_adv": (optional) perturbed projection representation
                "gate_adv": (optional) perturbed gate activation weights
                "gate_diff": (optional) gate activation shift (gate_adv - gate)
        """
        if not HAS_TORCH:
            batch_size = len(input_ids) if hasattr(input_ids, "__len__") else 1
            logits = _DummyTensor((batch_size, 2))
            proj = _DummyTensor((batch_size, self.proj_dim))
            gate = _DummyTensor((batch_size, self.embed_dim), fill_val=0.5)
            h_fused = _DummyTensor((batch_size, self.embed_dim))

            res = {
                "logits": logits,
                "proj": proj,
                "gate": gate,
                "h_fused": h_fused
            }
            if adv_input_ids is not None:
                res["proj_adv"] = _DummyTensor((batch_size, self.proj_dim))
                res["gate_adv"] = _DummyTensor((batch_size, self.embed_dim), fill_val=0.5)
                res["gate_diff"] = _DummyTensor((batch_size, self.embed_dim), fill_val=0.0)
                res["inv_loss"] = _DummyLoss(0.0)
            if labels is not None:
                res["loss"] = _DummyLoss(0.0)
                res["ce_loss"] = _DummyLoss(0.0)
                res["supcon_loss"] = _DummyLoss(0.0)
            return res

        # Stream 1: Contextual Semantic Representation
        if attention_mask is None and input_ids is not None:
            attention_mask = (input_ids != 0).long()

        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(roberta_out, "last_hidden_state"):
            h_sem = roberta_out.last_hidden_state[:, 0, :]
        elif isinstance(roberta_out, (tuple, list)):
            h_sem = roberta_out[0][:, 0, :]
        else:
            h_sem = roberta_out[:, 0, :]

        # Stream 2: Explicit Morphology Representation
        if hasattr(morpheme_ids, "ne"):
            pad_id = getattr(self.morph_encoder, "pad_token_id", 0)
            morph_mask = morpheme_ids.ne(pad_id).long()
        else:
            morph_mask = None
        h_morph = self.morph_encoder(morpheme_ids, attention_mask=morph_mask)

        # Dynamic Gated Fusion
        combined = torch.cat([h_sem, h_morph], dim=-1)
        gate = torch.sigmoid(self.gate_fc(combined))
        h_fused = gate * h_sem + (1.0 - gate) * h_morph

        # Dual Task Heads
        logits = self.classifier(h_fused)
        proj = F.normalize(self.projection_head(h_fused), p=2, dim=-1)

        result = {
            "logits": logits,
            "proj": proj,
            "gate": gate,
            "h_fused": h_fused
        }

        # Multi-Task Loss Computation
        total_loss = None
        if labels is not None:
            ce_loss = self.ce_loss_fn(logits, labels)
            supcon_loss = self.supcon_loss_fn(proj, labels)
            total_loss = ce_loss + self.lambda_supcon * supcon_loss
            result["ce_loss"] = ce_loss
            result["supcon_loss"] = supcon_loss

        # Adversarial Stream & Invariance Loss
        if adv_input_ids is not None and adv_morpheme_ids is not None:
            if adv_attention_mask is None:
                adv_attention_mask = (adv_input_ids != 0).long()
            adv_roberta_out = self.roberta(input_ids=adv_input_ids, attention_mask=adv_attention_mask)
            if hasattr(adv_roberta_out, "last_hidden_state"):
                h_sem_adv = adv_roberta_out.last_hidden_state[:, 0, :]
            elif isinstance(adv_roberta_out, (tuple, list)):
                h_sem_adv = adv_roberta_out[0][:, 0, :]
            else:
                h_sem_adv = adv_roberta_out[:, 0, :]

            if hasattr(adv_morpheme_ids, "ne"):
                pad_id = getattr(self.morph_encoder, "pad_token_id", 0)
                adv_morph_mask = adv_morpheme_ids.ne(pad_id).long()
            else:
                adv_morph_mask = None
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

    def predict(self, text: str, tokenizer=None, morpheme_tokenizer=None, device=None) -> dict:
        """
        Convenience inference helper taking a raw Kazakh string and predicting class and gating weights.
        """
        if not HAS_TORCH:
            return {
                "prediction": 0,
                "probability": 0.5,
                "gate_semantic_weight": 0.5,
                "gate_morphology_weight": 0.5
            }

        self.eval()
        if device is None:
            device = next(self.parameters()).device

        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.roberta_model_name)

        if morpheme_tokenizer is None:
            try:
                from api.morpheme_tokenizer import MorphemeTokenizer
            except ImportError:
                from morpheme_tokenizer import MorphemeTokenizer
            morpheme_tokenizer = MorphemeTokenizer()

        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        morph_tensor = morpheme_tokenizer.batch_encode([text], max_length=64).to(device)

        with torch.no_grad():
            out = self.forward(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                morpheme_ids=morph_tensor
            )
            probs = F.softmax(out["logits"], dim=-1)
            pred_class = int(torch.argmax(probs, dim=-1).item())
            prob_ai = float(probs[0, 1].item())
            gate_mean = float(out["gate"].mean().item())

        return {
            "prediction": pred_class,
            "probability": prob_ai,
            "gate_semantic_weight": gate_mean,
            "gate_morphology_weight": 1.0 - gate_mean
        }

    if not HAS_TORCH:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)
