"""
Supervised Contrastive Learning Loss Module.

Reference:
    Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    https://arxiv.org/abs/2004.11362
"""

import sys
import os

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    _BaseLoss = nn.Module
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False
    _BaseLoss = object


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


class SupConLoss(_BaseLoss):
    """
    Supervised Contrastive Learning loss (Khosla et al., NeurIPS 2020).
    Pulls representations of samples belonging to the same class together,
    while pushing representations of samples from different classes apart.

    Args:
        temperature (float): Temperature scaling factor tau for logits (default: 0.07).
        contrast_mode (str): 'all' or 'one' (default: 'all').
        base_temperature (float): Base temperature for scaling gradient (default: 0.07).
    """

    def __init__(self, temperature: float = 0.07, contrast_mode: str = "all", base_temperature: float = 0.07):
        if HAS_TORCH:
            super().__init__()
        self.temperature = float(temperature)
        self.contrast_mode = contrast_mode
        self.base_temperature = float(base_temperature)

    def forward(self, features, labels=None, mask=None):
        """
        Compute SupCon loss for model features.

        Args:
            features (torch.Tensor): Hidden representations of shape [batch_size, dim]
                                     or [batch_size, n_views, dim].
            labels (torch.Tensor, optional): Ground truth labels of shape [batch_size].
            mask (torch.Tensor, optional): Contrastive mask of shape [batch_size, batch_size],
                                           mask_{i, j}=1 if sample j has the same class as sample i.

        Returns:
            torch.Tensor: A scalar contrastive loss.
        """
        if not HAS_TORCH:
            # Graceful fallback when PyTorch is not available locally
            return _DummyLoss(0.0)

        device = features.device

        # Handle 2D [B, D] vs 3D [B, V, D] inputs
        if len(features.shape) < 2:
            raise ValueError(f"`features` needs to be at least 2-dimensional (received shape {features.shape})")

        if len(features.shape) == 2:
            # [B, D] -> [B, 1, D]
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        n_views = features.shape[1]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            # Unsupervised: each view of sample i is positive with all other views of sample i
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(f"Num of labels ({labels.shape[0]}) does not match num of features ({batch_size})")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        # Compute dot products / cosine similarities
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # For numerical stability: subtract max per row (detached so it doesn't affect autograd)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask to match anchor_count and contrast_count
        mask = mask.repeat(anchor_count, contrast_count)

        # Mask-out self-contrast (the diagonal where sample compares with itself)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        # Compute log probabilities: log( exp(z_i * z_p / tau) / sum_{a != i} exp(z_i * z_a / tau) )
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positive pairs per anchor
        mask_pos_pairs = mask.sum(1)
        # Avoid division by zero if an anchor has no positive pairs in the batch
        has_pos = (mask_pos_pairs > 0).float()
        safe_mask_pos = torch.where(mask_pos_pairs > 0, mask_pos_pairs, torch.ones_like(mask_pos_pairs))

        mean_log_prob_pos = (mask * log_prob).sum(1) / safe_mask_pos

        # Loss: - (temperature / base_temperature) * mean(mean_log_prob_pos)
        if has_pos.sum() > 0:
            loss = - (self.temperature / self.base_temperature) * ((mean_log_prob_pos * has_pos).sum() / has_pos.sum())
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        return loss

    if not HAS_TORCH:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)
