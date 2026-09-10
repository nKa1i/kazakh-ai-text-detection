"""
Domain-Stratified Mini-Batch Sampler for Tri-Domain Contrastive Training.

Partitions dataset indices by (domain, label) groups and constructs balanced
mini-batches ensuring equal domain representation (Reviews, News, Wikipedia)
and balanced class representation (Human vs AI).
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


try:
    import torch
    from torch.utils.data import Sampler
    _TORCH_AVAILABLE = True
except ImportError:
    class Sampler:  # type: ignore[no-redef]
        """Fallback Sampler base class when PyTorch is not installed."""
        def __init__(self, data_source: Optional[Any] = None) -> None:
            pass

        def __class_getitem__(cls, item: Any) -> type:
            return cls

    _TORCH_AVAILABLE = False


class DomainStratifiedBatchSampler(Sampler):
    """
    Mini-batch sampler ensuring balanced domain representation per batch.

    Partitions dataset indices by (domain, label) pairs. For each batch of size B,
    computes quotas per domain (B // num_domains + distributed remainder),
    allocating samples from both labels within each domain.

    Supports deterministic seeding with process-invariant random generators.
    Inherits from torch.utils.data.Sampler if PyTorch is installed, else object.
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
            raise ValueError(
                f"domains and labels must have the same length, got "
                f"{len(domains)} and {len(labels)}."
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        self.domains = list(domains)
        self.labels = list(labels)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        # Unique domains (sorted for deterministic ordering)
        self.unique_domains = sorted(list(set(self.domains)))
        self.num_domains = len(self.unique_domains)

        # Pre-group indices by (domain, label)
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
        """Set epoch for deterministic per-epoch shuffling."""
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

        # Create active pools for this epoch
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
                    # Collect all remaining unused indices across all domains
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

            # Form full balanced batch
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
                # First pass: take according to label quotas
                for lbl in labels_in_d:
                    needed = l_quotas[lbl]
                    avail = len(pools[(dom, lbl)]) - pointers[(dom, lbl)]
                    take = min(needed, avail)
                    ptr = pointers[(dom, lbl)]
                    dom_samples.extend(pools[(dom, lbl)][ptr : ptr + take])
                    pointers[(dom, lbl)] += take

                # Second pass: if shortfall due to label imbalance, take from other labels in domain
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
