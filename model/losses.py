from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import functional as F


def supervised_classification_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Cross-entropy on labeled examples only. Unlabeled entries use label -1."""
    mask = labels >= 0
    if not torch.any(mask):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], labels[mask])


def reward_weighted_policy_loss(logits: Tensor, actions: Tensor, advantages: Tensor) -> Tensor:
    """Policy-style loss using model actions and reward-derived advantages."""
    dist = Categorical(logits=logits)
    log_probs = dist.log_prob(actions)
    return -(advantages.detach() * log_probs).mean()


def entropy_regularization(logits: Tensor) -> Tensor:
    dist = Categorical(logits=logits)
    return dist.entropy().mean()


def temporal_smoothness_loss(projections: Tensor) -> Tensor:
    """Encourage consecutive projected points to move smoothly."""
    if projections.shape[0] < 2:
        return projections.sum() * 0.0
    deltas = projections[1:] - projections[:-1]
    return deltas.pow(2).sum(dim=-1).mean()
