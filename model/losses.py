from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import functional as F


def supervised_classification_loss(
    logits: Tensor,
    labels: Tensor,
    class_weights: Tensor | None = None,
    focal_gamma: float | None = None,
) -> Tensor:
    """
    Cross-entropy on labeled examples only.

    Unlabeled entries use label -1. Optional focal modulation can be enabled
    by setting focal_gamma > 0.
    """
    mask = labels >= 0
    if not torch.any(mask):
        return logits.sum() * 0.0

    logits_l = logits[mask]
    labels_l = labels[mask]

    if focal_gamma is None or focal_gamma <= 0.0:
        return F.cross_entropy(logits_l, labels_l, weight=class_weights)

    ce = F.cross_entropy(logits_l, labels_l, weight=class_weights, reduction="none")
    probs = torch.softmax(logits_l, dim=-1)
    p_t = probs.gather(dim=1, index=labels_l.unsqueeze(1)).squeeze(1)
    modulating = (1.0 - p_t).pow(float(focal_gamma))
    return (modulating * ce).mean()


def class_centroids(
    embeddings: Tensor,
    labels: Tensor,
    n_classes: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Return class centroids and a boolean mask for classes observed in the batch."""
    mask = labels >= 0
    if not torch.any(mask):
        n = int(n_classes or 1)
        zeros = embeddings.new_zeros((n, embeddings.shape[-1]))
        present = torch.zeros(n, dtype=torch.bool, device=embeddings.device)
        return zeros, present

    labels_l = labels[mask]
    emb_l = embeddings[mask]
    if n_classes is None:
        n_classes = int(labels_l.max().item()) + 1
    n_classes = max(1, int(n_classes))

    centroids = embeddings.new_zeros((n_classes, embeddings.shape[-1]))
    present = torch.zeros(n_classes, dtype=torch.bool, device=embeddings.device)
    for cls in range(n_classes):
        cls_mask = labels_l == cls
        if torch.any(cls_mask):
            centroids[cls] = emb_l[cls_mask].mean(dim=0)
            present[cls] = True
    return centroids, present


def compactness_loss(
    embeddings: Tensor,
    labels: Tensor,
    n_classes: int | None = None,
) -> Tensor:
    """Intra-class compactness: mean squared distance to class centroid."""
    mask = labels >= 0
    if not torch.any(mask):
        return embeddings.sum() * 0.0

    centroids, _ = class_centroids(embeddings, labels, n_classes=n_classes)
    emb_l = embeddings[mask]
    labels_l = labels[mask]
    targets = centroids[labels_l]
    return (emb_l - targets).pow(2).sum(dim=-1).mean()


def centroid_separation_loss(
    embeddings: Tensor,
    labels: Tensor,
    margin: float,
    n_classes: int | None = None,
) -> Tensor:
    """Inter-class centroid margin loss: penalize centroids that are too close."""
    centroids, present = class_centroids(embeddings, labels, n_classes=n_classes)
    active = centroids[present]
    if active.shape[0] < 2:
        return embeddings.sum() * 0.0

    dists = torch.cdist(active, active, p=2)
    idx = torch.triu_indices(dists.shape[0], dists.shape[1], offset=1)
    pair_dists = dists[idx[0], idx[1]]
    if pair_dists.numel() == 0:
        return embeddings.sum() * 0.0

    violations = torch.relu(float(margin) - pair_dists)
    return (violations.pow(2)).mean()


def class_conditional_temporal_loss(embeddings: Tensor, labels: Tensor) -> Tensor:
    """Temporal smoothness only across consecutive labeled samples of the same class."""
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0

    same = (labels[1:] == labels[:-1]) & (labels[1:] >= 0) & (labels[:-1] >= 0)
    if not torch.any(same):
        return embeddings.sum() * 0.0

    deltas = embeddings[1:] - embeddings[:-1]
    sq = deltas.pow(2).sum(dim=-1)
    return sq[same].mean()


def geometry_statistics(
    embeddings: Tensor,
    labels: Tensor,
    n_classes: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Return within-class variance, between-class variance, and Fisher ratio.

    Computed on labeled samples only.
    """
    mask = labels >= 0
    if not torch.any(mask):
        zero = embeddings.sum() * 0.0
        return zero, zero, zero

    emb_l = embeddings[mask]
    labels_l = labels[mask]
    global_center = emb_l.mean(dim=0)

    if n_classes is None:
        n_classes = int(labels_l.max().item()) + 1

    within_sum = embeddings.new_zeros(())
    between_sum = embeddings.new_zeros(())
    total = embeddings.new_zeros(())

    for cls in range(int(max(1, n_classes))):
        cls_mask = labels_l == cls
        if not torch.any(cls_mask):
            continue
        cls_points = emb_l[cls_mask]
        cls_n = float(cls_points.shape[0])
        cls_center = cls_points.mean(dim=0)

        within_sum = within_sum + (cls_points - cls_center).pow(2).sum()
        between_sum = between_sum + cls_n * (cls_center - global_center).pow(2).sum()
        total = total + cls_n

    denom = torch.clamp(total, min=1.0)
    within_var = within_sum / denom
    between_var = between_sum / denom
    fisher = between_var / (within_var + 1e-8)
    return within_var, between_var, fisher


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
