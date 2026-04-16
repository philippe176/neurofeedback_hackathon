import numpy as np
import torch

from model.losses import (
    centroid_separation_loss,
    class_conditional_temporal_loss,
    compactness_loss,
    entropy_regularization,
    geometry_statistics,
    reward_weighted_policy_loss,
    supervised_classification_loss,
    temporal_smoothness_loss,
)
from model.network import MovementDecoder


def test_decoder_output_shapes_and_probability_simplex() -> None:
    model = MovementDecoder(input_dim=16, hidden_dim=32, embedding_dim=8, n_classes=4, projection_dim=3)
    x = torch.randn(5, 16)
    out = model(x)

    assert out.logits.shape == (5, 4)
    assert out.probs.shape == (5, 4)
    assert out.projection_logits.shape == (5, 4)
    assert out.projection_probs.shape == (5, 4)
    assert out.penultimate.shape == (5, 8)
    assert out.projection.shape == (5, 3)

    row_sums = out.probs.sum(dim=1)
    proj_row_sums = out.projection_probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    assert torch.allclose(proj_row_sums, torch.ones_like(proj_row_sums), atol=1e-6)


def test_supervised_loss_ignores_unlabeled_rows() -> None:
    logits = torch.tensor(
        [[3.0, 0.5, -1.0, -1.0], [0.1, 0.2, 0.3, 0.4], [0.5, 2.0, -0.2, 0.1]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, -1, 1], dtype=torch.int64)

    loss = supervised_classification_loss(logits, labels)
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0


def test_policy_entropy_and_smoothness_losses_are_finite() -> None:
    logits = torch.randn(6, 4)
    actions = torch.tensor([0, 1, 2, 3, 1, 0], dtype=torch.int64)
    advantages = torch.tensor([0.3, -0.2, 0.8, 0.1, -0.5, 0.4], dtype=torch.float32)
    projections = torch.randn(6, 2)

    policy = reward_weighted_policy_loss(logits, actions, advantages)
    entropy = entropy_regularization(logits)
    smooth = temporal_smoothness_loss(projections)

    assert torch.isfinite(policy)
    assert torch.isfinite(entropy)
    assert torch.isfinite(smooth)


def test_supervised_manifold_losses_are_finite() -> None:
    embeddings = torch.tensor(
        [
            [0.90, 0.10],
            [0.85, 0.15],
            [0.12, 0.86],
            [0.10, 0.92],
            [0.55, 0.52],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1, -1], dtype=torch.int64)

    compact = compactness_loss(embeddings, labels, n_classes=4)
    sep = centroid_separation_loss(embeddings, labels, margin=0.5, n_classes=4)
    temp = class_conditional_temporal_loss(embeddings, labels)
    within, between, fisher = geometry_statistics(embeddings, labels, n_classes=4)

    assert torch.isfinite(compact)
    assert torch.isfinite(sep)
    assert torch.isfinite(temp)
    assert torch.isfinite(within)
    assert torch.isfinite(between)
    assert torch.isfinite(fisher)
    assert float(within.item()) >= 0.0
    assert float(between.item()) >= 0.0


def test_temporal_smoothness_single_point_is_zero() -> None:
    single = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    loss = temporal_smoothness_loss(single)
    assert float(loss.item()) == 0.0


def test_class_conditional_temporal_ignores_label_transitions() -> None:
    embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [2.0, 2.0],
            [2.1, 2.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    loss = class_conditional_temporal_loss(embeddings, labels)
    assert float(loss.item()) < 0.03
