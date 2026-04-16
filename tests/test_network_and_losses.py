import numpy as np
import torch

from model.losses import (
    entropy_regularization,
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
    assert out.penultimate.shape == (5, 8)
    assert out.projection.shape == (5, 3)

    row_sums = out.probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


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


def test_temporal_smoothness_single_point_is_zero() -> None:
    single = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    loss = temporal_smoothness_loss(single)
    assert float(loss.item()) == 0.0
