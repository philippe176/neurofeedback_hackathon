from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class StreamSample:
    sample_idx: int
    timestamp: float
    embedding: np.ndarray
    label: int | None
    label_name: str | None
    class_scale: float | None
    strategy_quality: float | None
    difficulty: str | None
    raw: dict[str, Any]


@dataclass(slots=True)
class ModelOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    penultimate: torch.Tensor
    projection: torch.Tensor


@dataclass(slots=True)
class Experience:
    sample_idx: int
    timestamp: float
    embedding: np.ndarray
    label: int
    action: int
    reward: float


@dataclass(slots=True)
class TrainingMetrics:
    update_applied: bool
    total_loss: float
    supervised_loss: float
    policy_loss: float
    entropy: float
    smoothness_loss: float
    labeled_in_batch: int
    rl_enabled: bool


@dataclass(slots=True)
class InferenceStep:
    sample_idx: int
    label: int | None
    predicted_class: int
    confidence: float
    reward: float
    probabilities: np.ndarray
    penultimate: np.ndarray
    projection: np.ndarray
    training: TrainingMetrics | None
