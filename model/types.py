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
    projection_logits: torch.Tensor
    projection_probs: torch.Tensor
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
    auxiliary_loss: float
    entropy: float
    smoothness_loss: float
    labeled_in_batch: int
    rl_enabled: bool
    manifold_supervised_loss: float = 0.0
    projection_supervised_loss: float = 0.0
    compactness_loss: float = 0.0
    separation_loss: float = 0.0
    temporal_consistency_loss: float = 0.0
    projection_compactness_loss: float = 0.0
    projection_separation_loss: float = 0.0
    projection_temporal_loss: float = 0.0
    within_class_var_z: float = 0.0
    between_class_var_z: float = 0.0
    fisher_ratio_z: float = 0.0
    within_class_var_m: float = 0.0
    between_class_var_m: float = 0.0
    fisher_ratio_m: float = 0.0
    balanced_accuracy: float = 0.0
    macro_f1: float = 0.0
    top2_accuracy: float = 0.0
    negative_log_likelihood: float = 0.0
    brier_score: float = 0.0
    expected_calibration_error: float = 0.0


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
    game_prompt_id: int | None = None
    game_target_class: int | None = None
    game_in_window: bool = False
    game_hit: bool = False
    game_label_correct: bool = False
    game_timing_hit: bool = False
    game_timing_error_s: float | None = None
    game_seconds_to_window_start: float | None = None
    game_next_target_class: int | None = None
    game_seconds_to_next_prompt_start: float | None = None
    game_prompt_progress: float | None = None
    game_margin: float | None = None
    game_level: int | None = None
    game_streak: int | None = None
    game_reward_components: dict[str, float] | None = None
