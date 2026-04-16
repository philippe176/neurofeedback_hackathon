from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ModelConfig:
    # Network
    input_dim: int | None = None
    hidden_dim: int = 128
    embedding_dim: int = 32
    projection_dim: int = 2
    n_classes: int = 4
    dropout: float = 0.10

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # Online updates
    buffer_size: int = 4000
    batch_size: int = 128
    update_every: int = 10
    min_buffer_before_updates: int = 256
    warmup_labeled_samples: int = 200
    reward_baseline_alpha: float = 0.05

    # Objective weights
    supervised_weight: float = 1.0
    policy_weight: float = 0.20
    entropy_weight: float = 0.01
    smoothness_weight: float = 0.02

    # Programmatic reward weights
    reward_confidence_weight: float = 0.35
    reward_correct_weight: float = 0.35
    reward_class_scale_weight: float = 0.20
    reward_strategy_quality_weight: float = 0.10
    reward_min: float = 0.0
    reward_max: float = 1.0

    # Streaming
    host: str = "localhost"
    port: int = 5555
    embedding_key: str = "data"
    queue_capacity: int = 2048
    receiver_timeout_ms: int = 500

    # Visualization
    viz_enabled: bool = True
    viz_history: int = 300
    viz_draw_every: int = 2
    viz_ema_alpha: float = 0.25

    # Runtime
    heartbeat_every: int = 20
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.projection_dim not in (2, 3):
            raise ValueError("projection_dim must be 2 or 3")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        if self.update_every < 1:
            raise ValueError("update_every must be >= 1")

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
