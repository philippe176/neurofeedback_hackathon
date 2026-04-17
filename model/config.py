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

    # Optimization - slightly higher LR for faster online adaptation
    lr: float = 8e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # Online updates - tuned for faster real-time adaptation
    buffer_size: int = 2000
    batch_size: int = 64
    update_every: int = 5
    min_buffer_before_updates: int = 64
    warmup_labeled_samples: int = 80
    reward_baseline_alpha: float = 0.08

    # Objective weights
    supervised_weight: float = 1.0
    policy_weight: float = 0.05
    entropy_weight: float = 0.01
    smoothness_weight: float = 0.0

    # ENZO supervised manifold objective.
    # Increased projection weights for faster visual separation learning
    lambda_cls: float = 1.0
    lambda_compact: float = 0.15
    lambda_sep: float = 0.15
    lambda_temp: float = 0.05
    lambda_proj_cls: float = 1.5       # Increased from 0.5 - projection needs strong supervision
    lambda_proj_compact: float = 0.20  # Increased from 0.05 - tighter clusters
    lambda_proj_sep: float = 0.25      # Increased from 0.05 - push clusters apart
    lambda_proj_temp: float = 0.02
    contrastive_weight: float = 0.10
    contrastive_temperature: float = 0.50

    latent_sep_margin: float = 2.0
    projection_sep_margin: float = 1.2
    classification_focal_gamma: float | None = 1.0  # Focal loss helps focus on hard examples
    class_weights: tuple[float, ...] | None = None

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
    viz_method: str = "neural"
    viz_fit_window: int = 300
    viz_refit_every: int = 10
    viz_use_penultimate: bool = True
    viz_tsne_perplexity: float = 20.0

    # Runtime
    heartbeat_every: int = 20
    device: str = "auto"

    def __post_init__(self) -> None:
        allowed_viz_methods = {"neural", "pca", "lda", "tsne", "umap"}
        if self.projection_dim not in (2, 3):
            raise ValueError("projection_dim must be 2 or 3")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        if self.update_every < 1:
            raise ValueError("update_every must be >= 1")
        if self.viz_method not in allowed_viz_methods:
            raise ValueError(f"viz_method must be one of {sorted(allowed_viz_methods)}")
        if self.viz_fit_window < 10:
            raise ValueError("viz_fit_window must be at least 10")
        if self.viz_refit_every < 1:
            raise ValueError("viz_refit_every must be >= 1")
        if self.viz_tsne_perplexity <= 0.0:
            raise ValueError("viz_tsne_perplexity must be > 0")
        if self.latent_sep_margin <= 0.0:
            raise ValueError("latent_sep_margin must be > 0")
        if self.projection_sep_margin <= 0.0:
            raise ValueError("projection_sep_margin must be > 0")
        if self.classification_focal_gamma is not None and self.classification_focal_gamma < 0.0:
            raise ValueError("classification_focal_gamma must be >= 0")
        if self.contrastive_weight < 0.0:
            raise ValueError("contrastive_weight must be >= 0")
        if self.contrastive_temperature <= 0.0:
            raise ValueError("contrastive_temperature must be > 0")
        if self.class_weights is not None:
            if len(self.class_weights) != self.n_classes:
                raise ValueError("class_weights must have length n_classes")
            if any(w <= 0.0 for w in self.class_weights):
                raise ValueError("class_weights entries must be > 0")

        for name in (
            "lambda_cls",
            "lambda_compact",
            "lambda_sep",
            "lambda_temp",
            "lambda_proj_cls",
            "lambda_proj_compact",
            "lambda_proj_sep",
            "lambda_proj_temp",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def class_weight_tensor(self, device: torch.device) -> torch.Tensor | None:
        if self.class_weights is None:
            return None
        return torch.tensor(self.class_weights, dtype=torch.float32, device=device)
