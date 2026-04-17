from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .losses import supervised_contrastive_loss
from .types import ModelOutput


class RunningStandardizer(nn.Module):
    """Online mean/variance tracker that normalizes inputs.

    Uses exponential moving average so it adapts to distribution drift
    without needing the full dataset upfront.
    """

    def __init__(self, dim: int, momentum: float = 0.01, eps: float = 1e-5) -> None:
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        self.register_buffer("n_seen", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.shape[0] > 1:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            if self.n_seen.item() == 0:
                self.running_mean.copy_(batch_mean)
                self.running_var.copy_(batch_var.clamp(min=self.eps))
            else:
                self.running_mean.lerp_(batch_mean, self.momentum)
                self.running_var.lerp_(batch_var, self.momentum)
            self.n_seen.add_(x.shape[0])

        return (x - self.running_mean) / (self.running_var.sqrt() + self.eps)


class _ProjectionDecoderBase(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_classes: int,
        projection_dim: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.projection_dim = projection_dim
        self.classifier_head = nn.Linear(embedding_dim, n_classes)
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, projection_dim),
        )
        self.projection_classifier_head = nn.Linear(projection_dim, n_classes)

    def _finalize_output(self, penultimate: torch.Tensor) -> ModelOutput:
        penultimate = F.normalize(penultimate, p=2, dim=-1)
        logits = self.classifier_head(penultimate)
        probs = torch.softmax(logits, dim=-1)
        projection = self.projection_head(penultimate)
        projection_logits = self.projection_classifier_head(projection)
        projection_probs = torch.softmax(projection_logits, dim=-1)

        return ModelOutput(
            logits=logits,
            probs=probs,
            projection_logits=projection_logits,
            projection_probs=projection_probs,
            penultimate=penultimate,
            projection=projection,
        )

    def score_penultimate(self, penultimate: torch.Tensor) -> torch.Tensor:
        """Given L2-normalized penultimate embeddings, return class probabilities."""
        return torch.softmax(self.classifier_head(penultimate), dim=-1)

    def auxiliary_loss(self, out: ModelOutput, labels: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
        return out.logits.sum() * 0.0


class MovementDecoder(_ProjectionDecoderBase):
    """
    DNN decoder for precomputed timestep embeddings.

    This remains the default dense baseline used in the app.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 32,
        n_classes: int = 4,
        projection_dim: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__(embedding_dim=embedding_dim, n_classes=n_classes, projection_dim=projection_dim)
        self.input_dim = input_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.penultimate_head = nn.Linear(hidden_dim, embedding_dim)
        self.standardizer = RunningStandardizer(input_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        x = self.standardizer(x)
        h = self.trunk(x)
        penultimate = self.penultimate_head(h)
        return self._finalize_output(penultimate)


class ConvMovementDecoder(_ProjectionDecoderBase):
    """1D CNN decoder that treats the input vector as a structured signal."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 32,
        n_classes: int = 4,
        projection_dim: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__(embedding_dim=embedding_dim, n_classes=n_classes, projection_dim=projection_dim)
        self.input_dim = input_dim
        pooled_len = min(8, max(4, input_dim // 16))
        mid_channels = max(16, hidden_dim // 4)
        out_channels = max(32, hidden_dim // 2)

        self.conv_stack = nn.Sequential(
            nn.Conv1d(1, mid_channels, kernel_size=7, padding=3),
            nn.GELU(),
            nn.GroupNorm(max(1, mid_channels // 8), mid_channels),
            nn.Conv1d(mid_channels, out_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.GroupNorm(max(1, out_channels // 8), out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(pooled_len)
        self.penultimate_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels * pooled_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.standardizer = RunningStandardizer(input_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = self.standardizer(x)
        features = self.conv_stack(x.unsqueeze(1))
        pooled = self.pool(features)
        penultimate = self.penultimate_head(pooled)
        return self._finalize_output(penultimate)


class _ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CEBRAMovementDecoder(_ProjectionDecoderBase):
    """
    CEBRA-inspired decoder with normalized latent codes and contrastive regularization.

    This is not the official CEBRA package; it uses a similar representation-learning
    bias inside the existing online classification pipeline.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 32,
        n_classes: int = 4,
        projection_dim: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__(embedding_dim=embedding_dim, n_classes=n_classes, projection_dim=projection_dim)
        self.input_dim = input_dim
        self.temperature = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            _ResidualMLPBlock(hidden_dim, dropout),
            nn.LayerNorm(hidden_dim),
            _ResidualMLPBlock(hidden_dim, dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.penultimate_head = nn.Linear(hidden_dim, embedding_dim)
        self.class_prototypes = nn.Parameter(torch.randn(n_classes, embedding_dim))
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, projection_dim),
        )
        self.projection_classifier_head = nn.Linear(projection_dim, n_classes)
        self.standardizer = RunningStandardizer(input_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        x = self.standardizer(x)
        h = self.input_proj(x)
        h = self.encoder(h)
        penultimate = F.normalize(self.penultimate_head(h), p=2, dim=-1)

        prototypes = F.normalize(self.class_prototypes, p=2, dim=-1)
        logit_scale = self.temperature.exp().clamp(min=1.0, max=30.0)
        logits = logit_scale * (penultimate @ prototypes.T)
        probs = torch.softmax(logits, dim=-1)

        projection = self.projection_head(penultimate)
        projection_logits = self.projection_classifier_head(projection)
        projection_probs = torch.softmax(projection_logits, dim=-1)

        return ModelOutput(
            logits=logits,
            probs=probs,
            projection_logits=projection_logits,
            projection_probs=projection_probs,
            penultimate=penultimate,
            projection=projection,
        )

    def score_penultimate(self, penultimate: torch.Tensor) -> torch.Tensor:
        prototypes = F.normalize(self.class_prototypes, p=2, dim=-1)
        logit_scale = self.temperature.exp().clamp(min=1.0, max=30.0)
        logits = logit_scale * (penultimate @ prototypes.T)
        return torch.softmax(logits, dim=-1)

    def auxiliary_loss(self, out: ModelOutput, labels: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
        return supervised_contrastive_loss(
            out.penultimate,
            labels,
            temperature=cfg.contrastive_temperature,
        )


def build_decoder(
    model_type: str,
    cfg: ModelConfig,
) -> nn.Module:
    normalized = str(model_type).strip().lower()
    if normalized == "neural":
        normalized = "dnn"

    common_kwargs = dict(
        input_dim=int(cfg.input_dim or 0),
        hidden_dim=cfg.hidden_dim,
        embedding_dim=cfg.embedding_dim,
        n_classes=cfg.n_classes,
        projection_dim=cfg.projection_dim,
        dropout=cfg.dropout,
    )

    if normalized == "dnn":
        return MovementDecoder(**common_kwargs)
    if normalized == "cnn":
        return ConvMovementDecoder(**common_kwargs)
    if normalized == "cebra":
        return CEBRAMovementDecoder(**common_kwargs)
    raise ValueError(f"Unknown decoder type: {model_type}")
