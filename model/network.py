from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .types import ModelOutput


class MovementDecoder(nn.Module):
    """
    Decoder for precomputed timestep embeddings.

    The model keeps a penultimate representation for manifold visualization and
    a classifier head for 4-way movement decoding.
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
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes

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
        self.classifier_head = nn.Linear(embedding_dim, n_classes)
        self.projection_head = nn.Linear(embedding_dim, projection_dim)
        self.projection_classifier_head = nn.Linear(projection_dim, n_classes)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        h = self.trunk(x)
        penultimate = self.penultimate_head(h)
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
