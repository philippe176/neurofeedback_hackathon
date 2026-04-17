"""
Linear 2D projection + centroid-distance softmax, matching receiver_gui_v2.

Load a snapshot written by the GUI "Save model" button (compressed .npz):
  mean        (D,)   — subtract before projection
  W           (2, D) — rows are projection axes; xy = (X - mean) @ W.T
  centroids   (4, 2) — class means in projected space (NaN = unknown class)
  meta_json   UTF-8 JSON: version, method, class_names
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL_FILENAME = "classic_brain_model.npz"


def softmax_from_centroid_distances(
    points: np.ndarray, centroids: dict[int, np.ndarray]
) -> np.ndarray:
    """
    points: (T, 2); centroids: class_idx -> (2,).
    Returns (T, 4) row-wise softmax; logits = −‖x − μ_c‖².
    """
    T = len(points)
    out = np.zeros((T, 4), dtype=float)
    if T == 0:
        return out

    centers = np.full((4, 2), np.nan, dtype=float)
    have = np.zeros(4, dtype=bool)
    for c in range(4):
        if c in centroids:
            centers[c] = centroids[c]
            have[c] = True

    if not have.any():
        out[:, :] = 0.25
        return out

    d2 = np.full((T, 4), np.inf, dtype=float)
    for c in range(4):
        if not have[c]:
            continue
        diff = points[:, :] - centers[c]
        d2[:, c] = np.sum(diff * diff, axis=1)

    logits = np.where(have, -d2, -1e9)
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / (ex.sum(axis=1, keepdims=True) + 1e-12)


def _centroids_dict_from_matrix(C: np.ndarray) -> dict[int, np.ndarray]:
    d: dict[int, np.ndarray] = {}
    for c in range(4):
        row = C[c]
        if np.any(np.isnan(row)):
            continue
        d[c] = row.astype(float).copy()
    return d


@dataclass(frozen=True)
class ClassicPredictResult:
    """One row from predict (single sample)."""

    probs: np.ndarray  # (4,)
    x: float
    y: float


class ClassicModel:
    """
    Loads mean, W, and centroids from an .npz saved by the receiver GUI.

    Use ``centroids`` for a dict view, ``centroids_matrix`` for the fixed
    (4, 2) layout (NaN rows = class not present at save time).
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as d:
            self.mean: np.ndarray = np.asarray(d["mean"], dtype=float).copy()
            self.W: np.ndarray = np.asarray(d["W"], dtype=float).copy()
            self.centroids_matrix: np.ndarray = np.asarray(
                d["centroids"], dtype=float
            ).copy()
            meta_raw = d["meta_json"]
            if meta_raw.dtype == np.uint8:
                meta = json.loads(meta_raw.tobytes().decode("utf-8"))
            else:
                meta = json.loads(bytes(meta_raw))

        if self.W.ndim != 2 or self.W.shape[0] != 2:
            raise ValueError(f"W must be (2, D), got {self.W.shape}")
        if self.mean.ndim != 1 or self.mean.shape[0] != self.W.shape[1]:
            raise ValueError(
                f"mean length {self.mean.shape} must match W columns {self.W.shape[1]}"
            )
        if self.centroids_matrix.shape != (4, 2):
            raise ValueError(
                f"centroids must be (4, 2), got {self.centroids_matrix.shape}"
            )

        self._meta: dict[str, Any] = meta
        self.method: str = str(meta.get("method", ""))
        self.class_names: list[str] = list(meta.get("class_names", []))
        self._centroids: dict[int, np.ndarray] = _centroids_dict_from_matrix(
            self.centroids_matrix
        )

    @property
    def centroids(self) -> dict[int, np.ndarray]:
        """Class index -> (2,) centroid in projected space (copy-safe dict)."""
        return {c: v.copy() for c, v in self._centroids.items()}

    @property
    def n_features(self) -> int:
        return int(self.mean.shape[0])

    def project(self, X: np.ndarray) -> np.ndarray:
        """(T, D) or (D,) -> (T, 2) projected coordinates (PC1, PC2)."""
        X2 = np.atleast_2d(np.asarray(X, dtype=float))
        if X2.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features per row, got {X2.shape[1]}"
            )
        return (X2 - self.mean) @ self.W.T

    def predict(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (probs, xy) with probs (T, 4) softmax over classes and
        xy (T, 2) where column 0 is projection x (PC1), column 1 is y (PC2).
        """
        xy = self.project(X)
        probs = softmax_from_centroid_distances(xy, self._centroids)
        return probs, xy

    def predict_one(self, x: np.ndarray) -> ClassicPredictResult:
        """Single feature vector (D,) -> structured result."""
        probs, xy = self.predict(x)
        return ClassicPredictResult(
            probs=probs[0].copy(), x=float(xy[0, 0]), y=float(xy[0, 1])
        )
