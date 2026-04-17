from __future__ import annotations

from collections import deque

import numpy as np

from .config import ModelConfig
from .reward import RewardProvider
from .types import InferenceStep, StreamSample, TrainingMetrics


class PCAOnlineDecoder:
    """
    Lightweight online decoder based on PCA projection plus class centroids.

    The model:
    - learns a PCA basis from a rolling buffer of recent observations
    - projects samples to 2D for visualization
    - classifies using nearest labeled class centroids in PCA space
    """

    def __init__(
        self,
        cfg: ModelConfig,
        reward_provider: RewardProvider,
        history_size: int | None = None,
        metric_window: int = 200,
    ) -> None:
        self.cfg = cfg
        self.reward_provider = reward_provider
        self.history_size = history_size or cfg.buffer_size
        self.metric_window = metric_window

        self.num_updates = 0
        self.labeled_seen = 0

        self._embeddings: deque[np.ndarray] = deque(maxlen=self.history_size)
        self._labels: deque[int] = deque(maxlen=self.history_size)
        self._recent_true: deque[int] = deque(maxlen=metric_window)
        self._recent_pred: deque[int] = deque(maxlen=metric_window)

        self._mean: np.ndarray | None = None
        self._components: np.ndarray | None = None
        self._explained_variance: np.ndarray | None = None
        self._class_centroids: dict[int, np.ndarray] = {}

    def process_sample(self, sample: StreamSample) -> InferenceStep:
        embedding = np.asarray(sample.embedding, dtype=np.float32)
        self._embeddings.append(embedding.copy())

        label = int(sample.label) if sample.label is not None else -1
        self._labels.append(label)
        if label >= 0:
            self.labeled_seen += 1

        training = self._fit_model()
        projection = self._project(embedding)
        probs = self._predict_probabilities(projection)
        pred = int(np.argmax(probs))
        confidence = float(probs[pred])
        reward = self.reward_provider.compute(sample, probs)

        if label >= 0:
            self._recent_true.append(label)
            self._recent_pred.append(pred)

        return InferenceStep(
            sample_idx=sample.sample_idx,
            label=sample.label,
            predicted_class=pred,
            confidence=confidence,
            reward=reward,
            probabilities=probs.copy(),
            penultimate=projection.copy(),
            projection=projection.copy(),
            training=training,
        )

    def _fit_model(self) -> TrainingMetrics:
        if len(self._embeddings) < 2:
            return self._metrics(update_applied=False)

        X = np.stack(self._embeddings, axis=0).astype(np.float32)
        self._mean = X.mean(axis=0)
        centered = X - self._mean

        try:
            _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            self._components = None
            self._explained_variance = None
            self._class_centroids = {}
            return self._metrics(update_applied=False)

        n_components = min(self.cfg.projection_dim, vh.shape[0], centered.shape[1])
        self._components = vh[:n_components].astype(np.float32)

        total_var = float(np.sum(singular_values ** 2))
        if total_var > 0.0:
            self._explained_variance = (singular_values[:n_components] ** 2 / total_var).astype(np.float32)
        else:
            self._explained_variance = np.zeros(n_components, dtype=np.float32)

        projected = centered @ self._components.T
        if projected.shape[1] < self.cfg.projection_dim:
            projected = _pad_projection(projected, self.cfg.projection_dim)

        labels = np.asarray(self._labels, dtype=np.int64)
        centroids: dict[int, np.ndarray] = {}
        for cls in range(self.cfg.n_classes):
            mask = labels == cls
            if np.any(mask):
                centroids[cls] = projected[mask].mean(axis=0).astype(np.float32)
        self._class_centroids = centroids
        self.num_updates += 1

        return self._metrics(update_applied=True, projected=projected, labels=labels)

    def _project(self, embedding: np.ndarray) -> np.ndarray:
        target_dim = self.cfg.projection_dim
        if self._mean is None or self._components is None:
            fallback = np.asarray(embedding[:target_dim], dtype=np.float32)
            if fallback.shape[0] < target_dim:
                fallback = _pad_vector(fallback, target_dim)
            return fallback

        centered = embedding.astype(np.float32) - self._mean
        projection = centered @ self._components.T
        projection = np.asarray(projection, dtype=np.float32)
        if projection.shape[0] < target_dim:
            projection = _pad_vector(projection, target_dim)
        return projection

    def _predict_probabilities(self, projection: np.ndarray) -> np.ndarray:
        if not self._class_centroids:
            return np.full(self.cfg.n_classes, 1.0 / self.cfg.n_classes, dtype=np.float32)

        distances = np.full(self.cfg.n_classes, 6.0, dtype=np.float32)
        for cls, centroid in self._class_centroids.items():
            distances[cls] = float(np.linalg.norm(projection - centroid))

        logits = -distances
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        denom = float(np.sum(exp_logits))
        if denom <= 0.0 or not np.isfinite(denom):
            return np.full(self.cfg.n_classes, 1.0 / self.cfg.n_classes, dtype=np.float32)
        return (exp_logits / denom).astype(np.float32)

    def _metrics(
        self,
        update_applied: bool,
        projected: np.ndarray | None = None,
        labels: np.ndarray | None = None,
    ) -> TrainingMetrics:
        balanced_accuracy, macro_f1 = _classification_metrics(
            np.asarray(self._recent_true, dtype=np.int64),
            np.asarray(self._recent_pred, dtype=np.int64),
            self.cfg.n_classes,
        )

        fisher_ratio = 0.0
        if projected is not None and labels is not None:
            fisher_ratio = _fisher_ratio(projected, labels, self.cfg.n_classes)

        explained = 0.0
        if self._explained_variance is not None:
            explained = float(np.sum(self._explained_variance))

        return TrainingMetrics(
            update_applied=update_applied,
            total_loss=0.0,
            supervised_loss=0.0,
            policy_loss=0.0,
            auxiliary_loss=0.0,
            entropy=0.0,
            smoothness_loss=0.0,
            labeled_in_batch=max(0, int(np.sum(np.asarray(self._labels, dtype=np.int64) >= 0))),
            rl_enabled=False,
            balanced_accuracy=balanced_accuracy if update_applied else 0.0,
            macro_f1=macro_f1 if update_applied else 0.0,
            projection_supervised_loss=explained,
            fisher_ratio_m=fisher_ratio,
        )


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> tuple[float, float]:
    if y_true.size == 0 or y_pred.size == 0 or y_true.shape != y_pred.shape:
        return 0.0, 0.0

    recalls: list[float] = []
    f1s: list[float] = []
    for cls in range(n_classes):
        true_mask = y_true == cls
        support = int(np.sum(true_mask))
        if support == 0:
            continue

        pred_mask = y_pred == cls
        tp = int(np.sum(true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))

        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        f1 = 2.0 * precision * recall / max(1e-8, precision + recall)

        recalls.append(recall)
        f1s.append(f1)

    return (
        float(np.mean(recalls)) if recalls else 0.0,
        float(np.mean(f1s)) if f1s else 0.0,
    )


def _fisher_ratio(projected: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    valid_mask = labels >= 0
    if not np.any(valid_mask):
        return 0.0

    x = projected[valid_mask]
    y = labels[valid_mask]
    overall_mean = x.mean(axis=0)

    between = 0.0
    within = 0.0
    for cls in range(n_classes):
        cls_mask = y == cls
        if not np.any(cls_mask):
            continue

        cls_points = x[cls_mask]
        cls_mean = cls_points.mean(axis=0)
        between += float(cls_points.shape[0] * np.sum((cls_mean - overall_mean) ** 2))
        within += float(np.sum((cls_points - cls_mean) ** 2))

    return between / max(within, 1e-6)


def _pad_projection(projected: np.ndarray, target_dim: int) -> np.ndarray:
    pad_width = target_dim - projected.shape[1]
    if pad_width <= 0:
        return projected.astype(np.float32)
    padding = np.zeros((projected.shape[0], pad_width), dtype=np.float32)
    return np.concatenate([projected.astype(np.float32), padding], axis=1)


def _pad_vector(vector: np.ndarray, target_dim: int) -> np.ndarray:
    pad_width = target_dim - vector.shape[0]
    if pad_width <= 0:
        return vector.astype(np.float32)
    return np.concatenate([vector.astype(np.float32), np.zeros(pad_width, dtype=np.float32)])
