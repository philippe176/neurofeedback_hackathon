"""Strategy exploration: cluster single-class penultimate embeddings and score via frozen classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


@dataclass(slots=True)
class StrategyCluster:
    cluster_id: int
    size: int
    centroid_2d: np.ndarray
    confidence: float
    is_best: bool


@dataclass(slots=True)
class ExplorationResult:
    target_class: int
    n_samples: int
    n_clusters: int
    points_2d: np.ndarray
    cluster_labels: np.ndarray
    clusters: list[StrategyCluster]
    best_cluster_id: int | None

    def to_dict(self) -> dict:
        return {
            "target_class": self.target_class,
            "n_samples": self.n_samples,
            "n_clusters": self.n_clusters,
            "points_2d": self.points_2d.tolist(),
            "cluster_labels": self.cluster_labels.tolist(),
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "size": c.size,
                    "centroid_2d": c.centroid_2d.tolist(),
                    "confidence": c.confidence,
                    "is_best": c.is_best,
                }
                for c in self.clusters
            ],
            "best_cluster_id": self.best_cluster_id,
        }


def analyze_strategies(
    penultimate: np.ndarray,
    model: torch.nn.Module,
    target_class: int,
    device: torch.device,
    max_clusters: int = 5,
    min_samples: int = 20,
) -> ExplorationResult | None:
    if penultimate.shape[0] < min_samples:
        return None

    n_components = min(2, penultimate.shape[0], penultimate.shape[1])
    pca = PCA(n_components=n_components)
    points_2d = pca.fit_transform(penultimate).astype(np.float32)
    if points_2d.shape[1] < 2:
        points_2d = np.column_stack([points_2d, np.zeros(points_2d.shape[0], dtype=np.float32)])

    k = min(max_clusters, max(2, penultimate.shape[0] // 15))
    kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
    cluster_labels = kmeans.fit_predict(penultimate)

    centroids_np = kmeans.cluster_centers_
    centroids_2d = pca.transform(centroids_np).astype(np.float32)
    if centroids_2d.shape[1] < 2:
        centroids_2d = np.column_stack([centroids_2d, np.zeros(centroids_2d.shape[0], dtype=np.float32)])

    with torch.no_grad():
        centroids_t = torch.from_numpy(centroids_np).to(device, dtype=torch.float32)
        centroids_t = torch.nn.functional.normalize(centroids_t, p=2, dim=-1)
        probs = model.score_penultimate(centroids_t).cpu().numpy()

    cluster_confs = probs[:, target_class]
    best_id = int(np.argmax(cluster_confs))

    sorted_confs = np.sort(cluster_confs)[::-1]
    margin = float(sorted_confs[0] - sorted_confs[1]) if len(sorted_confs) > 1 else 1.0
    flag_best = margin > 0.05

    clusters = []
    for i in range(k):
        mask = cluster_labels == i
        clusters.append(StrategyCluster(
            cluster_id=i,
            size=int(mask.sum()),
            centroid_2d=centroids_2d[i].copy(),
            confidence=float(cluster_confs[i]),
            is_best=(i == best_id and flag_best),
        ))

    return ExplorationResult(
        target_class=target_class,
        n_samples=penultimate.shape[0],
        n_clusters=k,
        points_2d=points_2d,
        cluster_labels=cluster_labels,
        clusters=clusters,
        best_cluster_id=best_id if flag_best else None,
    )
