from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _pad_projection(z: np.ndarray, target_dim: int) -> np.ndarray:
    arr = _as_2d(z)
    if arr.shape[1] == target_dim:
        return arr
    if arr.shape[1] > target_dim:
        return arr[:, :target_dim]
    pad = np.zeros((arr.shape[0], target_dim - arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


class BaseProjector:
    def __init__(self, projection_dim: int = 2) -> None:
        if projection_dim not in (2, 3):
            raise ValueError("projection_dim must be 2 or 3")
        self.projection_dim = projection_dim

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> BaseProjector:
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        self.fit(x, y=y)
        return self.transform(x)


class NeuralProjector(BaseProjector):
    def transform(self, x: np.ndarray) -> np.ndarray:
        return _pad_projection(x, self.projection_dim)


class PCAProjector(BaseProjector):
    def __init__(self, projection_dim: int = 2) -> None:
        super().__init__(projection_dim=projection_dim)
        self._model = None

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> PCAProjector:
        from sklearn.decomposition import PCA

        x_arr = _as_2d(x)
        if x_arr.shape[0] < 2:
            self._model = None
            return self
        n_components = max(1, min(self.projection_dim, x_arr.shape[0], x_arr.shape[1]))
        self._model = PCA(n_components=n_components)
        self._model.fit(x_arr)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_arr = _as_2d(x)
        if self._model is None:
            return _pad_projection(x_arr, self.projection_dim)
        return _pad_projection(self._model.transform(x_arr), self.projection_dim)


class LDAProjector(BaseProjector):
    def __init__(self, projection_dim: int = 2) -> None:
        super().__init__(projection_dim=projection_dim)
        self._model = None
        self._fallback = PCAProjector(projection_dim=projection_dim)

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> LDAProjector:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        x_arr = _as_2d(x)
        y_arr = None if y is None else np.asarray(y, dtype=np.int64).reshape(-1)
        if y_arr is None or y_arr.shape[0] != x_arr.shape[0]:
            self._model = None
            self._fallback.fit(x_arr)
            return self

        classes = np.unique(y_arr)
        max_components = min(self.projection_dim, x_arr.shape[1], max(0, classes.size - 1))
        if classes.size < 2 or max_components < 1 or x_arr.shape[0] <= classes.size:
            self._model = None
            self._fallback.fit(x_arr)
            return self

        self._model = LinearDiscriminantAnalysis(n_components=max_components)
        self._model.fit(x_arr, y_arr)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_arr = _as_2d(x)
        if self._model is None:
            return self._fallback.transform(x_arr)
        return _pad_projection(self._model.transform(x_arr), self.projection_dim)


class _SnapshotProjector(BaseProjector):
    def __init__(self, projection_dim: int = 2, n_neighbors: int = 6) -> None:
        super().__init__(projection_dim=projection_dim)
        self.n_neighbors = max(1, int(n_neighbors))
        self._reference_x: np.ndarray | None = None
        self._reference_z: np.ndarray | None = None
        self._neighbors = None
        self._fallback = PCAProjector(projection_dim=projection_dim)

    def _set_snapshot(self, x: np.ndarray, z: np.ndarray) -> None:
        from sklearn.neighbors import NearestNeighbors

        x_arr = _as_2d(x)
        z_arr = _pad_projection(z, self.projection_dim)
        self._reference_x = x_arr
        self._reference_z = z_arr
        self._fallback.fit(x_arr)

        n_neighbors = min(self.n_neighbors, x_arr.shape[0])
        self._neighbors = NearestNeighbors(n_neighbors=n_neighbors)
        self._neighbors.fit(x_arr)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_arr = _as_2d(x)
        if self._reference_x is None or self._reference_z is None or self._neighbors is None:
            return self._fallback.transform(x_arr)

        distances, indices = self._neighbors.kneighbors(x_arr)
        weights = 1.0 / np.maximum(distances, 1e-6)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        projected = np.einsum("ij,ijk->ik", weights, self._reference_z[indices])
        return _pad_projection(projected, self.projection_dim)


class TSNEProjector(_SnapshotProjector):
    def __init__(
        self,
        projection_dim: int = 2,
        perplexity: float = 20.0,
        random_state: int = 7,
    ) -> None:
        super().__init__(projection_dim=projection_dim)
        self.perplexity = float(perplexity)
        self.random_state = int(random_state)

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> TSNEProjector:
        from sklearn.manifold import TSNE

        x_arr = _as_2d(x)
        if x_arr.shape[0] < 4:
            self._set_snapshot(x_arr, self._fallback.fit_transform(x_arr))
            return self

        perplexity = min(self.perplexity, max(2.0, float(x_arr.shape[0] - 1)))
        tsne = TSNE(
            n_components=self.projection_dim,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=self.random_state,
        )
        self._set_snapshot(x_arr, tsne.fit_transform(x_arr))
        return self

    def fit_transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        self.fit(x, y=y)
        if self._reference_z is None:
            return self._fallback.transform(x)
        return self._reference_z.copy()


class UMAPProjector(_SnapshotProjector):
    def __init__(
        self,
        projection_dim: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 7,
    ) -> None:
        super().__init__(projection_dim=projection_dim, n_neighbors=n_neighbors)
        self.min_dist = float(min_dist)
        self.random_state = int(random_state)
        self._model = None

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> UMAPProjector:
        try:
            import umap
        except ImportError as exc:
            raise ImportError("UMAP visualization requires the optional 'umap-learn' package") from exc

        x_arr = _as_2d(x)
        n_neighbors = min(self.n_neighbors, max(2, x_arr.shape[0] - 1))
        self._model = umap.UMAP(
            n_components=self.projection_dim,
            n_neighbors=n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
        )
        z = self._model.fit_transform(x_arr, y=y)
        self._set_snapshot(x_arr, z)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_arr = _as_2d(x)
        if self._model is not None and hasattr(self._model, "transform"):
            return _pad_projection(self._model.transform(x_arr), self.projection_dim)
        return super().transform(x_arr)


def build_projector(
    method: str,
    projection_dim: int = 2,
    tsne_perplexity: float = 20.0,
    random_state: int = 7,
) -> BaseProjector:
    normalized = str(method).strip().lower()
    if normalized == "neural":
        return NeuralProjector(projection_dim=projection_dim)
    if normalized == "pca":
        return PCAProjector(projection_dim=projection_dim)
    if normalized == "lda":
        return LDAProjector(projection_dim=projection_dim)
    if normalized == "tsne":
        return TSNEProjector(
            projection_dim=projection_dim,
            perplexity=tsne_perplexity,
            random_state=random_state,
        )
    if normalized == "umap":
        return UMAPProjector(projection_dim=projection_dim, random_state=random_state)
    raise ValueError(f"Unknown viz_method: {method}")
