from __future__ import annotations

import collections

import matplotlib.pyplot as plt
import numpy as np

from .types import InferenceStep


CLASS_COLORS = {
    0: "#6495ed",
    1: "#ffa500",
    2: "#32cd32",
    3: "#dc143c",
    None: "#787878",
}


class RealtimeManifoldVisualizer:
    """Simple matplotlib visualizer for 2D or 3D projected embeddings."""

    def __init__(
        self,
        projection_dim: int = 2,
        history_len: int = 300,
        ema_alpha: float = 0.25,
        draw_every: int = 2,
    ) -> None:
        if projection_dim not in (2, 3):
            raise ValueError("projection_dim must be 2 or 3")
        self.projection_dim = projection_dim
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.draw_every = max(1, int(draw_every))

        self._points: collections.deque[np.ndarray] = collections.deque(maxlen=history_len)
        self._labels: collections.deque[int | None] = collections.deque(maxlen=history_len)
        self._confidences: collections.deque[float] = collections.deque(maxlen=history_len)
        self._rewards: collections.deque[float] = collections.deque(maxlen=history_len)
        self._last_ema: np.ndarray | None = None
        self._counter = 0

        plt.ion()
        self.fig = plt.figure(figsize=(8, 6))
        if projection_dim == 3:
            self.ax = self.fig.add_subplot(111, projection="3d")
        else:
            self.ax = self.fig.add_subplot(111)

    def update(self, step: InferenceStep) -> None:
        point = np.asarray(step.projection, dtype=float).reshape(-1)
        if self._last_ema is None:
            smoothed = point
        else:
            smoothed = self.ema_alpha * point + (1.0 - self.ema_alpha) * self._last_ema
        self._last_ema = smoothed

        self._points.append(smoothed)
        self._labels.append(step.label)
        self._confidences.append(step.confidence)
        self._rewards.append(step.reward)

        self._counter += 1
        if self._counter % self.draw_every == 0:
            self._draw(step)

    def close(self) -> None:
        plt.close(self.fig)

    def _draw(self, step: InferenceStep) -> None:
        if not self._points:
            return

        points = np.stack(self._points, axis=0)
        labels = list(self._labels)
        confs = np.asarray(self._confidences, dtype=float)

        sizes = 18.0 + 72.0 * np.clip(confs, 0.0, 1.0)
        colors = [CLASS_COLORS.get(lbl, CLASS_COLORS[None]) for lbl in labels]

        self.ax.cla()

        if self.projection_dim == 2:
            if len(points) > 1:
                self.ax.plot(points[:, 0], points[:, 1], color="#ffffff", alpha=0.25, linewidth=1.0)
            self.ax.scatter(points[:, 0], points[:, 1], c=colors, s=sizes, alpha=0.85, linewidths=0)
            self.ax.scatter(points[-1, 0], points[-1, 1], c="white", s=120, linewidths=0)
            self.ax.set_xlabel("manifold x")
            self.ax.set_ylabel("manifold y")
        else:
            if len(points) > 1:
                self.ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#ffffff", alpha=0.25, linewidth=1.0)
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=sizes, alpha=0.85, linewidths=0)
            self.ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], c="white", s=120, linewidths=0)
            self.ax.set_xlabel("manifold x")
            self.ax.set_ylabel("manifold y")
            self.ax.set_zlabel("manifold z")

        self.ax.grid(True, alpha=0.2)
        self.ax.set_title(
            f"sample={step.sample_idx} pred={step.predicted_class} "
            f"conf={step.confidence:.2f} reward={step.reward:.2f}"
        )

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        backend = plt.get_backend().lower()
        if "agg" not in backend:
            plt.pause(0.001)
