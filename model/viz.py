from __future__ import annotations

import collections

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from .types import InferenceStep


CLASS_COLORS = {
    0: "#6495ed",
    1: "#ffa500",
    2: "#32cd32",
    3: "#dc143c",
    None: "#787878",
}


class RealtimeManifoldVisualizer:
    """Interactive visualization monitor for non-game real-time decoding mode."""

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
        self._preds: collections.deque[int] = collections.deque(maxlen=history_len)
        self._confidences: collections.deque[float] = collections.deque(maxlen=history_len)
        self._rewards: collections.deque[float] = collections.deque(maxlen=history_len)
        self._probs: collections.deque[np.ndarray] = collections.deque(maxlen=history_len)
        self._balanced_accuracy: collections.deque[float] = collections.deque(maxlen=history_len)
        self._macro_f1: collections.deque[float] = collections.deque(maxlen=history_len)
        self._manual_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float] | None] | None = None
        self.auto_tracking = True
        self._last_ema: np.ndarray | None = None
        self._counter = 0

        plt.ion()
        self.fig = plt.figure(figsize=(12.5, 7.5), facecolor="#0b1220", constrained_layout=False)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2.25, 1.0], height_ratios=[1.0, 1.0], wspace=0.20, hspace=0.25)
        if projection_dim == 3:
            self.ax = self.fig.add_subplot(gs[:, 0], projection="3d")
        else:
            self.ax = self.fig.add_subplot(gs[:, 0])
        self.ax_probs = self.fig.add_subplot(gs[0, 1])
        self.ax_metrics = self.fig.add_subplot(gs[1, 1])

        self.ax_track_button = self.fig.add_axes([0.82, 0.93, 0.15, 0.05])
        self.btn_track = Button(self.ax_track_button, "")
        self.btn_track.on_clicked(self._on_toggle_tracking)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self._refresh_button()

    def update(self, step: InferenceStep) -> None:
        point = np.asarray(step.projection, dtype=float).reshape(-1)
        if self._last_ema is None:
            smoothed = point
        else:
            smoothed = self.ema_alpha * point + (1.0 - self.ema_alpha) * self._last_ema
        self._last_ema = smoothed

        self._points.append(smoothed)
        self._labels.append(step.label)
        self._preds.append(int(step.predicted_class))
        self._confidences.append(step.confidence)
        self._rewards.append(step.reward)
        self._probs.append(np.asarray(step.probabilities, dtype=float).copy())
        if step.training is not None and step.training.update_applied:
            self._balanced_accuracy.append(float(step.training.balanced_accuracy))
            self._macro_f1.append(float(step.training.macro_f1))
        else:
            self._balanced_accuracy.append(np.nan)
            self._macro_f1.append(np.nan)

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
        preds = np.asarray(self._preds, dtype=int)
        confs = np.asarray(self._confidences, dtype=float)
        rewards = np.asarray(self._rewards, dtype=float)
        probs_latest = self._probs[-1] if self._probs else np.zeros(4, dtype=float)

        sizes = 18.0 + 72.0 * np.clip(confs, 0.0, 1.0)
        colors = [CLASS_COLORS.get(lbl, CLASS_COLORS[None]) for lbl in labels]

        self.ax.cla()

        self.ax.set_facecolor("#0f1830")
        if self.projection_dim == 2:
            if len(points) > 1:
                self.ax.plot(points[:, 0], points[:, 1], color="#d8e0ff", alpha=0.22, linewidth=1.0)
            self.ax.scatter(points[:, 0], points[:, 1], c=colors, s=sizes, alpha=0.88, linewidths=0)
            self.ax.scatter(points[-1, 0], points[-1, 1], c="white", s=130, linewidths=0)
            self._apply_2d_limits(points)
            self.ax.set_xlabel("Projection X", color="#a6b0cf")
            self.ax.set_ylabel("Projection Y", color="#a6b0cf")
        else:
            if len(points) > 1:
                self.ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#d8e0ff", alpha=0.22, linewidth=1.0)
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=sizes, alpha=0.88, linewidths=0)
            self.ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], c="white", s=130, linewidths=0)
            self._apply_3d_limits(points)
            self.ax.set_xlabel("Projection X", color="#a6b0cf")
            self.ax.set_ylabel("Projection Y", color="#a6b0cf")
            self.ax.set_zlabel("Projection Z", color="#a6b0cf")

        self.ax.grid(True, alpha=0.22)
        self.ax.tick_params(colors="#9aa7d8")
        self.ax.set_title(
            f"Live Manifold  |  sample={step.sample_idx} pred={step.predicted_class} conf={step.confidence:.2f}",
            color="#dbe2ff",
        )

        self._draw_probabilities(step, probs_latest)
        self._draw_metrics(step, rewards, preds, confs)

        self.fig.canvas.draw_idle()
        backend = plt.get_backend().lower()
        if "agg" not in backend:
            plt.pause(0.001)

    def _draw_probabilities(self, step: InferenceStep, probs: np.ndarray) -> None:
        self.ax_probs.cla()
        self.ax_probs.set_facecolor("#131f39")
        n = min(4, int(probs.shape[0]))
        xs = np.arange(n)
        colors = [CLASS_COLORS.get(i, "#7782a6") for i in range(n)]
        bars = self.ax_probs.bar(xs, probs[:n], color=colors, alpha=0.86)
        for i, bar in enumerate(bars):
            if i == int(step.predicted_class):
                bar.set_hatch("//")

        self.ax_probs.set_ylim(0.0, 1.0)
        self.ax_probs.set_xticks(xs)
        self.ax_probs.set_xticklabels(["0", "1", "2", "3"][:n], color="#c6d2ff")
        self.ax_probs.tick_params(axis="y", colors="#9aa7d8")
        self.ax_probs.grid(alpha=0.18, axis="y")
        self.ax_probs.set_title("Decoder probabilities", color="#dbe2ff", fontsize=10)
        self.ax_probs.text(
            0.02,
            0.96,
            f"pred={step.predicted_class}  conf={step.confidence:.2f}",
            transform=self.ax_probs.transAxes,
            color="#dbe2ff",
            fontsize=9,
            va="top",
        )

    def _draw_metrics(
        self,
        step: InferenceStep,
        rewards: np.ndarray,
        preds: np.ndarray,
        confs: np.ndarray,
    ) -> None:
        self.ax_metrics.cla()
        self.ax_metrics.set_facecolor("#131f39")

        x = np.arange(len(rewards))
        self.ax_metrics.plot(x, self._rolling(rewards, 14), color="#7dcfff", linewidth=1.6, label="reward")
        self.ax_metrics.plot(x, self._rolling(confs, 14), color="#9ece6a", linewidth=1.5, label="confidence")

        if len(self._balanced_accuracy) > 0:
            ba = self._rolling(np.asarray(self._balanced_accuracy, dtype=float), 12)
            f1 = self._rolling(np.asarray(self._macro_f1, dtype=float), 12)
            self.ax_metrics.plot(np.arange(len(ba)), ba, color="#f5b971", linewidth=1.3, label="balanced-acc")
            self.ax_metrics.plot(np.arange(len(f1)), f1, color="#f98aa4", linewidth=1.3, label="macro-f1")

        self.ax_metrics.set_ylim(-0.05, 1.05)
        self.ax_metrics.grid(alpha=0.18)
        self.ax_metrics.tick_params(colors="#9aa7d8")
        self.ax_metrics.set_title("Progress monitor (rolling)", color="#dbe2ff", fontsize=10)
        self.ax_metrics.legend(loc="upper right", fontsize=8, framealpha=0.15)

        if step.training is not None and step.training.update_applied:
            self.ax_metrics.text(
                0.02,
                0.95,
                (
                    f"BA={step.training.balanced_accuracy:.2f}  "
                    f"F1={step.training.macro_f1:.2f}  "
                    f"ECE={step.training.expected_calibration_error:.2f}"
                ),
                transform=self.ax_metrics.transAxes,
                color="#dbe2ff",
                fontsize=8.5,
                va="top",
            )

    def _rolling(self, values: np.ndarray, window: int) -> np.ndarray:
        if values.size == 0:
            return values
        window = max(1, int(window))
        out = np.zeros_like(values, dtype=float)
        for i in range(values.size):
            lo = max(0, i - window + 1)
            segment = values[lo : i + 1]
            finite = segment[np.isfinite(segment)]
            out[i] = float(np.mean(finite)) if finite.size > 0 else np.nan
        return out

    def _apply_2d_limits(self, points: np.ndarray) -> None:
        if self.auto_tracking:
            view = points[-min(120, points.shape[0]) :]
            xmin, xmax = float(np.min(view[:, 0])), float(np.max(view[:, 0]))
            ymin, ymax = float(np.min(view[:, 1])), float(np.max(view[:, 1]))
            dx = max(0.4, xmax - xmin)
            dy = max(0.4, ymax - ymin)
            self.ax.set_xlim(xmin - 0.22 * dx, xmax + 0.22 * dx)
            self.ax.set_ylim(ymin - 0.22 * dy, ymax + 0.22 * dy)
            return

        if self._manual_limits is None:
            xmin, xmax = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
            ymin, ymax = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
            dx = max(0.5, xmax - xmin)
            dy = max(0.5, ymax - ymin)
            self._manual_limits = ((xmin - 0.25 * dx, xmax + 0.25 * dx), (ymin - 0.25 * dy, ymax + 0.25 * dy), None)

        xlim, ylim, _ = self._manual_limits
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)

    def _apply_3d_limits(self, points: np.ndarray) -> None:
        if self.auto_tracking:
            view = points[-min(120, points.shape[0]) :]
            mins = np.min(view, axis=0)
            maxs = np.max(view, axis=0)
            span = np.maximum(maxs - mins, 0.4)
            pad = 0.22 * span
            self.ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
            self.ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
            self.ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
            return

        if self._manual_limits is None:
            mins = np.min(points, axis=0)
            maxs = np.max(points, axis=0)
            span = np.maximum(maxs - mins, 0.5)
            pad = 0.25 * span
            self._manual_limits = (
                (mins[0] - pad[0], maxs[0] + pad[0]),
                (mins[1] - pad[1], maxs[1] + pad[1]),
                (mins[2] - pad[2], maxs[2] + pad[2]),
            )

        xlim, ylim, zlim = self._manual_limits
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        if zlim is not None:
            self.ax.set_zlim(*zlim)

    def _refresh_button(self) -> None:
        color = "#3a7a5f" if self.auto_tracking else "#9a7433"
        self.ax_track_button.set_facecolor(color)
        self.btn_track.color = color
        self.btn_track.hovercolor = "#6ca58f" if self.auto_tracking else "#c8a15b"
        self.btn_track.label.set_text("Tracking ON" if self.auto_tracking else "Tracking OFF")
        self.btn_track.label.set_color("#f5f8ff")
        self.btn_track.label.set_fontsize(9)

    def _on_toggle_tracking(self, _event) -> None:
        self.auto_tracking = not self.auto_tracking
        if self.auto_tracking:
            self._manual_limits = None
        self._refresh_button()

    def _on_key_press(self, event) -> None:
        key = (getattr(event, "key", "") or "").lower()
        if key == "t":
            self._on_toggle_tracking(None)
