from __future__ import annotations

import collections

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from .projectors import build_projector
from .types import InferenceStep


CLASS_COLORS = {
    0: "#5c9fff",
    1: "#ffb347",
    2: "#77dd77",
    3: "#ff6b6b",
    None: "#7a7a8c",
}

CLASS_NAMES = {
    0: "Left Hand",
    1: "Right Hand",
    2: "Left Leg",
    3: "Right Leg",
}

THEME = {
    "bg_dark": "#0a0e1a",
    "bg_medium": "#111827",
    "bg_light": "#1e293b",
    "text_primary": "#f1f5f9",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
    "accent": "#3b82f6",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "border": "#334155",
}

DEFAULT_CENTROID_WINDOW = 50


class RealtimeManifoldVisualizer:
    """Interactive visualization monitor for non-game real-time decoding mode."""

    def __init__(
        self,
        projection_dim: int = 2,
        history_len: int = 300,
        ema_alpha: float = 0.25,
        draw_every: int = 2,
        centroid_window: int = DEFAULT_CENTROID_WINDOW,
        viz_method: str = "neural",
        fit_window: int | None = None,
        refit_every: int = 10,
        use_penultimate: bool = True,
        tsne_perplexity: float = 20.0,
    ) -> None:
        if projection_dim not in (2, 3):
            raise ValueError("projection_dim must be 2 or 3")
        self.projection_dim = projection_dim
        self.history_len = history_len
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.draw_every = max(1, int(draw_every))
        self.centroid_window = max(10, min(centroid_window, history_len))
        self.viz_method = str(viz_method).strip().lower()
        self.fit_window = max(10, min(history_len, int(fit_window or history_len)))
        self.refit_every = max(1, int(refit_every))
        self.use_penultimate = bool(use_penultimate)
        self.projector = build_projector(
            self.viz_method,
            projection_dim=projection_dim,
            tsne_perplexity=tsne_perplexity,
        )

        self._neural_points: collections.deque[np.ndarray] = collections.deque(maxlen=history_len)
        self._penultimate: collections.deque[np.ndarray] = collections.deque(maxlen=history_len)
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
        self._last_refit_counter = -1
        self.n_classes = 4

        plt.ion()
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(14, 8), facecolor=THEME["bg_dark"], constrained_layout=False)
        self.fig.suptitle(self._figure_title(), fontsize=16, fontweight="bold", color=THEME["text_primary"], y=0.98)

        gs = self.fig.add_gridspec(
            2,
            2,
            width_ratios=[2.2, 1.0],
            height_ratios=[1.0, 1.0],
            wspace=0.18,
            hspace=0.28,
            left=0.06,
            right=0.94,
            top=0.92,
            bottom=0.12,
        )
        if projection_dim == 3:
            self.ax = self.fig.add_subplot(gs[:, 0], projection="3d")
        else:
            self.ax = self.fig.add_subplot(gs[:, 0])
        self.ax_probs = self.fig.add_subplot(gs[0, 1])
        self.ax_metrics = self.fig.add_subplot(gs[1, 1])

        self.ax_track_button = self.fig.add_axes([0.82, 0.94, 0.12, 0.04])
        self.btn_track = Button(self.ax_track_button, "", color=THEME["bg_light"], hovercolor=THEME["accent"])
        self.btn_track.on_clicked(self._on_toggle_tracking)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self._refresh_button()

        self.ax_slider = self.fig.add_axes([0.06, 0.03, 0.45, 0.025], facecolor=THEME["bg_light"])
        self.slider = Slider(
            self.ax_slider,
            "Centroid Window",
            valmin=10,
            valmax=history_len,
            valinit=self.centroid_window,
            valstep=5,
            color=THEME["accent"],
        )
        self.slider.label.set_color(THEME["text_secondary"])
        self.slider.valtext.set_color(THEME["text_primary"])
        self.slider.on_changed(self._on_slider_change)

    def _figure_title(self) -> str:
        source = "penultimate" if self.use_penultimate else "projection"
        if self.viz_method == "neural":
            return "Neural Decoder - Live Manifold"
        return f"Neural Decoder - {self.viz_method.upper()} View ({source})"

    def _on_slider_change(self, val: float) -> None:
        self.centroid_window = int(val)

    def update(self, step: InferenceStep) -> None:
        neural_point = np.asarray(step.projection, dtype=float).reshape(-1)
        if self._last_ema is None:
            smoothed = neural_point
        else:
            smoothed = self.ema_alpha * neural_point + (1.0 - self.ema_alpha) * self._last_ema
        self._last_ema = smoothed

        self._neural_points.append(smoothed)
        self._penultimate.append(np.asarray(step.penultimate, dtype=float).reshape(-1))
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

    def _display_window(self) -> int:
        return min(self.fit_window, len(self._neural_points))

    def _projection_labels(self, labels: list[int | None], preds: np.ndarray) -> np.ndarray:
        return np.asarray(
            [int(pred) if label is None else int(label) for label, pred in zip(labels, preds)],
            dtype=np.int64,
        )

    def _project_points(self, window_size: int, labels: list[int | None], preds: np.ndarray) -> np.ndarray:
        neural_points = np.stack(list(self._neural_points)[-window_size:], axis=0)
        if self.viz_method == "neural":
            return neural_points

        source_deque = self._penultimate if self.use_penultimate else self._neural_points
        x = np.stack(list(source_deque)[-window_size:], axis=0)
        y = self._projection_labels(labels, preds)

        should_refit = (
            self._last_refit_counter < 0
            or (self._counter - self._last_refit_counter) >= self.refit_every
        )
        if should_refit:
            points = self.projector.fit_transform(x, y=y)
            self._last_refit_counter = self._counter
            return points
        return self.projector.transform(x)

    def _compute_centroids(self, points: np.ndarray, preds: np.ndarray) -> dict[int, np.ndarray]:
        if points.size == 0:
            return {}

        window = min(self.centroid_window, points.shape[0])
        points_arr = points[-window:]
        preds_arr = preds[-window:]

        centroids = {}
        for cls in range(self.n_classes):
            mask = preds_arr == cls
            if np.sum(mask) >= 3:
                centroids[cls] = np.mean(points_arr[mask], axis=0)
        return centroids

    def _compute_projection_summary(
        self,
        points: np.ndarray,
        preds: np.ndarray,
        centroids: dict[int, np.ndarray],
    ) -> tuple[float, float, float]:
        if len(centroids) < 2:
            min_sep = 0.0
            mean_sep = 0.0
        else:
            dists = []
            keys = list(centroids.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    dists.append(float(np.linalg.norm(centroids[keys[i]] - centroids[keys[j]])))
            min_sep = float(np.min(dists))
            mean_sep = float(np.mean(dists))

        spreads = []
        for cls, centroid in centroids.items():
            mask = preds == cls
            if np.sum(mask) < 2:
                continue
            dists = np.linalg.norm(points[mask] - centroid, axis=1)
            spreads.append(float(np.mean(dists)))
        mean_spread = float(np.mean(spreads)) if spreads else 0.0
        return min_sep, mean_sep, mean_spread

    def _draw(self, step: InferenceStep) -> None:
        if not self._neural_points:
            return

        window_size = self._display_window()
        labels = list(self._labels)[-window_size:]
        preds = np.asarray(list(self._preds)[-window_size:], dtype=int)
        confs = np.asarray(list(self._confidences)[-window_size:], dtype=float)
        rewards = np.asarray(list(self._rewards)[-window_size:], dtype=float)
        probs_latest = self._probs[-1] if self._probs else np.zeros(4, dtype=float)
        balanced_accuracy = np.asarray(list(self._balanced_accuracy)[-window_size:], dtype=float)
        macro_f1 = np.asarray(list(self._macro_f1)[-window_size:], dtype=float)
        points = self._project_points(window_size, labels, preds)

        sizes = 25.0 + 85.0 * np.clip(confs, 0.0, 1.0)
        colors = [CLASS_COLORS.get(pred, CLASS_COLORS[None]) for pred in preds]

        self.fig.suptitle(self._figure_title(), fontsize=16, fontweight="bold", color=THEME["text_primary"], y=0.98)
        self.ax.cla()
        self.ax.set_facecolor(THEME["bg_medium"])

        centroids = self._compute_centroids(points, preds)
        min_sep, mean_sep, mean_spread = self._compute_projection_summary(points, preds, centroids)

        if self.projection_dim == 2:
            if len(points) > 1:
                self.ax.plot(points[:, 0], points[:, 1], color=THEME["text_muted"], alpha=0.20, linewidth=1.2)
                recent = points[-min(40, len(points)) :]
                self.ax.plot(recent[:, 0], recent[:, 1], color=THEME["text_secondary"], alpha=0.55, linewidth=1.8)

            scatter_window = min(self.centroid_window, len(points))
            recent_points = points[-scatter_window:]
            recent_colors = colors[-scatter_window:]
            recent_sizes = sizes[-scatter_window:]
            alphas = np.linspace(0.3, 0.95, scatter_window)

            for pt, col, sz, alpha in zip(recent_points, recent_colors, recent_sizes, alphas):
                self.ax.scatter(pt[0], pt[1], c=[col], s=sz, alpha=alpha, linewidths=0, zorder=2)

            for cls, centroid in centroids.items():
                color = CLASS_COLORS.get(cls, THEME["text_muted"])
                self.ax.scatter(centroid[0], centroid[1], c=[color], s=600, alpha=0.20, linewidths=0, zorder=5, marker="o")
                self.ax.scatter(
                    centroid[0],
                    centroid[1],
                    c=[color],
                    s=250,
                    alpha=0.95,
                    linewidths=2.5,
                    edgecolors="white",
                    zorder=6,
                    marker="o",
                )
                short_name = ["LH", "RH", "LL", "RL"][cls]
                self.ax.annotate(
                    short_name,
                    (centroid[0], centroid[1]),
                    textcoords="offset points",
                    xytext=(12, 8),
                    fontsize=10,
                    fontweight="bold",
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=THEME["bg_dark"], alpha=0.8, edgecolor=color),
                )

            self.ax.scatter(points[-1, 0], points[-1, 1], c="white", s=180, alpha=0.25, linewidths=0, zorder=7)
            self.ax.scatter(points[-1, 0], points[-1, 1], c="white", s=100, alpha=0.95, linewidths=0, zorder=8)

            self._apply_2d_limits(points)
            self.ax.set_xlabel("Projection X", color=THEME["text_secondary"], fontsize=11)
            self.ax.set_ylabel("Projection Y", color=THEME["text_secondary"], fontsize=11)
        else:
            if len(points) > 1:
                self.ax.plot(points[:, 0], points[:, 1], points[:, 2], color=THEME["text_muted"], alpha=0.20, linewidth=1.2)
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=sizes, alpha=0.85, linewidths=0)

            for cls, centroid in centroids.items():
                color = CLASS_COLORS.get(cls, THEME["text_muted"])
                self.ax.scatter(
                    centroid[0],
                    centroid[1],
                    centroid[2],
                    c=[color],
                    s=300,
                    alpha=0.95,
                    linewidths=2,
                    edgecolors="white",
                    zorder=6,
                    marker="o",
                )

            self.ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], c="white", s=140, linewidths=0)
            self._apply_3d_limits(points)
            self.ax.set_xlabel("Projection X", color=THEME["text_secondary"], fontsize=10)
            self.ax.set_ylabel("Projection Y", color=THEME["text_secondary"], fontsize=10)
            self.ax.set_zlabel("Projection Z", color=THEME["text_secondary"], fontsize=10)

        self.ax.grid(True, alpha=0.15, color=THEME["border"])
        self.ax.tick_params(colors=THEME["text_muted"], labelsize=9)

        pred_name = CLASS_NAMES.get(step.predicted_class, str(step.predicted_class))
        self.ax.set_title(
            f"{self.viz_method.upper()} Projection  |  {len(centroids)}/4 Clusters",
            color=THEME["text_primary"],
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        info_text = (
            f"Sample: {step.sample_idx:,}  |  Prediction: {pred_name}  |  Confidence: {step.confidence:.1%}  |  "
            f"Method: {self.viz_method}  |  Window: {window_size}"
        )
        if self.projection_dim == 2:
            self.ax.text(0.5, -0.08, info_text, transform=self.ax.transAxes, ha="center", color=THEME["text_secondary"], fontsize=10)

        self._draw_probabilities(step, probs_latest)
        self._draw_metrics(step, rewards, confs, balanced_accuracy, macro_f1, len(centroids), min_sep, mean_sep, mean_spread)

        self.fig.canvas.draw_idle()
        backend = plt.get_backend().lower()
        if "agg" not in backend:
            plt.pause(0.001)

    def _draw_probabilities(self, step: InferenceStep, probs: np.ndarray) -> None:
        self.ax_probs.cla()
        self.ax_probs.set_facecolor(THEME["bg_medium"])
        n = min(4, int(probs.shape[0]))
        xs = np.arange(n)
        colors = [CLASS_COLORS.get(i, THEME["text_muted"]) for i in range(n)]

        bars = self.ax_probs.bar(xs, probs[:n], color=colors, alpha=0.90, width=0.65, edgecolor=THEME["bg_dark"], linewidth=1)

        for i, bar in enumerate(bars):
            if i == int(step.predicted_class):
                bar.set_edgecolor("white")
                bar.set_linewidth(2.5)

            height = probs[i]
            self.ax_probs.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.03,
                f"{height:.0%}",
                ha="center",
                va="bottom",
                color=THEME["text_primary"],
                fontsize=9,
                fontweight="bold",
            )

        self.ax_probs.set_ylim(0.0, 1.15)
        self.ax_probs.set_xticks(xs)
        self.ax_probs.set_xticklabels(["LH", "RH", "LL", "RL"][:n], color=THEME["text_secondary"], fontsize=10, fontweight="bold")
        self.ax_probs.tick_params(axis="y", colors=THEME["text_muted"], labelsize=9)
        self.ax_probs.grid(alpha=0.12, axis="y", color=THEME["border"])
        self.ax_probs.set_title("Class Probabilities", color=THEME["text_primary"], fontsize=12, fontweight="bold", pad=10)

        pred_name = CLASS_NAMES.get(step.predicted_class, "?")
        self.ax_probs.text(
            0.98,
            0.98,
            f"Predicted: {pred_name}",
            transform=self.ax_probs.transAxes,
            color=CLASS_COLORS.get(step.predicted_class, THEME["text_primary"]),
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="right",
        )

    def _draw_metrics(
        self,
        step: InferenceStep,
        rewards: np.ndarray,
        confs: np.ndarray,
        balanced_accuracy: np.ndarray,
        macro_f1: np.ndarray,
        cluster_count: int,
        min_sep: float,
        mean_sep: float,
        mean_spread: float,
    ) -> None:
        self.ax_metrics.cla()
        self.ax_metrics.set_facecolor(THEME["bg_medium"])

        x = np.arange(len(rewards))
        reward_smooth = self._rolling(rewards, 14)
        conf_smooth = self._rolling(confs, 14)

        self.ax_metrics.fill_between(x, 0, reward_smooth, color="#38bdf8", alpha=0.15)
        self.ax_metrics.plot(x, reward_smooth, color="#38bdf8", linewidth=2.0, label="Reward")

        self.ax_metrics.fill_between(x, 0, conf_smooth, color="#22c55e", alpha=0.10)
        self.ax_metrics.plot(x, conf_smooth, color="#22c55e", linewidth=1.8, label="Confidence")

        if balanced_accuracy.size > 0:
            ba = self._rolling(balanced_accuracy, 12)
            f1 = self._rolling(macro_f1, 12)
            self.ax_metrics.plot(np.arange(len(ba)), ba, color="#fbbf24", linewidth=1.6, label="Bal. Accuracy", linestyle="--")
            self.ax_metrics.plot(np.arange(len(f1)), f1, color="#f472b6", linewidth=1.6, label="Macro F1", linestyle="--")

        self.ax_metrics.set_ylim(-0.02, 1.08)
        self.ax_metrics.set_xlim(0, max(len(rewards), 1))
        self.ax_metrics.grid(alpha=0.12, color=THEME["border"])
        self.ax_metrics.tick_params(colors=THEME["text_muted"], labelsize=9)
        self.ax_metrics.set_title("Training Progress", color=THEME["text_primary"], fontsize=12, fontweight="bold", pad=10)
        self.ax_metrics.legend(loc="upper right", fontsize=8, framealpha=0.3, facecolor=THEME["bg_dark"], edgecolor=THEME["border"])

        cluster_info = (
            f"Clusters: {cluster_count}/4  |  MinSep: {min_sep:.2f}  |  "
            f"MeanSep: {mean_sep:.2f}  |  Spread: {mean_spread:.2f}"
        )

        if step.training is not None and step.training.update_applied:
            metrics_text = (
                f"Bal.Acc: {step.training.balanced_accuracy:.1%}  |  "
                f"F1: {step.training.macro_f1:.1%}  |  "
                f"ECE: {step.training.expected_calibration_error:.3f}"
            )
            self.ax_metrics.text(
                0.02,
                0.02,
                metrics_text,
                transform=self.ax_metrics.transAxes,
                color=THEME["text_secondary"],
                fontsize=9,
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=THEME["bg_dark"], alpha=0.8, edgecolor=THEME["border"]),
            )

        sep_color = THEME["success"] if min_sep > 0.3 else THEME["warning"] if min_sep > 0.1 else THEME["text_muted"]
        self.ax_metrics.text(
            0.98,
            0.02,
            cluster_info,
            transform=self.ax_metrics.transAxes,
            color=sep_color,
            fontsize=9,
            va="bottom",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=THEME["bg_dark"], alpha=0.8, edgecolor=sep_color),
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
        if self.auto_tracking:
            color = "#059669"
            hover = "#10b981"
            text = "Auto-Track ON"
        else:
            color = "#d97706"
            hover = "#f59e0b"
            text = "Auto-Track OFF"

        self.ax_track_button.set_facecolor(color)
        self.btn_track.color = color
        self.btn_track.hovercolor = hover
        self.btn_track.label.set_text(text)
        self.btn_track.label.set_color(THEME["text_primary"])
        self.btn_track.label.set_fontsize(9)
        self.btn_track.label.set_fontweight("bold")

    def _on_toggle_tracking(self, _event) -> None:
        self.auto_tracking = not self.auto_tracking
        if self.auto_tracking:
            self._manual_limits = None
        self._refresh_button()

    def _on_key_press(self, event) -> None:
        key = (getattr(event, "key", "") or "").lower()
        if key == "t":
            self._on_toggle_tracking(None)
