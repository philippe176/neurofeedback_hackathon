from __future__ import annotations

import collections

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from .types import InferenceStep


# Enhanced color palette with better contrast
CLASS_COLORS = {
    0: "#5c9fff",  # Soft blue - left hand
    1: "#ffb347",  # Soft orange - right hand
    2: "#77dd77",  # Soft green - left leg
    3: "#ff6b6b",  # Soft coral - right leg
    None: "#7a7a8c",
}

CLASS_NAMES = {
    0: "Left Hand",
    1: "Right Hand",
    2: "Left Leg",
    3: "Right Leg",
}

# Theme colors
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

# Default centroid window size
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
    ) -> None:
        if projection_dim not in (2, 3):
            raise ValueError("projection_dim must be 2 or 3")
        self.projection_dim = projection_dim
        self.history_len = history_len
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.draw_every = max(1, int(draw_every))
        self.centroid_window = max(10, min(centroid_window, history_len))

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
        self.n_classes = 4

        plt.ion()
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 8), facecolor=THEME["bg_dark"], constrained_layout=False)
        self.fig.suptitle("Neural Decoder - Live Manifold", fontsize=16, fontweight="bold",
                         color=THEME["text_primary"], y=0.98)

        # Adjust grid to leave space for slider at the bottom
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1.0, 1.0],
                                   wspace=0.18, hspace=0.28, left=0.06, right=0.94, top=0.92, bottom=0.12)
        if projection_dim == 3:
            self.ax = self.fig.add_subplot(gs[:, 0], projection="3d")
        else:
            self.ax = self.fig.add_subplot(gs[:, 0])
        self.ax_probs = self.fig.add_subplot(gs[0, 1])
        self.ax_metrics = self.fig.add_subplot(gs[1, 1])

        # Auto-track button
        self.ax_track_button = self.fig.add_axes([0.82, 0.94, 0.12, 0.04])
        self.btn_track = Button(self.ax_track_button, "", color=THEME["bg_light"], hovercolor=THEME["accent"])
        self.btn_track.on_clicked(self._on_toggle_tracking)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self._refresh_button()

        # Centroid window slider
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

    def _on_slider_change(self, val: float) -> None:
        """Update centroid window when slider changes."""
        self.centroid_window = int(val)

    def _compute_centroids(self) -> dict[int, np.ndarray]:
        """Compute class centroids using only the last N points (sliding window)."""
        if len(self._points) == 0:
            return {}

        # Use only the last centroid_window samples
        window_size = min(self.centroid_window, len(self._points))
        points_list = list(self._points)[-window_size:]
        preds_list = list(self._preds)[-window_size:]

        points_arr = np.stack(points_list, axis=0)
        preds_arr = np.array(preds_list, dtype=int)

        centroids = {}
        for cls in range(self.n_classes):
            mask = preds_arr == cls
            if np.sum(mask) >= 3:  # Need at least 3 points to compute centroid
                centroids[cls] = np.mean(points_arr[mask], axis=0)

        return centroids

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

        sizes = 25.0 + 85.0 * np.clip(confs, 0.0, 1.0)
        # Color by PREDICTIONS to show what the model is learning
        # This is more informative than ground truth labels which may not be available
        colors = [CLASS_COLORS.get(pred, CLASS_COLORS[None]) for pred in preds]

        self.ax.cla()

        self.ax.set_facecolor(THEME["bg_medium"])

        # Compute centroids using sliding window
        centroids = self._compute_centroids()

        if self.projection_dim == 2:
            # Draw trajectory with gradient fade
            if len(points) > 1:
                self.ax.plot(points[:, 0], points[:, 1], color=THEME["text_muted"], alpha=0.20, linewidth=1.2)
                # Recent trajectory highlight
                recent = points[-min(40, len(points)):]
                self.ax.plot(recent[:, 0], recent[:, 1], color=THEME["text_secondary"], alpha=0.55, linewidth=1.8)

            # Only show points from the centroid window (recent data)
            window_size = min(self.centroid_window, len(points))
            recent_points = points[-window_size:]
            recent_colors = colors[-window_size:]
            recent_sizes = sizes[-window_size:]

            # Fade old points, bright recent points
            alphas = np.linspace(0.3, 0.95, window_size)

            # Main scatter - use window data
            for i, (pt, col, sz, alpha) in enumerate(zip(recent_points, recent_colors, recent_sizes, alphas)):
                self.ax.scatter(pt[0], pt[1], c=[col], s=sz, alpha=alpha, linewidths=0, zorder=2)

            # Draw centroids with distinctive markers
            for cls, centroid in centroids.items():
                color = CLASS_COLORS.get(cls, THEME["text_muted"])
                # Outer glow
                self.ax.scatter(centroid[0], centroid[1], c=[color], s=600, alpha=0.20, linewidths=0, zorder=5, marker='o')
                # Inner marker
                self.ax.scatter(centroid[0], centroid[1], c=[color], s=250, alpha=0.95, linewidths=2.5,
                               edgecolors='white', zorder=6, marker='o')
                # Label
                short_name = ["LH", "RH", "LL", "RL"][cls]
                self.ax.annotate(short_name, (centroid[0], centroid[1]), textcoords="offset points",
                                xytext=(12, 8), fontsize=10, fontweight='bold', color=color,
                                bbox=dict(boxstyle="round,pad=0.2", facecolor=THEME["bg_dark"], alpha=0.8, edgecolor=color))

            # Current point with glow effect
            self.ax.scatter(points[-1, 0], points[-1, 1], c="white", s=180, alpha=0.25, linewidths=0, zorder=7)
            self.ax.scatter(points[-1, 0], points[-1, 1], c="white", s=100, alpha=0.95, linewidths=0, zorder=8)

            self._apply_2d_limits(points)
            self.ax.set_xlabel("Projection X", color=THEME["text_secondary"], fontsize=11)
            self.ax.set_ylabel("Projection Y", color=THEME["text_secondary"], fontsize=11)
        else:
            if len(points) > 1:
                self.ax.plot(points[:, 0], points[:, 1], points[:, 2], color=THEME["text_muted"], alpha=0.20, linewidth=1.2)
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=sizes, alpha=0.85, linewidths=0)

            # Draw centroids in 3D
            for cls, centroid in centroids.items():
                color = CLASS_COLORS.get(cls, THEME["text_muted"])
                self.ax.scatter(centroid[0], centroid[1], centroid[2], c=[color], s=300, alpha=0.95,
                               linewidths=2, edgecolors='white', zorder=6, marker='o')

            self.ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], c="white", s=140, linewidths=0)
            self._apply_3d_limits(points)
            self.ax.set_xlabel("Projection X", color=THEME["text_secondary"], fontsize=10)
            self.ax.set_ylabel("Projection Y", color=THEME["text_secondary"], fontsize=10)
            self.ax.set_zlabel("Projection Z", color=THEME["text_secondary"], fontsize=10)

        self.ax.grid(True, alpha=0.15, color=THEME["border"])
        self.ax.tick_params(colors=THEME["text_muted"], labelsize=9)

        # Enhanced title with predicted class name and centroid info
        pred_name = CLASS_NAMES.get(step.predicted_class, str(step.predicted_class))
        pred_color = CLASS_COLORS.get(step.predicted_class, THEME["text_primary"])
        n_centroids = len(centroids)
        self.ax.set_title(
            f"Latent Space Projection  |  {n_centroids}/4 Clusters",
            color=THEME["text_primary"],
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        # Add info text (only for 2D - 3D axes have different text API)
        info_text = f"Sample: {step.sample_idx:,}  |  Prediction: {pred_name}  |  Confidence: {step.confidence:.1%}  |  Window: {self.centroid_window}"
        if self.projection_dim == 2:
            self.ax.text(0.5, -0.08, info_text, transform=self.ax.transAxes, ha='center',
                        color=THEME["text_secondary"], fontsize=10)

        self._draw_probabilities(step, probs_latest)
        self._draw_metrics(step, rewards, preds, confs, centroids)

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

        # Draw bars with rounded appearance
        bars = self.ax_probs.bar(xs, probs[:n], color=colors, alpha=0.90, width=0.65,
                                  edgecolor=THEME["bg_dark"], linewidth=1)

        # Highlight predicted class
        for i, bar in enumerate(bars):
            if i == int(step.predicted_class):
                bar.set_edgecolor("white")
                bar.set_linewidth(2.5)

            # Add value label on top of bars
            height = probs[i]
            self.ax_probs.text(bar.get_x() + bar.get_width() / 2., height + 0.03,
                              f'{height:.0%}', ha='center', va='bottom',
                              color=THEME["text_primary"], fontsize=9, fontweight='bold')

        self.ax_probs.set_ylim(0.0, 1.15)
        self.ax_probs.set_xticks(xs)
        short_names = ["LH", "RH", "LL", "RL"]
        self.ax_probs.set_xticklabels(short_names[:n], color=THEME["text_secondary"], fontsize=10, fontweight='bold')
        self.ax_probs.tick_params(axis="y", colors=THEME["text_muted"], labelsize=9)
        self.ax_probs.grid(alpha=0.12, axis="y", color=THEME["border"])
        self.ax_probs.set_title("Class Probabilities", color=THEME["text_primary"],
                               fontsize=12, fontweight="bold", pad=10)

        # Add legend for class names
        pred_name = CLASS_NAMES.get(step.predicted_class, "?")
        self.ax_probs.text(
            0.98, 0.98,
            f"Predicted: {pred_name}",
            transform=self.ax_probs.transAxes,
            color=CLASS_COLORS.get(step.predicted_class, THEME["text_primary"]),
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="right",
        )

    def _compute_cluster_separation(self, centroids: dict[int, np.ndarray]) -> tuple[float, float]:
        """Compute min and mean inter-centroid distances."""
        if len(centroids) < 2:
            return 0.0, 0.0

        dists = []
        keys = list(centroids.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                d = np.linalg.norm(centroids[keys[i]] - centroids[keys[j]])
                dists.append(d)

        return float(np.min(dists)), float(np.mean(dists))

    def _draw_metrics(
        self,
        step: InferenceStep,
        rewards: np.ndarray,
        preds: np.ndarray,
        confs: np.ndarray,
        centroids: dict[int, np.ndarray],
    ) -> None:
        self.ax_metrics.cla()
        self.ax_metrics.set_facecolor(THEME["bg_medium"])

        x = np.arange(len(rewards))

        # Draw filled areas under lines for better visibility
        reward_smooth = self._rolling(rewards, 14)
        conf_smooth = self._rolling(confs, 14)

        self.ax_metrics.fill_between(x, 0, reward_smooth, color="#38bdf8", alpha=0.15)
        self.ax_metrics.plot(x, reward_smooth, color="#38bdf8", linewidth=2.0, label="Reward")

        self.ax_metrics.fill_between(x, 0, conf_smooth, color="#22c55e", alpha=0.10)
        self.ax_metrics.plot(x, conf_smooth, color="#22c55e", linewidth=1.8, label="Confidence")

        if len(self._balanced_accuracy) > 0:
            ba = self._rolling(np.asarray(self._balanced_accuracy, dtype=float), 12)
            f1 = self._rolling(np.asarray(self._macro_f1, dtype=float), 12)
            self.ax_metrics.plot(np.arange(len(ba)), ba, color="#fbbf24", linewidth=1.6, label="Bal. Accuracy", linestyle="--")
            self.ax_metrics.plot(np.arange(len(f1)), f1, color="#f472b6", linewidth=1.6, label="Macro F1", linestyle="--")

        self.ax_metrics.set_ylim(-0.02, 1.08)
        self.ax_metrics.set_xlim(0, max(len(rewards), 1))
        self.ax_metrics.grid(alpha=0.12, color=THEME["border"])
        self.ax_metrics.tick_params(colors=THEME["text_muted"], labelsize=9)
        self.ax_metrics.set_title("Training Progress", color=THEME["text_primary"],
                                 fontsize=12, fontweight="bold", pad=10)
        self.ax_metrics.legend(loc="upper right", fontsize=8, framealpha=0.3,
                              facecolor=THEME["bg_dark"], edgecolor=THEME["border"])

        # Current metrics display
        min_sep, mean_sep = self._compute_cluster_separation(centroids)
        cluster_info = f"Clusters: {len(centroids)}/4  |  MinSep: {min_sep:.2f}  |  MeanSep: {mean_sep:.2f}"

        if step.training is not None and step.training.update_applied:
            metrics_text = (
                f"Bal.Acc: {step.training.balanced_accuracy:.1%}  |  "
                f"F1: {step.training.macro_f1:.1%}  |  "
                f"ECE: {step.training.expected_calibration_error:.3f}"
            )
            self.ax_metrics.text(
                0.02, 0.02,
                metrics_text,
                transform=self.ax_metrics.transAxes,
                color=THEME["text_secondary"],
                fontsize=9,
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=THEME["bg_dark"], alpha=0.8, edgecolor=THEME["border"]),
            )

        # Cluster separation info (always show)
        sep_color = THEME["success"] if min_sep > 0.3 else THEME["warning"] if min_sep > 0.1 else THEME["text_muted"]
        self.ax_metrics.text(
            0.98, 0.02,
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
            color = "#059669"  # Emerald green
            hover = "#10b981"
            text = "Auto-Track ON"
        else:
            color = "#d97706"  # Amber
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
