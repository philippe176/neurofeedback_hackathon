from __future__ import annotations

from collections import deque
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from model.types import InferenceStep


CLASS_NAMES = {
    0: "Left Hand",
    1: "Right Hand",
    2: "Left Leg",
    3: "Right Leg",
}

# Enhanced vibrant color palette
CLASS_COLORS = {
    0: "#5c9fff",  # Vivid blue - left hand
    1: "#ffb347",  # Soft orange - right hand
    2: "#77dd77",  # Fresh green - left leg
    3: "#ff6b6b",  # Coral red - right leg
}

SHORT_CLASS_NAMES = {
    0: "LH",
    1: "RH",
    2: "LL",
    3: "RL",
}

CLASS_HINTS = {
    0: "Focus: LEFT HAND movement",
    1: "Focus: RIGHT HAND movement",
    2: "Focus: LEFT LEG movement",
    3: "Focus: RIGHT LEG movement",
}

# Theme colors for a modern dark UI
THEME = {
    "bg_dark": "#0a0e1a",
    "bg_medium": "#111827",
    "bg_light": "#1e293b",
    "bg_card": "#1a1f2e",
    "text_primary": "#f1f5f9",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
    "accent": "#3b82f6",
    "accent_light": "#60a5fa",
    "success": "#22c55e",
    "success_light": "#4ade80",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "border": "#334155",
    "border_light": "#475569",
}


class RhythmConsoleHUD:
    """Minimal terminal HUD for rhythm prompts and hit feedback."""

    def __init__(self, print_every: int = 10) -> None:
        self.print_every = max(1, int(print_every))

    def maybe_render(self, step: InferenceStep, processed_count: int) -> None:
        if step.game_prompt_id is None:
            return
        if processed_count % self.print_every != 0:
            return

        hit_flag = "HIT" if step.game_hit else "..."
        target = step.game_target_class if step.game_target_class is not None else -1
        next_target = step.game_next_target_class if step.game_next_target_class is not None else -1
        level = step.game_level if step.game_level is not None else -1
        streak = step.game_streak if step.game_streak is not None else 0
        to_window = (
            float(step.game_seconds_to_window_start)
            if step.game_seconds_to_window_start is not None
            else 0.0
        )
        to_next = (
            float(step.game_seconds_to_next_prompt_start)
            if step.game_seconds_to_next_prompt_start is not None
            else 0.0
        )

        print(
            "[game] "
            f"prompt={step.game_prompt_id:04d} target={target} lvl={level} "
            f"pred={step.predicted_class} conf={step.confidence:.2f} "
            f"reward={step.reward:.2f} streak={streak} {hit_flag} "
            f"window_in={to_window:.2f}s next={next_target} next_in={to_next:.2f}s"
        )


class RhythmGameDashboard:
    """
    Interactive control-room style dashboard for rhythm training.

    The dashboard focuses on clarity for first-time users:
    - what to imagine now vs next,
    - how the decoder is responding,
    - why reward changed,
    - whether manifold clusters are improving.

    Includes runtime controls for:
    - automatic manifold tracking window,
    - automatic behavior simulation (auto-perform).
    """

    def __init__(
        self,
        history_len: int = 320,
        draw_every: int = 2,
        on_toggle_simulation: Callable[[bool], None] | None = None,
        get_simulation_enabled: Callable[[], bool] | None = None,
    ) -> None:
        self.history_len = max(40, int(history_len))
        self.draw_every = max(1, int(draw_every))
        self.auto_tracking = True

        self._set_simulation_enabled = on_toggle_simulation
        self._get_simulation_enabled = get_simulation_enabled
        self.simulation_enabled = (
            bool(get_simulation_enabled())
            if get_simulation_enabled is not None
            else False
        )

        self._counter = 0
        self._points: deque[np.ndarray] = deque(maxlen=self.history_len)
        self._preds: deque[int] = deque(maxlen=self.history_len)
        self._targets: deque[int] = deque(maxlen=self.history_len)
        self._confidences: deque[float] = deque(maxlen=self.history_len)
        self._rewards: deque[float] = deque(maxlen=self.history_len)
        self._margins: deque[float] = deque(maxlen=self.history_len)
        self._correct: deque[bool] = deque(maxlen=self.history_len)
        self._timing_hits: deque[bool] = deque(maxlen=self.history_len)
        self._balanced_accuracy: deque[float] = deque(maxlen=self.history_len)
        self._macro_f1: deque[float] = deque(maxlen=self.history_len)
        self._top2: deque[float] = deque(maxlen=self.history_len)
        self._ece: deque[float] = deque(maxlen=self.history_len)
        self._nll: deque[float] = deque(maxlen=self.history_len)
        self._manual_xlim: tuple[float, float] | None = None
        self._manual_ylim: tuple[float, float] | None = None

        plt.ion()
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(18, 11), constrained_layout=False, facecolor=THEME["bg_dark"])
        self.fig.suptitle("Neurofeedback Training Dashboard", fontsize=18, fontweight="bold",
                         color=THEME["text_primary"], y=0.985)

        gs = self.fig.add_gridspec(
            6, 2,
            width_ratios=[2.2, 1.4],
            height_ratios=[0.32, 1.15, 0.95, 0.95, 0.95, 0.95],
            wspace=0.20,
            hspace=0.22,
            left=0.05,
            right=0.95,
            top=0.95,
            bottom=0.04,
        )
        self.ax_status = self.fig.add_subplot(gs[0, :])
        self.ax_space = self.fig.add_subplot(gs[1:, 0])
        self.ax_cue = self.fig.add_subplot(gs[1, 1])
        self.ax_decoder = self.fig.add_subplot(gs[2, 1])
        self.ax_components = self.fig.add_subplot(gs[3, 1])
        self.ax_metrics = self.fig.add_subplot(gs[4, 1])
        self.ax_trend = self.fig.add_subplot(gs[5, 1])

        self.ax_track_button = self.fig.add_axes([0.74, 0.955, 0.10, 0.032])
        self.ax_sim_button = self.fig.add_axes([0.85, 0.955, 0.10, 0.032])
        self.btn_track = Button(self.ax_track_button, "", color=THEME["bg_light"], hovercolor=THEME["accent"])
        self.btn_sim = Button(self.ax_sim_button, "", color=THEME["bg_light"], hovercolor=THEME["accent"])
        self.btn_track.on_clicked(self._on_toggle_tracking)
        self.btn_sim.on_clicked(self._on_toggle_simulation)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self._refresh_control_buttons()

    def update(self, step: InferenceStep) -> None:
        if step.game_prompt_id is None:
            return

        point = np.asarray(step.projection, dtype=float).reshape(-1)
        if point.size < 2:
            return

        self._points.append(point[:2])
        self._preds.append(int(step.predicted_class))
        self._targets.append(int(step.game_target_class if step.game_target_class is not None else -1))
        self._confidences.append(float(step.confidence))
        self._rewards.append(float(step.reward))
        self._margins.append(float(step.game_margin if step.game_margin is not None else 0.0))
        self._correct.append(bool(step.game_label_correct))
        self._timing_hits.append(bool(step.game_timing_hit))
        if step.training is not None and step.training.update_applied:
            self._balanced_accuracy.append(float(step.training.balanced_accuracy))
            self._macro_f1.append(float(step.training.macro_f1))
            self._top2.append(float(step.training.top2_accuracy))
            self._ece.append(float(step.training.expected_calibration_error))
            self._nll.append(float(step.training.negative_log_likelihood))
        else:
            self._balanced_accuracy.append(np.nan)
            self._macro_f1.append(np.nan)
            self._top2.append(np.nan)
            self._ece.append(np.nan)
            self._nll.append(np.nan)

        self._counter += 1
        if self._counter % self.draw_every == 0:
            self._draw(step)

    def close(self) -> None:
        plt.close(self.fig)

    def _draw(self, step: InferenceStep) -> None:
        if self._get_simulation_enabled is not None:
            self.simulation_enabled = bool(self._get_simulation_enabled())
        self._refresh_control_buttons()

        points = np.stack(self._points, axis=0)
        preds = np.asarray(self._preds, dtype=int)
        targets = np.asarray(self._targets, dtype=int)
        rewards = np.asarray(self._rewards, dtype=float)
        margins = np.asarray(self._margins, dtype=float)
        correct = np.asarray(self._correct, dtype=bool)
        timing_hits = np.asarray(self._timing_hits, dtype=bool)
        confidences = np.asarray(self._confidences, dtype=float)

        self._draw_status(step, rewards, correct)
        self._draw_space(points, preds, correct, confidences)
        self._draw_cue(step)
        self._draw_decoder(step)
        self._draw_components(step)
        self._draw_metrics(preds, targets, correct, timing_hits, margins)
        self._draw_trends(rewards, margins, correct)

        self.fig.canvas.draw_idle()
        backend = plt.get_backend().lower()
        if "agg" not in backend:
            plt.pause(0.001)

    def _draw_status(self, step: InferenceStep, rewards: np.ndarray, correct: np.ndarray) -> None:
        self.ax_status.cla()
        self.ax_status.set_facecolor(THEME["bg_medium"])
        self.ax_status.set_xlim(0, 1)
        self.ax_status.set_ylim(0, 1)
        self.ax_status.set_xticks([])
        self.ax_status.set_yticks([])

        # Remove spines for cleaner look
        for spine in self.ax_status.spines.values():
            spine.set_visible(False)

        rolling_reward = float(np.mean(rewards[-50:])) if rewards.size > 0 else 0.0
        rolling_correct = float(np.mean(correct[-50:])) if correct.size > 0 else 0.0
        mode = "SIMULATION" if self.simulation_enabled else "LIVE"
        track = "TRACKING ON" if self.auto_tracking else "TRACKING OFF"

        # Keyboard hints
        self.ax_status.text(
            0.01, 0.50,
            "Keys: [T] tracking  |  [S] simulation",
            color=THEME["text_muted"],
            fontsize=9,
            va="center",
        )

        # Status chips with better styling
        self._chip(self.ax_status, 0.28, 0.50, f"Prompt #{step.game_prompt_id:04d}", THEME["accent"])
        self._chip(self.ax_status, 0.46, 0.50, mode,
                   THEME["warning"] if self.simulation_enabled else THEME["success"])
        self._chip(self.ax_status, 0.60, 0.50, track,
                   THEME["success"] if self.auto_tracking else THEME["warning"])

        # Performance metrics
        reward_color = THEME["success"] if rolling_reward > 0.6 else (THEME["warning"] if rolling_reward > 0.4 else THEME["error"])
        correct_color = THEME["success"] if rolling_correct > 0.7 else (THEME["warning"] if rolling_correct > 0.5 else THEME["error"])
        streak_color = THEME["success"] if (step.game_streak or 0) >= 5 else THEME["text_secondary"]

        self._chip(self.ax_status, 0.74, 0.50, f"Reward: {rolling_reward:.0%}", reward_color)
        self._chip(self.ax_status, 0.88, 0.50, f"x{step.game_streak or 0}", streak_color)

    def _draw_space(
        self,
        points: np.ndarray,
        preds: np.ndarray,
        correct: np.ndarray,
        confidences: np.ndarray,
    ) -> None:
        self.ax_space.cla()
        self.ax_space.set_facecolor(THEME["bg_medium"])

        # Remove top and right spines for cleaner look
        self.ax_space.spines['top'].set_visible(False)
        self.ax_space.spines['right'].set_visible(False)
        self.ax_space.spines['bottom'].set_color(THEME["border"])
        self.ax_space.spines['left'].set_color(THEME["border"])

        if points.shape[0] > 1:
            # Faded trajectory
            self.ax_space.plot(
                points[:, 0], points[:, 1],
                color=THEME["text_muted"],
                alpha=0.15,
                linewidth=1.0,
            )
            # Recent trajectory highlighted
            recent = points[-min(50, points.shape[0]):]
            self.ax_space.plot(
                recent[:, 0], recent[:, 1],
                color=THEME["text_secondary"],
                alpha=0.50,
                linewidth=1.8,
            )

        # Draw class clusters with better styling
        for cls in sorted(CLASS_COLORS):
            mask = preds == cls
            if not np.any(mask):
                continue

            sizes = 30 + 80 * np.clip(confidences[mask], 0.0, 1.0)
            self.ax_space.scatter(
                points[mask, 0], points[mask, 1],
                s=sizes,
                color=CLASS_COLORS[cls],
                alpha=0.80,
                linewidths=0,
                label=CLASS_NAMES[cls],
                zorder=2,
            )

            # Class centroid marker
            if np.sum(mask) >= 3:
                centroid = points[mask].mean(axis=0)
                self.ax_space.scatter(
                    [centroid[0]], [centroid[1]],
                    marker="o",
                    facecolors="none",
                    edgecolors=CLASS_COLORS[cls],
                    s=200,
                    linewidths=2.5,
                    zorder=3,
                )
                self.ax_space.scatter(
                    [centroid[0]], [centroid[1]],
                    marker="+",
                    color=CLASS_COLORS[cls],
                    s=100,
                    linewidths=2,
                    zorder=4,
                )

        # Correct predictions ring
        if np.any(correct):
            self.ax_space.scatter(
                points[correct, 0], points[correct, 1],
                facecolors="none",
                edgecolors="white",
                linewidths=1.2,
                s=100,
                alpha=0.6,
                zorder=3,
            )

        # Current point with glow effect
        self.ax_space.scatter(
            [points[-1, 0]], [points[-1, 1]],
            color="white",
            s=220,
            alpha=0.20,
            linewidths=0,
            zorder=5,
        )
        self.ax_space.scatter(
            [points[-1, 0]], [points[-1, 1]],
            color="white",
            s=100,
            alpha=0.95,
            linewidths=0,
            zorder=6,
        )

        self._apply_tracking_window(points)

        sep_idx = self._separability_index(points, preds)
        rolling_correct = float(np.mean(correct[-50:])) if correct.size > 0 else 0.0

        self.ax_space.set_title(
            "Latent Space Projection",
            color=THEME["text_primary"],
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        self.ax_space.set_xlabel("Projection X", color=THEME["text_secondary"], fontsize=11)
        self.ax_space.set_ylabel("Projection Y", color=THEME["text_secondary"], fontsize=11)
        self.ax_space.grid(alpha=0.12, color=THEME["border"])
        self.ax_space.tick_params(colors=THEME["text_muted"], labelsize=9)

        # Legend with better styling
        legend = self.ax_space.legend(loc="upper right", fontsize=9, framealpha=0.85,
                                      facecolor=THEME["bg_dark"], edgecolor=THEME["border"])
        legend.get_frame().set_linewidth(0.5)

        # Info panel
        info_text = f"Samples: {points.shape[0]}  |  Separability: {sep_idx:.2f}  |  Accuracy: {rolling_correct:.0%}"
        self.ax_space.text(
            0.5, -0.06,
            info_text,
            transform=self.ax_space.transAxes,
            color=THEME["text_secondary"],
            fontsize=10,
            ha="center",
        )

    def _draw_cue(self, step: InferenceStep) -> None:
        self.ax_cue.cla()
        self.ax_cue.set_facecolor(THEME["bg_card"])
        self.ax_cue.set_xlim(0, 1)
        self.ax_cue.set_ylim(0, 1)
        self.ax_cue.set_xticks([])
        self.ax_cue.set_yticks([])

        # Remove all spines for card look
        for spine in self.ax_cue.spines.values():
            spine.set_visible(False)

        target = step.game_target_class if step.game_target_class is not None else -1
        target_name = CLASS_NAMES.get(target, "---")
        target_color = CLASS_COLORS.get(target, THEME["text_muted"])
        next_target = step.game_next_target_class if step.game_next_target_class is not None else -1
        next_target_name = CLASS_NAMES.get(next_target, "---")
        next_target_color = CLASS_COLORS.get(next_target, THEME["text_muted"])
        to_window = (
            float(step.game_seconds_to_window_start)
            if step.game_seconds_to_window_start is not None
            else 0.0
        )
        to_next = (
            float(step.game_seconds_to_next_prompt_start)
            if step.game_seconds_to_next_prompt_start is not None
            else 0.0
        )

        # Header
        self.ax_cue.text(
            0.03, 0.95,
            "Mental Task Coach",
            color=THEME["text_primary"],
            fontsize=12,
            fontweight="bold",
        )

        # Current target - larger, more prominent
        self.ax_cue.text(0.03, 0.82, "NOW", color=THEME["text_muted"], fontsize=9, fontweight="bold")
        self.ax_cue.text(
            0.03, 0.68,
            target_name,
            color=target_color,
            fontsize=18,
            fontweight="bold",
        )
        self.ax_cue.text(
            0.03, 0.58,
            CLASS_HINTS.get(target, "Focus on your intention"),
            color=THEME["text_secondary"],
            fontsize=9,
        )

        # Next target - smaller, secondary
        self.ax_cue.text(0.58, 0.82, "NEXT", color=THEME["text_muted"], fontsize=9, fontweight="bold")
        self.ax_cue.text(
            0.58, 0.68,
            next_target_name,
            color=next_target_color,
            fontsize=13,
            fontweight="bold",
        )
        self.ax_cue.text(
            0.58, 0.58,
            f"In {to_next:.1f}s",
            color=THEME["text_muted"],
            fontsize=9,
        )

        # Progress bar with better styling
        progress = float(step.game_prompt_progress) if step.game_prompt_progress is not None else 0.0
        progress = float(np.clip(progress, 0.0, 1.0))

        bar_y = 0.46
        bar_height = 0.06
        # Background
        self.ax_cue.barh([bar_y], [1.0], height=bar_height, color=THEME["bg_dark"], left=0.03)
        # Progress fill with gradient effect
        prog_color = THEME["success"] if step.game_in_window else THEME["accent"]
        self.ax_cue.barh([bar_y], [progress * 0.94], height=bar_height, color=prog_color, left=0.03, alpha=0.9)

        # Window status indicator
        if step.game_in_window:
            win_text = ">>> WINDOW OPEN - Act Now! <<<"
            win_color = THEME["success_light"]
        else:
            win_text = f"Window opens in {to_window:.1f}s"
            win_color = THEME["warning"]

        self.ax_cue.text(0.03, 0.34, win_text, color=win_color, fontsize=11, fontweight="bold")

        # Status feedback
        if step.game_label_correct:
            status = "[OK] Correct!"
            status_color = THEME["success"]
            if step.game_timing_hit:
                status += " + Timing Bonus"
        else:
            status = "[ ] Focus on target class..."
            status_color = THEME["text_muted"]

        self.ax_cue.text(
            0.03, 0.20,
            status,
            color=status_color,
            fontsize=10,
            fontweight="bold",
        )

        # Tips at bottom
        self.ax_cue.text(
            0.03, 0.06,
            "Tip: Focus on distinguishing the target from other classes",
            color=THEME["text_muted"],
            fontsize=8,
            style="italic",
        )

    def _draw_decoder(self, step: InferenceStep) -> None:
        self.ax_decoder.cla()
        self.ax_decoder.set_facecolor(THEME["bg_card"])

        # Remove spines
        for spine in self.ax_decoder.spines.values():
            spine.set_visible(False)

        probs = np.asarray(step.probabilities, dtype=float).reshape(-1)
        n = min(4, probs.size)
        xs = np.arange(n)
        colors = [CLASS_COLORS.get(i, THEME["text_muted"]) for i in range(n)]

        target = int(step.game_target_class) if step.game_target_class is not None else -1
        pred = int(step.predicted_class)

        # Draw bars with better styling
        bars = self.ax_decoder.bar(xs, probs[:n], color=colors, alpha=0.85, width=0.6,
                                    edgecolor=THEME["bg_dark"], linewidth=1)

        for i, bar in enumerate(bars):
            # Target indicator (thick white border)
            if i == target:
                bar.set_edgecolor("white")
                bar.set_linewidth(3)

            # Predicted indicator (checkmark above)
            if i == pred:
                height = probs[i]
                marker = "▼" if i == target else "▽"
                self.ax_decoder.text(
                    bar.get_x() + bar.get_width() / 2., height + 0.08,
                    marker, ha='center', va='bottom',
                    color="white" if i == target else THEME["text_secondary"],
                    fontsize=10,
                )

            # Value labels
            height = probs[i]
            self.ax_decoder.text(
                bar.get_x() + bar.get_width() / 2., height + 0.02,
                f'{height:.0%}', ha='center', va='bottom',
                color=THEME["text_primary"], fontsize=9, fontweight='bold',
            )

        self.ax_decoder.set_ylim(0.0, 1.20)
        self.ax_decoder.set_xticks(xs)
        self.ax_decoder.set_xticklabels([SHORT_CLASS_NAMES.get(i, str(i)) for i in xs],
                                        color=THEME["text_secondary"], fontsize=10, fontweight='bold')
        self.ax_decoder.tick_params(axis="y", colors=THEME["text_muted"], labelsize=8)
        self.ax_decoder.grid(alpha=0.10, axis="y", color=THEME["border"])

        self.ax_decoder.set_title(
            "Decoder Output",
            color=THEME["text_primary"],
            fontsize=11,
            fontweight="bold",
            pad=8,
        )

        # Legend text
        if target >= 0:
            legend_text = f"Target: {CLASS_NAMES.get(target, '?')} (white border)  |  Predicted: {CLASS_NAMES.get(pred, '?')} (▼)"
        else:
            legend_text = f"Predicted: {CLASS_NAMES.get(pred, '?')}"
        self.ax_decoder.text(
            0.5, -0.08,
            legend_text,
            transform=self.ax_decoder.transAxes,
            color=THEME["text_muted"],
            fontsize=8,
            ha="center",
        )

        # Training metrics badge
        if step.training is not None and step.training.update_applied:
            metrics = f"Acc: {step.training.balanced_accuracy:.0%}  F1: {step.training.macro_f1:.0%}"
            self.ax_decoder.text(
                0.98, 0.98,
                metrics,
                transform=self.ax_decoder.transAxes,
                color=THEME["text_secondary"],
                fontsize=8,
                va="top",
                ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=THEME["bg_dark"], alpha=0.8, edgecolor=THEME["border"]),
            )

    def _draw_components(self, step: InferenceStep) -> None:
        self.ax_components.cla()
        self.ax_components.set_facecolor(THEME["bg_card"])

        # Remove spines
        for spine in self.ax_components.spines.values():
            spine.set_visible(False)

        components = dict(step.game_reward_components or {})
        if not components:
            self.ax_components.set_xticks([])
            self.ax_components.set_yticks([])
            self.ax_components.text(
                0.5, 0.5,
                "Waiting for reward data...",
                transform=self.ax_components.transAxes,
                color=THEME["text_muted"],
                fontsize=10,
                va="center",
                ha="center",
            )
            self.ax_components.set_title("Reward Breakdown", color=THEME["text_primary"],
                                        fontsize=11, fontweight="bold", pad=8)
            return

        ordered_keys = [
            "correctness",
            "margin",
            "separability",
            "timing",
            "stability",
            "hit_bonus",
            "confusion_penalty",
        ]
        display_names = {
            "correctness": "Correct Class",
            "margin": "Confidence Margin",
            "separability": "Class Separation",
            "timing": "Timing Accuracy",
            "stability": "Signal Stability",
            "hit_bonus": "Hit Bonus",
            "confusion_penalty": "Confusion Penalty",
        }
        labels: list[str] = []
        values: list[float] = []
        for key in ordered_keys:
            if key not in components:
                continue
            value = float(components[key])
            if key == "confusion_penalty":
                value = -value
            labels.append(display_names.get(key, key))
            values.append(float(np.clip(value, -1.0, 1.0)))

        y = np.arange(len(values))
        colors = [THEME["error"] if v < 0.0 else THEME["success"] for v in values]

        # Draw bars
        bars = self.ax_components.barh(y, values, color=colors, alpha=0.85, height=0.65)
        self.ax_components.axvline(0.0, color=THEME["border"], linewidth=1.5, alpha=0.7)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            x_pos = val + 0.05 if val >= 0 else val - 0.05
            ha = 'left' if val >= 0 else 'right'
            self.ax_components.text(
                x_pos, bar.get_y() + bar.get_height() / 2,
                f'{val:+.2f}', va='center', ha=ha,
                color=THEME["text_secondary"], fontsize=8,
            )

        self.ax_components.set_yticks(y)
        self.ax_components.set_yticklabels(labels, color=THEME["text_secondary"], fontsize=9)
        self.ax_components.set_xlim(-1.1, 1.1)
        self.ax_components.tick_params(axis="x", colors=THEME["text_muted"], labelsize=8)
        self.ax_components.grid(alpha=0.10, axis="x", color=THEME["border"])

        # Title with total reward
        reward_color = THEME["success"] if step.reward > 0.6 else (THEME["warning"] if step.reward > 0.3 else THEME["error"])
        self.ax_components.set_title(
            f"Reward Breakdown",
            color=THEME["text_primary"],
            fontsize=11,
            fontweight="bold",
            pad=8,
        )
        self.ax_components.text(
            0.98, 0.98,
            f"Total: {step.reward:.2f}",
            transform=self.ax_components.transAxes,
            color=reward_color,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="right",
        )

    def _draw_metrics(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        correct: np.ndarray,
        timing_hits: np.ndarray,
        margins: np.ndarray,
    ) -> None:
        self.ax_metrics.cla()
        self.ax_metrics.set_facecolor(THEME["bg_card"])

        cm = np.zeros((4, 4), dtype=float)
        valid = targets >= 0
        if np.any(valid):
            t = targets[valid]
            p = preds[valid]
            for tt, pp in zip(t, p):
                if 0 <= tt < 4 and 0 <= pp < 4:
                    cm[tt, pp] += 1.0

        display = cm.copy()
        row_sum = display.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        display = display / row_sum

        # Custom colormap for better visibility
        self.ax_metrics.imshow(display, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto", alpha=0.9)

        self.ax_metrics.set_xticks(np.arange(4))
        self.ax_metrics.set_yticks(np.arange(4))
        self.ax_metrics.set_xticklabels([SHORT_CLASS_NAMES[c] for c in range(4)],
                                        color=THEME["text_secondary"], fontsize=9, fontweight="bold")
        self.ax_metrics.set_yticklabels([SHORT_CLASS_NAMES[c] for c in range(4)],
                                        color=THEME["text_secondary"], fontsize=9, fontweight="bold")
        self.ax_metrics.set_xlabel("Predicted", color=THEME["text_muted"], fontsize=9)
        self.ax_metrics.set_ylabel("Target", color=THEME["text_muted"], fontsize=9)
        self.ax_metrics.tick_params(axis="both", labelsize=9)

        # Cell values with better contrast
        for i in range(4):
            for j in range(4):
                val = int(cm[i, j])
                text_color = THEME["bg_dark"] if display[i, j] > 0.5 else THEME["text_primary"]
                weight = "bold" if i == j else "normal"
                self.ax_metrics.text(
                    j, i,
                    str(val) if val > 0 else "·",
                    ha="center", va="center",
                    color=text_color,
                    fontsize=9,
                    fontweight=weight,
                )

        corr = float(np.mean(correct[-50:])) if correct.size > 0 else 0.0
        timing = float(np.mean(timing_hits[-50:])) if timing_hits.size > 0 else 0.0
        margin_mean = float(np.mean(margins[-50:])) if margins.size > 0 else 0.0

        self.ax_metrics.set_title(
            "Confusion Matrix",
            color=THEME["text_primary"],
            fontsize=11,
            fontweight="bold",
            pad=8,
        )

        # Stats badge
        stats_text = f"Acc: {corr:.0%}  Timing: {timing:.0%}  Margin: {margin_mean:.2f}"
        self.ax_metrics.text(
            0.5, -0.12,
            stats_text,
            transform=self.ax_metrics.transAxes,
            color=THEME["text_secondary"],
            fontsize=8,
            ha="center",
        )

        confusion = self._dominant_confusion(targets, preds)
        self.ax_metrics.text(
            0.02, 0.02,
            confusion,
            transform=self.ax_metrics.transAxes,
            color=THEME["warning"] if "none" not in confusion.lower() else THEME["text_muted"],
            fontsize=8,
            va="bottom",
        )

    def _draw_trends(self, rewards: np.ndarray, margins: np.ndarray, correct: np.ndarray) -> None:
        self.ax_trend.cla()
        self.ax_trend.set_facecolor(THEME["bg_card"])

        # Remove top and right spines
        self.ax_trend.spines['top'].set_visible(False)
        self.ax_trend.spines['right'].set_visible(False)
        self.ax_trend.spines['bottom'].set_color(THEME["border"])
        self.ax_trend.spines['left'].set_color(THEME["border"])

        x = np.arange(len(rewards))

        # Draw with filled areas for better visual effect
        reward_smooth = self._rolling(rewards, 16)
        correct_smooth = self._rolling(correct.astype(float), 16)
        margin_smooth = self._rolling(margins, 16)

        self.ax_trend.fill_between(x, 0, reward_smooth, color="#38bdf8", alpha=0.12)
        self.ax_trend.plot(x, reward_smooth, color="#38bdf8", linewidth=2.0, label="Reward")

        self.ax_trend.fill_between(x, 0, correct_smooth, color="#22c55e", alpha=0.10)
        self.ax_trend.plot(x, correct_smooth, color="#22c55e", linewidth=1.8, label="Accuracy")

        self.ax_trend.plot(x, margin_smooth, color="#a78bfa", linewidth=1.6, label="Margin", linestyle="--")

        if len(self._balanced_accuracy) > 0:
            ba = self._rolling(np.asarray(self._balanced_accuracy, dtype=float), 14)
            f1 = self._rolling(np.asarray(self._macro_f1, dtype=float), 14)
            self.ax_trend.plot(np.arange(len(ba)), ba, color="#fbbf24", linewidth=1.4, label="Bal. Acc", linestyle=":")
            self.ax_trend.plot(np.arange(len(f1)), f1, color="#f472b6", linewidth=1.4, label="F1", linestyle=":")

        self.ax_trend.set_ylim(-0.05, 1.08)
        self.ax_trend.set_xlim(0, max(len(rewards), 1))
        self.ax_trend.tick_params(colors=THEME["text_muted"], labelsize=8)
        self.ax_trend.grid(alpha=0.10, color=THEME["border"])
        self.ax_trend.set_title("Training Progress", color=THEME["text_primary"],
                               fontsize=11, fontweight="bold", pad=8)
        self.ax_trend.legend(loc="upper right", fontsize=8, framealpha=0.85,
                            facecolor=THEME["bg_dark"], edgecolor=THEME["border"], ncol=2)

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

    def _apply_tracking_window(self, points: np.ndarray) -> None:
        if points.shape[0] == 0:
            return

        if self.auto_tracking:
            view = points[-min(120, points.shape[0]) :]
            xmin, xmax = float(np.min(view[:, 0])), float(np.max(view[:, 0]))
            ymin, ymax = float(np.min(view[:, 1])), float(np.max(view[:, 1]))
            dx = max(0.4, xmax - xmin)
            dy = max(0.4, ymax - ymin)
            px = 0.22 * dx
            py = 0.22 * dy
            self.ax_space.set_xlim(xmin - px, xmax + px)
            self.ax_space.set_ylim(ymin - py, ymax + py)
            return

        if self._manual_xlim is None or self._manual_ylim is None:
            xmin, xmax = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
            ymin, ymax = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
            dx = max(0.5, xmax - xmin)
            dy = max(0.5, ymax - ymin)
            self._manual_xlim = (xmin - 0.25 * dx, xmax + 0.25 * dx)
            self._manual_ylim = (ymin - 0.25 * dy, ymax + 0.25 * dy)

        self.ax_space.set_xlim(*self._manual_xlim)
        self.ax_space.set_ylim(*self._manual_ylim)

    def _chip(self, ax, x: float, y: float, text: str, bg: str, icon: str = "") -> None:
        display_text = f"{icon}  {text}" if icon else text
        ax.text(
            x, y,
            display_text,
            color=THEME["text_primary"],
            fontsize=9,
            fontweight="bold",
            va="center",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.35,rounding_size=0.25",
                "facecolor": bg,
                "edgecolor": "none",
                "alpha": 0.92,
            },
        )

    def _refresh_control_buttons(self) -> None:
        # Tracking button
        if self.auto_tracking:
            track_color = "#059669"  # Emerald green
            track_hover = "#10b981"
            track_text = "Track: ON"
        else:
            track_color = "#d97706"  # Amber
            track_hover = "#f59e0b"
            track_text = "Track: OFF"

        self.ax_track_button.set_facecolor(track_color)
        self.btn_track.color = track_color
        self.btn_track.hovercolor = track_hover
        self.btn_track.label.set_text(track_text)
        self.btn_track.label.set_color(THEME["text_primary"])
        self.btn_track.label.set_fontsize(9)
        self.btn_track.label.set_fontweight("bold")

        # Simulation button
        if self._set_simulation_enabled is None:
            sim_label = "Sim: N/A"
            sim_color = THEME["bg_light"]
            sim_hover = THEME["border"]
        else:
            if self.simulation_enabled:
                sim_label = "Sim: ON"
                sim_color = "#7c3aed"  # Violet
                sim_hover = "#8b5cf6"
            else:
                sim_label = "Sim: OFF"
                sim_color = THEME["bg_light"]
                sim_hover = THEME["border_light"]

        self.ax_sim_button.set_facecolor(sim_color)
        self.btn_sim.color = sim_color
        self.btn_sim.hovercolor = sim_hover
        self.btn_sim.label.set_text(sim_label)
        self.btn_sim.label.set_color(THEME["text_primary"])
        self.btn_sim.label.set_fontsize(9)
        self.btn_sim.label.set_fontweight("bold")

    def _on_toggle_tracking(self, _event) -> None:
        self.auto_tracking = not self.auto_tracking
        if self.auto_tracking:
            self._manual_xlim = None
            self._manual_ylim = None
        self._refresh_control_buttons()

    def _on_toggle_simulation(self, _event) -> None:
        if self._set_simulation_enabled is None:
            return

        desired = not self.simulation_enabled
        self._set_simulation_enabled(desired)
        if self._get_simulation_enabled is not None:
            self.simulation_enabled = bool(self._get_simulation_enabled())
        else:
            self.simulation_enabled = desired
        self._refresh_control_buttons()

    def _on_key_press(self, event) -> None:
        key = (getattr(event, "key", "") or "").lower()
        if key == "t":
            self._on_toggle_tracking(None)
        elif key == "s":
            self._on_toggle_simulation(None)

    def _separability_index(self, points: np.ndarray, preds: np.ndarray) -> float:
        unique = np.unique(preds)
        if unique.size < 2:
            return 0.0

        grand = points.mean(axis=0)
        between = 0.0
        within = 0.0

        for cls in unique:
            cls_points = points[preds == cls]
            if cls_points.shape[0] < 2:
                continue
            mu = cls_points.mean(axis=0)
            between += cls_points.shape[0] * float(np.sum((mu - grand) ** 2))
            within += float(np.sum((cls_points - mu) ** 2))

        return float(np.clip(between / (within + 1e-8), 0.0, 10.0))

    def _dominant_confusion(self, targets: np.ndarray, preds: np.ndarray) -> str:
        valid = targets >= 0
        if not np.any(valid):
            return "Confusion: waiting for target statistics"

        t = targets[valid]
        p = preds[valid]
        cm = np.zeros((4, 4), dtype=int)
        for tt, pp in zip(t, p):
            if 0 <= tt < 4 and 0 <= pp < 4:
                cm[tt, pp] += 1
        np.fill_diagonal(cm, 0)

        if int(cm.max()) == 0:
            return "Confusion: none in recent window"

        i, j = np.unravel_index(np.argmax(cm), cm.shape)
        return f"Top confusion: {CLASS_NAMES[i]} -> {CLASS_NAMES[j]} ({cm[i, j]} samples)"
