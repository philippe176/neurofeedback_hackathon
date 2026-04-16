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

CLASS_COLORS = {
    0: "#7aa2f7",
    1: "#ff9e64",
    2: "#9ece6a",
    3: "#f7768e",
}

SHORT_CLASS_NAMES = {
    0: "LH",
    1: "RH",
    2: "LL",
    3: "RL",
}

CLASS_HINTS = {
    0: "Imagine LEFT HAND movement",
    1: "Imagine RIGHT HAND movement",
    2: "Imagine LEFT LEG movement",
    3: "Imagine RIGHT LEG movement",
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
        self.fig = plt.figure(figsize=(16, 10), constrained_layout=False, facecolor="#0b1220")
        gs = self.fig.add_gridspec(
            6,
            2,
            width_ratios=[2.25, 1.35],
            height_ratios=[0.36, 1.2, 1.0, 1.0, 1.0, 1.0],
            wspace=0.24,
            hspace=0.26,
        )
        self.ax_status = self.fig.add_subplot(gs[0, :])
        self.ax_space = self.fig.add_subplot(gs[1:, 0])
        self.ax_cue = self.fig.add_subplot(gs[1, 1])
        self.ax_decoder = self.fig.add_subplot(gs[2, 1])
        self.ax_components = self.fig.add_subplot(gs[3, 1])
        self.ax_metrics = self.fig.add_subplot(gs[4, 1])
        self.ax_trend = self.fig.add_subplot(gs[5, 1])

        self.ax_track_button = self.fig.add_axes([0.76, 0.94, 0.10, 0.04])
        self.ax_sim_button = self.fig.add_axes([0.87, 0.94, 0.11, 0.04])
        self.btn_track = Button(self.ax_track_button, "")
        self.btn_sim = Button(self.ax_sim_button, "")
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
        self.ax_status.set_facecolor("#101a32")
        self.ax_status.set_xlim(0, 1)
        self.ax_status.set_ylim(0, 1)
        self.ax_status.set_xticks([])
        self.ax_status.set_yticks([])

        rolling_reward = float(np.mean(rewards[-50:])) if rewards.size > 0 else 0.0
        rolling_correct = float(np.mean(correct[-50:])) if correct.size > 0 else 0.0
        mode = "SIMULATED BEHAVIOR" if self.simulation_enabled else "LIVE MODEL BEHAVIOR"
        track = "AUTO TRACK ON" if self.auto_tracking else "AUTO TRACK OFF"

        self.ax_status.text(
            0.015,
            0.63,
            "NEUROFEEDBACK CONTROL ROOM",
            color="#eef3ff",
            fontsize=13,
            fontweight="bold",
            va="center",
        )
        self.ax_status.text(
            0.015,
            0.24,
            "Hotkeys: [T] toggle auto tracking, [S] toggle simulation",
            color="#a9b9df",
            fontsize=9,
            va="center",
        )

        self._chip(self.ax_status, 0.43, 0.65, f"prompt #{step.game_prompt_id}", "#29446f")
        self._chip(self.ax_status, 0.57, 0.65, mode, "#4b6cb8" if self.simulation_enabled else "#2b3c64")
        self._chip(self.ax_status, 0.76, 0.65, track, "#2e7d5a" if self.auto_tracking else "#8a6d2d")
        self._chip(self.ax_status, 0.43, 0.23, f"rolling reward {rolling_reward:.2f}", "#3b2d5e")
        self._chip(self.ax_status, 0.62, 0.23, f"rolling correct {rolling_correct:.2f}", "#2e5d5a")
        self._chip(self.ax_status, 0.80, 0.23, f"streak {step.game_streak}", "#5e2f48")

    def _draw_space(
        self,
        points: np.ndarray,
        preds: np.ndarray,
        correct: np.ndarray,
        confidences: np.ndarray,
    ) -> None:
        self.ax_space.cla()
        self.ax_space.set_facecolor("#0f1830")

        if points.shape[0] > 1:
            self.ax_space.plot(
                points[:, 0],
                points[:, 1],
                color="#d2dcff",
                alpha=0.18,
                linewidth=1.0,
            )
            recent = points[-min(60, points.shape[0]) :]
            self.ax_space.plot(
                recent[:, 0],
                recent[:, 1],
                color="#f5f7ff",
                alpha=0.65,
                linewidth=1.6,
            )

        for cls in sorted(CLASS_COLORS):
            mask = preds == cls
            if not np.any(mask):
                continue
            self.ax_space.scatter(
                points[mask, 0],
                points[mask, 1],
                s=20 + 60 * np.clip(confidences[mask], 0.0, 1.0),
                color=CLASS_COLORS[cls],
                alpha=0.70,
                linewidths=0,
                label=CLASS_NAMES[cls],
            )

            if np.sum(mask) >= 3:
                centroid = points[mask].mean(axis=0)
                self.ax_space.scatter(
                    [centroid[0]],
                    [centroid[1]],
                    marker="x",
                    color=CLASS_COLORS[cls],
                    s=120,
                    linewidths=2,
                )

        if np.any(correct):
            self.ax_space.scatter(
                points[correct, 0],
                points[correct, 1],
                facecolors="none",
                edgecolors="white",
                linewidths=0.9,
                s=85,
            )

        self.ax_space.scatter(
            [points[-1, 0]],
            [points[-1, 1]],
            color="white",
            s=150,
            linewidths=0,
            zorder=5,
        )

        self._apply_tracking_window(points)

        sep_idx = self._separability_index(points, preds)
        rolling_correct = float(np.mean(correct[-50:])) if correct.size > 0 else 0.0
        self.ax_space.set_title(
            (
                "Live Manifold Map"
                f"  |  separability={sep_idx:.2f}"
                f"  |  label-correct={rolling_correct:.2f}"
            ),
            color="#dbe2ff",
            fontsize=12,
        )
        self.ax_space.set_xlabel("Projection X", color="#a6b0cf")
        self.ax_space.set_ylabel("Projection Y", color="#a6b0cf")
        self.ax_space.grid(alpha=0.24)
        self.ax_space.tick_params(colors="#9aa7d8")
        self.ax_space.legend(loc="upper right", fontsize=8, framealpha=0.18)
        self.ax_space.text(
            0.02,
            0.98,
            f"Samples shown: {points.shape[0]}",
            transform=self.ax_space.transAxes,
            color="#c9d4f8",
            fontsize=9,
            va="top",
        )

    def _draw_cue(self, step: InferenceStep) -> None:
        self.ax_cue.cla()
        self.ax_cue.set_facecolor("#131f39")
        self.ax_cue.set_xlim(0, 1)
        self.ax_cue.set_ylim(0, 1)
        self.ax_cue.set_xticks([])
        self.ax_cue.set_yticks([])

        target = step.game_target_class if step.game_target_class is not None else -1
        target_name = CLASS_NAMES.get(target, "---")
        target_color = CLASS_COLORS.get(target, "#a9b3d9")
        next_target = step.game_next_target_class if step.game_next_target_class is not None else -1
        next_target_name = CLASS_NAMES.get(next_target, "---")
        next_target_color = CLASS_COLORS.get(next_target, "#a9b3d9")
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

        self.ax_cue.text(
            0.04,
            0.92,
            "MENTAL TASK COACH",
            color="#dbe2ff",
            fontsize=12,
            fontweight="bold",
        )
        self.ax_cue.text(
            0.04,
            0.80,
            "NOW:",
            color="#b7c3ee",
            fontsize=11,
            fontweight="bold",
        )
        self.ax_cue.text(
            0.04,
            0.67,
            target_name,
            color=target_color,
            fontsize=20,
            fontweight="bold",
        )
        self.ax_cue.text(
            0.04,
            0.59,
            CLASS_HINTS.get(target, "Hold a stable class-specific intention"),
            color="#d4dcfb",
            fontsize=9,
        )

        self.ax_cue.text(
            0.57,
            0.80,
            "UP NEXT:",
            color="#b7c3ee",
            fontsize=10,
            fontweight="bold",
        )
        self.ax_cue.text(
            0.56,
            0.67,
            next_target_name,
            color=next_target_color,
            fontsize=14,
            fontweight="bold",
        )
        self.ax_cue.text(
            0.56,
            0.59,
            CLASS_HINTS.get(next_target, "Prepare next class intention"),
            color="#d4dcfb",
            fontsize=8,
        )

        self.ax_cue.text(
            0.56,
            0.52,
            f"Starts in {to_next:.2f}s",
            color="#c7d1f2",
            fontsize=9,
        )

        progress = float(step.game_prompt_progress) if step.game_prompt_progress is not None else 0.0
        progress = float(np.clip(progress, 0.0, 1.0))

        self.ax_cue.text(0.04, 0.48, "Prompt timeline", color="#b7c3ee", fontsize=9)
        self.ax_cue.barh([0.43], [1.0], height=0.08, color="#232d4d")
        self.ax_cue.barh([0.43], [progress], height=0.08, color="#4cc9f0")
        win_text = "WINDOW OPEN" if step.game_in_window else "WINDOW CLOSED"
        win_color = "#8be9a3" if step.game_in_window else "#f8c555"
        countdown = "Act now" if step.game_in_window else f"Window opens in {to_window:.2f}s"
        self.ax_cue.text(0.04, 0.34, win_text, color=win_color, fontsize=10, fontweight="bold")
        self.ax_cue.text(0.28, 0.34, countdown, color="#e5ecff", fontsize=9)

        if step.game_label_correct:
            status = "Correct label captured"
            status_color = "#8be9a3"
        else:
            status = "Separate target from competing classes"
            status_color = "#ff8b8b"

        if step.game_timing_hit:
            status += "  |  timing bonus"

        self.ax_cue.text(0.04, 0.21, status, color=status_color, fontsize=10)
        self.ax_cue.text(
            0.04,
            0.10,
            "1) Follow NOW class in the active window  2) Prime UP NEXT early",
            color="#c7d1f2",
            fontsize=9,
        )
        self.ax_cue.text(
            0.04,
            0.04,
            "Primary objective: class distinction. Timing adds bonus only.",
            color="#9fb0df",
            fontsize=8,
        )

    def _draw_decoder(self, step: InferenceStep) -> None:
        self.ax_decoder.cla()
        self.ax_decoder.set_facecolor("#131f39")

        probs = np.asarray(step.probabilities, dtype=float).reshape(-1)
        n = min(4, probs.size)
        xs = np.arange(n)
        colors = [CLASS_COLORS.get(i, "#7f8ab3") for i in range(n)]
        bars = self.ax_decoder.bar(xs, probs[:n], color=colors, alpha=0.85)

        target = int(step.game_target_class) if step.game_target_class is not None else -1
        pred = int(step.predicted_class)
        for i, bar in enumerate(bars):
            if i == target:
                bar.set_linewidth(2.0)
                bar.set_edgecolor("#f7fbff")
            if i == pred:
                bar.set_hatch("//")

        self.ax_decoder.set_ylim(0.0, 1.0)
        self.ax_decoder.set_xticks(xs)
        self.ax_decoder.set_xticklabels([SHORT_CLASS_NAMES.get(i, str(i)) for i in xs], color="#c6d2ff")
        self.ax_decoder.tick_params(axis="y", colors="#9aa7d8")
        self.ax_decoder.grid(alpha=0.18, axis="y")

        legend = "white border: target, hatched: predicted"
        self.ax_decoder.set_title(
            f"Decoder confidence  |  {legend}",
            color="#dbe2ff",
            fontsize=10,
        )

        y_text = 0.95
        self.ax_decoder.text(
            0.02,
            y_text,
            f"pred={CLASS_NAMES.get(pred, str(pred))} ({step.confidence:.2f})",
            transform=self.ax_decoder.transAxes,
            color="#dbe2ff",
            fontsize=9,
            va="top",
        )
        if target >= 0:
            self.ax_decoder.text(
                0.02,
                y_text - 0.12,
                f"target={CLASS_NAMES.get(target, str(target))}",
                transform=self.ax_decoder.transAxes,
                color=CLASS_COLORS.get(target, "#dbe2ff"),
                fontsize=9,
                va="top",
            )

        if step.training is not None and step.training.update_applied:
            self.ax_decoder.text(
                0.56,
                0.95,
                (
                    f"BA={step.training.balanced_accuracy:.2f}\n"
                    f"F1={step.training.macro_f1:.2f}\n"
                    f"Top2={step.training.top2_accuracy:.2f}\n"
                    f"ECE={step.training.expected_calibration_error:.2f}"
                ),
                transform=self.ax_decoder.transAxes,
                color="#dbe2ff",
                fontsize=8.5,
                va="top",
                linespacing=1.35,
            )

    def _draw_components(self, step: InferenceStep) -> None:
        self.ax_components.cla()
        self.ax_components.set_facecolor("#131f39")

        components = dict(step.game_reward_components or {})
        if not components:
            self.ax_components.set_xticks([])
            self.ax_components.set_yticks([])
            self.ax_components.text(
                0.04,
                0.5,
                "Reward components not available yet",
                transform=self.ax_components.transAxes,
                color="#c7d1f2",
                fontsize=9,
                va="center",
            )
            self.ax_components.set_title("Reward breakdown", color="#dbe2ff", fontsize=10)
            return

        ordered_keys = [
            "correctness",
            "margin",
            "separability",
            "timing",
            "stability",
            "aux",
            "hit_bonus",
            "confusion_penalty",
        ]
        labels: list[str] = []
        values: list[float] = []
        for key in ordered_keys:
            if key not in components:
                continue
            value = float(components[key])
            if key == "confusion_penalty":
                value = -value
            labels.append(key.replace("_", " "))
            values.append(float(np.clip(value, -1.0, 1.0)))

        y = np.arange(len(values))
        colors = ["#ff8b8b" if v < 0.0 else "#8be9a3" for v in values]
        self.ax_components.barh(y, values, color=colors, alpha=0.85)
        self.ax_components.axvline(0.0, color="#93a3d6", linewidth=1.0, alpha=0.6)
        self.ax_components.set_yticks(y)
        self.ax_components.set_yticklabels(labels, color="#c6d2ff", fontsize=8)
        self.ax_components.set_xlim(-1.0, 1.0)
        self.ax_components.tick_params(axis="x", colors="#9aa7d8")
        self.ax_components.grid(alpha=0.15, axis="x")
        self.ax_components.set_title(
            f"Reward breakdown (sample reward={step.reward:.2f})",
            color="#dbe2ff",
            fontsize=10,
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
        self.ax_metrics.set_facecolor("#131f39")

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

        self.ax_metrics.imshow(display, vmin=0.0, vmax=1.0, cmap="YlGnBu", aspect="auto")
        self.ax_metrics.set_xticks(np.arange(4))
        self.ax_metrics.set_yticks(np.arange(4))
        self.ax_metrics.set_xticklabels([SHORT_CLASS_NAMES[c] for c in range(4)], color="#c6d2ff")
        self.ax_metrics.set_yticklabels([SHORT_CLASS_NAMES[c] for c in range(4)], color="#c6d2ff")
        self.ax_metrics.set_xlabel("Predicted", color="#9aa7d8", fontsize=8)
        self.ax_metrics.set_ylabel("Target", color="#9aa7d8", fontsize=8)
        self.ax_metrics.tick_params(axis="both", labelsize=8)

        for i in range(4):
            for j in range(4):
                val = int(cm[i, j])
                self.ax_metrics.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    color="#0b1220" if display[i, j] > 0.45 else "#eef3ff",
                    fontsize=7.5,
                    fontweight="bold" if i == j else "normal",
                )

        corr = float(np.mean(correct[-50:])) if correct.size > 0 else 0.0
        timing = float(np.mean(timing_hits[-50:])) if timing_hits.size > 0 else 0.0
        margin_mean = float(np.mean(margins[-50:])) if margins.size > 0 else 0.0
        self.ax_metrics.set_title(
            (
                "Confusion matrix (row-normalized)"
                f"  |  correct={corr:.2f} timing={timing:.2f} margin={margin_mean:.2f}"
            ),
            color="#dbe2ff",
            fontsize=10,
        )

        confusion = self._dominant_confusion(targets, preds)
        self.ax_metrics.text(
            0.02,
            0.02,
            confusion,
            transform=self.ax_metrics.transAxes,
            color="#c7d1f2",
            fontsize=8,
            verticalalignment="bottom",
        )

    def _draw_trends(self, rewards: np.ndarray, margins: np.ndarray, correct: np.ndarray) -> None:
        self.ax_trend.cla()
        self.ax_trend.set_facecolor("#131f39")

        x = np.arange(len(rewards))
        self.ax_trend.plot(x, self._rolling(rewards, 16), color="#7dcfff", linewidth=1.6, label="reward")
        self.ax_trend.plot(x, self._rolling(margins, 16), color="#bb9af7", linewidth=1.6, label="margin")
        self.ax_trend.plot(
            x,
            self._rolling(correct.astype(float), 16),
            color="#9ece6a",
            linewidth=1.6,
            label="label-correct",
        )

        if len(self._balanced_accuracy) > 0:
            ba = self._rolling(np.asarray(self._balanced_accuracy, dtype=float), 14)
            f1 = self._rolling(np.asarray(self._macro_f1, dtype=float), 14)
            self.ax_trend.plot(np.arange(len(ba)), ba, color="#f5b971", linewidth=1.3, label="balanced-acc")
            self.ax_trend.plot(np.arange(len(f1)), f1, color="#f98aa4", linewidth=1.3, label="macro-f1")

        self.ax_trend.set_ylim(-0.1, 1.05)
        self.ax_trend.tick_params(colors="#9aa7d8")
        self.ax_trend.grid(alpha=0.18)
        self.ax_trend.set_title("Progress trends (rolling)", color="#dbe2ff", fontsize=10)
        self.ax_trend.legend(loc="upper right", fontsize=8, framealpha=0.15)

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

    def _chip(self, ax, x: float, y: float, text: str, bg: str) -> None:
        ax.text(
            x,
            y,
            text,
            color="#eef3ff",
            fontsize=8.8,
            va="center",
            ha="left",
            bbox={"boxstyle": "round,pad=0.32", "facecolor": bg, "edgecolor": "none", "alpha": 0.94},
        )

    def _refresh_control_buttons(self) -> None:
        track_color = "#3a7a5f" if self.auto_tracking else "#9a7433"
        self.ax_track_button.set_facecolor(track_color)
        self.btn_track.color = track_color
        self.btn_track.hovercolor = "#6ca58f" if self.auto_tracking else "#c8a15b"
        self.btn_track.label.set_text("Tracking ON" if self.auto_tracking else "Tracking OFF")
        self.btn_track.label.set_color("#f5f8ff")
        self.btn_track.label.set_fontsize(9)

        if self._set_simulation_enabled is None:
            sim_label = "Simulation N/A"
            sim_color = "#4b4f5d"
            sim_hover = "#626877"
        else:
            sim_label = "Simulation ON" if self.simulation_enabled else "Simulation OFF"
            sim_color = "#425e9a" if self.simulation_enabled else "#374565"
            sim_hover = "#607ec2" if self.simulation_enabled else "#4d628f"

        self.ax_sim_button.set_facecolor(sim_color)
        self.btn_sim.color = sim_color
        self.btn_sim.hovercolor = sim_hover
        self.btn_sim.label.set_text(sim_label)
        self.btn_sim.label.set_color("#f5f8ff")
        self.btn_sim.label.set_fontsize(9)

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
