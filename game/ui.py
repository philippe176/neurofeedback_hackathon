from __future__ import annotations

from collections import deque

import matplotlib.pyplot as plt
import numpy as np

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
    Rich game dashboard that keeps the prompt cue and embedding-space evolution
    visible at the same time.
    """

    def __init__(self, history_len: int = 320, draw_every: int = 2) -> None:
        self.history_len = max(40, int(history_len))
        self.draw_every = max(1, int(draw_every))

        self._counter = 0
        self._points: deque[np.ndarray] = deque(maxlen=self.history_len)
        self._preds: deque[int] = deque(maxlen=self.history_len)
        self._targets: deque[int] = deque(maxlen=self.history_len)
        self._confidences: deque[float] = deque(maxlen=self.history_len)
        self._rewards: deque[float] = deque(maxlen=self.history_len)
        self._margins: deque[float] = deque(maxlen=self.history_len)
        self._correct: deque[bool] = deque(maxlen=self.history_len)
        self._timing_hits: deque[bool] = deque(maxlen=self.history_len)

        plt.ion()
        self.fig = plt.figure(figsize=(15, 10), constrained_layout=False)
        gs = self.fig.add_gridspec(
            5,
            2,
            width_ratios=[2.15, 1.3],
            height_ratios=[1.2, 1.0, 1.0, 1.0, 1.0],
        )
        self.ax_space = self.fig.add_subplot(gs[:, 0])
        self.ax_cue = self.fig.add_subplot(gs[0, 1])
        self.ax_decoder = self.fig.add_subplot(gs[1, 1])
        self.ax_components = self.fig.add_subplot(gs[2, 1])
        self.ax_metrics = self.fig.add_subplot(gs[3, 1])
        self.ax_trend = self.fig.add_subplot(gs[4, 1])

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

        self._counter += 1
        if self._counter % self.draw_every == 0:
            self._draw(step)

    def close(self) -> None:
        plt.close(self.fig)

    def _draw(self, step: InferenceStep) -> None:
        points = np.stack(self._points, axis=0)
        preds = np.asarray(self._preds, dtype=int)
        targets = np.asarray(self._targets, dtype=int)
        rewards = np.asarray(self._rewards, dtype=float)
        margins = np.asarray(self._margins, dtype=float)
        correct = np.asarray(self._correct, dtype=bool)
        timing_hits = np.asarray(self._timing_hits, dtype=bool)
        confidences = np.asarray(self._confidences, dtype=float)

        self._draw_space(points, preds, correct, confidences)
        self._draw_cue(step)
        self._draw_decoder(step)
        self._draw_components(step)
        self._draw_metrics(preds, targets, correct, timing_hits, margins)
        self._draw_trends(rewards, margins, correct)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        backend = plt.get_backend().lower()
        if "agg" not in backend:
            plt.pause(0.001)

    def _draw_space(
        self,
        points: np.ndarray,
        preds: np.ndarray,
        correct: np.ndarray,
        confidences: np.ndarray,
    ) -> None:
        self.ax_space.cla()
        self.ax_space.set_facecolor("#11152a")

        for cls in sorted(CLASS_COLORS):
            mask = preds == cls
            if not np.any(mask):
                continue
            self.ax_space.scatter(
                points[mask, 0],
                points[mask, 1],
                s=20 + 60 * np.clip(confidences[mask], 0.0, 1.0),
                color=CLASS_COLORS[cls],
                alpha=0.55,
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

        sep_idx = self._separability_index(points, preds)
        rolling_correct = float(np.mean(correct[-50:])) if correct.size > 0 else 0.0
        self.ax_space.set_title(
            f"Embedding Space  |  separability={sep_idx:.2f}  |  label-correct={rolling_correct:.2f}",
            color="#dbe2ff",
            fontsize=11,
        )
        self.ax_space.set_xlabel("Projection x", color="#a6b0cf")
        self.ax_space.set_ylabel("Projection y", color="#a6b0cf")
        self.ax_space.grid(alpha=0.18)
        self.ax_space.tick_params(colors="#9aa7d8")
        self.ax_space.legend(loc="upper right", fontsize=8, framealpha=0.15)

    def _draw_cue(self, step: InferenceStep) -> None:
        self.ax_cue.cla()
        self.ax_cue.set_facecolor("#141a30")
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
            "TASK COACH",
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
        self.ax_decoder.set_facecolor("#141a30")

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

        legend = "white border = target, hatched = model prediction"
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

    def _draw_components(self, step: InferenceStep) -> None:
        self.ax_components.cla()
        self.ax_components.set_facecolor("#141a30")

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
        self.ax_metrics.set_facecolor("#141a30")

        per_class = []
        for cls in range(4):
            mask = targets == cls
            if np.any(mask):
                per_class.append(float(np.mean(preds[mask] == cls)))
            else:
                per_class.append(0.0)

        xs = np.arange(4)
        colors = [CLASS_COLORS[c] for c in range(4)]
        self.ax_metrics.bar(xs, per_class, color=colors, alpha=0.85)
        self.ax_metrics.set_ylim(0.0, 1.0)
        self.ax_metrics.set_xticks(xs)
        self.ax_metrics.set_xticklabels([SHORT_CLASS_NAMES[c] for c in range(4)], color="#c6d2ff")
        self.ax_metrics.tick_params(axis="y", colors="#9aa7d8")
        self.ax_metrics.grid(alpha=0.18, axis="y")

        corr = float(np.mean(correct[-50:])) if correct.size > 0 else 0.0
        timing = float(np.mean(timing_hits[-50:])) if timing_hits.size > 0 else 0.0
        margin_mean = float(np.mean(margins[-50:])) if margins.size > 0 else 0.0
        self.ax_metrics.set_title(
            f"Class distinction quality (recent)  |  correct={corr:.2f}  timing={timing:.2f}  margin={margin_mean:.2f}",
            color="#dbe2ff",
            fontsize=10,
        )

        confusion = self._dominant_confusion(targets, preds)
        self.ax_metrics.text(
            0.02,
            0.94,
            confusion,
            transform=self.ax_metrics.transAxes,
            color="#c7d1f2",
            fontsize=9,
            verticalalignment="top",
        )

    def _draw_trends(self, rewards: np.ndarray, margins: np.ndarray, correct: np.ndarray) -> None:
        self.ax_trend.cla()
        self.ax_trend.set_facecolor("#141a30")

        x = np.arange(len(rewards))
        self.ax_trend.plot(x, self._rolling(rewards, 12), color="#7dcfff", linewidth=1.6, label="reward")
        self.ax_trend.plot(x, self._rolling(margins, 12), color="#bb9af7", linewidth=1.6, label="margin")
        self.ax_trend.plot(
            x,
            self._rolling(correct.astype(float), 12),
            color="#9ece6a",
            linewidth=1.6,
            label="label-correct",
        )
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
            out[i] = float(np.mean(values[lo : i + 1]))
        return out

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
