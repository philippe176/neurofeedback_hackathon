from __future__ import annotations

from collections import deque

import numpy as np

from model.types import StreamSample

from .config import LevelPolicy, RhythmGameConfig
from .scoring import (
    aux_quality,
    confusion_penalty,
    margin_score,
    separability_score,
    stability_score,
    target_margin,
    timing_score,
)
from .timeline import PromptTimeline
from .types import GameFeedback, GamePreview, PromptEvent, RewardComponents


class GameSession:
    """
    Tracks prompt timeline, timing windows, streaks, and adaptive difficulty state.

    v1 defaults to fixed strictness. Adaptive policy can be enabled via config.
    """

    def __init__(self, cfg: RhythmGameConfig) -> None:
        self.cfg = cfg
        self.timeline = PromptTimeline(cfg)

        self.current_level = cfg.start_level
        self.current_prompt: PromptEvent | None = None
        self._current_prompt_correct = False
        self._current_prompt_timing_hit = False
        self._current_prompt_best_margin = -1.0

        self._origin_time_s: float | None = None
        self._margin_ema = 0.0
        self._margin_ema_initialized = False

        self._target_prob_history: deque[float] = deque(maxlen=cfg.stability_window)

        self.streak = 0
        self.best_streak = 0
        self.total_prompts = 0
        self.total_hits = 0

        self._prompts_since_eval = 0
        self._hits_since_eval = 0
        self._margins_since_eval: list[float] = []

        self.last_preview: GamePreview | None = None
        self.last_feedback: GameFeedback | None = None

    @property
    def level_policy(self) -> LevelPolicy:
        return self.cfg.levels[self.current_level]

    def snapshot(self) -> dict[str, float | int]:
        hit_rate = self.total_hits / max(1, self.total_prompts)
        return {
            "level": self.current_level,
            "streak": self.streak,
            "best_streak": self.best_streak,
            "total_prompts": self.total_prompts,
            "total_hits": self.total_hits,
            "hit_rate": hit_rate,
        }

    def step(self, sample: StreamSample, probs: np.ndarray) -> GameFeedback:
        probs = np.asarray(probs, dtype=float)
        preview = self.preview(sample.timestamp)
        rel_time_s = preview.relative_time_s

        prompt = self._require_current_prompt()
        target_class = prompt.target_class

        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        margin = target_margin(probs, target_class)

        if not self._margin_ema_initialized:
            self._margin_ema = margin
            self._margin_ema_initialized = True

        sep_score = separability_score(margin, self._margin_ema)
        self._margin_ema = (1.0 - self.cfg.margin_ema_alpha) * self._margin_ema + self.cfg.margin_ema_alpha * margin

        timing, timing_error_s, in_window = timing_score(rel_time_s, prompt)

        target_prob = float(probs[target_class])
        self._target_prob_history.append(target_prob)
        stability = stability_score(self._target_prob_history)

        aux = aux_quality(sample.class_scale, sample.strategy_quality)
        confusion = confusion_penalty(
            probs=probs,
            target_class=target_class,
            in_window=in_window,
            threshold=self.cfg.confusion_confidence_threshold,
        )

        policy = self.level_policy
        threshold_ok = confidence >= policy.min_confidence and margin >= policy.min_margin
        label_correct_now = bool(predicted_class == target_class and threshold_ok)
        timing_hit_now = bool(label_correct_now and in_window)

        new_correct = label_correct_now and not self._current_prompt_correct
        new_timing_hit = timing_hit_now and not self._current_prompt_timing_hit

        hit_bonus = 0.0
        if new_correct:
            self._current_prompt_correct = True
            self.total_hits += 1
            self._hits_since_eval += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
            hit_bonus = self.cfg.hit_bonus
        if new_timing_hit:
            self._current_prompt_timing_hit = True

        self._current_prompt_best_margin = max(self._current_prompt_best_margin, margin)

        components = RewardComponents(
            correctness=1.0 if label_correct_now else 0.0,
            margin=margin_score(margin),
            separability=sep_score,
            timing=timing if label_correct_now else 0.0,
            stability=stability,
            aux=aux,
            confusion_penalty=confusion,
            hit_bonus=hit_bonus,
        )

        reward = self._compose_reward(components)

        feedback = GameFeedback(
            sample_idx=sample.sample_idx,
            timestamp=sample.timestamp,
            prompt_id=prompt.prompt_id,
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            margin=float(margin),
            in_window=in_window,
            label_correct=label_correct_now,
            timing_hit=timing_hit_now,
            timing_error_s=float(timing_error_s),
            seconds_to_window_start=preview.seconds_to_window_start,
            next_target_class=preview.next_target_class,
            seconds_to_next_prompt_start=preview.seconds_to_next_prompt_start,
            hit=bool(new_correct),
            streak=self.streak,
            level=self.current_level,
            reward=reward,
            components=components.as_dict(),
            prompt_progress=preview.prompt_progress,
        )
        self.last_feedback = feedback
        return feedback

    def preview(self, timestamp: float) -> GamePreview:
        rel_time_s = self._relative_time(timestamp)
        self._ensure_active_prompt(rel_time_s)

        prompt = self._require_current_prompt()
        policy = self.level_policy

        in_window = prompt.hit_window_start_s <= rel_time_s <= prompt.hit_window_end_s
        prompt_progress = float(
            np.clip((rel_time_s - prompt.start_s) / max(self.cfg.prompt_duration_s, 1e-6), 0.0, 1.0)
        )

        next_start_s = prompt.start_s + policy.beat_interval_s
        preview = GamePreview(
            relative_time_s=rel_time_s,
            prompt_id=prompt.prompt_id,
            target_class=prompt.target_class,
            in_window=bool(in_window),
            prompt_progress=prompt_progress,
            seconds_to_window_start=float(max(0.0, prompt.hit_window_start_s - rel_time_s)),
            seconds_to_prompt_end=float(max(0.0, prompt.end_s - rel_time_s)),
            next_target_class=self.timeline.peek_next_target(last_target_override=prompt.target_class),
            seconds_to_next_prompt_start=float(max(0.0, next_start_s - rel_time_s)),
        )
        self.last_preview = preview
        return preview

    def _relative_time(self, timestamp: float) -> float:
        if self._origin_time_s is None:
            self._origin_time_s = timestamp
        return float(timestamp - self._origin_time_s)

    def _ensure_active_prompt(self, rel_time_s: float) -> None:
        if self.current_prompt is None:
            self._start_prompt(start_s=0.0)

        prompt = self._require_current_prompt()
        while rel_time_s > prompt.end_s:
            prev_prompt = prompt
            prev_policy = self.cfg.levels[prev_prompt.level]
            self._finalize_prompt()

            next_start_s = prev_prompt.start_s + prev_policy.beat_interval_s
            self._start_prompt(start_s=next_start_s)
            prompt = self._require_current_prompt()

    def _start_prompt(self, start_s: float) -> None:
        self.current_prompt = self.timeline.make_prompt(
            start_s=start_s,
            level=self.current_level,
            level_policy=self.level_policy,
        )
        self._current_prompt_correct = False
        self._current_prompt_timing_hit = False
        self._current_prompt_best_margin = -1.0

    def _finalize_prompt(self) -> None:
        self.total_prompts += 1
        self._prompts_since_eval += 1
        self._margins_since_eval.append(self._current_prompt_best_margin)

        if not self._current_prompt_correct:
            self.streak = 0

        if self.cfg.enable_adaptation and self._prompts_since_eval >= self.cfg.adaptation_eval_prompts:
            self._apply_adaptation()

    def _apply_adaptation(self) -> None:
        hit_rate = self._hits_since_eval / max(1, self._prompts_since_eval)
        margins = np.asarray(self._margins_since_eval, dtype=float)
        mean_margin = float(np.mean(margins)) if margins.size > 0 else 0.0

        should_promote = (
            hit_rate >= self.cfg.promote_hit_rate
            and mean_margin >= self.cfg.promote_margin
            and self.current_level < len(self.cfg.levels) - 1
        )
        should_demote = (
            (hit_rate <= self.cfg.demote_hit_rate or mean_margin <= self.cfg.demote_margin)
            and self.current_level > 0
        )

        if should_promote:
            self.current_level += 1
        elif should_demote:
            self.current_level -= 1

        self._prompts_since_eval = 0
        self._hits_since_eval = 0
        self._margins_since_eval.clear()

    def _compose_reward(self, c: RewardComponents) -> float:
        # Distinction-first reward: correct label, margin, and separability dominate.
        reward = (
            self.cfg.w_correctness * c.correctness
            + self.cfg.w_margin * c.margin
            + self.cfg.w_separability * c.separability
            + self.cfg.w_timing * c.timing
            + self.cfg.w_stability * c.stability
            + self.cfg.w_aux * c.aux
            + c.hit_bonus
            - self.cfg.w_confusion_penalty * c.confusion_penalty
        )
        return float(np.clip(reward, self.cfg.reward_min, self.cfg.reward_max))

    def _require_current_prompt(self) -> PromptEvent:
        if self.current_prompt is None:
            raise RuntimeError("GameSession has no active prompt")
        return self.current_prompt
