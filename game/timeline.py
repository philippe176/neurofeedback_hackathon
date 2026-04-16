from __future__ import annotations

import numpy as np

from .config import LevelPolicy, RhythmGameConfig
from .types import PromptEvent


class PromptTimeline:
    """Deterministic prompt generator with no immediate class repetition."""

    def __init__(self, cfg: RhythmGameConfig) -> None:
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        self._last_target: int | None = None
        self._next_prompt_id = 0

    def _choose_target(self, last_target: int | None) -> int:
        candidates = np.arange(self.cfg.n_classes)
        if last_target is not None and self.cfg.n_classes > 1:
            candidates = candidates[candidates != last_target]
        return int(self._rng.choice(candidates))

    def next_target_class(self) -> int:
        target = self._choose_target(self._last_target)
        self._last_target = target
        return target

    def peek_next_target(self, last_target_override: int | None = None) -> int:
        rng_state = self._rng.bit_generator.state
        try:
            last = self._last_target if last_target_override is None else last_target_override
            return self._choose_target(last)
        finally:
            self._rng.bit_generator.state = rng_state

    def make_prompt(self, start_s: float, level: int, level_policy: LevelPolicy) -> PromptEvent:
        target = self.next_target_class()
        end_s = start_s + self.cfg.prompt_duration_s
        center_s = start_s + 0.5 * self.cfg.prompt_duration_s
        half_window = 0.5 * level_policy.hit_window_s
        hit_start = max(start_s, center_s - half_window)
        hit_end = min(end_s, center_s + half_window)

        prompt = PromptEvent(
            prompt_id=self._next_prompt_id,
            target_class=target,
            level=level,
            start_s=start_s,
            end_s=end_s,
            center_s=center_s,
            hit_window_start_s=hit_start,
            hit_window_end_s=hit_end,
        )
        self._next_prompt_id += 1
        return prompt
