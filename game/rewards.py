from __future__ import annotations

import numpy as np

from model.config import ModelConfig
from model.types import StreamSample

from .config import RhythmGameConfig
from .session import GameSession
from .types import GameFeedback, GamePreview


class GameRewardProvider:
    """Reward provider that scores each sample against the active rhythm prompt."""

    def __init__(self, model_cfg: ModelConfig, game_cfg: RhythmGameConfig) -> None:
        self.model_cfg = model_cfg
        self.session = GameSession(game_cfg)
        self.last_feedback: GameFeedback | None = None

    def preview(self, timestamp: float) -> GamePreview:
        return self.session.preview(timestamp)

    def adjust_probabilities(self, sample: StreamSample, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        cfg = self.session.cfg
        if not cfg.auto_perform:
            return probs

        preview = self.preview(sample.timestamp)
        strength = self._auto_strength(preview)
        if strength <= 0.0:
            return probs

        n = int(probs.size)
        if n < 2 or not (0 <= preview.target_class < n):
            return probs

        target_dist = np.full(n, (1.0 - strength) / max(1, n - 1), dtype=float)
        target_dist[preview.target_class] = strength

        blend = float(np.clip(cfg.auto_blend, 0.0, 1.0))
        adjusted = (1.0 - blend) * probs + blend * target_dist
        adjusted = np.clip(adjusted, 1e-8, None)
        return adjusted / np.sum(adjusted)

    def compute(self, sample: StreamSample, probs: np.ndarray) -> float:
        feedback = self.session.step(sample, probs)
        self.last_feedback = feedback
        sample.raw.setdefault("game", {}).update(feedback.as_dict())

        return float(np.clip(feedback.reward, self.model_cfg.reward_min, self.model_cfg.reward_max))

    def _auto_strength(self, preview: GamePreview) -> float:
        cfg = self.session.cfg
        if preview.in_window:
            return float(np.clip(cfg.auto_perform_strength, 0.0, 1.0))

        if preview.seconds_to_window_start > 0.0:
            if cfg.auto_anticipation_s <= 0.0:
                return 0.0
            if preview.seconds_to_window_start > cfg.auto_anticipation_s:
                return 0.0

            phase = 1.0 - (preview.seconds_to_window_start / max(cfg.auto_anticipation_s, 1e-6))
            strength = cfg.auto_prewindow_strength + phase * (
                cfg.auto_perform_strength - cfg.auto_prewindow_strength
            )
            return float(np.clip(strength, 0.0, 1.0))

        if preview.seconds_to_prompt_end > 0.0:
            return float(np.clip(max(cfg.auto_prewindow_strength, cfg.auto_perform_strength * 0.9), 0.0, 1.0))

        return 0.0
