from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import ModelConfig
from .types import StreamSample


class RewardProvider(Protocol):
    def compute(self, sample: StreamSample, probs: np.ndarray) -> float:
        """Return a scalar reward for one timestep."""


class ProgrammaticReward:
    """
    Default reward used for online adaptation before external game rewards exist.

    Components can be computed purely from stream metadata and model confidence.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self.w_conf = cfg.reward_confidence_weight
        self.w_correct = cfg.reward_correct_weight
        self.w_class_scale = cfg.reward_class_scale_weight
        self.w_strategy = cfg.reward_strategy_quality_weight
        self.r_min = cfg.reward_min
        self.r_max = cfg.reward_max

    def compute(self, sample: StreamSample, probs: np.ndarray) -> float:
        probs = np.asarray(probs, dtype=float)
        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted])

        correctness = 0.0
        if sample.label is not None:
            correctness = 1.0 if predicted == int(sample.label) else 0.0

        class_scale = float(sample.class_scale) if sample.class_scale is not None else 0.0
        strategy = float(sample.strategy_quality) if sample.strategy_quality is not None else 0.0

        reward = (
            self.w_conf * confidence
            + self.w_correct * correctness
            + self.w_class_scale * class_scale
            + self.w_strategy * strategy
        )
        return float(np.clip(reward, self.r_min, self.r_max))
