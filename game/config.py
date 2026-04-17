from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class LevelPolicy:
    hit_window_s: float
    beat_interval_s: float
    min_confidence: float
    min_margin: float

    def __post_init__(self) -> None:
        if self.hit_window_s <= 0.0:
            raise ValueError("hit_window_s must be > 0")
        if self.beat_interval_s <= 0.0:
            raise ValueError("beat_interval_s must be > 0")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be in [0, 1]")


@dataclass(slots=True)
class RhythmGameConfig:
    n_classes: int = 4
    prompt_duration_s: float = 2.0
    base_hit_window_s: float = 0.75
    base_beat_interval_s: float = 2.4
    seed: int = 7

    # Automatic high-performance simulation for demos/training warm starts.
    auto_perform: bool = False
    auto_perform_strength: float = 0.92
    auto_prewindow_strength: float = 0.80
    auto_blend: float = 0.90
    auto_anticipation_s: float = 1.0

    # Reward composition with strong label-distinction priority.
    w_correctness: float = 0.40
    w_margin: float = 0.28
    w_separability: float = 0.20
    w_timing: float = 0.04
    w_stability: float = 0.06
    w_aux: float = 0.02
    w_confusion_penalty: float = 0.35
    hit_bonus: float = 0.06

    reward_min: float = 0.0
    reward_max: float = 1.0
    confusion_confidence_threshold: float = 0.65

    # Session smoothing.
    margin_ema_alpha: float = 0.08
    stability_window: int = 8

    # Adaptation policy (disabled by default for v1 fixed strictness).
    enable_adaptation: bool = False
    adaptation_eval_prompts: int = 20
    promote_hit_rate: float = 0.80
    promote_margin: float = 0.22
    demote_hit_rate: float = 0.55
    demote_margin: float = 0.08

    start_level: int = 0
    levels: tuple[LevelPolicy, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if self.prompt_duration_s <= 0.0:
            raise ValueError("prompt_duration_s must be > 0")
        if self.base_hit_window_s <= 0.0:
            raise ValueError("base_hit_window_s must be > 0")
        if self.base_beat_interval_s <= 0.0:
            raise ValueError("base_beat_interval_s must be > 0")
        if self.stability_window < 2:
            raise ValueError("stability_window must be >= 2")
        if not (0.0 <= self.auto_perform_strength <= 1.0):
            raise ValueError("auto_perform_strength must be in [0,1]")
        if not (0.0 <= self.auto_prewindow_strength <= 1.0):
            raise ValueError("auto_prewindow_strength must be in [0,1]")
        if not (0.0 <= self.auto_blend <= 1.0):
            raise ValueError("auto_blend must be in [0,1]")
        if self.auto_anticipation_s < 0.0:
            raise ValueError("auto_anticipation_s must be >= 0")
        if not (0.0 <= self.confusion_confidence_threshold <= 1.0):
            raise ValueError("confusion_confidence_threshold must be in [0,1]")
        if not (0.0 <= self.margin_ema_alpha <= 1.0):
            raise ValueError("margin_ema_alpha must be in [0,1]")
        if not (0.0 <= self.promote_hit_rate <= 1.0):
            raise ValueError("promote_hit_rate must be in [0,1]")
        if not (0.0 <= self.demote_hit_rate <= 1.0):
            raise ValueError("demote_hit_rate must be in [0,1]")

        if not self.levels:
            self.levels = (
                LevelPolicy(
                    hit_window_s=max(0.45, self.base_hit_window_s * 1.20),
                    beat_interval_s=max(self.prompt_duration_s * 1.05, self.base_beat_interval_s * 1.18),
                    min_confidence=0.30,
                    min_margin=-0.02,
                ),
                LevelPolicy(
                    hit_window_s=self.base_hit_window_s,
                    beat_interval_s=max(self.prompt_duration_s, self.base_beat_interval_s),
                    min_confidence=0.36,
                    min_margin=0.00,
                ),
                LevelPolicy(
                    hit_window_s=max(0.35, self.base_hit_window_s * 0.75),
                    beat_interval_s=max(self.prompt_duration_s * 0.95, self.base_beat_interval_s * 0.83),
                    min_confidence=0.44,
                    min_margin=0.04,
                ),
            )

        if not (0 <= self.start_level < len(self.levels)):
            raise ValueError("start_level is out of range for configured levels")
        if self.adaptation_eval_prompts < 1:
            raise ValueError("adaptation_eval_prompts must be >= 1")
        if self.reward_max < self.reward_min:
            raise ValueError("reward_max must be >= reward_min")
