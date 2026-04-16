from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PromptEvent:
    prompt_id: int
    target_class: int
    level: int
    start_s: float
    end_s: float
    center_s: float
    hit_window_start_s: float
    hit_window_end_s: float


@dataclass(slots=True)
class GamePreview:
    relative_time_s: float
    prompt_id: int
    target_class: int
    in_window: bool
    prompt_progress: float
    seconds_to_window_start: float
    seconds_to_prompt_end: float
    next_target_class: int
    seconds_to_next_prompt_start: float


@dataclass(slots=True)
class RewardComponents:
    correctness: float
    margin: float
    separability: float
    timing: float
    stability: float
    aux: float
    confusion_penalty: float
    hit_bonus: float

    def as_dict(self) -> dict[str, float]:
        return {
            "correctness": self.correctness,
            "margin": self.margin,
            "separability": self.separability,
            "timing": self.timing,
            "stability": self.stability,
            "aux": self.aux,
            "confusion_penalty": self.confusion_penalty,
            "hit_bonus": self.hit_bonus,
        }


@dataclass(slots=True)
class GameFeedback:
    sample_idx: int
    timestamp: float
    prompt_id: int
    target_class: int
    predicted_class: int
    confidence: float
    margin: float
    in_window: bool
    label_correct: bool
    timing_hit: bool
    timing_error_s: float
    seconds_to_window_start: float
    next_target_class: int
    seconds_to_next_prompt_start: float
    hit: bool
    streak: int
    level: int
    reward: float
    components: dict[str, float]
    prompt_progress: float

    def as_dict(self) -> dict[str, float | int | bool]:
        data: dict[str, float | int | bool] = {
            "prompt_id": self.prompt_id,
            "target_class": self.target_class,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "margin": self.margin,
            "in_window": self.in_window,
            "label_correct": self.label_correct,
            "timing_hit": self.timing_hit,
            "timing_error_s": self.timing_error_s,
            "seconds_to_window_start": self.seconds_to_window_start,
            "next_target_class": self.next_target_class,
            "seconds_to_next_prompt_start": self.seconds_to_next_prompt_start,
            "hit": self.hit,
            "streak": self.streak,
            "level": self.level,
            "reward": self.reward,
            "prompt_progress": self.prompt_progress,
        }
        for key, value in self.components.items():
            data[f"component_{key}"] = value
        return data
