from __future__ import annotations

from collections import deque

import numpy as np

from .types import PromptEvent


def target_margin(probs: np.ndarray, target_class: int) -> float:
    target_prob = float(probs[target_class])
    mask = np.ones_like(probs, dtype=bool)
    mask[target_class] = False
    competitor = float(np.max(probs[mask]))
    return target_prob - competitor


def margin_score(margin: float) -> float:
    return float(np.clip((margin + 1.0) * 0.5, 0.0, 1.0))


def separability_score(margin: float, margin_ema: float) -> float:
    delta = margin - margin_ema
    return float(np.clip(0.5 + 2.0 * delta, 0.0, 1.0))


def timing_score(relative_time_s: float, prompt: PromptEvent) -> tuple[float, float, bool]:
    in_window = prompt.hit_window_start_s <= relative_time_s <= prompt.hit_window_end_s
    timing_error_s = abs(relative_time_s - prompt.center_s)

    half_window = 0.5 * (prompt.hit_window_end_s - prompt.hit_window_start_s)
    if not in_window or half_window <= 0.0:
        return 0.0, float(timing_error_s), bool(in_window)

    score = np.clip(1.0 - (timing_error_s / half_window), 0.0, 1.0)
    return float(score), float(timing_error_s), True


def stability_score(target_prob_history: deque[float]) -> float:
    if not target_prob_history:
        return 0.0
    if len(target_prob_history) == 1:
        return float(np.clip(target_prob_history[-1], 0.0, 1.0))

    stdev = float(np.std(np.array(target_prob_history, dtype=float)))
    return float(np.clip(1.0 - stdev / 0.25, 0.0, 1.0))


def aux_quality(class_scale: float | None, strategy_quality: float | None) -> float:
    values = [v for v in (class_scale, strategy_quality) if v is not None]
    if not values:
        return 0.0
    return float(np.clip(float(np.mean(values)), 0.0, 1.0))


def confusion_penalty(
    probs: np.ndarray,
    target_class: int,
    in_window: bool,
    threshold: float,
) -> float:
    if not in_window:
        return 0.0

    pred = int(np.argmax(probs))
    confidence = float(probs[pred])
    if pred != target_class and confidence >= threshold:
        return float(np.clip(confidence, 0.0, 1.0))
    return 0.0
