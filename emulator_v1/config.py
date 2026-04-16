"""
Difficulty configurations for the Brain Emulator.

Each difficulty defines a per-class temporal disturbance pattern applied to
z_strategy every step.  The player must learn to counter-balance it with arrows
to keep z_strategy near (0, 0) — the single optimal strategy for all classes.

disturbance_fn(t, class_idx) -> np.ndarray([dx, dy])
  Returns a unit-ish direction vector.  Scaled by disturbance_amplitude each step.
  t is time in seconds, class_idx is 0-3.

Design principle: each class gets an "analogical" version of the same pattern
(rotated direction or shifted phase) so the motor challenge is symmetric.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Disturbance patterns (one per difficulty)
# ---------------------------------------------------------------------------

def _d1(t: float, cls: int) -> np.ndarray:
    """
    Single-axis regular pulses.
    2 pulses/second, 300 ms ON / 200 ms OFF.
    Each class has a different cardinal direction: R, U, L, D.
    Counter: tap the opposing arrow ~2x/second in sync.
    """
    dirs   = np.array([[1,0], [0,1], [-1,0], [0,-1]], float)
    period, on = 0.5, 0.30
    active = (t % period) < on
    return dirs[cls] * (1.0 if active else 0.0)


def _d2(t: float, cls: int) -> np.ndarray:
    """
    Diagonal pulses with 2:1 axis ratio.
    1.5 pulses/second, 350 ms ON.
    Each class has a different diagonal corner direction.
    Counter: hold diagonal arrows (both axes) at the right rhythm, ratio 2:1.
    """
    dirs   = np.array([[-1,-2], [1,-2], [-1,2], [1,2]], float)
    dirs   = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)  # normalise
    period, on = 1.0 / 1.5, 0.35
    active = (t % period) < on
    return dirs[cls] * (1.0 if active else 0.0)


def _d3(t: float, cls: int) -> np.ndarray:
    """
    Double-tap rhythm: two short 120 ms bursts separated by 100 ms, then
    750 ms rest (total period 1 s).
    Classes are rotated 90° from d1 (R→D, U→L, L→U, D→R) so the player
    must learn a different axis mapping on top of the new rhythm.
    Counter: two quick taps in sync, then release — holding continuously
    over-corrects during the rest phase.
    """
    dirs  = np.array([[1,0], [0,-1], [-1,0], [0,1]], float)
    phase = t % 1.0
    active = (0.00 <= phase < 0.12) or (0.22 <= phase < 0.34)
    return dirs[cls] * (1.0 if active else 0.0)


def _d4(t: float, cls: int) -> np.ndarray:
    """
    Continuously rotating disturbance — 1 full rotation per 2.5 s.
    Each class starts at a different angle (0°, 90°, 180°, 270°).
    Counter: smoothly rotate through arrow combinations; no single direction works.
    This is the hardest purely directional challenge.
    """
    start = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    omega = 2.0 * np.pi / 2.5
    angle = omega * t + start[cls]
    return np.array([np.cos(angle), np.sin(angle)])


def _d5(t: float, cls: int) -> np.ndarray:
    """
    Two independent axes driven at different frequencies (square waves).
    x and y must be controlled independently at their own rhythms.
    Each class has a different frequency pair — the player must learn
    two simultaneous counter-rhythms.
    freq pairs (fx, fy): (3,1.5), (2,2.5), (1.5,3), (2.5,2)
    Counter: tap L/R at fx and U/D at fy independently.
    """
    pairs = [(3.0, 1.5), (2.0, 2.5), (1.5, 3.0), (2.5, 2.0)]
    fx, fy = pairs[cls]
    dx = 1.0 if np.sin(2 * np.pi * fx * t) >= 0 else -1.0
    dy = 1.0 if np.sin(2 * np.pi * fy * t) >= 0 else -1.0
    return np.array([dx, dy]) * 0.5   # half amplitude each axis (total ≈ 0.7)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class DifficultyConfig:
    name: str

    # Temporal disturbance applied to z_strategy per step (see functions above)
    disturbance_fn:        Callable = field(compare=False)
    disturbance_amplitude: float = 0.07   # scale on disturbance_fn output per step

    # Observation noise (added to final 256-dim signal)
    gaussian_noise_std: float = 0.5

    # Latent noise on z_class (background neural noise, always present)
    latent_noise_std: float = 0.05

    # How fast z_class is pulled toward the active class centroid
    class_pull_strength: float = 0.28

    # How fast arrow keys move z_strategy
    strategy_speed: float = 0.08

    # How steeply rotation reacts to z_strategy deviation from (0, 0)
    strategy_sensitivity: float = 2.0

    n_classes: int = 4


# ---------------------------------------------------------------------------
# Five difficulty levels
# ---------------------------------------------------------------------------

DIFFICULTIES: dict[str, DifficultyConfig] = {
    "d1": DifficultyConfig(
        name="d1",
        disturbance_fn=_d1,
        disturbance_amplitude=0.065,
        gaussian_noise_std=0.35,
        latent_noise_std=0.04,
        class_pull_strength=0.30,
        strategy_speed=0.09,
        strategy_sensitivity=1.8,
    ),
    "d2": DifficultyConfig(
        name="d2",
        disturbance_fn=_d2,
        disturbance_amplitude=0.075,
        gaussian_noise_std=0.50,
        latent_noise_std=0.06,
        class_pull_strength=0.26,
        strategy_speed=0.08,
        strategy_sensitivity=2.0,
    ),
    "d3": DifficultyConfig(
        name="d3",
        disturbance_fn=_d3,
        disturbance_amplitude=0.080,
        gaussian_noise_std=0.65,
        latent_noise_std=0.08,
        class_pull_strength=0.24,
        strategy_speed=0.08,
        strategy_sensitivity=2.2,
    ),
    "d4": DifficultyConfig(
        name="d4",
        disturbance_fn=_d4,
        disturbance_amplitude=0.090,
        gaussian_noise_std=0.85,
        latent_noise_std=0.10,
        class_pull_strength=0.22,
        strategy_speed=0.09,
        strategy_sensitivity=2.5,
    ),
    "d5": DifficultyConfig(
        name="d5",
        disturbance_fn=_d5,
        disturbance_amplitude=0.095,
        gaussian_noise_std=1.10,
        latent_noise_std=0.12,
        class_pull_strength=0.20,
        strategy_speed=0.09,
        strategy_sensitivity=2.8,
    ),
}

# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

CLASS_NAMES: dict[int, str] = {
    0: "left_hand",
    1: "right_hand",
    2: "left_leg",
    3: "right_leg",
}

CLASS_COLORS: dict[int | None, tuple[int, int, int]] = {
    0:    (100, 149, 237),   # cornflower blue  — left_hand
    1:    (255, 165,   0),   # orange           — right_hand
    2:    ( 50, 205,  50),   # lime green       — left_leg
    3:    (220,  20,  60),   # crimson          — right_leg
    None: (120, 120, 120),
}
