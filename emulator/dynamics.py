"""
Latent space dynamics for the brain emulator, version 2.

State vector z (8-dimensional):
  dim 0    -> z_coarse   : coarse cluster signal (hands vs legs), never rotated
  dims 1-2 -> z_fine     : within-cluster class signals
  dims 3-4 -> z_strategy : user-controlled strategy state
  dims 5-7 -> z_noise    : slow random-walk noise

The updated emulator uses per-class optimal strategy corners instead of a
single shared optimum. The patient has to move toward and hold the active
class corner with arrow keys while a spring pulls the strategy back to center.
"""

from __future__ import annotations

import numpy as np

from .config import DifficultyConfig


# Class centroids in z_class = [coarse, fine_A, fine_B]
CLASS_CENTROIDS = np.array(
    [
        [+2.0, +1.5, 0.0],   # left hand
        [+2.0, -1.5, 0.0],   # right hand
        [-2.0, 0.0, +1.5],   # left leg
        [-2.0, 0.0, -1.5],   # right leg
    ],
    dtype=float,
)

# Per-class optimal strategy corners in the 2D strategy plane.
OPTIMAL_STRATEGIES = np.array(
    [
        [+0.2, +0.2],   # left hand
        [-0.2, +0.2],   # right hand
        [+0.2, -0.2],   # left leg
        [-0.2, -0.2],   # right leg
    ],
    dtype=float,
)

SCALE_TAU = 3.0

N_CLASS_DIMS = 3
N_STRATEGY_DIMS = 2
N_NOISE_DIMS = 3
N_LATENT = N_CLASS_DIMS + N_STRATEGY_DIMS + N_NOISE_DIMS


def _givens(n: int, i: int, j: int, theta: float) -> np.ndarray:
    rot = np.eye(n)
    c, s = np.cos(theta), np.sin(theta)
    rot[i, i] = c
    rot[i, j] = -s
    rot[j, i] = s
    rot[j, j] = c
    return rot


class LatentDynamics:
    """
    Simulates the patient's latent brain state.

    Strategy behavior:
    - arrow keys move z_strategy
    - a spring always pulls z_strategy back toward (0, 0)
    - each class has its own optimal corner
    - the fine class dimensions become more readable near that corner
    """

    def __init__(self, config: DifficultyConfig, sample_rate: float = 10.0, seed: int = 42):
        self.cfg = config
        self.dt = 1.0 / sample_rate
        self.t = 0.0

        rng = np.random.default_rng(seed)

        self.z_class = np.zeros(N_CLASS_DIMS)
        self.z_strategy = np.zeros(N_STRATEGY_DIMS)
        self.z_noise = np.zeros(N_NOISE_DIMS)

        self.current_class: int | None = None
        self._scale_integrated = 0.0
        self._noise_rng = rng

    def set_class(self, class_idx: int | None) -> None:
        self.current_class = class_idx

    def update_strategy(self, delta: np.ndarray) -> None:
        """Move z_strategy by one arrow-key update, clamped to [-1, 1]^2."""
        self.z_strategy = np.clip(
            self.z_strategy + np.asarray(delta, dtype=float) * self.cfg.strategy_speed,
            -1.0,
            1.0,
        )

    def step(self) -> dict[str, object]:
        cfg = self.cfg

        # Spring pull back to center. Releasing keys recenters the strategy.
        decay = np.exp(-cfg.spring_rate * self.dt)
        self.z_strategy = self.z_strategy * decay

        if self.current_class is not None:
            target = CLASS_CENTROIDS[self.current_class]
            self.z_class += cfg.class_pull_strength * (target - self.z_class)
        else:
            self.z_class *= 1.0 - cfg.class_pull_strength * 0.4

        self.z_class += self._noise_rng.normal(0.0, cfg.latent_noise_std, N_CLASS_DIMS)
        self.z_noise = 0.85 * self.z_noise + self._noise_rng.normal(0.0, 0.25, N_NOISE_DIMS)

        target_scale = self.strategy_quality ** 3
        alpha = self.dt / SCALE_TAU
        self._scale_integrated += alpha * (target_scale - self._scale_integrated)

        self.t += self.dt

        return {
            "z_class": self.z_class.copy(),
            "z_strategy": self.z_strategy.copy(),
            "z_noise": self.z_noise.copy(),
            "current_class": self.current_class,
            "strategy_quality": self.strategy_quality,
            "class_scale": self.class_scale,
            "t": self.t,
        }

    def get_rotation(self) -> np.ndarray:
        """
        Build an 8x8 rotation matrix.

        Dim 0 is never rotated, so the coarse hands-vs-legs split remains.
        Fine dimensions rotate into noise depending on the distance from the
        active class's optimal strategy corner.
        """
        if self.current_class is not None:
            err = self.z_strategy - OPTIMAL_STRATEGIES[self.current_class]
        else:
            err = np.array([1.0, 1.0], dtype=float)

        scale = self.cfg.strategy_sensitivity
        half_pi = np.pi / 2.0

        theta1 = half_pi * np.tanh(scale * err[0])
        theta2 = half_pi * np.tanh(scale * err[1])
        theta3 = half_pi * np.tanh(scale * (err[0] + err[1]) / 2.0)

        return (
            _givens(N_LATENT, 1, 5, theta1)
            @ _givens(N_LATENT, 2, 6, theta2)
            @ _givens(N_LATENT, 1, 7, theta3)
        )

    @property
    def z_full(self) -> np.ndarray:
        return np.concatenate([self.z_class, self.z_strategy, self.z_noise])

    @property
    def strategy_quality(self) -> float:
        """Quality is 1 at the active class optimum and 0 with no active class."""
        if self.current_class is None:
            return 0.0
        error = np.linalg.norm(self.z_strategy - OPTIMAL_STRATEGIES[self.current_class])
        return float(np.exp(-2.5 * error))

    @property
    def class_scale(self) -> float:
        """Integrated signal scale in [0, 1]."""
        return float(self._scale_integrated)

    @property
    def optimal_strategy(self) -> np.ndarray:
        """Return the active class's optimal strategy corner, or zeros at rest."""
        if self.current_class is None:
            return np.zeros(2, dtype=float)
        return OPTIMAL_STRATEGIES[self.current_class].copy()
