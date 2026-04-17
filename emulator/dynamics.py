"""
Latent space dynamics for the brain emulator.

State vector z (8-dimensional):
  dim 0    -> z_coarse   : coarse cluster signal (hands vs legs), never rotated
  dims 1-2 -> z_fine     : within-cluster class signals
  dims 3-4 -> z_strategy : user-controlled strategy state
  dims 5-7 -> z_noise    : slow random-walk noise

This version supports multiple strategy optima for a label. Each class is
always anchored to a coarse cluster, while the exact strategy can pull the
latent state toward a more specific centroid. Some optional strategies share a
merge centroid, which makes those labels intentionally harder to distinguish.
"""

from __future__ import annotations

import numpy as np

from .config import DifficultyConfig


# Class-conditional strategy optima in the 2D strategy plane.
OPTIMAL_STRATEGIES: dict[int, np.ndarray] = {
    0: np.array(
        [
            [+0.50, +0.50],  # left hand primary
            [+0.45, -0.20],  # left hand merge-style alternative
        ],
        dtype=float,
    ),
    1: np.array(
        [
            [-0.50, +0.50],  # right hand primary
        ],
        dtype=float,
    ),
    2: np.array(
        [
            [+0.50, -0.50],  # left leg primary
        ],
        dtype=float,
    ),
    3: np.array(
        [
            [-0.50, -0.50],  # right leg primary
            [-0.45, +0.20],  # right leg merge-style alternative
            [+0.15, -0.05],  # right leg alternate path
        ],
        dtype=float,
    ),
}

_MERGE = np.array([0.0, 0.0, 0.0], dtype=float)

# Strategy-specific class centroids in the z_class subspace [coarse, fine_a, fine_b].
STRATEGY_CENTROIDS: dict[int, np.ndarray] = {
    0: np.array(
        [
            [+2.0, +1.5, 0.0],  # left hand primary
            _MERGE,             # merge-style confusion with class 3
        ],
        dtype=float,
    ),
    1: np.array(
        [
            [+2.0, -1.5, 0.0],  # right hand primary
        ],
        dtype=float,
    ),
    2: np.array(
        [
            [-2.0, 0.0, +1.5],  # left leg primary
        ],
        dtype=float,
    ),
    3: np.array(
        [
            [-2.0, 0.0, -1.5],  # right leg primary
            _MERGE,             # merge-style confusion with class 0
            [-2.0, 0.0, -1.5],  # alternate path, same class centroid
        ],
        dtype=float,
    ),
}

# Coarse cluster anchors used when strategy quality is low.
CLUSTER_CENTROIDS = np.array(
    [
        [+2.0, 0.0, 0.0],   # left hand
        [+2.0, 0.0, 0.0],   # right hand
        [-2.0, 0.0, 0.0],   # left leg
        [-2.0, 0.0, 0.0],   # right leg
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
    - a spring pulls z_strategy back toward (0, 0)
    - each class has one or more strategy optima
    - low strategy quality collapses a class toward its coarse cluster anchor
    - high strategy quality reveals the strategy-specific class centroid
    """

    def __init__(self, config: DifficultyConfig, sample_rate: float = 10.0, seed: int = 42):
        if sample_rate <= 0.0:
            raise ValueError("sample_rate must be > 0")
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
        self._nearest_strategy_idx = 0

    def set_class(self, class_idx: int | None) -> None:
        if class_idx is None:
            self.current_class = None
            return
        class_idx = int(class_idx)
        if class_idx not in OPTIMAL_STRATEGIES:
            raise ValueError(f"class_idx must be one of {sorted(OPTIMAL_STRATEGIES)} or None")
        self.current_class = class_idx

    def update_strategy(self, delta: np.ndarray) -> None:
        """Move z_strategy by one arrow-key update, clamped to [-1, 1]^2."""
        delta_arr = np.asarray(delta, dtype=float).reshape(-1)
        if delta_arr.shape != (N_STRATEGY_DIMS,):
            raise ValueError(f"delta must have shape ({N_STRATEGY_DIMS},)")
        if not np.all(np.isfinite(delta_arr)):
            raise ValueError("delta must contain only finite values")
        self.z_strategy = np.clip(
            self.z_strategy + delta_arr * self.cfg.strategy_speed,
            -1.0,
            1.0,
        )

    def step(self) -> dict[str, object]:
        cfg = self.cfg

        # Evaluate the nearest strategy before the spring is applied so a held
        # optimum gives the strongest pull toward its paired centroid.
        if self.current_class is not None:
            cls = self.current_class
            optima = OPTIMAL_STRATEGIES[cls]
            dists = np.linalg.norm(optima - self.z_strategy, axis=1)
            self._nearest_strategy_idx = int(np.argmin(dists))
            quality = float(np.exp(-2.5 * dists[self._nearest_strategy_idx]))

            cluster_centroid = CLUSTER_CENTROIDS[cls]
            strategy_centroid = STRATEGY_CENTROIDS[cls][self._nearest_strategy_idx]
            effective_centroid = cluster_centroid + quality * (strategy_centroid - cluster_centroid)
            self.z_class += cfg.class_pull_strength * (effective_centroid - self.z_class)
        else:
            self._nearest_strategy_idx = 0
            self.z_class *= 1.0 - cfg.class_pull_strength * 0.4

        decay = np.exp(-cfg.spring_rate * self.dt)
        self.z_strategy = self.z_strategy * decay

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
            "nearest_strategy_idx": self._nearest_strategy_idx,
            "strategy_quality": self.strategy_quality,
            "class_scale": self.class_scale,
            "t": self.t,
        }

    def get_rotation(self) -> np.ndarray:
        """
        Build an 8x8 rotation matrix.

        Dim 0 is never rotated, so the coarse hands-vs-legs split remains.
        Fine dimensions rotate into noise depending on the distance from the
        nearest active-class strategy optimum.
        """
        if self.current_class is not None:
            cls = self.current_class
            target = OPTIMAL_STRATEGIES[cls][self._nearest_strategy_idx]
            err = self.z_strategy - target
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
        """Quality is 1 at the nearest active-class optimum and 0 with no class."""
        if self.current_class is None:
            return 0.0
        optima = OPTIMAL_STRATEGIES[self.current_class]
        dists = np.linalg.norm(optima - self.z_strategy, axis=1)
        return float(np.exp(-2.5 * dists.min()))

    @property
    def class_scale(self) -> float:
        """Integrated signal scale in [0, 1]."""
        return float(self._scale_integrated)

    @property
    def optimal_strategies(self) -> np.ndarray:
        """All strategy optima for the active class. Shape (K, 2)."""
        if self.current_class is None:
            return np.zeros((0, 2), dtype=float)
        return OPTIMAL_STRATEGIES[self.current_class].copy()

    @property
    def optimal_strategy(self) -> np.ndarray:
        """Nearest strategy optimum for the active class, or zeros at rest."""
        if self.current_class is None:
            return np.zeros(2, dtype=float)
        optima = OPTIMAL_STRATEGIES[self.current_class]
        dists = np.linalg.norm(optima - self.z_strategy, axis=1)
        return optima[int(np.argmin(dists))].copy()
