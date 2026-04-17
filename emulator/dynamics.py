"""
Latent space dynamics for the brain emulator — Version 4.

State vector z (8-dimensional):
  dim  0     →  z_coarse   : coarse cluster signal (A vs B), NEVER rotated
  dims 1-2   →  z_fine     : fine within-cluster signals
                               dim 1 (fine_A): separates class 0 (+) from class 1 (-)
                               dim 2 (fine_B): separates class 2 (+) from class 3 (-)
  dims 3-4   →  z_strategy : position in strategy space, controlled by arrow keys
  dims 5-7   →  z_noise    : slow AR(1) random walk

Cluster hierarchy
-----------------
  Cluster A (classes 0 & 1): z_coarse = +2.0
  Cluster B (classes 2 & 3): z_coarse = -2.0

  Within each cluster classes are ONLY separable when z_strategy is at one of
  the class-specific optimal positions.  Away from all optima, z_class is pulled
  toward the cluster centroid (fine dims = 0) so within-cluster classes look
  identical in any projection.

Per-class optimal strategies
-----------------------------
  Each class has one or more optimal positions in z_strategy space ([-1,1]²).
  Each optimal position is paired with a specific z_class centroid target.

  class 0 (left_hand)  — 2 strategies:
      primary  (+0.25, +0.25)  →  [+2.0, +1.5,  0.0]  separable from class 1
      merge    (+0.25, -0.25)  →  [  0,    0,    0  ]  same centroid as class 3 merge

  class 1 (right_hand) — 1 strategy:
      primary  (-0.25, +0.25)  →  [+2.0, -1.5,  0.0]  separable from class 0

  class 2 (left_leg)   — 1 strategy:
      primary  (-0.25, -0.25)  →  [-2.0,  0.0, +1.5]  separable from class 3

  class 3 (right_leg)  — 3 strategies:
      primary  (-0.20, -0.20)  →  [-2.0,  0.0, -1.5]  separable from class 2
      merge    (-0.25, +0.25)  →  [  0,    0,    0  ]  same centroid as class 0 merge
      alt      (+0.20,  0.00)  →  [-2.0,  0.0, -1.5]  separable from class 2 (alt path)

Merge mechanic
--------------
  When class 0 uses its merge strategy, or class 3 uses its merge strategy,
  z_class is pulled toward [0, 0, 0].  Both classes then produce the same
  observation in x — they project to the same point in any LDA/PCA view.
  This makes class 0 and class 3 visually indistinguishable when either
  uses the merge strategy.

Spring mechanics
----------------
  z_strategy is always attracted back to (0, 0) with a configurable spring rate.
  Releasing arrow keys always recovers the neutral state.
"""

import numpy as np
from .config import DifficultyConfig


# ---------------------------------------------------------------------------
# Per-class optimal strategy positions in z_strategy space ([-1,1]²)
# Each row is one (x, y) optimal position for that class.
# ---------------------------------------------------------------------------
OPTIMAL_STRATEGIES: dict[int, np.ndarray] = {
    0: np.array(
        [
            [+0.5, +0.5],  # primary: class 0 separable from class 1
            # [+0.25, -0.25],   # merge:   class 0 looks same as class 3 merge
        ]
    ),
    1: np.array(
        [
            [-0.5, +0.5],  # primary: class 1 separable from class 0
        ]
    ),
    2: np.array(
        [
            [-0.5, -0.5],  # primary: class 2 separable from class 3
        ]
    ),
    3: np.array(
        [
            # [-0.20, -0.20],  # primary: class 3 separable from class 2
            [0.5, -0.5],  # merge:   class 3 looks same as class 0 merge
            # [
            #     +0.20,
            #     0.00,
            # ],  # alt:     class 3 separable from class 2 (alternate path)
        ]
    ),
}

# ---------------------------------------------------------------------------
# z_class centroid targets paired with the above optimal strategies.
# STRATEGY_CENTROIDS[cls][k] is the centroid when class cls uses strategy k.
# MERGE_CENTROID is shared — it's what makes class 0 + class 3 project together.
# ---------------------------------------------------------------------------
_MERGE = np.array([0.0, 0.0, 0.0])

STRATEGY_CENTROIDS: dict[int, np.ndarray] = {
    0: np.array(
        [
            [+2.0, +1.5, 0.0],  # class 0 primary
            _MERGE,  # class 0 merge  ← same as class 3 merge
        ]
    ),
    1: np.array(
        [
            [+2.0, -1.5, 0.0],  # class 1 primary
        ]
    ),
    2: np.array(
        [
            [-2.0, 0.0, +1.5],  # class 2 primary
        ]
    ),
    3: np.array(
        [
            [-2.0, 0.0, -1.5],  # class 3 primary
            _MERGE,  # class 3 merge  ← same as class 0 merge
            [
                -2.0,
                0.0,
                -1.5,
            ],  # class 3 alt (same centroid, different strategy position)
        ]
    ),
}

# Cluster-level centroids — z_class is pulled here when NOT at any optimal.
# Fine dims are zero so within-cluster classes are indistinguishable.
CLUSTER_CENTROIDS = np.array(
    [
        [+2.0, 0.0, 0.0],  # class 0 — cluster A, no fine separation
        [+2.0, 0.0, 0.0],  # class 1 — cluster A, no fine separation
        [-2.0, 0.0, 0.0],  # class 2 — cluster B, no fine separation
        [-2.0, 0.0, 0.0],  # class 3 — cluster B, no fine separation
    ]
)

# Integration time constant: class_scale builds up over ~SCALE_TAU seconds
SCALE_TAU = 3.0

N_CLASS_DIMS = 3
N_STRATEGY_DIMS = 2
N_NOISE_DIMS = 3
N_LATENT = N_CLASS_DIMS + N_STRATEGY_DIMS + N_NOISE_DIMS  # 8


def _givens(n: int, i: int, j: int, theta: float) -> np.ndarray:
    R = np.eye(n)
    c, s = np.cos(theta), np.sin(theta)
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    return R


class LatentDynamics:
    """
    Simulates the patient's latent brain state — Version 4.

    Public interface
    ----------------
    set_class(k)           — set current movement intention (0-3) or None for rest
    update_strategy(delta) — arrow-key input, delta is a 2-vector
    step() → dict          — advance one time step; returns a state snapshot
    get_rotation()         — 8×8 strategy-dependent rotation matrix
    strategy_quality       — float in [0,1]: 1 = z_strategy at nearest optimal
    class_scale            — leaky-integrated signal strength (0-1)
    optimal_strategies     — (K,2) all optimal positions for the active class
    optimal_strategy       — (2,) nearest optimal position (for display)
    z_full                 — full 8-dim latent vector
    """

    def __init__(
        self,
        config: DifficultyConfig,
        sample_rate: float = 10.0,
        seed: int = 42,
    ):
        self.cfg = config
        self.dt = 1.0 / sample_rate
        self.t = 0.0

        rng = np.random.default_rng(seed)

        self.z_class = np.zeros(N_CLASS_DIMS)
        self.z_strategy = np.zeros(N_STRATEGY_DIMS)
        self.z_noise = np.zeros(N_NOISE_DIMS)

        self.current_class: int | None = None
        self._scale_integrated: float = 0.0
        self._noise_rng = rng
        self._nearest_strategy_idx: int = (
            0  # index into current class's strategy list
        )

    # ------------------------------------------------------------------
    # Public controls
    # ------------------------------------------------------------------

    def set_class(self, class_idx: int | None) -> None:
        self.current_class = class_idx

    def update_strategy(self, delta: np.ndarray) -> None:
        """Move z_strategy by delta (arrow keys). Clamped to [-1, 1]²."""
        self.z_strategy = np.clip(
            self.z_strategy
            + np.asarray(delta, float) * self.cfg.strategy_speed,
            -1.0,
            1.0,
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> dict:
        cfg = self.cfg

        # 1. Pull z_class toward the effective centroid.
        #    Quality is evaluated BEFORE the spring fires so that holding z_strategy
        #    exactly at an optimal corner gives sq = 1.0, ensuring perfect convergence
        #    to the strategy-specific centroid (including the shared merge centroid).
        if self.current_class is not None:
            cls = self.current_class
            optima = OPTIMAL_STRATEGIES[cls]  # (K, 2)
            dists = np.linalg.norm(optima - self.z_strategy, axis=1)
            self._nearest_strategy_idx = int(np.argmin(dists))
            sq = float(np.exp(-2.5 * dists[self._nearest_strategy_idx]))

            cluster_cent = CLUSTER_CENTROIDS[cls]
            specific_cent = STRATEGY_CENTROIDS[cls][self._nearest_strategy_idx]
            effective_cent = cluster_cent + sq * (specific_cent - cluster_cent)

            self.z_class += cfg.class_pull_strength * (
                effective_cent - self.z_class
            )
        else:
            self._nearest_strategy_idx = 0
            self.z_class *= 1.0 - cfg.class_pull_strength * 0.4

        # 2. Spring: z_strategy decays toward (0, 0) every step.
        decay = np.exp(-cfg.spring_rate * self.dt)
        self.z_strategy = self.z_strategy * decay

        # 3. Latent Gaussian noise on z_class
        self.z_class += self._noise_rng.normal(
            0, cfg.latent_noise_std, N_CLASS_DIMS
        )

        # 4. Noise dims: slow AR(1) random walk
        self.z_noise = 0.85 * self.z_noise + self._noise_rng.normal(
            0, 0.25, N_NOISE_DIMS
        )

        # 5. Leaky integrator: class_scale rises when strategy quality is high
        target_scale = self.strategy_quality**3
        alpha = self.dt / SCALE_TAU
        self._scale_integrated += alpha * (
            target_scale - self._scale_integrated
        )

        # 6. Advance time
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

    # ------------------------------------------------------------------
    # Rotation matrix
    # ------------------------------------------------------------------

    def get_rotation(self) -> np.ndarray:
        """
        Build an 8×8 rotation matrix.

        Dim 0 (coarse cluster signal) is NEVER rotated.
        Dims 1-2 (fine signals) are mixed into noise dims 5-7 based on the
        error between z_strategy and the nearest optimal for the active class.
        """
        if self.current_class is not None:
            cls = self.current_class
            target = OPTIMAL_STRATEGIES[cls][self._nearest_strategy_idx]
            err = self.z_strategy - target
        else:
            err = np.array([1.0, 1.0])

        scale = self.cfg.strategy_sensitivity
        half_pi = np.pi / 2

        theta1 = half_pi * np.tanh(scale * err[0])
        theta2 = half_pi * np.tanh(scale * err[1])
        theta3 = half_pi * np.tanh(scale * (err[0] + err[1]) / 2)

        R = (
            _givens(N_LATENT, 1, 5, theta1)
            @ _givens(N_LATENT, 2, 6, theta2)
            @ _givens(N_LATENT, 1, 7, theta3)
        )
        return R

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def z_full(self) -> np.ndarray:
        return np.concatenate([self.z_class, self.z_strategy, self.z_noise])

    @property
    def strategy_quality(self) -> float:
        """1.0 at the nearest optimal position, 0.0 when no class active."""
        if self.current_class is None:
            return 0.0
        optima = OPTIMAL_STRATEGIES[self.current_class]
        dists = np.linalg.norm(optima - self.z_strategy, axis=1)
        return float(np.exp(-2.5 * dists.min()))

    @property
    def class_scale(self) -> float:
        """Integrated signal scale (0-1). Builds over ~SCALE_TAU seconds."""
        return float(self._scale_integrated)

    @property
    def optimal_strategies(self) -> np.ndarray:
        """All optimal z_strategy positions for the active class. Shape (K, 2)."""
        if self.current_class is None:
            return np.zeros((0, 2))
        return OPTIMAL_STRATEGIES[self.current_class].copy()

    @property
    def optimal_strategy(self) -> np.ndarray:
        """The nearest optimal z_strategy position for the active class (or zeros)."""
        if self.current_class is None:
            return np.zeros(2)
        optima = OPTIMAL_STRATEGIES[self.current_class]
        dists = np.linalg.norm(optima - self.z_strategy, axis=1)
        return optima[np.argmin(dists)].copy()
