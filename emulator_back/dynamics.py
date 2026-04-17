"""
Latent space dynamics for the brain emulator — Version 2.

State vector z (8-dimensional):
  dim  0     →  z_coarse   : coarse cluster signal (A vs B), NEVER rotated
  dims 1-2   →  z_fine     : fine within-cluster signals
                               dim 1 (fine_A): separates class 0 (+) from class 1 (-)
                               dim 2 (fine_B): separates class 2 (+) from class 3 (-)
  dims 3-4   →  z_strategy : position in strategy space, controlled by arrow keys
  dims 5-7   →  z_noise    : slow AR(1) random walk

Hierarchy
---------
  Cluster A (classes 0 & 1): z_coarse = +2.0  →  always separable from B
  Cluster B (classes 2 & 3): z_coarse = -2.0  →  always separable from A

  Within cluster A: fine_A (dim 1) = ±1.5 separates 0 from 1
  Within cluster B: fine_B (dim 2) = ±1.5 separates 2 from 3

  The fine dims are mixed into noise dims 5-7 when z_strategy is far from the
  active class's optimal corner.  Dim 0 is never touched by the rotation, so
  the two coarse clusters remain separable at any strategy position.

Per-class optimal strategy corners
-----------------------------------
  class 0 (left_hand)  → (+0.7, +0.7)
  class 1 (right_hand) → (-0.7, +0.7)
  class 2 (left_leg)   → (+0.7, -0.7)
  class 3 (right_leg)  → (-0.7, -0.7)

  strategy_quality = 1 when z_strategy == active-class optimal, decays with distance.
  class_scale integrates quality over time (~SCALE_TAU seconds) before the fine
  signal reaches full strength.

Spring mechanics
----------------
  z_strategy is always attracted back to (0, 0) with a configurable spring rate.
  Doing nothing → z_strategy decays to the origin within a few seconds.
  The patient must actively hold arrow keys to reach and maintain the optimal corner.
  This ensures no "stranded far away" situations: releasing arrows always recovers
  the neutral state immediately.
"""

import numpy as np
from .config import DifficultyConfig


# ---------------------------------------------------------------------------
# Class centroids in the 3-dim z_class subspace [coarse, fine_A, fine_B]
# ---------------------------------------------------------------------------
CLASS_CENTROIDS = np.array(
    [
        [+2.0, +1.5, 0.0],  # 0 — left_hand  (cluster A, fine_A = +)
        [+2.0, -1.5, 0.0],  # 1 — right_hand (cluster A, fine_A = -)
        [-2.0, 0.0, +1.5],  # 2 — left_leg   (cluster B, fine_B = +)
        [-2.0, 0.0, -1.5],  # 3 — right_leg  (cluster B, fine_B = -)
    ],
    dtype=float,
)

# Per-class optimal strategy positions — the four corners of [-1,1]²
OPTIMAL_STRATEGIES = np.array(
    [
        [+0.5, +0.5],  # 0 — left_hand
        [-0.5, +0.5],  # 1 — right_hand
        [+0.5, -0.5],  # 2 — left_leg
        [-0.5, -0.5],  # 3 — right_leg
    ],
    dtype=float,
)

# Integration time constant: class_scale builds up over ~SCALE_TAU seconds
SCALE_TAU = 3.0

N_CLASS_DIMS = 3  # coarse + fine_A + fine_B
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
    Simulates the patient's latent brain state — Version 2.

    Public interface
    ----------------
    set_class(k)           — set current movement intention (0-3) or None for rest
    update_strategy(delta) — arrow-key input, delta is a 2-vector
    step() → dict          — advance one time step; returns a state snapshot
    get_rotation()         — 8×8 strategy-dependent rotation matrix
    strategy_quality       — float in [0,1]: 1 = z_strategy at active class optimal
    class_scale            — leaky-integrated signal strength (0-1)
    optimal_strategy       — 2-vector: current class's optimal corner (or zeros)
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

        # 1. Spring: z_strategy decays toward (0, 0) every step.
        #    The player must actively hold arrow keys to counteract this.
        #    Doing nothing always returns the point to the neutral center.
        decay = np.exp(-cfg.spring_rate * self.dt)
        self.z_strategy = self.z_strategy * decay

        # 2. Pull z_class toward the active class centroid (or decay to rest)
        if self.current_class is not None:
            target = CLASS_CENTROIDS[self.current_class]
            self.z_class += cfg.class_pull_strength * (target - self.z_class)
        else:
            self.z_class *= 1.0 - cfg.class_pull_strength * 0.4

        # 3. Latent Gaussian noise on z_class (background neural noise)
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

        Dim 0 (coarse cluster signal) is NEVER rotated — the two clusters
        remain separable at any strategy position.

        Dims 1-2 (fine signals) are mixed into noise dims 5-7 based on the
        error between z_strategy and the active class's optimal corner.
        When at optimal: error = 0 → rotation = identity → fine signals readable.
        When far from optimal: fine signals fully scrambled into noise.
        """
        if self.current_class is not None:
            err = self.z_strategy - OPTIMAL_STRATEGIES[self.current_class]
        else:
            # No active class: maximum rotation (fine dims fully scrambled)
            err = np.array([1.0, 1.0])

        scale = self.cfg.strategy_sensitivity
        half_pi = np.pi / 2

        theta1 = half_pi * np.tanh(scale * err[0])  # fine_A (1) ↔ noise (5)
        theta2 = half_pi * np.tanh(scale * err[1])  # fine_B (2) ↔ noise (6)
        theta3 = half_pi * np.tanh(
            scale * (err[0] + err[1]) / 2
        )  # cross-mix (1) ↔ (7)

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
        """1.0 at the active class's optimal corner, 0 when no class active."""
        if self.current_class is None:
            return 0.0
        error = np.linalg.norm(
            self.z_strategy - OPTIMAL_STRATEGIES[self.current_class]
        )
        return float(np.exp(-2.5 * error))

    @property
    def class_scale(self) -> float:
        """Integrated signal scale (0-1). Builds over ~SCALE_TAU seconds."""
        return float(self._scale_integrated)

    @property
    def optimal_strategy(self) -> np.ndarray:
        """The optimal z_strategy position for the active class (or zeros if rest)."""
        if self.current_class is None:
            return np.zeros(2)
        return OPTIMAL_STRATEGIES[self.current_class].copy()
