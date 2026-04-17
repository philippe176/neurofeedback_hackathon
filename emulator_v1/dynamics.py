"""
Latent space dynamics for the brain emulator.

State vector z (8-dimensional):
  dims 0-2  →  z_class    : class signal (pulled toward the active class centroid)
  dims 3-4  →  z_strategy : position in strategy space, controlled by arrow keys
  dims 5-7  →  z_noise    : slow AR(1) random walk

Design
------
Optimal strategy is always (0, 0) — the same for every class.

Each difficulty has a per-class temporal disturbance function that pushes
z_strategy away from (0, 0) at every step.  The player must learn to
counter-balance this with arrow keys to keep z_strategy near the origin.

class_scale is a leaky integrator: the class signal builds up only when
z_strategy is held near (0, 0) for several seconds.  Arriving at (0, 0)
is not enough — you must stay there.
"""

import numpy as np
from .config import DifficultyConfig


# ---------------------------------------------------------------------------
# Class centroids in the 3-dim z_class subspace
# hand vs leg: large gap (dim 0) → easy to separate
# left vs right: small gap (dim 1) → hard to separate
# ---------------------------------------------------------------------------
CLASS_CENTROIDS = np.array(
    [
        [ 2.0,  0.9,  0.0],   # 0 — left_hand
        [ 2.0, -0.9,  0.0],   # 1 — right_hand
        [-2.0,  0.9,  0.0],   # 2 — left_leg
        [-2.0, -0.9,  0.0],   # 3 — right_leg
    ],
    dtype=float,
)

# Integration time constant: class_scale builds up over ~SCALE_TAU seconds
# when strategy quality is high.
SCALE_TAU = 3.0

N_CLASS_DIMS    = 3
N_STRATEGY_DIMS = 2
N_NOISE_DIMS    = 3
N_LATENT        = N_CLASS_DIMS + N_STRATEGY_DIMS + N_NOISE_DIMS   # 8


def _givens(n: int, i: int, j: int, theta: float) -> np.ndarray:
    R = np.eye(n)
    c, s = np.cos(theta), np.sin(theta)
    R[i, i] =  c;  R[i, j] = -s
    R[j, i] =  s;  R[j, j] =  c
    return R


class LatentDynamics:
    """
    Simulates the patient's latent brain state.

    Public interface
    ----------------
    set_class(k)           — set current movement intention (0-3) or None for rest
    update_strategy(delta) — arrow-key input, delta is a 2-vector
    step() → dict          — advance one time step; returns a state snapshot
    get_rotation()         — 8×8 strategy-dependent rotation matrix
    strategy_quality       — float in [0,1]: 1 = z_strategy at (0,0)
    class_scale            — leaky-integrated signal strength (0-1)
    current_disturbance    — current disturbance vector (dx, dy) for display
    z_full                 — full 8-dim latent vector
    """

    def __init__(self, config: DifficultyConfig, sample_rate: float = 10.0, seed: int = 42):
        self.cfg         = config
        self.dt          = 1.0 / sample_rate
        self.t           = 0.0

        rng = np.random.default_rng(seed)

        # --- Latent state ---
        self.z_class    = np.zeros(N_CLASS_DIMS)
        self.z_strategy = np.zeros(N_STRATEGY_DIMS)
        self.z_noise    = np.zeros(N_NOISE_DIMS)

        # --- Current intention ---
        self.current_class: int | None = None

        # --- Leaky integrator for class signal strength ---
        self._scale_integrated: float = 0.0

        # --- Background latent noise on z_class ---
        # (small independent noise dims, always present regardless of difficulty)
        self._noise_rng = rng

        # --- Current disturbance vector (for GUI display) ---
        self.current_disturbance = np.zeros(2)

    # ------------------------------------------------------------------
    # Public controls
    # ------------------------------------------------------------------

    def set_class(self, class_idx: int | None) -> None:
        self.current_class = class_idx

    def update_strategy(self, delta: np.ndarray) -> None:
        """Move z_strategy by delta (arrow keys). Clamped to [-1, 1]²."""
        self.z_strategy = np.clip(
            self.z_strategy + np.asarray(delta, float) * self.cfg.strategy_speed,
            -1.0, 1.0,
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> dict:
        cfg = self.cfg

        # 1. Apply class-specific temporal disturbance to z_strategy
        if self.current_class is not None:
            disturb = cfg.disturbance_fn(self.t, self.current_class)
            self.current_disturbance = disturb * cfg.disturbance_amplitude
            self.z_strategy = np.clip(
                self.z_strategy + self.current_disturbance,
                -1.0, 1.0,
            )
        else:
            self.current_disturbance = np.zeros(2)

        # 2. Pull z_class toward the active class centroid (or decay to rest)
        if self.current_class is not None:
            target = CLASS_CENTROIDS[self.current_class]
            self.z_class += cfg.class_pull_strength * (target - self.z_class)
        else:
            self.z_class *= 1.0 - cfg.class_pull_strength * 0.4

        # 3. Latent Gaussian noise on z_class (background neural noise)
        self.z_class += self._noise_rng.normal(0, cfg.latent_noise_std, N_CLASS_DIMS)

        # 4. Noise dims: slow AR(1) random walk
        self.z_noise = (0.85 * self.z_noise
                        + self._noise_rng.normal(0, 0.25, N_NOISE_DIMS))

        # 5. Leaky integrator: class_scale rises when near (0,0), decays otherwise
        target_scale           = self.strategy_quality ** 3
        alpha                  = self.dt / SCALE_TAU
        self._scale_integrated += alpha * (target_scale - self._scale_integrated)

        # 6. Advance time
        self.t += self.dt

        return {
            "z_class":           self.z_class.copy(),
            "z_strategy":        self.z_strategy.copy(),
            "z_noise":           self.z_noise.copy(),
            "current_class":     self.current_class,
            "strategy_quality":  self.strategy_quality,
            "class_scale":       self.class_scale,
            "disturbance":       self.current_disturbance.copy(),
            "t":                 self.t,
        }

    # ------------------------------------------------------------------
    # Rotation matrix
    # ------------------------------------------------------------------

    def get_rotation(self) -> np.ndarray:
        """
        Build an 8×8 rotation matrix.
        Optimal at z_strategy = (0, 0); rotation increases with distance.

        Rotation mixes class dims with noise dims:
          plane (0, 5), (1, 6), (2, 7)
        """
        err     = self.z_strategy          # distance from origin = error
        scale   = self.cfg.strategy_sensitivity
        half_pi = np.pi / 2

        theta1 = half_pi * np.tanh(scale * err[0])
        theta2 = half_pi * np.tanh(scale * err[1])
        theta3 = half_pi * np.tanh(scale * (err[0] + err[1]) / 2)

        R = (
            _givens(N_LATENT, 0, 5, theta1)
            @ _givens(N_LATENT, 1, 6, theta2)
            @ _givens(N_LATENT, 2, 7, theta3)
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
        """1.0 at z_strategy = (0,0), decays with distance. 0 when no class active."""
        if self.current_class is None:
            return 0.0
        error = np.linalg.norm(self.z_strategy)
        return float(np.exp(-2.5 * error))

    @property
    def class_scale(self) -> float:
        """Integrated class signal scale (0-1). Builds over ~SCALE_TAU seconds."""
        return float(self._scale_integrated)
