"""
Generative (observation) model — Version 2.

Maps the 8-dim rotated latent vector to the n_obs-dim observed signal:

    x = A @ R(z_strategy) @ z + ε,   ε ~ N(0, noise_std²·I)

A is a fixed random n_obs × n_latent mixing matrix with unit-norm columns,
generated once at construction time and unknown to the students.

The rotation R comes from LatentDynamics.get_rotation().  In Version 2 the
rotation only touches the fine dims (1-2) and noise dims (5-7); dim 0 (the
coarse cluster signal) is never rotated so the two clusters are always
separable regardless of strategy.

class_scale suppresses only the fine dims (1-2).  Dim 0 (coarse) is always
at full strength so clusters remain visible even before strategy is learned.
"""

import numpy as np


class GenerativeModel:
    def __init__(self, n_obs: int = 256, n_latent: int = 8, seed: int = 99):
        self.n_obs    = n_obs
        self.n_latent = n_latent

        rng = np.random.default_rng(seed)

        # Random mixing matrix: columns are unit-norm so every latent dim
        # contributes equally before scaling.
        A_raw = rng.standard_normal((n_obs, n_latent))
        norms = np.linalg.norm(A_raw, axis=0, keepdims=True)
        self.A: np.ndarray = A_raw / norms   # shape (n_obs, n_latent)

    def observe(
        self,
        z: np.ndarray,          # (n_latent,) full latent vector
        R: np.ndarray,          # (n_latent, n_latent) rotation from dynamics
        noise_std: float,
        class_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Return one n_obs-dim observation sample.

        class_scale (0–1) multiplies the z_class dims (indices 0-2) before
        mixing.  At 0 the class signal vanishes and all classes overlap in x.
        At 1 the class signal is at full strength; combined with the rotation,
        the right projection reveals the classes.
        """
        z_scaled = z.copy()
        z_scaled[1:3] *= class_scale         # suppress only fine dims; coarse (dim 0) always present
        z_rotated = R @ z_scaled
        x = self.A @ z_rotated
        x += np.random.normal(0.0, noise_std, self.n_obs)
        return x
