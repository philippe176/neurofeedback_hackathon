"""
Generative (observation) model.

Maps the 8-dim rotated latent vector to the n_obs-dim observed signal:

    x = A @ R(z_strategy) @ z + ε,   ε ~ N(0, noise_std²·I)

A is a fixed random n_obs × n_latent mixing matrix with unit-norm columns,
generated once at construction time and unknown to the students.

The rotation R comes from LatentDynamics.get_rotation(); it is what makes
the problem hard: the same linear projection W that separates classes at one
strategy position will fail at another.
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
        z_scaled[:3] *= class_scale          # suppress class dims when strategy is poor
        z_rotated = R @ z_scaled
        x = self.A @ z_rotated
        x += np.random.normal(0.0, noise_std, self.n_obs)
        return x
