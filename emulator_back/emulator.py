"""
BrainEmulator: ties dynamics + generative model together and publishes
samples via ZMQ PUB socket.

Message format (JSON, one per sample):
{
    "timestamp":   float,   # unix time
    "sample_idx":  int,     # monotonically increasing
    "data":        [float, ...],  # n_dims values — the raw 256-dim signal
    "label":       int | null,    # current intention (0-3) or null for rest
    "label_name":  str,           # "left_hand" / "right_hand" / "left_leg" / "right_leg" / "rest"
    "n_dims":      int,
    "sample_rate": int,
    "difficulty":  str
}
"""

import json
import time

import numpy as np
import zmq

from .config import CLASS_NAMES, DIFFICULTIES, DifficultyConfig
from .dynamics import LatentDynamics
from .generative import GenerativeModel


class BrainEmulator:
    def __init__(
        self,
        difficulty: str = "medium",
        n_dims: int = 256,
        port: int = 5555,
        sample_rate: float = 10.0,
    ):
        self.cfg: DifficultyConfig = DIFFICULTIES[difficulty]
        self.n_dims      = n_dims
        self.sample_rate = sample_rate

        self.dynamics  = LatentDynamics(self.cfg, sample_rate=sample_rate)
        self.gen_model = GenerativeModel(n_obs=n_dims)

        # ZMQ publisher
        self._context = zmq.Context()
        self._socket  = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://*:{port}")
        self._port = port

        self.sample_count = 0

    # ------------------------------------------------------------------
    # Controls (called by the GUI on key events)
    # ------------------------------------------------------------------

    def set_class(self, class_idx: int | None) -> None:
        self.dynamics.set_class(class_idx)

    def update_strategy(self, delta: np.ndarray) -> None:
        self.dynamics.update_strategy(delta)

    # ------------------------------------------------------------------
    # Generate + publish one sample
    # ------------------------------------------------------------------

    def step(self) -> dict:
        state = self.dynamics.step()
        R     = self.dynamics.get_rotation()
        z     = self.dynamics.z_full

        x     = self.gen_model.observe(z, R, self.cfg.gaussian_noise_std,
                                       class_scale=self.dynamics.class_scale)

        label = state["current_class"]
        msg   = {
            "timestamp":      time.time(),
            "sample_idx":     self.sample_count,
            "data":           x.tolist(),
            "label":          label,
            "label_name":     CLASS_NAMES[label] if label is not None else "rest",
            "n_dims":         self.n_dims,
            "sample_rate":    int(self.sample_rate),
            "difficulty":     self.cfg.name,
            "class_scale":    round(self.dynamics.class_scale, 3),
            "strategy_quality": round(self.dynamics.strategy_quality, 3),
        }

        self._socket.send_string(json.dumps(msg))
        self.sample_count += 1
        return msg

    # ------------------------------------------------------------------

    def close(self) -> None:
        self._socket.close()
        self._context.term()

    @property
    def port(self) -> int:
        return self._port
