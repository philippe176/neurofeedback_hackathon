"""
Bridge between the web application and the emulator/model components.

This module provides direct integration without ZMQ overhead, making the
system more responsive and easier to debug.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np
import torch

from emulator.dynamics import LatentDynamics, CLASS_CENTROIDS
from emulator.generative import GenerativeModel
from emulator.config import DIFFICULTIES, CLASS_NAMES
from model.network import MovementDecoder
from model.config import ModelConfig
from model.trainer import OnlineTrainer
from model.reward import ProgrammaticReward
from model.types import StreamSample


CLASS_NAME_MAP = {
    0: "Left Hand",
    1: "Right Hand",
    2: "Left Leg",
    3: "Right Leg",
}


class EmulatorBridge:
    """
    Direct bridge between emulator and model for web visualization.

    This class:
    - Generates simulated brain signals using the emulator
    - Processes them through the model for classification
    - Maintains history for visualization
    - Supports auto-tracking mode where optimal strategy is maintained
    """

    def __init__(
        self,
        difficulty: str = "d1",
        n_dims: int = 256,
        history_len: int = 300,
        centroid_window: int = 50,
    ):
        self.difficulty = difficulty
        self.n_dims = n_dims
        self.history_len = history_len
        self.centroid_window = centroid_window

        # Emulator components
        self.cfg = DIFFICULTIES[difficulty]
        self.dynamics = LatentDynamics(self.cfg, sample_rate=10.0, seed=42)
        self.gen_model = GenerativeModel(n_obs=n_dims, n_latent=8, seed=99)

        # Model components
        self.model_cfg = ModelConfig(input_dim=n_dims)
        self.device = torch.device("cpu")
        self.model = MovementDecoder(
            input_dim=self.model_cfg.input_dim,
            hidden_dim=self.model_cfg.hidden_dim,
            embedding_dim=self.model_cfg.embedding_dim,
            n_classes=self.model_cfg.n_classes,
            projection_dim=self.model_cfg.projection_dim,
            dropout=self.model_cfg.dropout,
        )
        self.reward_provider = ProgrammaticReward(self.model_cfg)
        self.trainer = OnlineTrainer(
            model=self.model,
            cfg=self.model_cfg,
            reward_provider=self.reward_provider,
            device=self.device,
        )

        # State
        self.current_class: int | None = None
        self.auto_tracking = False
        self.sample_count = 0
        self.start_time = time.time()

        # History buffers for visualization
        self._points: deque[list[float]] = deque(maxlen=history_len)
        self._preds: deque[int] = deque(maxlen=history_len)
        self._confs: deque[float] = deque(maxlen=history_len)
        self._rewards: deque[float] = deque(maxlen=history_len)
        self._accuracies: deque[float] = deque(maxlen=history_len)

        # Auto-tracking state
        self._auto_class_timer = 0.0
        self._auto_current_class = 0
        self._auto_class_duration = 3.0  # seconds per class

    def set_class(self, class_idx: int | None) -> None:
        """Set the current mental state class (0-3 or None for rest)."""
        self.current_class = class_idx
        self.dynamics.set_class(class_idx)

    def step(self) -> dict[str, Any]:
        """
        Perform one step of emulation and model inference.

        Returns a dictionary with all visualization data.
        """
        # Handle auto-tracking mode
        if self.auto_tracking:
            self._handle_auto_tracking()

        # Step the dynamics
        self.dynamics.step()

        # In auto-tracking mode, keep strategy near origin for optimal signal
        if self.auto_tracking and self.current_class is not None:
            # Apply counter-movement to keep z_strategy near (0, 0)
            counter = -self.dynamics.z_strategy * 0.3
            self.dynamics.update_strategy(counter)

        # Generate observation
        z = self.dynamics.z_full
        R = self.dynamics.get_rotation()
        obs = self.gen_model.observe(
            z, R,
            noise_std=self.cfg.gaussian_noise_std,
            class_scale=self.dynamics.class_scale,
        )

        # Create stream sample
        sample = StreamSample(
            sample_idx=self.sample_count,
            timestamp=time.time() - self.start_time,
            embedding=obs,
            label=self.current_class,
            label_name=CLASS_NAMES.get(self.current_class, "rest") if self.current_class is not None else "rest",
            class_scale=self.dynamics.class_scale,
            strategy_quality=self.dynamics.strategy_quality,
            difficulty=self.difficulty,
            raw={},
        )

        # Process through model
        result = self.trainer.process_sample(sample)
        self.sample_count += 1

        # Store in history
        self._points.append(result.projection.tolist())
        self._preds.append(int(result.predicted_class))
        self._confs.append(float(result.confidence))
        self._rewards.append(float(result.reward))

        if result.training and result.training.update_applied:
            self._accuracies.append(float(result.training.balanced_accuracy))
        else:
            # Keep last accuracy or 0
            last_acc = self._accuracies[-1] if self._accuracies else 0.25
            self._accuracies.append(last_acc)

        # Compute centroids from recent data
        centroids = self._compute_centroids()
        min_sep, mean_sep = self._compute_separation(centroids)

        # Build response
        return {
            "sample_idx": self.sample_count,
            "timestamp": sample.timestamp,

            # Current state
            "current_class": self.current_class,
            "current_class_name": CLASS_NAME_MAP.get(self.current_class, "Rest"),
            "auto_tracking": self.auto_tracking,

            # Model output
            "predicted_class": int(result.predicted_class),
            "predicted_class_name": CLASS_NAME_MAP.get(result.predicted_class, "Unknown"),
            "confidence": float(result.confidence),
            "probabilities": result.probabilities.tolist(),
            "reward": float(result.reward),

            # Projection for visualization
            "projection": result.projection.tolist(),

            # Emulator state
            "class_scale": float(self.dynamics.class_scale),
            "strategy_quality": float(self.dynamics.strategy_quality),
            "z_strategy": self.dynamics.z_strategy.tolist(),
            "disturbance": self.dynamics.current_disturbance.tolist(),

            # History for plotting
            "points": list(self._points),
            "predictions": list(self._preds),
            "confidences": list(self._confs),
            "rewards": list(self._rewards),
            "accuracies": list(self._accuracies),

            # Centroids
            "centroids": {str(k): v.tolist() for k, v in centroids.items()},
            "centroid_window": self.centroid_window,
            "min_separation": min_sep,
            "mean_separation": mean_sep,

            # Training metrics
            "training": {
                "update_applied": result.training.update_applied if result.training else False,
                "total_loss": result.training.total_loss if result.training else 0,
                "balanced_accuracy": result.training.balanced_accuracy if result.training and result.training.update_applied else None,
                "macro_f1": result.training.macro_f1 if result.training and result.training.update_applied else None,
                "rl_enabled": result.training.rl_enabled if result.training else False,
                "num_updates": self.trainer.num_updates,
                "labeled_seen": self.trainer.labeled_seen,
            },
        }

    def _handle_auto_tracking(self) -> None:
        """Handle automatic class cycling in auto-tracking mode."""
        self._auto_class_timer += 0.1  # 10 Hz

        if self._auto_class_timer >= self._auto_class_duration:
            self._auto_class_timer = 0.0
            self._auto_current_class = (self._auto_current_class + 1) % 4

        # Set the class
        if self.current_class != self._auto_current_class:
            self.current_class = self._auto_current_class
            self.dynamics.set_class(self._auto_current_class)

    def _compute_centroids(self) -> dict[int, np.ndarray]:
        """Compute class centroids from recent predictions."""
        if len(self._points) == 0:
            return {}

        window = min(self.centroid_window, len(self._points))
        points = list(self._points)[-window:]
        preds = list(self._preds)[-window:]

        points_arr = np.array(points)
        preds_arr = np.array(preds)

        centroids = {}
        for cls in range(4):
            mask = preds_arr == cls
            if np.sum(mask) >= 3:
                centroids[cls] = np.mean(points_arr[mask], axis=0)

        return centroids

    def _compute_separation(self, centroids: dict[int, np.ndarray]) -> tuple[float, float]:
        """Compute min and mean inter-centroid distances."""
        if len(centroids) < 2:
            return 0.0, 0.0

        dists = []
        keys = list(centroids.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                d = float(np.linalg.norm(centroids[keys[i]] - centroids[keys[j]]))
                dists.append(d)

        return float(np.min(dists)), float(np.mean(dists))
