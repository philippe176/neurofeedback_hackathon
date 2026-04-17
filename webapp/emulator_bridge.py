"""
Bridge between the web application and the emulator/model components.

This module provides direct integration without ZMQ overhead, making the
system more responsive and easier to debug.
"""

from __future__ import annotations

import importlib.util
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from emulator.dynamics import LatentDynamics
from emulator.generative import GenerativeModel
from emulator.config import DIFFICULTIES, CLASS_NAMES
from model.network import build_decoder
from model.config import ModelConfig
from model.projectors import build_projector
from model.trainer import OnlineTrainer
from model.reward import ProgrammaticReward
from model.types import StreamSample


CLASS_NAME_MAP = {
    0: "Left Hand",
    1: "Right Hand",
    2: "Left Leg",
    3: "Right Leg",
}

AVAILABLE_MODELS = {
    "dnn": "DNN Decoder",
    "cnn": "CNN Decoder",
    "cebra": "CEBRA-Inspired Decoder",
}

AVAILABLE_VIZ_METHODS = {
    "neural": "Neural Projection",
    "pca": "PCA Projection",
    "lda": "LDA Projection",
    "tsne": "t-SNE Projection",
}
if importlib.util.find_spec("umap") is not None:
    AVAILABLE_VIZ_METHODS["umap"] = "UMAP Projection"

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "saved_models"


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
        model_type: str = "dnn",
        viz_method: str = "neural",
        viz_fit_window: int = 200,
        viz_refit_every: int = 10,
    ):
        self.difficulty = difficulty
        self.n_dims = n_dims
        self.history_len = history_len
        self.centroid_window = centroid_window
        self.model_type = "dnn"
        self.viz_method = "neural"
        self.viz_fit_window = max(10, min(viz_fit_window, history_len))
        self.viz_refit_every = max(1, viz_refit_every)

        # Emulator components
        self.cfg = DIFFICULTIES[difficulty]
        self.dynamics = LatentDynamics(self.cfg, sample_rate=10.0, seed=42)
        self.gen_model = GenerativeModel(n_obs=n_dims, n_latent=8, seed=99)

        # Model components
        self.model_cfg = ModelConfig(input_dim=n_dims)
        self.device = torch.device("cpu")
        self.model = None
        self.reward_provider = ProgrammaticReward(self.model_cfg)
        self.trainer = None
        self.projector = build_projector("neural", projection_dim=self.model_cfg.projection_dim)

        # State
        self.current_class: int | None = None
        self.auto_tracking = False
        self.sample_count = 0
        self.start_time = time.time()

        # History buffers for visualization
        self._neural_points: deque[np.ndarray] = deque(maxlen=history_len)
        self._penultimate: deque[np.ndarray] = deque(maxlen=history_len)
        self._labels: deque[int | None] = deque(maxlen=history_len)
        self._points: deque[list[float]] = deque(maxlen=history_len)
        self._preds: deque[int] = deque(maxlen=history_len)
        self._confs: deque[float] = deque(maxlen=history_len)
        self._rewards: deque[float] = deque(maxlen=history_len)
        self._accuracies: deque[float] = deque(maxlen=history_len)
        self._last_viz_refit_sample = -1

        # Auto-tracking state
        self._auto_class_timer = 0.0
        self._auto_current_class = 0
        self._auto_class_duration = 3.0  # seconds per class

        self.set_model(model_type)
        self.set_viz_method(viz_method)

    def set_class(self, class_idx: int | None) -> None:
        """Set the current mental state class (0-3 or None for rest)."""
        self.current_class = class_idx
        self.dynamics.set_class(class_idx)

    def set_model(self, model_type: str) -> None:
        """Switch the active decoder implementation."""
        normalized = str(model_type).strip().lower()
        if normalized == "neural":
            normalized = "dnn"
        if normalized not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model_type = normalized
        self._initialize_decoder()
        self._reset_history()

    def set_viz_method(self, viz_method: str) -> None:
        """Switch the active visualization projection backend."""
        normalized = str(viz_method).strip().lower()
        if normalized not in AVAILABLE_VIZ_METHODS:
            raise ValueError(f"Unknown visualization method: {viz_method}")

        if normalized == "umap" and importlib.util.find_spec("umap") is None:
            raise ValueError("UMAP visualization requires the optional 'umap-learn' package")

        self.viz_method = normalized
        self.projector = build_projector(
            normalized,
            projection_dim=self.model_cfg.projection_dim,
            tsne_perplexity=self.model_cfg.viz_tsne_perplexity,
        )
        self._last_viz_refit_sample = -1
        self._refresh_projected_history()

    @property
    def model_name(self) -> str:
        return AVAILABLE_MODELS[self.model_type]

    def available_models(self) -> dict[str, str]:
        return dict(AVAILABLE_MODELS)

    @property
    def viz_name(self) -> str:
        return AVAILABLE_VIZ_METHODS[self.viz_method]

    def available_viz_methods(self) -> dict[str, str]:
        return dict(AVAILABLE_VIZ_METHODS)

    def default_checkpoint_name(self) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return f"{self.model_type}_{self.viz_method}_{timestamp}"

    def save_model_snapshot(self, name: str | None = None) -> Path:
        """Save the current decoder state and runtime metadata."""
        checkpoint_name = self._normalize_checkpoint_name(name or self.default_checkpoint_name())
        checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_type": self.model_type,
            "viz_method": self.viz_method,
            "sample_count": self.sample_count,
            "difficulty": self.difficulty,
            "current_class": self.current_class,
            "auto_tracking": self.auto_tracking,
            "timestamp": time.time(),
            "model_config": {
                "input_dim": self.model_cfg.input_dim,
                "hidden_dim": self.model_cfg.hidden_dim,
                "embedding_dim": self.model_cfg.embedding_dim,
                "projection_dim": self.model_cfg.projection_dim,
                "n_classes": self.model_cfg.n_classes,
                "dropout": self.model_cfg.dropout,
            },
            "trainer_state": {
                "num_updates": getattr(self.trainer, "num_updates", 0),
                "labeled_seen": getattr(self.trainer, "labeled_seen", 0),
                "reward_baseline": float(getattr(self.trainer, "reward_baseline", 0.0)),
            },
        }

        if self.model is not None:
            checkpoint["model_state_dict"] = self.model.state_dict()
        optimizer = getattr(self.trainer, "optimizer", None)
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

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
        self._neural_points.append(np.asarray(result.projection, dtype=float).copy())
        self._penultimate.append(np.asarray(result.penultimate, dtype=float).copy())
        self._labels.append(self.current_class)
        self._preds.append(int(result.predicted_class))
        self._confs.append(float(result.confidence))
        self._rewards.append(float(result.reward))
        self._refresh_projected_history()

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
            "model_type": self.model_type,
            "model_name": self.model_name,
            "viz_method": self.viz_method,
            "viz_name": self.viz_name,

            # Model output
            "predicted_class": int(result.predicted_class),
            "predicted_class_name": CLASS_NAME_MAP.get(result.predicted_class, "Unknown"),
            "confidence": float(result.confidence),
            "probabilities": result.probabilities.tolist(),
            "reward": float(result.reward),

            # Projection for visualization
            "projection": self._points[-1] if self._points else result.projection.tolist(),

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
            "mean_spread": self._compute_spread(centroids),

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

    def _initialize_decoder(self) -> None:
        if self.model_type in {"dnn", "cnn", "cebra"}:
            self.model = build_decoder(self.model_type, self.model_cfg)
            self.trainer = OnlineTrainer(
                model=self.model,
                cfg=self.model_cfg,
                reward_provider=self.reward_provider,
                device=self.device,
            )
            return

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def _reset_history(self) -> None:
        self._neural_points.clear()
        self._penultimate.clear()
        self._labels.clear()
        self._points.clear()
        self._preds.clear()
        self._confs.clear()
        self._rewards.clear()
        self._accuracies.clear()
        self._last_viz_refit_sample = -1

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

    def _compute_spread(self, centroids: dict[int, np.ndarray]) -> float:
        """Compute mean within-class spread in the active visualization space."""
        if len(self._points) == 0 or len(centroids) == 0:
            return 0.0

        window = min(self.centroid_window, len(self._points))
        points_arr = np.array(list(self._points)[-window:], dtype=float)
        preds_arr = np.array(list(self._preds)[-window:], dtype=int)

        spreads = []
        for cls, centroid in centroids.items():
            mask = preds_arr == cls
            if np.sum(mask) < 2:
                continue
            dists = np.linalg.norm(points_arr[mask] - centroid, axis=1)
            spreads.append(float(np.mean(dists)))

        return float(np.mean(spreads)) if spreads else 0.0

    def _projection_labels(self) -> np.ndarray:
        labels = []
        for label, pred in zip(self._labels, self._preds):
            labels.append(int(pred) if label is None else int(label))
        return np.asarray(labels, dtype=np.int64)

    def _refresh_projected_history(self) -> None:
        if len(self._neural_points) == 0:
            self._points.clear()
            return

        if self.viz_method == "neural":
            projected = np.stack(list(self._neural_points), axis=0)
        else:
            embeddings = np.stack(list(self._penultimate), axis=0)
            labels = self._projection_labels()
            fit_window = min(self.viz_fit_window, embeddings.shape[0])
            fit_x = embeddings[-fit_window:]
            fit_y = labels[-fit_window:]
            should_refit = (
                self._last_viz_refit_sample < 0
                or (self.sample_count - self._last_viz_refit_sample) >= self.viz_refit_every
            )
            if should_refit:
                self.projector.fit(fit_x, y=fit_y)
                self._last_viz_refit_sample = self.sample_count
            projected = self.projector.transform(embeddings)

        self._points.clear()
        for point in projected:
            self._points.append(np.asarray(point, dtype=float).tolist())

    def _normalize_checkpoint_name(self, raw_name: str) -> str:
        name = raw_name.strip()
        if name.endswith(".pt"):
            name = name[:-3]
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
        return safe or self.default_checkpoint_name()
