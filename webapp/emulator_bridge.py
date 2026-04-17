"""
Bridge between the web application and the emulator/model components.

This module provides direct integration without ZMQ overhead, making the
system more responsive and easier to debug.
"""

from __future__ import annotations

import importlib.util
import re
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from emulator.config import CLASS_NAMES, DIFFICULTIES
from emulator.dynamics import LatentDynamics
from emulator.generative import GenerativeModel
from game.config import LevelPolicy, RhythmGameConfig
from game.rewards import GameRewardProvider
from game.scoring import target_margin
from model.config import ModelConfig
from model.network import build_decoder
from model.projectors import build_projector
from model.reward import ProgrammaticReward
from model.trainer import OnlineTrainer
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

AVAILABLE_CONTROL_MODES = {
    "buttons": "Button Controls",
    "keyboard": "Keyboard Emulator",
}

AVAILABLE_TRAINING_PHASES = {
    "calibration": "Guided Calibration",
    "feedback": "Neurofeedback Coach",
    "manual": "Manual Sandbox",
}

AVAILABLE_DIFFICULTIES = {
    "d1": "D1 - Cardinal Pulses",
    "d2": "D2 - Diagonal Pulses",
    "d3": "D3 - Double-Tap Rhythm",
    "d4": "D4 - Rotating Force",
    "d5": "D5 - Dual-Frequency Drive",
}

TRAINING_PHASE_DESCRIPTIONS = {
    "calibration": (
        "Follow the shown task cue and repeat a stable strategy until each label forms a distinct cluster."
    ),
    "feedback": (
        "Treat the task prompts like a game and reinforce the mental strategy that improves reward and separation."
    ),
    "manual": (
        "Free-play sandbox for manually selecting classes and testing control strategies."
    ),
}

ARROW_KEY_MAP = {
    "ArrowLeft": np.array([-1.0, 0.0], dtype=float),
    "ArrowRight": np.array([1.0, 0.0], dtype=float),
    "ArrowUp": np.array([0.0, 1.0], dtype=float),
    "ArrowDown": np.array([0.0, -1.0], dtype=float),
}

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "saved_models"


class EmulatorBridge:
    """
    Direct bridge between emulator and model for web visualization.

    This class:
    - Generates simulated brain signals using the emulator
    - Processes them through the model for classification
    - Maintains history for visualization
    - Supports manual, calibration, and neurofeedback coaching phases
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
        training_phase: str = "calibration",
    ):
        self.sample_rate = 10.0
        self.difficulty = difficulty
        self.n_dims = n_dims
        self.history_len = history_len
        self.centroid_window = centroid_window
        self.model_type = "dnn"
        self.viz_method = "neural"
        self.control_mode = "buttons"
        self.training_phase = "manual"
        self.viz_fit_window = max(10, min(viz_fit_window, history_len))
        self.viz_refit_every = max(1, viz_refit_every)

        # Emulator components
        self.cfg = DIFFICULTIES[difficulty]
        self.dynamics = LatentDynamics(self.cfg, sample_rate=self.sample_rate, seed=42)
        self.gen_model = GenerativeModel(n_obs=n_dims, n_latent=8, seed=99)

        # Model components
        self.model_cfg = ModelConfig(input_dim=n_dims)
        self.device = torch.device("cpu")
        self.model = None
        self.reward_provider = ProgrammaticReward(self.model_cfg)
        self.trainer = None
        self.projector = build_projector("neural", projection_dim=self.model_cfg.projection_dim)

        # Runtime state
        self.current_class: int | None = None
        self.auto_tracking = False
        self.sample_count = 0
        self.start_time = time.time()
        self._pressed_arrows: set[str] = set()

        # History buffers for visualization and coaching
        self._neural_points: deque[np.ndarray] = deque(maxlen=history_len)
        self._penultimate: deque[np.ndarray] = deque(maxlen=history_len)
        self._labels: deque[int | None] = deque(maxlen=history_len)
        self._points: deque[list[float]] = deque(maxlen=history_len)
        self._preds: deque[int] = deque(maxlen=history_len)
        self._confs: deque[float] = deque(maxlen=history_len)
        self._rewards: deque[float] = deque(maxlen=history_len)
        self._accuracies: deque[float] = deque(maxlen=history_len)
        self._class_scales: deque[float] = deque(maxlen=history_len)
        self._strategy_qualities: deque[float] = deque(maxlen=history_len)
        self._margins: deque[float] = deque(maxlen=history_len)
        self._target_correct: deque[float] = deque(maxlen=history_len)
        self._last_viz_refit_sample = -1

        # Auto-tracking state
        self._auto_class_timer = 0.0
        self._auto_current_class = 0
        self._auto_class_duration = 3.0

        self.set_model(model_type)
        self.set_viz_method(viz_method)
        self.set_training_phase(training_phase)

    def set_class(self, class_idx: int | None) -> None:
        """Set the current mental state class (0-3 or None for rest)."""
        if self.training_phase != "manual":
            self.set_training_phase("manual")
        self.current_class = class_idx
        self.dynamics.set_class(class_idx)

    def set_control_mode(self, control_mode: str) -> None:
        """Switch between button-based control and keyboard emulator mode."""
        normalized = str(control_mode).strip().lower()
        if normalized not in AVAILABLE_CONTROL_MODES:
            raise ValueError(f"Unknown control mode: {control_mode}")

        self.control_mode = normalized
        self.release_all_controls()
        if normalized == "keyboard":
            self.auto_tracking = False

    def set_training_phase(self, training_phase: str) -> None:
        """Switch between manual play, guided calibration, and neurofeedback coaching."""
        normalized = str(training_phase).strip().lower()
        if normalized not in AVAILABLE_TRAINING_PHASES:
            raise ValueError(f"Unknown training phase: {training_phase}")

        self.training_phase = normalized
        self._set_reward_provider(self._build_reward_provider(normalized))

        if normalized != "manual":
            self.current_class = None
            self.dynamics.set_class(None)

    def set_difficulty(self, difficulty: str) -> None:
        """Rebuild the emulator dynamics for a new disturbance pattern."""
        normalized = str(difficulty).strip().lower()
        if normalized not in AVAILABLE_DIFFICULTIES:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        self.difficulty = normalized
        self.cfg = DIFFICULTIES[normalized]
        self.dynamics = LatentDynamics(self.cfg, sample_rate=self.sample_rate, seed=42)
        self.sample_count = 0
        self._auto_class_timer = 0.0
        self._auto_current_class = 0
        self.release_all_controls()
        self._initialize_decoder()
        self._set_reward_provider(self._build_reward_provider(self.training_phase))
        self._reset_history()

    def available_control_modes(self) -> dict[str, str]:
        return dict(AVAILABLE_CONTROL_MODES)

    def available_training_phases(self) -> dict[str, str]:
        return dict(AVAILABLE_TRAINING_PHASES)

    def available_difficulties(self) -> dict[str, str]:
        return dict(AVAILABLE_DIFFICULTIES)

    def set_arrow_pressed(self, key: str, pressed: bool) -> None:
        """Track held arrow keys for browser-driven emulator control."""
        if key not in ARROW_KEY_MAP:
            raise ValueError(f"Unsupported control key: {key}")

        if pressed:
            self._pressed_arrows.add(key)
            if self.control_mode == "keyboard":
                self.auto_tracking = False
        else:
            self._pressed_arrows.discard(key)

    def pressed_arrows(self) -> list[str]:
        """Return active arrow keys in a stable display order."""
        return [key for key in ARROW_KEY_MAP if key in self._pressed_arrows]

    def release_all_controls(self) -> None:
        """Clear any held keyboard state to avoid sticky controls."""
        self._pressed_arrows.clear()

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

    @property
    def viz_name(self) -> str:
        return AVAILABLE_VIZ_METHODS[self.viz_method]

    @property
    def training_phase_name(self) -> str:
        return AVAILABLE_TRAINING_PHASES[self.training_phase]

    @property
    def difficulty_name(self) -> str:
        return AVAILABLE_DIFFICULTIES[self.difficulty]

    def available_models(self) -> dict[str, str]:
        return dict(AVAILABLE_MODELS)

    def available_viz_methods(self) -> dict[str, str]:
        return dict(AVAILABLE_VIZ_METHODS)

    def default_checkpoint_name(self) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return f"{self.model_type}_{self.viz_method}_{self.training_phase}_{timestamp}"

    def status_snapshot(self) -> dict[str, Any]:
        prompt = self.preview_prompt_state()
        calibration = self.calibration_snapshot()
        session = self.session_snapshot()
        coach = self.coach_snapshot()
        return {
            "current_class": self.current_class,
            "current_class_name": CLASS_NAME_MAP.get(self.current_class, "Rest"),
            "auto_tracking": self.auto_tracking,
            "control_mode": self.control_mode,
            "control_mode_name": AVAILABLE_CONTROL_MODES[self.control_mode],
            "available_control_modes": self.available_control_modes(),
            "training_phase": self.training_phase,
            "training_phase_name": self.training_phase_name,
            "training_phase_description": TRAINING_PHASE_DESCRIPTIONS[self.training_phase],
            "available_training_phases": self.available_training_phases(),
            "difficulty": self.difficulty,
            "difficulty_name": self.difficulty_name,
            "available_difficulties": self.available_difficulties(),
            "pressed_arrows": self.pressed_arrows(),
            "sample_count": self.sample_count,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "available_models": self.available_models(),
            "viz_method": self.viz_method,
            "viz_name": self.viz_name,
            "available_viz_methods": self.available_viz_methods(),
            "prompt": prompt,
            "session": session,
            "calibration": calibration,
            "coach": coach,
        }

    def save_model_snapshot(self, name: str | None = None) -> Path:
        """Save the current decoder state and runtime metadata."""
        checkpoint_name = self._normalize_checkpoint_name(name or self.default_checkpoint_name())
        checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_type": self.model_type,
            "viz_method": self.viz_method,
            "training_phase": self.training_phase,
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

    def preview_prompt_state(self, timestamp: float | None = None) -> dict[str, Any]:
        preview_fn = getattr(self.reward_provider, "preview", None)
        if not callable(preview_fn):
            return {
                "guided": False,
                "prompt_id": None,
                "target_class": self.current_class,
                "target_class_name": CLASS_NAME_MAP.get(self.current_class, "Rest"),
                "next_target_class": None,
                "next_target_class_name": None,
                "window_open": False,
                "progress": 0.0,
                "seconds_to_window_start": 0.0,
                "seconds_to_prompt_end": 0.0,
                "seconds_to_next_prompt_start": 0.0,
                "level": None,
            }

        current_time = self._current_sample_time() if timestamp is None else float(timestamp)
        preview = preview_fn(current_time)
        session = getattr(self.reward_provider, "session", None)
        level = getattr(session, "current_level", None)
        return {
            "guided": True,
            "prompt_id": int(preview.prompt_id),
            "target_class": int(preview.target_class),
            "target_class_name": CLASS_NAME_MAP.get(int(preview.target_class), "Unknown"),
            "next_target_class": int(preview.next_target_class),
            "next_target_class_name": CLASS_NAME_MAP.get(int(preview.next_target_class), "Unknown"),
            "window_open": bool(preview.in_window),
            "progress": float(preview.prompt_progress),
            "seconds_to_window_start": float(preview.seconds_to_window_start),
            "seconds_to_prompt_end": float(preview.seconds_to_prompt_end),
            "seconds_to_next_prompt_start": float(preview.seconds_to_next_prompt_start),
            "level": None if level is None else int(level),
        }

    def session_snapshot(self) -> dict[str, Any]:
        rolling_reward = self._rolling_mean(self._rewards)
        rolling_accuracy = self._compute_recent_accuracy()
        session = getattr(self.reward_provider, "session", None)
        if session is None:
            return {
                "guided": False,
                "level": None,
                "streak": 0,
                "best_streak": 0,
                "total_prompts": 0,
                "total_hits": 0,
                "hit_rate": 0.0,
                "rolling_reward": rolling_reward,
                "rolling_accuracy": rolling_accuracy,
            }

        snapshot = session.snapshot()
        return {
            "guided": True,
            "level": int(snapshot["level"]),
            "streak": int(snapshot["streak"]),
            "best_streak": int(snapshot["best_streak"]),
            "total_prompts": int(snapshot["total_prompts"]),
            "total_hits": int(snapshot["total_hits"]),
            "hit_rate": float(snapshot["hit_rate"]),
            "rolling_reward": rolling_reward,
            "rolling_accuracy": rolling_accuracy,
        }

    def calibration_snapshot(self) -> dict[str, Any]:
        counts = Counter(int(label) for label in self._labels if label is not None)
        counts_by_class = {str(cls): int(counts.get(cls, 0)) for cls in range(4)}
        ready_classes = sum(1 for cls in range(4) if counts_by_class[str(cls)] >= 12)

        label_centroids = self._compute_true_label_centroids()
        label_min_sep, label_mean_sep = self._compute_separation(label_centroids)
        rolling_accuracy = self._compute_recent_accuracy()
        signal_fit = 0.5 * (
            self._rolling_mean(self._class_scales) + self._rolling_mean(self._strategy_qualities)
        )
        separation_fit = float(np.clip(label_mean_sep / 1.6, 0.0, 1.0))
        readiness = float(
            np.clip(
                0.30 * (ready_classes / 4.0)
                + 0.30 * separation_fit
                + 0.25 * rolling_accuracy
                + 0.15 * signal_fit,
                0.0,
                1.0,
            )
        )
        ready = bool(
            ready_classes == 4
            and label_mean_sep >= 0.80
            and rolling_accuracy >= 0.55
        )

        if ready:
            message = "All four tasks are separating reliably. You can move into neurofeedback coaching."
        elif ready_classes < 4:
            message = "Keep collecting attempts for every task so each label forms its own cluster."
        elif label_mean_sep < 0.80:
            message = "The labels are present, but the latent clusters still overlap. Reinforce the strategies that increase separation."
        else:
            message = "The model is learning the classes. Stay consistent a little longer to stabilize the manifold."

        return {
            "ready": ready,
            "readiness": readiness,
            "label_counts": counts_by_class,
            "classes_ready": ready_classes,
            "min_label_separation": label_min_sep,
            "mean_label_separation": label_mean_sep,
            "rolling_accuracy": rolling_accuracy,
            "signal_fit": signal_fit,
            "message": message,
        }

    def coach_snapshot(self) -> dict[str, Any]:
        calibration = self.calibration_snapshot()
        session = self.session_snapshot()
        signal_fit = 0.5 * (
            self._rolling_mean(self._class_scales) + self._rolling_mean(self._strategy_qualities)
        )

        if self.training_phase == "manual":
            headline = "Manual sandbox"
            message = "Pick a class manually or switch to Guided Calibration to start building a labeled latent space."
            state = "idle"
        elif self.training_phase == "calibration":
            headline = "Calibration in progress"
            message = calibration["message"]
            state = "ready" if calibration["ready"] else "hold"
        else:
            headline = "Neurofeedback coach"
            message = "Watch the reward and target bar. Keep the strategy that makes the target class dominate and the clusters spread apart."
            state = "hold"

        score = float(
            np.clip(
                0.35 * session["rolling_reward"]
                + 0.35 * calibration["rolling_accuracy"]
                + 0.30 * signal_fit,
                0.0,
                1.0,
            )
        )
        return {
            "state": state,
            "headline": headline,
            "message": message,
            "score": score,
            "score_label": "Guidance Score",
            "target_margin": self._rolling_mean(self._margins),
            "rolling_reward": session["rolling_reward"],
            "rolling_accuracy": calibration["rolling_accuracy"],
        }

    def step(self) -> dict[str, Any]:
        """
        Perform one step of emulation and model inference.

        Returns a dictionary with all visualization and coaching data.
        """
        sample_timestamp = self._current_sample_time()

        if self.auto_tracking and self.training_phase == "manual":
            self._handle_auto_tracking()

        prompt_preview = self.preview_prompt_state(sample_timestamp)
        if prompt_preview["guided"] and prompt_preview["target_class"] is not None:
            self.current_class = int(prompt_preview["target_class"])
            self.dynamics.set_class(self.current_class)

        self._apply_keyboard_strategy()
        self.dynamics.step()

        if self.auto_tracking and self.current_class is not None:
            counter = -self.dynamics.z_strategy * 0.3
            self.dynamics.update_strategy(counter)

        z = self.dynamics.z_full
        rotation = self.dynamics.get_rotation()
        obs = self.gen_model.observe(
            z,
            rotation,
            noise_std=self.cfg.gaussian_noise_std,
            class_scale=self.dynamics.class_scale,
        )

        sample = StreamSample(
            sample_idx=self.sample_count,
            timestamp=sample_timestamp,
            embedding=obs,
            label=self.current_class,
            label_name=CLASS_NAMES.get(self.current_class, "rest") if self.current_class is not None else "rest",
            class_scale=self.dynamics.class_scale,
            strategy_quality=self.dynamics.strategy_quality,
            difficulty=self.difficulty,
            raw={},
        )

        result = self.trainer.process_sample(sample)
        self.sample_count += 1

        self._neural_points.append(np.asarray(result.projection, dtype=float).copy())
        self._penultimate.append(np.asarray(result.penultimate, dtype=float).copy())
        self._labels.append(self.current_class)
        self._preds.append(int(result.predicted_class))
        self._confs.append(float(result.confidence))
        self._rewards.append(float(result.reward))
        self._class_scales.append(float(self.dynamics.class_scale))
        self._strategy_qualities.append(float(self.dynamics.strategy_quality))

        prompt = self._build_prompt_payload(result, sample_timestamp)
        margin = self._resolve_target_margin(result, prompt["target_class"])
        self._margins.append(margin)
        self._target_correct.append(1.0 if self._is_target_correct(result, prompt["target_class"]) else 0.0)

        self._refresh_projected_history()

        if result.training and result.training.update_applied:
            self._accuracies.append(float(result.training.balanced_accuracy))
        else:
            last_acc = self._accuracies[-1] if self._accuracies else 0.25
            self._accuracies.append(last_acc)

        centroids = self._compute_centroids()
        min_sep, mean_sep = self._compute_separation(centroids)
        mean_spread = self._compute_spread(centroids)
        session = self.session_snapshot()
        calibration = self.calibration_snapshot()
        coach = self._build_coach_payload(
            result=result,
            prompt=prompt,
            session=session,
            calibration=calibration,
            mean_sep=mean_sep,
        )

        return {
            "sample_idx": self.sample_count,
            "timestamp": sample.timestamp,
            "current_class": self.current_class,
            "current_class_name": CLASS_NAME_MAP.get(self.current_class, "Rest"),
            "auto_tracking": self.auto_tracking,
            "control_mode": self.control_mode,
            "control_mode_name": AVAILABLE_CONTROL_MODES[self.control_mode],
            "training_phase": self.training_phase,
            "training_phase_name": self.training_phase_name,
            "training_phase_description": TRAINING_PHASE_DESCRIPTIONS[self.training_phase],
            "difficulty": self.difficulty,
            "difficulty_name": self.difficulty_name,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "viz_method": self.viz_method,
            "viz_name": self.viz_name,
            "predicted_class": int(result.predicted_class),
            "predicted_class_name": CLASS_NAME_MAP.get(result.predicted_class, "Unknown"),
            "confidence": float(result.confidence),
            "probabilities": result.probabilities.tolist(),
            "reward": float(result.reward),
            "projection": self._points[-1] if self._points else result.projection.tolist(),
            "class_scale": float(self.dynamics.class_scale),
            "strategy_quality": float(self.dynamics.strategy_quality),
            "z_strategy": self.dynamics.z_strategy.tolist(),
            "disturbance": self.dynamics.current_disturbance.tolist(),
            "pressed_arrows": self.pressed_arrows(),
            "points": list(self._points),
            "labels": [None if label is None else int(label) for label in self._labels],
            "predictions": list(self._preds),
            "confidences": list(self._confs),
            "rewards": list(self._rewards),
            "accuracies": list(self._accuracies),
            "centroids": {str(k): v.tolist() for k, v in centroids.items()},
            "centroid_window": self.centroid_window,
            "min_separation": min_sep,
            "mean_separation": mean_sep,
            "mean_spread": mean_spread,
            "prompt": prompt,
            "session": session,
            "calibration": calibration,
            "coach": coach,
            "training": {
                "update_applied": result.training.update_applied if result.training else False,
                "total_loss": result.training.total_loss if result.training else 0,
                "balanced_accuracy": (
                    result.training.balanced_accuracy
                    if result.training and result.training.update_applied
                    else None
                ),
                "macro_f1": (
                    result.training.macro_f1
                    if result.training and result.training.update_applied
                    else None
                ),
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

    def _set_reward_provider(self, reward_provider: ProgrammaticReward | GameRewardProvider) -> None:
        self.reward_provider = reward_provider
        if self.trainer is not None:
            self.trainer.reward_provider = reward_provider

    def _build_reward_provider(self, training_phase: str) -> ProgrammaticReward | GameRewardProvider:
        if training_phase == "manual":
            return ProgrammaticReward(self.model_cfg)

        if training_phase == "calibration":
            cfg = RhythmGameConfig(
                n_classes=self.model_cfg.n_classes,
                prompt_duration_s=3.4,
                base_hit_window_s=1.6,
                base_beat_interval_s=4.2,
                enable_adaptation=False,
                start_level=0,
                w_correctness=0.40,
                w_margin=0.24,
                w_separability=0.18,
                w_timing=0.02,
                w_stability=0.09,
                w_aux=0.07,
                w_confusion_penalty=0.18,
                hit_bonus=0.04,
                levels=(
                    LevelPolicy(
                        hit_window_s=1.6,
                        beat_interval_s=4.2,
                        min_confidence=0.24,
                        min_margin=-0.08,
                    ),
                ),
            )
            return GameRewardProvider(self.model_cfg, cfg)

        cfg = RhythmGameConfig(
            n_classes=self.model_cfg.n_classes,
            prompt_duration_s=2.6,
            base_hit_window_s=0.9,
            base_beat_interval_s=3.0,
            enable_adaptation=True,
            adaptation_eval_prompts=12,
            start_level=0,
            w_correctness=0.38,
            w_margin=0.28,
            w_separability=0.22,
            w_timing=0.04,
            w_stability=0.05,
            w_aux=0.03,
            w_confusion_penalty=0.28,
            hit_bonus=0.05,
        )
        return GameRewardProvider(self.model_cfg, cfg)

    def _reset_history(self) -> None:
        self._neural_points.clear()
        self._penultimate.clear()
        self._labels.clear()
        self._points.clear()
        self._preds.clear()
        self._confs.clear()
        self._rewards.clear()
        self._accuracies.clear()
        self._class_scales.clear()
        self._strategy_qualities.clear()
        self._margins.clear()
        self._target_correct.clear()
        self._last_viz_refit_sample = -1

    def _handle_auto_tracking(self) -> None:
        """Handle automatic class cycling in auto-tracking mode."""
        self._auto_class_timer += 1.0 / self.sample_rate

        if self._auto_class_timer >= self._auto_class_duration:
            self._auto_class_timer = 0.0
            self._auto_current_class = (self._auto_current_class + 1) % 4

        if self.current_class != self._auto_current_class:
            self.current_class = self._auto_current_class
            self.dynamics.set_class(self._auto_current_class)

    def _apply_keyboard_strategy(self) -> None:
        """Apply held arrow keys once per sample, matching the legacy emulator."""
        if self.control_mode != "keyboard" or not self._pressed_arrows:
            return

        for key in self.pressed_arrows():
            self.dynamics.update_strategy(ARROW_KEY_MAP[key])

    def _build_prompt_payload(self, result, timestamp: float) -> dict[str, Any]:
        preview = self.preview_prompt_state(timestamp)
        prompt_id = result.game_prompt_id if result.game_prompt_id is not None else preview["prompt_id"]
        target_class = (
            int(result.game_target_class)
            if result.game_target_class is not None
            else preview["target_class"]
        )
        next_target_class = (
            int(result.game_next_target_class)
            if result.game_next_target_class is not None
            else preview["next_target_class"]
        )
        level = result.game_level if result.game_level is not None else preview["level"]
        return {
            "guided": bool(preview["guided"]),
            "prompt_id": prompt_id,
            "target_class": target_class,
            "target_class_name": CLASS_NAME_MAP.get(target_class, "Rest"),
            "next_target_class": next_target_class,
            "next_target_class_name": CLASS_NAME_MAP.get(next_target_class, "Rest"),
            "window_open": bool(result.game_in_window if result.game_prompt_id is not None else preview["window_open"]),
            "progress": (
                float(result.game_prompt_progress)
                if result.game_prompt_progress is not None
                else float(preview["progress"])
            ),
            "seconds_to_window_start": (
                float(result.game_seconds_to_window_start)
                if result.game_seconds_to_window_start is not None
                else float(preview["seconds_to_window_start"])
            ),
            "seconds_to_prompt_end": float(preview["seconds_to_prompt_end"]),
            "seconds_to_next_prompt_start": (
                float(result.game_seconds_to_next_prompt_start)
                if result.game_seconds_to_next_prompt_start is not None
                else float(preview["seconds_to_next_prompt_start"])
            ),
            "level": None if level is None else int(level),
        }

    def _build_coach_payload(
        self,
        result,
        prompt: dict[str, Any],
        session: dict[str, Any],
        calibration: dict[str, Any],
        mean_sep: float,
    ) -> dict[str, Any]:
        target_class = prompt["target_class"]
        target_name = CLASS_NAME_MAP.get(target_class, "Rest")
        predicted_name = CLASS_NAME_MAP.get(result.predicted_class, "Unknown")
        margin = self._resolve_target_margin(result, target_class)
        signal_fit = float(np.clip(0.5 * (self.dynamics.class_scale + self.dynamics.strategy_quality), 0.0, 1.0))
        margin_fit = float(np.clip(0.5 * (margin + 1.0), 0.0, 1.0))
        separation_fit = float(np.clip(mean_sep / 1.6, 0.0, 1.0))
        score = float(
            np.clip(
                0.35 * float(result.confidence)
                + 0.25 * margin_fit
                + 0.20 * signal_fit
                + 0.20 * separation_fit,
                0.0,
                1.0,
            )
        )

        if self.training_phase == "manual":
            if self.current_class is None:
                state = "idle"
                headline = "Choose a task"
                message = "Select one of the four imagined movements or switch to Guided Calibration to begin training."
            elif result.predicted_class == self.current_class and score >= 0.65:
                state = "good"
                headline = "That pattern is working"
                message = "The decoder is following your selected class. Hold the strategy steady and watch the cluster stay separated."
            elif result.predicted_class == self.current_class:
                state = "hold"
                headline = "Hold it a little longer"
                message = "You have the right class, but the signal is still stabilizing. Keep the same strategy so the latent point settles."
            else:
                state = "adjust"
                headline = "Try a clearer strategy"
                message = (
                    f"The model is reading {predicted_name} instead of {target_name}. "
                    "Shift the strategy until the target probability becomes dominant."
                )
        else:
            target_correct = self._is_target_correct(result, target_class)
            if target_correct and calibration["ready"] and self.training_phase == "calibration":
                state = "ready"
                headline = "Calibration is ready"
                message = "All four labels are separating well. You can move into Neurofeedback Coach mode."
            elif target_correct and self.dynamics.class_scale < 0.55:
                state = "hold"
                headline = "Hold the winning strategy"
                message = (
                    f"The decoder is tracking {target_name}. Keep the same mental pattern steady so the signal has time to build."
                )
            elif target_correct and score >= 0.72:
                state = "good"
                headline = "Reinforce this pattern"
                message = (
                    f"{target_name} is separating from the other classes. Repeat this strategy consistently so the cluster moves even farther away."
                )
            elif target_correct:
                state = "adjust"
                headline = "Right class, sharper signal"
                message = (
                    "You found the correct class, but it is still close to competitors. Make the mental pattern a bit more distinct and keep it stable."
                )
            elif self.dynamics.strategy_quality < 0.40:
                state = "recover"
                headline = "Recover the strategy"
                message = (
                    "The hidden strategy is drifting. Counter the disturbance and bring it back toward center before trying to separate the target again."
                )
            elif margin > -0.05:
                state = "adjust"
                headline = "Almost there"
                message = (
                    f"The model is close but still leans toward {predicted_name}. Nudge the strategy until the target bar clearly overtakes it."
                )
            else:
                state = "recover"
                headline = "Change strategy"
                message = (
                    f"The decoder is reading {predicted_name} instead of {target_name}. Try a different mental strategy and keep the one that raises reward."
                )

        return {
            "state": state,
            "headline": headline,
            "message": message,
            "score": score,
            "score_label": "Neurofeedback Score",
            "target_margin": margin,
            "rolling_reward": session["rolling_reward"],
            "rolling_accuracy": calibration["rolling_accuracy"],
        }

    def _compute_centroids(self) -> dict[int, np.ndarray]:
        """Compute recent class centroids using labels when available."""
        if len(self._points) == 0:
            return {}

        window = min(self.centroid_window, len(self._points))
        points_arr = np.array(list(self._points)[-window:], dtype=float)
        labels_arr = self._plot_labels_array()[-window:]

        centroids: dict[int, np.ndarray] = {}
        for cls in range(4):
            mask = labels_arr == cls
            if int(np.sum(mask)) >= 3:
                centroids[cls] = np.mean(points_arr[mask], axis=0)

        return centroids

    def _compute_true_label_centroids(self) -> dict[int, np.ndarray]:
        if len(self._points) == 0:
            return {}

        window = min(self.centroid_window, len(self._points))
        points_arr = np.array(list(self._points)[-window:], dtype=float)
        labels = list(self._labels)[-window:]

        centroids: dict[int, np.ndarray] = {}
        for cls in range(4):
            indices = [idx for idx, label in enumerate(labels) if label == cls]
            if len(indices) < 3:
                continue
            centroids[cls] = np.mean(points_arr[indices], axis=0)
        return centroids

    def _compute_separation(self, centroids: dict[int, np.ndarray]) -> tuple[float, float]:
        """Compute min and mean inter-centroid distances."""
        if len(centroids) < 2:
            return 0.0, 0.0

        dists = []
        keys = list(centroids.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                dist = float(np.linalg.norm(centroids[keys[i]] - centroids[keys[j]]))
                dists.append(dist)

        return float(np.min(dists)), float(np.mean(dists))

    def _compute_spread(self, centroids: dict[int, np.ndarray]) -> float:
        """Compute mean within-class spread in the active visualization space."""
        if len(self._points) == 0 or len(centroids) == 0:
            return 0.0

        window = min(self.centroid_window, len(self._points))
        points_arr = np.array(list(self._points)[-window:], dtype=float)
        labels_arr = self._plot_labels_array()[-window:]

        spreads = []
        for cls, centroid in centroids.items():
            mask = labels_arr == cls
            if int(np.sum(mask)) < 2:
                continue
            dists = np.linalg.norm(points_arr[mask] - centroid, axis=1)
            spreads.append(float(np.mean(dists)))

        return float(np.mean(spreads)) if spreads else 0.0

    def _projection_labels(self) -> np.ndarray:
        labels = []
        for label, pred in zip(self._labels, self._preds):
            labels.append(int(pred) if label is None else int(label))
        return np.asarray(labels, dtype=np.int64)

    def _plot_labels_array(self) -> np.ndarray:
        return self._projection_labels()

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

    def _compute_recent_accuracy(self, window: int = 60) -> float:
        if len(self._labels) == 0:
            return 0.0

        labels = list(self._labels)[-window:]
        preds = list(self._preds)[-window:]
        pairs = [(label, pred) for label, pred in zip(labels, preds) if label is not None]
        if not pairs:
            return 0.0
        correct = [1.0 if int(label) == int(pred) else 0.0 for label, pred in pairs]
        return float(np.mean(correct))

    def _rolling_mean(self, values: deque[float], window: int = 50) -> float:
        if not values:
            return 0.0
        arr = np.asarray(list(values)[-window:], dtype=float)
        return float(np.mean(arr)) if arr.size else 0.0

    def _resolve_target_margin(self, result, target_class: int | None) -> float:
        if result.game_margin is not None:
            return float(result.game_margin)
        if target_class is None:
            return 0.0
        return float(target_margin(np.asarray(result.probabilities, dtype=float), int(target_class)))

    def _is_target_correct(self, result, target_class: int | None) -> bool:
        if result.game_prompt_id is not None:
            return bool(result.game_label_correct)
        if target_class is None:
            return False
        return int(result.predicted_class) == int(target_class)

    def _current_sample_time(self) -> float:
        return float(self.sample_count / self.sample_rate)

    def _normalize_checkpoint_name(self, raw_name: str) -> str:
        name = raw_name.strip()
        if name.endswith(".pt"):
            name = name[:-3]
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
        return safe or self.default_checkpoint_name()
