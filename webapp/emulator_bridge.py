"""
Stream-driven bridge between the web application and the decoder stack.

The webapp no longer synthesizes emulator samples itself. Instead it listens to
the external emulator's ZMQ stream, feeds samples through the online decoder,
and exposes an interface centered on the neurofeedback loop:

- what task the user is trying to send
- what the model currently thinks
- whether the user should keep or change strategy
- where the current point sits in the latent "brain zone" map
"""

from __future__ import annotations

import copy
import importlib.util
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from game.config import LevelPolicy, RhythmGameConfig
from game.rewards import GameRewardProvider
from model.config import ModelConfig
from model.network import build_decoder
from model.projectors import build_projector
from model.reward import ProgrammaticReward
from model.stream import ZMQEmbeddingReceiver
from model.trainer import OnlineTrainer
from model.types import InferenceStep, StreamSample


class StreamReceiver(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def get(self, timeout: float = 0.2) -> StreamSample | None: ...


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

AVAILABLE_TRAINING_PHASES = {
    "calibration": "Teach",
    "feedback": "Practice",
    "exploration": "Explore",
}

TRAINING_PHASE_DESCRIPTIONS = {
    "calibration": (
        "Show the decoder what each movement looks like by repeating tasks consistently."
    ),
    "feedback": (
        "The decoder coaches you in real time. Adjust until its prediction matches your intention."
    ),
    "exploration": (
        "Model is frozen. Focus on one movement and discover which strategy the decoder reads best."
    ),
}

AVAILABLE_DIFFICULTIES = {
    "d1": "D1 - Cardinal Pulses",
    "d2": "D2 - Diagonal Pulses",
    "d3": "D3 - Double-Tap Rhythm",
    "d4": "D4 - Rotating Force",
    "d5": "D5 - Dual-Frequency Drive",
}

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "saved_models"


class EmulatorBridge:
    """
    Stream-aware bridge used by the web dashboard.

    The bridge:
    - subscribes to the emulator's ZMQ stream
    - lazily initializes the decoder from the first received sample
    - runs online training/inference
    - tracks manifold history and calibration readiness
    - turns decoder outputs into intuitive neurofeedback guidance
    """

    def __init__(
        self,
        stream_host: str = "localhost",
        stream_port: int = 5555,
        embedding_key: str = "data",
        receiver: StreamReceiver | None = None,
        history_len: int = 480,
        calibration_samples_per_class: int = 200,
        transition_ignore_samples: int = 40,
        centroid_window: int = 120,
        centroid_min_samples_per_class: int = 12,
        display_window: int = 180,
        display_min_samples_per_class: int = 24,
        model_type: str = "dnn",
        viz_method: str = "neural",
        viz_fit_window: int = 200,
        viz_refit_every: int = 10,
        training_phase: str = "calibration",
    ) -> None:
        self.stream_host = stream_host
        self.stream_port = int(stream_port)
        self.embedding_key = embedding_key
        self.history_len = history_len
        self.calibration_samples_per_class = max(10, int(calibration_samples_per_class))
        self.transition_ignore_samples = max(0, int(transition_ignore_samples))
        self.centroid_window = centroid_window
        self.centroid_min_samples_per_class = max(1, int(centroid_min_samples_per_class))
        self.display_window = max(10, min(int(display_window), history_len))
        self.display_min_samples_per_class = max(1, int(display_min_samples_per_class))
        self.viz_fit_window = max(10, min(viz_fit_window, history_len))
        self.viz_refit_every = max(1, viz_refit_every)
        self.exploration_max_points = 3000

        self.receiver: StreamReceiver = receiver or ZMQEmbeddingReceiver(
            host=stream_host,
            port=stream_port,
            embedding_key=embedding_key,
        )
        self.receiver_started = False

        self.model_cfg = ModelConfig(input_dim=None)
        self.device = torch.device("cpu")
        self.programmatic_reward = ProgrammaticReward(self.model_cfg)
        self.game_reward = self._make_game_reward_provider()
        self.reward_provider = self.programmatic_reward
        self.model = None
        self.trainer: OnlineTrainer | None = None
        self.projector = build_projector(
            "neural",
            projection_dim=self.model_cfg.projection_dim,
            tsne_perplexity=self.model_cfg.viz_tsne_perplexity,
        )

        self.model_type = "dnn"
        self.viz_method = "neural"
        self.training_phase = "calibration"

        self.n_dims: int | None = None
        self.current_class: int | None = None
        self.difficulty: str | None = None
        self.sample_count = 0
        self.last_source_sample_idx: int | None = None
        self.last_sample_wall_time: float | None = None
        self.last_stream_timestamp: float | None = None
        self.stream_sample_rate_hz: float | None = None

        self._neural_points: deque[np.ndarray] = deque(maxlen=history_len)
        self._penultimate: deque[np.ndarray] = deque(maxlen=history_len)
        self._labels: deque[int | None] = deque(maxlen=history_len)
        self._points: deque[list[float]] = deque(maxlen=history_len)
        self._preds: deque[int] = deque(maxlen=history_len)
        self._confs: deque[float] = deque(maxlen=history_len)
        self._rewards: deque[float] = deque(maxlen=history_len)
        self._accuracies: deque[float] = deque(maxlen=history_len)
        self._agreements: deque[float] = deque(maxlen=history_len)
        self._class_scales: deque[float] = deque(maxlen=history_len)
        self._strategy_qualities: deque[float] = deque(maxlen=history_len)
        self._target_margins: deque[float] = deque(maxlen=history_len)
        self._last_viz_refit_sample = -1
        self._last_seen_label: int | None = None
        self._samples_since_label_change = 0
        self._transition_ignored = False
        self._transition_samples_remaining = 0

        self._calibration_neural_by_class = {
            cls: deque(maxlen=self.calibration_samples_per_class)
            for cls in range(4)
        }
        self._calibration_penultimate_by_class = {
            cls: deque(maxlen=self.calibration_samples_per_class)
            for cls in range(4)
        }
        self._reference_neural_by_class = {
            cls: []
            for cls in range(4)
        }
        self._reference_penultimate_by_class = {
            cls: []
            for cls in range(4)
        }
        self._reference_snapshot_ready = False

        self.exploration_target_class: int | None = None
        self._exploration_penultimate: deque[np.ndarray] = deque(maxlen=self.exploration_max_points)
        self._exploration_reanalyze_every: int = 10
        self._exploration_last_analysis: int = 0
        self._exploration_result: dict | None = None

        self._last_payload = self._build_waiting_payload()

        self.set_model(model_type)
        self.set_viz_method(viz_method)
        self.set_training_phase(training_phase)

    def start_stream(self) -> None:
        if self.receiver_started:
            return
        self.receiver.start()
        self.receiver_started = True

    def stop_stream(self) -> None:
        if not self.receiver_started:
            return
        self.receiver.stop()
        self.receiver_started = False

    def close(self) -> None:
        self.stop_stream()

    def set_training_phase(self, training_phase: str) -> None:
        normalized = str(training_phase).strip().lower()
        if normalized not in AVAILABLE_TRAINING_PHASES:
            raise ValueError(f"Unknown training phase: {training_phase}")
        previous = self.training_phase
        self.training_phase = normalized

        if previous == "calibration" and normalized in {"feedback", "exploration"}:
            self.game_reward = self._make_game_reward_provider()

        if normalized == "exploration":
            self._exploration_penultimate = deque(maxlen=self.exploration_max_points)
            self._exploration_last_analysis = 0
            self._exploration_result = None
            if self.exploration_target_class is None:
                self.exploration_target_class = self.current_class if self.current_class is not None else 0
        else:
            self._exploration_result = None

        self._sync_reward_provider()

    def set_exploration_class(self, class_idx: int) -> None:
        if class_idx not in range(4):
            raise ValueError(f"Invalid class index: {class_idx}")
        self.exploration_target_class = class_idx
        self._exploration_penultimate = deque(maxlen=self.exploration_max_points)
        self._exploration_last_analysis = 0
        self._exploration_result = None

    def _build_game_config(self) -> RhythmGameConfig:
        fixed_level = LevelPolicy(
            hit_window_s=0.75,
            beat_interval_s=2.4,
            min_confidence=0.36,
            min_margin=0.0,
        )
        return RhythmGameConfig(
            n_classes=self.model_cfg.n_classes,
            prompt_duration_s=2.0,
            base_hit_window_s=fixed_level.hit_window_s,
            base_beat_interval_s=fixed_level.beat_interval_s,
            enable_adaptation=False,
            reward_min=self.model_cfg.reward_min,
            reward_max=self.model_cfg.reward_max,
            levels=(fixed_level,),
            start_level=0,
        )

    def _make_game_reward_provider(self) -> GameRewardProvider:
        return GameRewardProvider(model_cfg=self.model_cfg, game_cfg=self._build_game_config())

    def _active_game_provider(self) -> GameRewardProvider | None:
        if isinstance(self.reward_provider, GameRewardProvider):
            return self.reward_provider
        return None

    def _sync_reward_provider(self) -> None:
        self.reward_provider = (
            self.game_reward
            if self.training_phase in {"feedback", "exploration"}
            else self.programmatic_reward
        )
        if self.trainer is not None:
            self.trainer.reward_provider = self.reward_provider
            self.trainer.frozen = (self.training_phase == "exploration")

    def set_model(self, model_type: str) -> None:
        normalized = str(model_type).strip().lower()
        if normalized == "neural":
            normalized = "dnn"
        if normalized not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model_type = normalized
        if self.n_dims is not None:
            self._initialize_decoder()
        self._reset_history()

    def set_viz_method(self, viz_method: str) -> None:
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
    def training_phase_description(self) -> str:
        return TRAINING_PHASE_DESCRIPTIONS[self.training_phase]

    @property
    def difficulty_name(self) -> str:
        if self.difficulty is None:
            return "Waiting for emulator"
        return AVAILABLE_DIFFICULTIES.get(self.difficulty, self.difficulty.upper())

    def available_models(self) -> dict[str, str]:
        return dict(AVAILABLE_MODELS)

    def available_viz_methods(self) -> dict[str, str]:
        return dict(AVAILABLE_VIZ_METHODS)

    def available_training_phases(self) -> dict[str, str]:
        return dict(AVAILABLE_TRAINING_PHASES)

    def default_checkpoint_name(self) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        phase = self.training_phase or "phase"
        return f"{self.model_type}_{self.viz_method}_{phase}_{timestamp}"

    def save_model_snapshot(self, name: str | None = None) -> Path:
        checkpoint_name = self._normalize_checkpoint_name(name or self.default_checkpoint_name())
        checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_type": self.model_type,
            "viz_method": self.viz_method,
            "training_phase": self.training_phase,
            "sample_count": self.sample_count,
            "difficulty": self.difficulty,
            "stream_host": self.stream_host,
            "stream_port": self.stream_port,
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

    def status_snapshot(self) -> dict[str, Any]:
        session = self.session_snapshot()
        calibration = self.calibration_snapshot()
        coach = self.coach_snapshot()
        return {
            "stream": self.stream_snapshot(),
            "sample_count": self.sample_count,
            "source_sample_idx": self.last_source_sample_idx,
            "current_class": self.current_class,
            "current_class_name": self._class_name(self.current_class),
            "difficulty": self.difficulty,
            "difficulty_name": self.difficulty_name,
            "n_dims": self.n_dims,
            "training_phase": self.training_phase,
            "training_phase_name": self.training_phase_name,
            "training_phase_description": self.training_phase_description,
            "available_training_phases": self.available_training_phases(),
            "model_type": self.model_type,
            "model_name": self.model_name,
            "available_models": self.available_models(),
            "viz_method": self.viz_method,
            "viz_name": self.viz_name,
            "available_viz_methods": self.available_viz_methods(),
            "session": session,
            "game": self.game_snapshot(),
            "calibration": calibration,
            "coach": coach,
            "calibration_samples_per_class": self.calibration_samples_per_class,
            "transition_ignore_samples": self.transition_ignore_samples,
            "transition_ignored": self._transition_ignored,
            "transition_samples_remaining": self._transition_samples_remaining,
            "graph_frozen": self.training_phase != "calibration",
            "display_window": self.display_window,
            "display_min_samples_per_class": self.display_min_samples_per_class,
            "centroid_min_samples_per_class": self.centroid_min_samples_per_class,
        }

    def step(self, timeout: float = 0.05) -> dict[str, Any]:
        if not self.receiver_started:
            self.start_stream()

        sample = self.receiver.get(timeout=timeout)
        if sample is None:
            payload = self._build_stale_payload()
            self._last_payload = payload
            return payload

        self._ensure_decoder_ready(sample.embedding.shape[0])
        self._process_sample_metadata(sample)

        if self.trainer is None:
            payload = self._build_stale_payload()
            self._last_payload = payload
            return payload

        training_label = self._training_label_for_sample(sample)
        result = self.trainer.process_sample(sample, training_label=training_label)
        self.sample_count += 1
        self._append_history(sample, result)

        payload = self._build_live_payload(sample, result)
        self._last_payload = payload
        return payload

    def stream_snapshot(self) -> dict[str, Any]:
        age = self._last_sample_age_s()
        waiting_for_stream = self.receiver_started and self.last_sample_wall_time is None
        healthy = age is not None and age <= 5.0

        if not self.receiver_started:
            state = "idle"
            message = "Click Start Listening to subscribe to the emulator stream."
        elif waiting_for_stream:
            state = "waiting"
            message = (
                "Waiting for emulator data. Start `python -m emulator` and interact with the emulator window."
            )
        elif healthy:
            state = "live"
            message = "Live samples are arriving from the emulator."
        else:
            state = "stale"
            message = "The stream connected before, but new samples have paused."

        return {
            "host": self.stream_host,
            "port": self.stream_port,
            "sample_rate_hz": self.stream_sample_rate_hz,
            "listening": self.receiver_started,
            "connected": self.last_sample_wall_time is not None,
            "waiting_for_stream": waiting_for_stream,
            "healthy": healthy,
            "state": state,
            "message": message,
            "last_sample_age_s": age,
        }

    def session_snapshot(self) -> dict[str, Any]:
        labeled_samples = sum(1 for label in self._labels if label is not None)
        snapshot = {
            "rolling_reward": self._rolling_mean(self._rewards),
            "rolling_alignment": self._compute_recent_alignment(),
            "rolling_confidence": self._rolling_mean(self._confs),
            "rolling_signal": 0.5 * (
                self._rolling_mean(self._class_scales) + self._rolling_mean(self._strategy_qualities)
            ),
            "total_samples": self.sample_count,
            "labeled_samples": labeled_samples,
        }
        game = self.game_snapshot()
        if game is not None:
            snapshot["game"] = game
        return snapshot

    def calibration_snapshot(self) -> dict[str, Any]:
        counts_by_class = self._calibration_bank_counts()
        ready_classes = sum(
            1
            for cls in range(4)
            if counts_by_class[str(cls)] >= self.calibration_samples_per_class
        )

        label_centroids = self._compute_true_label_centroids()
        label_min_sep, label_mean_sep = self._compute_separation(label_centroids)
        rolling_alignment = self._compute_recent_alignment()
        signal_fit = 0.5 * (
            self._rolling_mean(self._class_scales) + self._rolling_mean(self._strategy_qualities)
        )
        separation_fit = float(np.clip(label_mean_sep / 1.6, 0.0, 1.0))
        readiness = float(
            np.clip(
                0.30 * (ready_classes / 4.0)
                + 0.30 * separation_fit
                + 0.25 * rolling_alignment
                + 0.15 * signal_fit,
                0.0,
                1.0,
            )
        )
        ready = bool(
            ready_classes == 4
            and label_mean_sep >= 0.80
            and rolling_alignment >= 0.55
        )

        if ready:
            message = "All four tasks are separating reliably. The decoder is ready for stronger neurofeedback."
        elif ready_classes < 4:
            message = (
                "Keep collecting stable attempts for every task. The first "
                f"{self.transition_ignore_samples} samples after a label switch are ignored, "
                "then each class fills its own calibration bank."
            )
        elif label_mean_sep < 0.80:
            message = "The tasks exist, but their zones still overlap. Keep the strategies that pull them farther apart."
        else:
            message = "The decoder is learning the task labels. Stay consistent a little longer to stabilize the map."

        return {
            "ready": ready,
            "readiness": readiness,
            "label_counts": counts_by_class,
            "classes_ready": ready_classes,
            "target_per_class": self.calibration_samples_per_class,
            "reference_ready": self._reference_snapshot_ready,
            "min_label_separation": label_min_sep,
            "mean_label_separation": label_mean_sep,
            "rolling_alignment": rolling_alignment,
            "signal_fit": signal_fit,
            "message": message,
        }

    def coach_snapshot(self) -> dict[str, Any]:
        if not self._preds:
            return {
                "state": "idle",
                "headline": "Waiting for the first sample",
                "message": (
                    "Start the emulator, pick a task with 1-4, and use the arrow keys there. "
                    "The dashboard will start coaching as soon as the stream arrives."
                ),
                "score": 0.0,
                "score_label": "Decoder Match",
                "target_margin": 0.0,
            }

        if self.training_phase in {"feedback", "exploration"}:
            return self._build_game_coach_snapshot()

        current_label = self._labels[-1]
        predicted_class = self._preds[-1]
        probabilities = self._last_probabilities()
        calibration = self.calibration_snapshot()
        session = self.session_snapshot()
        return self._build_coach_snapshot(
            current_label=current_label,
            predicted_class=predicted_class,
            probabilities=probabilities,
            calibration=calibration,
            session=session,
        )

    def _build_coach_snapshot(
        self,
        current_label: int | None,
        predicted_class: int,
        probabilities: np.ndarray,
        calibration: dict[str, Any],
        session: dict[str, Any],
    ) -> dict[str, Any]:
        probabilities = np.asarray(probabilities, dtype=float)

        if current_label is None:
            headline = "Pick a task in the emulator"
            message = "Use 1-4 in the emulator to declare the task you want to train, then search for a stable strategy with the arrows."
            return {
                "state": "idle",
                "headline": headline,
                "message": message,
                "score": session["rolling_alignment"],
                "score_label": "Decoder Match",
                "target_margin": 0.0,
            }

        target_prob = float(probabilities[int(current_label)])
        target_margin = self._target_margin(probabilities, int(current_label))
        signal_fit = float(
            np.clip(
                0.5 * (
                    (self._class_scales[-1] if self._class_scales else 0.0)
                    + (self._strategy_qualities[-1] if self._strategy_qualities else 0.0)
                ),
                0.0,
                1.0,
            )
        )
        separation_fit = float(np.clip(calibration["mean_label_separation"] / 1.6, 0.0, 1.0))
        score = float(
            np.clip(
                0.40 * target_prob
                + 0.25 * session["rolling_alignment"]
                + 0.20 * signal_fit
                + 0.15 * separation_fit,
                0.0,
                1.0,
            )
        )

        target_name = self._class_name(current_label)
        predicted_name = self._class_name(predicted_class)

        if predicted_class == current_label and score >= 0.74:
            state = "good"
            headline = "Great match"
            message = f"Model reads {target_name}. Hold this strategy."
        elif predicted_class == current_label:
            state = "hold"
            headline = "Hold steady"
            message = f"Reading {target_name} — signal still settling."
        elif target_margin > -0.10:
            state = "adjust"
            headline = "Adjust now"
            message = f"Close — nudge away from {predicted_name}."
        elif self._strategy_qualities and self._strategy_qualities[-1] < 0.40:
            state = "recover"
            headline = "Reset strategy"
            message = "Signal unstable. Re-center arrows first."
        else:
            state = "recover"
            headline = "Change strategy"
            message = f"Reads {predicted_name}, not {target_name}. Keep adjusting."

        if self.training_phase == "calibration" and calibration["ready"]:
            state = "ready"
            headline = "Ready to practice"
            message = "Four task zones formed. Switch to Practice mode."

        return {
            "state": state,
            "headline": headline,
            "message": message,
            "score": score,
            "score_label": "Decoder Match",
            "target_margin": target_margin,
        }

    def _build_game_coach_snapshot(self) -> dict[str, Any]:
        game = self.game_snapshot()
        if not game:
            return {
                "state": "idle",
                "headline": "Waiting for game timing",
                "message": "Live samples will start the prompt sequence as soon as the stream is active.",
                "score": 0.0,
                "score_label": "Prompt Match",
                "target_margin": 0.0,
            }

        target_class = game["target_class"]
        predicted_class = self._preds[-1] if self._preds else None
        probabilities = self._last_probabilities()
        target_prob = float(probabilities[int(target_class)]) if target_class is not None else 0.0
        target_margin = self._target_margin(probabilities, target_class)
        hit_rate = float(game.get("hit_rate") or 0.0)
        score = float(np.clip(0.65 * target_prob + 0.20 * hit_rate + 0.15 * self._compute_recent_alignment(), 0.0, 1.0))

        target_name = self._class_name(target_class)
        predicted_name = self._class_name(predicted_class)
        in_window = bool(game.get("in_window"))
        seconds_to_window_start = game.get("seconds_to_window_start")
        feedback = game.get("last_feedback") or {}

        if self.training_phase == "exploration":
            if predicted_class == target_class and target_prob >= 0.75:
                state = "good"
                headline = "Strong frozen readout"
                message = f"Frozen model reliably reads {target_name}. Keep this strategy."
            elif predicted_class == target_class:
                state = "hold"
                headline = "Useful exploration pattern"
                message = f"The frozen model reads {target_name}. Hold it a bit longer and compare nearby strategies."
            else:
                state = "adjust"
                headline = "Search a different strategy"
                message = f"Frozen model reads {predicted_name}, not {target_name}. Shift your strategy and watch the map move."
            return {
                "state": state,
                "headline": headline,
                "message": message,
                "score": score,
                "score_label": "Frozen Readout",
                "target_margin": target_margin,
            }

        if not in_window and isinstance(seconds_to_window_start, (int, float)) and seconds_to_window_start > 0.12:
            state = "hold"
            headline = f"Prepare {target_name}"
            message = f"Next scoring window opens in {seconds_to_window_start:.2f}s. Line up the strategy before the hit window."
        elif feedback.get("timing_hit"):
            state = "good"
            headline = "Nice hit"
            message = f"You matched {target_name} in the scoring window. Keep that strategy for the next beat."
        elif predicted_class == target_class:
            state = "hold"
            headline = "Model has the right class"
            message = f"Decoder reads {target_name}. Hold steady and try to land it inside the hit window."
        elif target_margin > -0.10:
            state = "adjust"
            headline = "Almost there"
            message = f"Prompt wants {target_name}, but the readout leans toward {predicted_name}. Nudge the strategy now."
        else:
            state = "recover"
            headline = "Change strategy"
            message = f"Prompt wants {target_name}. The decoder still reads {predicted_name}, so keep adjusting before the window closes."

        return {
            "state": state,
            "headline": headline,
            "message": message,
            "score": score,
            "score_label": "Prompt Match",
            "target_margin": target_margin,
        }

    def _initialize_decoder(self) -> None:
        self.model = build_decoder(self.model_type, self.model_cfg)
        self.trainer = OnlineTrainer(
            model=self.model,
            cfg=self.model_cfg,
            reward_provider=self.reward_provider,
            device=self.device,
        )
        self._sync_reward_provider()

    def _ensure_decoder_ready(self, input_dim: int) -> None:
        input_dim = int(input_dim)
        if self.n_dims == input_dim and self.trainer is not None:
            return

        self.n_dims = input_dim
        self.model_cfg.input_dim = input_dim
        self._initialize_decoder()
        self._reset_history()

    def _process_sample_metadata(self, sample: StreamSample) -> None:
        self._update_label_transition(sample.label)
        self.current_class = sample.label
        self.last_source_sample_idx = int(sample.sample_idx)
        self.last_sample_wall_time = time.time()
        self.last_stream_timestamp = float(sample.timestamp)
        sample_rate = sample.raw.get("sample_rate") if isinstance(sample.raw, dict) else None
        if sample_rate is not None:
            try:
                rate = float(sample_rate)
                if rate > 0.0:
                    self.stream_sample_rate_hz = rate
            except (TypeError, ValueError):
                pass
        if sample.difficulty:
            self.difficulty = str(sample.difficulty)

    def _update_label_transition(self, label: int | None) -> None:
        if label is None:
            self._last_seen_label = None
            self._samples_since_label_change = 0
            self._transition_ignored = False
            self._transition_samples_remaining = 0
            return

        if label != self._last_seen_label:
            self._last_seen_label = int(label)
            self._samples_since_label_change = 1
        else:
            self._samples_since_label_change += 1

        ignore_phase = self.training_phase in {"calibration", "exploration"}
        self._transition_ignored = (
            ignore_phase
            and self._samples_since_label_change <= self.transition_ignore_samples
        )
        self._transition_samples_remaining = max(
            0,
            self.transition_ignore_samples - self._samples_since_label_change + 1,
        ) if self._transition_ignored else 0

    def _training_label_for_sample(self, sample: StreamSample) -> int | None:
        if self.training_phase == "exploration":
            return None
        if sample.label is None:
            return None
        if self._transition_ignored:
            return None
        return int(sample.label)

    def _append_history(self, sample: StreamSample, result: InferenceStep) -> None:
        self._neural_points.append(np.asarray(result.projection, dtype=float).copy())
        self._penultimate.append(np.asarray(result.penultimate, dtype=float).copy())
        self._labels.append(sample.label)
        self._preds.append(int(result.predicted_class))
        self._confs.append(float(result.confidence))
        self._rewards.append(float(result.reward))
        self._class_scales.append(float(sample.class_scale or 0.0))
        self._strategy_qualities.append(float(sample.strategy_quality or 0.0))

        feedback_target = self._feedback_target_class(sample, result)
        target_margin = self._target_margin(result.probabilities, feedback_target)
        self._target_margins.append(float(target_margin))
        agreement = 1.0 if feedback_target is not None and int(feedback_target) == int(result.predicted_class) else 0.0
        self._agreements.append(agreement)

        if result.training and result.training.update_applied:
            self._accuracies.append(float(result.training.balanced_accuracy))
        else:
            last_acc = self._accuracies[-1] if self._accuracies else agreement
            self._accuracies.append(last_acc)

        self._append_calibration_bank(sample, result)
        self._refresh_projected_history()

        if (
            self.training_phase == "exploration"
            and self.exploration_target_class is not None
            and sample.label == self.exploration_target_class
            and not self._transition_ignored
        ):
            self._exploration_penultimate.append(
                np.asarray(result.penultimate, dtype=float).copy()
            )
            n = len(self._exploration_penultimate)
            if n >= 20 and (n - self._exploration_last_analysis) >= self._exploration_reanalyze_every:
                self._run_exploration_analysis()
                self._exploration_last_analysis = n

    def _append_calibration_bank(self, sample: StreamSample, result: InferenceStep) -> None:
        if self.training_phase != "calibration":
            return
        if sample.label is None or self._transition_ignored:
            return

        label = int(sample.label)
        self._calibration_neural_by_class[label].append(
            np.asarray(result.projection, dtype=float).copy()
        )
        self._calibration_penultimate_by_class[label].append(
            np.asarray(result.penultimate, dtype=float).copy()
        )

        if self._reference_snapshot_ready:
            return
        if all(
            len(self._calibration_penultimate_by_class[cls]) >= self.calibration_samples_per_class
            for cls in range(4)
        ):
            for cls in range(4):
                self._reference_neural_by_class[cls] = [
                    np.asarray(point, dtype=float).copy()
                    for point in self._calibration_neural_by_class[cls]
                ]
                self._reference_penultimate_by_class[cls] = [
                    np.asarray(point, dtype=float).copy()
                    for point in self._calibration_penultimate_by_class[cls]
                ]
            self._reference_snapshot_ready = True

    def _run_exploration_analysis(self) -> None:
        from model.exploration import analyze_strategies

        if not self._exploration_penultimate or self.model is None:
            self._exploration_result = None
            return

        penultimate = np.stack(self._exploration_penultimate, axis=0)
        result = analyze_strategies(
            penultimate=penultimate,
            model=self.model,
            target_class=self.exploration_target_class,
            device=self.device,
        )
        self._exploration_result = result.to_dict() if result is not None else None

    def _build_exploration_payload(self) -> dict | None:
        if self.training_phase != "exploration":
            return None
        return {
            "target_class": self.exploration_target_class,
            "target_class_name": self._class_name(self.exploration_target_class),
            "n_collected": len(self._exploration_penultimate),
            "analysis": self._exploration_result,
        }

    def _build_live_payload(self, sample: StreamSample, result: InferenceStep) -> dict[str, Any]:
        centroids = self._compute_centroids()
        min_sep, mean_sep = self._compute_separation(centroids)
        mean_spread = self._compute_spread(centroids)
        cluster_points, cluster_labels = self._cluster_display_points()
        session = self.session_snapshot()
        calibration = self.calibration_snapshot()
        game = self.game_snapshot(timestamp=sample.timestamp)
        intended_class = self._ui_target_class(sample, result, game)
        coach = self.coach_snapshot()
        zone = self._zone_snapshot(centroids)

        return {
            "sample_idx": self.sample_count,
            "source_sample_idx": self.last_source_sample_idx,
            "timestamp": sample.timestamp,
            "stream": self.stream_snapshot(),
            "intended_class": intended_class,
            "intended_class_name": self._class_name(intended_class),
            "current_class": sample.label,
            "current_class_name": self._class_name(sample.label),
            "emulator_label": sample.label,
            "emulator_label_name": self._class_name(sample.label),
            "difficulty": self.difficulty,
            "difficulty_name": self.difficulty_name,
            "training_phase": self.training_phase,
            "training_phase_name": self.training_phase_name,
            "training_phase_description": self.training_phase_description,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "viz_method": self.viz_method,
            "viz_name": self.viz_name,
            "predicted_class": int(result.predicted_class),
            "predicted_class_name": self._class_name(result.predicted_class),
            "confidence": float(result.confidence),
            "probabilities": result.probabilities.tolist(),
            "reward": float(result.reward),
            "agreement": float(self._agreements[-1]) if self._agreements else 0.0,
            "target_margin": float(self._target_margins[-1]) if self._target_margins else 0.0,
            "projection": self._points[-1] if self._points else result.projection.tolist(),
            "class_scale": float(sample.class_scale or 0.0),
            "strategy_quality": float(sample.strategy_quality or 0.0),
            "signal_score": float(
                np.clip(0.5 * ((sample.class_scale or 0.0) + (sample.strategy_quality or 0.0)), 0.0, 1.0)
            ),
            "zone": zone,
            "points": list(self._points),
            "labels": [None if label is None else int(label) for label in self._labels],
            "predictions": list(self._preds),
            "display_indices": self._display_indices(),
            "cluster_points": cluster_points,
            "cluster_labels": cluster_labels,
            "display_window": self.display_window,
            "display_min_samples_per_class": self.display_min_samples_per_class,
            "confidences": list(self._confs),
            "rewards": list(self._rewards),
            "agreements": list(self._agreements),
            "accuracies": list(self._accuracies),
            "centroids": {str(k): v.tolist() for k, v in centroids.items()},
            "reference_centroids": {str(k): v.tolist() for k, v in self._reference_centroids().items()},
            "centroid_window": self.centroid_window,
            "centroid_min_samples_per_class": self.centroid_min_samples_per_class,
            "calibration_samples_per_class": self.calibration_samples_per_class,
            "min_separation": min_sep,
            "mean_separation": mean_sep,
            "mean_spread": mean_spread,
            "transition_ignored": self._transition_ignored,
            "transition_samples_remaining": self._transition_samples_remaining,
            "graph_frozen": self.training_phase != "calibration",
            "session": session,
            "game": game,
            "calibration": calibration,
            "coach": coach,
            "training": {
                "update_applied": result.training.update_applied if result.training else False,
                "total_loss": result.training.total_loss if result.training else 0.0,
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
                "num_updates": self.trainer.num_updates if self.trainer else 0,
                "labeled_seen": self.trainer.labeled_seen if self.trainer else 0,
            },
            "exploration": self._build_exploration_payload(),
        }

    def _build_waiting_payload(self) -> dict[str, Any]:
        game = self.game_snapshot()
        return {
            "sample_idx": self.sample_count,
            "source_sample_idx": self.last_source_sample_idx,
            "timestamp": self.last_stream_timestamp,
            "stream": self.stream_snapshot(),
            "intended_class": None,
            "intended_class_name": "Waiting",
            "current_class": self.current_class,
            "current_class_name": self._class_name(self.current_class),
            "emulator_label": self.current_class,
            "emulator_label_name": self._class_name(self.current_class),
            "difficulty": self.difficulty,
            "difficulty_name": self.difficulty_name,
            "training_phase": self.training_phase,
            "training_phase_name": self.training_phase_name,
            "training_phase_description": self.training_phase_description,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "viz_method": self.viz_method,
            "viz_name": self.viz_name,
            "predicted_class": None,
            "predicted_class_name": "Waiting",
            "confidence": 0.0,
            "probabilities": [0.0, 0.0, 0.0, 0.0],
            "reward": 0.0,
            "agreement": 0.0,
            "target_margin": 0.0,
            "projection": [0.0, 0.0],
            "class_scale": 0.0,
            "strategy_quality": 0.0,
            "signal_score": 0.0,
            "zone": {
                "class_idx": None,
                "class_name": "Waiting for stream",
                "distance": None,
                "source": "none",
            },
            "points": [],
            "labels": [],
            "predictions": [],
            "display_indices": [],
            "cluster_points": [],
            "cluster_labels": [],
            "display_window": self.display_window,
            "display_min_samples_per_class": self.display_min_samples_per_class,
            "confidences": [],
            "rewards": [],
            "agreements": [],
            "accuracies": [],
            "centroids": {},
            "reference_centroids": {},
            "centroid_window": self.centroid_window,
            "centroid_min_samples_per_class": self.centroid_min_samples_per_class,
            "calibration_samples_per_class": self.calibration_samples_per_class,
            "min_separation": 0.0,
            "mean_separation": 0.0,
            "mean_spread": 0.0,
            "transition_ignored": self._transition_ignored,
            "transition_samples_remaining": self._transition_samples_remaining,
            "graph_frozen": self.training_phase != "calibration",
            "session": self.session_snapshot(),
            "game": game,
            "calibration": self.calibration_snapshot(),
            "coach": self.coach_snapshot(),
            "training": {
                "update_applied": False,
                "total_loss": 0.0,
                "balanced_accuracy": None,
                "macro_f1": None,
                "rl_enabled": False,
                "num_updates": self.trainer.num_updates if self.trainer else 0,
                "labeled_seen": self.trainer.labeled_seen if self.trainer else 0,
            },
            "exploration": None,
        }

    def _build_stale_payload(self) -> dict[str, Any]:
        payload = copy.deepcopy(self._last_payload)
        payload["stream"] = self.stream_snapshot()
        payload["training_phase"] = self.training_phase
        payload["training_phase_name"] = self.training_phase_name
        payload["training_phase_description"] = self.training_phase_description
        payload["difficulty"] = self.difficulty
        payload["difficulty_name"] = self.difficulty_name
        payload["model_type"] = self.model_type
        payload["model_name"] = self.model_name
        payload["viz_method"] = self.viz_method
        payload["viz_name"] = self.viz_name
        payload["graph_frozen"] = self.training_phase != "calibration"
        payload["transition_ignored"] = self._transition_ignored
        payload["transition_samples_remaining"] = self._transition_samples_remaining
        payload["session"] = self.session_snapshot()
        payload["game"] = self.game_snapshot()
        payload["calibration"] = self.calibration_snapshot()
        payload["coach"] = self.coach_snapshot()
        payload["training"] = {
            "update_applied": False,
            "total_loss": 0.0,
            "balanced_accuracy": None,
            "macro_f1": None,
            "rl_enabled": bool(getattr(self.trainer, "labeled_seen", 0) >= self.model_cfg.warmup_labeled_samples),
            "num_updates": self.trainer.num_updates if self.trainer else 0,
            "labeled_seen": self.trainer.labeled_seen if self.trainer else 0,
        }
        payload["exploration"] = self._build_exploration_payload()
        return payload

    def _zone_snapshot(self, centroids: dict[int, np.ndarray]) -> dict[str, Any]:
        if not self._points:
            return {
                "class_idx": None,
                "class_name": "Waiting for stream",
                "distance": None,
                "source": "none",
            }

        current = np.asarray(self._points[-1], dtype=float)
        if centroids:
            best_cls, best_dist = min(
                ((cls, float(np.linalg.norm(current - centroid))) for cls, centroid in centroids.items()),
                key=lambda item: item[1],
            )
            return {
                "class_idx": int(best_cls),
                "class_name": self._class_name(int(best_cls)),
                "distance": best_dist,
                "source": "centroid",
            }

        predicted = self._preds[-1] if self._preds else None
        return {
            "class_idx": predicted,
            "class_name": self._class_name(predicted),
            "distance": None,
            "source": "prediction",
        }

    def _compute_centroids(self) -> dict[int, np.ndarray]:
        points_arr, labels_arr = self._calibration_projected_points()
        if points_arr.size == 0:
            return {}
        centroids: dict[int, np.ndarray] = {}
        for cls in range(4):
            mask = labels_arr == cls
            if int(np.sum(mask)) >= 3:
                centroids[cls] = np.mean(points_arr[mask], axis=0)
        return centroids

    def _compute_true_label_centroids(self) -> dict[int, np.ndarray]:
        return self._compute_centroids()

    def _compute_separation(self, centroids: dict[int, np.ndarray]) -> tuple[float, float]:
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
        if not centroids:
            return 0.0

        points_arr, labels_arr = self._calibration_projected_points()
        if points_arr.size == 0:
            return 0.0

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

    def _display_indices(self) -> list[int]:
        plot_labels = self._plot_labels_array()
        return self._window_indices_with_class_floor(
            labels=plot_labels.tolist(),
            window=self.display_window,
            min_per_class=self.display_min_samples_per_class,
        )

    def _calibration_bank_counts(self) -> dict[str, int]:
        return {
            str(cls): int(len(self._calibration_penultimate_by_class[cls]))
            for cls in range(4)
        }

    def _calibration_projected_points(self) -> tuple[np.ndarray, np.ndarray]:
        labels: list[int] = []

        if self.viz_method == "neural":
            points = [
                np.asarray(point, dtype=float)
                for cls in range(4)
                for point in self._calibration_neural_by_class[cls]
            ]
            for cls in range(4):
                labels.extend([cls] * len(self._calibration_neural_by_class[cls]))
            if not points:
                return np.empty((0, self.model_cfg.projection_dim), dtype=float), np.empty((0,), dtype=np.int64)
            return np.asarray(points, dtype=float), np.asarray(labels, dtype=np.int64)

        embeddings = [
            np.asarray(point, dtype=float)
            for cls in range(4)
            for point in self._calibration_penultimate_by_class[cls]
        ]
        for cls in range(4):
            labels.extend([cls] * len(self._calibration_penultimate_by_class[cls]))
        if not embeddings:
            return np.empty((0, self.model_cfg.projection_dim), dtype=float), np.empty((0,), dtype=np.int64)

        embedding_arr = np.asarray(embeddings, dtype=float)
        try:
            projected = self.projector.transform(embedding_arr)
        except Exception:
            projected = embedding_arr[:, : self.model_cfg.projection_dim]
        return np.asarray(projected, dtype=float), np.asarray(labels, dtype=np.int64)

    def _cluster_display_points(self) -> tuple[list[list[float]], list[int]]:
        points_arr, labels_arr = self._calibration_projected_points()
        if points_arr.size == 0:
            return [], []
        points = [np.asarray(point, dtype=float).tolist() for point in points_arr]
        labels = [int(label) for label in labels_arr.tolist()]
        return points, labels

    def _projector_fit_data(self) -> tuple[np.ndarray, np.ndarray]:
        embeddings = [
            np.asarray(point, dtype=float)
            for cls in range(4)
            for point in self._calibration_penultimate_by_class[cls]
        ]
        labels = [
            cls
            for cls in range(4)
            for _ in self._calibration_penultimate_by_class[cls]
        ]
        if not embeddings:
            return np.empty((0, 0), dtype=float), np.empty((0,), dtype=np.int64)

        fit_x = np.asarray(embeddings, dtype=float)
        fit_y = np.asarray(labels, dtype=np.int64)
        fit_window = min(self.viz_fit_window, fit_x.shape[0])
        return fit_x[-fit_window:], fit_y[-fit_window:]

    def _reference_centroids(self) -> dict[int, np.ndarray]:
        labels: list[int] = []

        if self.viz_method == "neural":
            points = [
                np.asarray(point, dtype=float)
                for cls in range(4)
                for point in self._reference_neural_by_class[cls]
            ]
            for cls in range(4):
                labels.extend([cls] * len(self._reference_neural_by_class[cls]))
            if not points:
                return {}
            points_arr = np.asarray(points, dtype=float)
        else:
            embeddings = [
                np.asarray(point, dtype=float)
                for cls in range(4)
                for point in self._reference_penultimate_by_class[cls]
            ]
            for cls in range(4):
                labels.extend([cls] * len(self._reference_penultimate_by_class[cls]))
            if not embeddings:
                return {}
            embedding_arr = np.asarray(embeddings, dtype=float)
            try:
                points_arr = np.asarray(self.projector.transform(embedding_arr), dtype=float)
            except Exception:
                points_arr = embedding_arr[:, : self.model_cfg.projection_dim]

        labels_arr = np.asarray(labels, dtype=np.int64)
        centroids: dict[int, np.ndarray] = {}
        for cls in range(4):
            mask = labels_arr == cls
            if int(np.sum(mask)) >= 3:
                centroids[cls] = np.mean(points_arr[mask], axis=0)
        return centroids

    def _window_indices_with_class_floor(
        self,
        labels: list[int | None],
        window: int,
        min_per_class: int,
    ) -> list[int]:
        total = len(labels)
        if total == 0:
            return []

        start = max(0, total - max(1, int(window)))
        selected = set(range(start, total))

        for cls in range(4):
            class_indices = [idx for idx, label in enumerate(labels) if label == cls]
            if not class_indices:
                continue
            take = min(len(class_indices), max(1, int(min_per_class)))
            selected.update(class_indices[-take:])

        return sorted(selected)

    def _refresh_projected_history(self) -> None:
        if not self._neural_points:
            self._points.clear()
            return

        if self.viz_method == "neural":
            projected = np.stack(list(self._neural_points), axis=0)
        else:
            embeddings = np.stack(list(self._penultimate), axis=0)
            fit_x, fit_y = self._projector_fit_data()
            if fit_x.size == 0:
                fit_x = embeddings
                fit_y = self._projection_labels()
            should_refit = (
                self._last_viz_refit_sample < 0
                or (
                    self.training_phase == "calibration"
                    and (self.sample_count - self._last_viz_refit_sample) >= self.viz_refit_every
                )
            )
            if should_refit:
                self.projector.fit(fit_x, y=fit_y if fit_y.size else None)
                self._last_viz_refit_sample = self.sample_count
            projected = self.projector.transform(embeddings)

        self._points.clear()
        for point in projected:
            self._points.append(np.asarray(point, dtype=float).tolist())

    def _reset_history(self) -> None:
        self._neural_points.clear()
        self._penultimate.clear()
        self._labels.clear()
        self._points.clear()
        self._preds.clear()
        self._confs.clear()
        self._rewards.clear()
        self._accuracies.clear()
        self._agreements.clear()
        self._class_scales.clear()
        self._strategy_qualities.clear()
        self._target_margins.clear()
        self._last_seen_label = None
        self._samples_since_label_change = 0
        self._transition_ignored = False
        self._transition_samples_remaining = 0
        for cls in range(4):
            self._calibration_neural_by_class[cls].clear()
            self._calibration_penultimate_by_class[cls].clear()
            self._reference_neural_by_class[cls] = []
            self._reference_penultimate_by_class[cls] = []
        self._reference_snapshot_ready = False
        self._exploration_penultimate = deque(maxlen=self.exploration_max_points)
        self._exploration_last_analysis = 0
        self._exploration_result = None
        self._last_viz_refit_sample = -1
        self._last_payload = self._build_waiting_payload()

    def _rolling_mean(self, values: deque[float], window: int = 50) -> float:
        if not values:
            return 0.0
        arr = np.asarray(list(values)[-window:], dtype=float)
        return float(np.mean(arr)) if arr.size else 0.0

    def _compute_recent_alignment(self, window: int = 60) -> float:
        if not self._agreements:
            return 0.0
        return self._rolling_mean(self._agreements, window=window)

    def _last_probabilities(self) -> np.ndarray:
        last = self._last_payload.get("probabilities", [0.0, 0.0, 0.0, 0.0])
        return np.asarray(last, dtype=float)

    def _target_margin(self, probs: np.ndarray | list[float], label: int | None) -> float:
        if label is None:
            return 0.0
        arr = np.asarray(probs, dtype=float).reshape(-1)
        target = float(arr[int(label)])
        others = np.delete(arr, int(label))
        if others.size == 0:
            return target
        return float(target - np.max(others))

    def _last_sample_age_s(self) -> float | None:
        if self.last_sample_wall_time is None:
            return None
        return max(0.0, time.time() - self.last_sample_wall_time)

    def game_snapshot(self, timestamp: float | None = None) -> dict[str, Any] | None:
        if self.training_phase not in {"feedback", "exploration"}:
            return None

        provider = self._active_game_provider()
        if provider is None:
            return None

        ts = timestamp if timestamp is not None else self.last_stream_timestamp
        if ts is None:
            return None

        preview = provider.preview(float(ts))
        session = provider.session.snapshot()
        feedback = provider.last_feedback

        return {
            "prompt_id": preview.prompt_id,
            "target_class": preview.target_class,
            "target_class_name": self._class_name(preview.target_class),
            "next_target_class": preview.next_target_class,
            "next_target_class_name": self._class_name(preview.next_target_class),
            "in_window": bool(preview.in_window),
            "prompt_progress": float(preview.prompt_progress),
            "seconds_to_window_start": float(preview.seconds_to_window_start),
            "seconds_to_prompt_end": float(preview.seconds_to_prompt_end),
            "seconds_to_next_prompt_start": float(preview.seconds_to_next_prompt_start),
            "level": int(session["level"]),
            "streak": int(session["streak"]),
            "best_streak": int(session["best_streak"]),
            "total_prompts": int(session["total_prompts"]),
            "total_hits": int(session["total_hits"]),
            "hit_rate": float(session["hit_rate"]),
            "last_feedback": (
                {
                    "predicted_class": feedback.predicted_class,
                    "predicted_class_name": self._class_name(feedback.predicted_class),
                    "confidence": float(feedback.confidence),
                    "margin": float(feedback.margin),
                    "label_correct": bool(feedback.label_correct),
                    "timing_hit": bool(feedback.timing_hit),
                    "timing_error_s": float(feedback.timing_error_s),
                    "hit": bool(feedback.hit),
                    "reward": float(feedback.reward),
                    "components": dict(feedback.components),
                }
                if feedback is not None
                else None
            ),
        }

    def _feedback_target_class(self, sample: StreamSample, result: InferenceStep) -> int | None:
        if self.training_phase in {"feedback", "exploration"} and result.game_target_class is not None:
            return int(result.game_target_class)
        if sample.label is None:
            return None
        return int(sample.label)

    def _ui_target_class(
        self,
        sample: StreamSample,
        result: InferenceStep,
        game: dict[str, Any] | None,
    ) -> int | None:
        if game is not None and game.get("target_class") is not None:
            return int(game["target_class"])
        return self._feedback_target_class(sample, result)

    def _class_name(self, class_idx: int | None) -> str:
        if class_idx is None:
            return "Rest / No Label"
        return CLASS_NAME_MAP.get(int(class_idx), "Unknown")

    def _normalize_checkpoint_name(self, raw_name: str) -> str:
        name = raw_name.strip()
        if name.endswith(".pt"):
            name = name[:-3]
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
        return safe or self.default_checkpoint_name()
