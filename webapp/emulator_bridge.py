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
from collections import Counter, deque
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

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
    "calibration": "Guided Calibration",
    "feedback": "Neurofeedback Coach",
}

TRAINING_PHASE_DESCRIPTIONS = {
    "calibration": (
        "Follow the requested task in the emulator, repeat it consistently, "
        "and build four clearly separated latent clusters."
    ),
    "feedback": (
        "Watch what the model thinks you are doing. If it reads the wrong task, "
        "change strategy until the prediction and zone both move where you want."
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
        history_len: int = 300,
        centroid_window: int = 50,
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
        self.centroid_window = centroid_window
        self.viz_fit_window = max(10, min(viz_fit_window, history_len))
        self.viz_refit_every = max(1, viz_refit_every)

        self.receiver: StreamReceiver = receiver or ZMQEmbeddingReceiver(
            host=stream_host,
            port=stream_port,
            embedding_key=embedding_key,
        )
        self.receiver_started = False

        self.model_cfg = ModelConfig(input_dim=None)
        self.device = torch.device("cpu")
        self.reward_provider = ProgrammaticReward(self.model_cfg)
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
        self.training_phase = normalized

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
            "calibration": calibration,
            "coach": coach,
        }

    def step(self, timeout: float = 0.05) -> dict[str, Any]:
        if not self.receiver_started:
            self.start_stream()

        sample = self.receiver.get(timeout=timeout)
        if sample is None:
            payload = self._build_stale_payload()
            self._last_payload = payload
            return payload

        self._process_sample_metadata(sample)
        self._ensure_decoder_ready(sample.embedding.shape[0])

        if self.trainer is None:
            payload = self._build_stale_payload()
            self._last_payload = payload
            return payload

        result = self.trainer.process_sample(sample)
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
        return {
            "rolling_reward": self._rolling_mean(self._rewards),
            "rolling_alignment": self._compute_recent_alignment(),
            "rolling_confidence": self._rolling_mean(self._confs),
            "rolling_signal": 0.5 * (
                self._rolling_mean(self._class_scales) + self._rolling_mean(self._strategy_qualities)
            ),
            "total_samples": self.sample_count,
            "labeled_samples": labeled_samples,
        }

    def calibration_snapshot(self) -> dict[str, Any]:
        counts = Counter(int(label) for label in self._labels if label is not None)
        counts_by_class = {str(cls): int(counts.get(cls, 0)) for cls in range(4)}
        ready_classes = sum(1 for cls in range(4) if counts_by_class[str(cls)] >= 12)

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
        ready = bool(ready_classes == 4 and label_mean_sep >= 0.80 and rolling_alignment >= 0.55)

        if ready:
            message = "All four tasks are separating reliably. The decoder is ready for stronger neurofeedback."
        elif ready_classes < 4:
            message = "Collect stable attempts for every task so each label gets its own zone."
        elif label_mean_sep < 0.80:
            message = "The tasks exist, but their zones still overlap. Keep the strategies that pull them farther apart."
        else:
            message = "The decoder is learning the task labels. Stay consistent a little longer to stabilize the map."

        return {
            "ready": ready,
            "readiness": readiness,
            "label_counts": counts_by_class,
            "classes_ready": ready_classes,
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
            headline = "Keep this strategy"
            message = (
                f"The model agrees that you are using {target_name}. Hold the same pattern so the current zone stays there and the cluster keeps separating."
            )
        elif predicted_class == current_label:
            state = "hold"
            headline = "Hold it a little longer"
            message = (
                f"The model is reading {target_name}, but the signal is still settling. "
                "Stay with the same strategy until the target bar becomes more dominant."
            )
        elif target_margin > -0.10:
            state = "adjust"
            headline = "Small adjustment"
            message = (
                f"The model is close, but it still leans toward {predicted_name}. "
                "Nudge the strategy until your intended task becomes the highest bar."
            )
        elif self._strategy_qualities and self._strategy_qualities[-1] < 0.40:
            state = "recover"
            headline = "Recover the strategy"
            message = (
                f"The model is reading {predicted_name}, and the strategy looks unstable. "
                "Re-center with the arrows, then try a cleaner version of the intended task."
            )
        else:
            state = "recover"
            headline = "Change strategy"
            message = (
                f"The model thinks you are doing {predicted_name}, not {target_name}. "
                "If you do not like that readout, keep changing strategy until the prediction matches your intention."
            )

        if self.training_phase == "calibration" and calibration["ready"]:
            state = "ready"
            headline = "Calibration is ready"
            message = (
                "The four tasks are forming distinct zones. You can switch to Neurofeedback Coach mode and focus on real-time correction."
            )

        return {
            "state": state,
            "headline": headline,
            "message": message,
            "score": score,
            "score_label": "Decoder Match",
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

    def _ensure_decoder_ready(self, input_dim: int) -> None:
        input_dim = int(input_dim)
        if self.n_dims == input_dim and self.trainer is not None:
            return

        self.n_dims = input_dim
        self.model_cfg.input_dim = input_dim
        self._initialize_decoder()
        self._reset_history()

    def _process_sample_metadata(self, sample: StreamSample) -> None:
        self.current_class = sample.label
        self.last_source_sample_idx = int(sample.sample_idx)
        self.last_sample_wall_time = time.time()
        self.last_stream_timestamp = float(sample.timestamp)
        if sample.difficulty:
            self.difficulty = str(sample.difficulty)

    def _append_history(self, sample: StreamSample, result: InferenceStep) -> None:
        self._neural_points.append(np.asarray(result.projection, dtype=float).copy())
        self._penultimate.append(np.asarray(result.penultimate, dtype=float).copy())
        self._labels.append(sample.label)
        self._preds.append(int(result.predicted_class))
        self._confs.append(float(result.confidence))
        self._rewards.append(float(result.reward))
        self._class_scales.append(float(sample.class_scale or 0.0))
        self._strategy_qualities.append(float(sample.strategy_quality or 0.0))

        target_margin = self._target_margin(result.probabilities, sample.label)
        self._target_margins.append(float(target_margin))
        agreement = 1.0 if sample.label is not None and int(sample.label) == int(result.predicted_class) else 0.0
        self._agreements.append(agreement)

        if result.training and result.training.update_applied:
            self._accuracies.append(float(result.training.balanced_accuracy))
        else:
            last_acc = self._accuracies[-1] if self._accuracies else agreement
            self._accuracies.append(last_acc)

        self._refresh_projected_history()

    def _build_live_payload(self, sample: StreamSample, result: InferenceStep) -> dict[str, Any]:
        centroids = self._compute_centroids()
        min_sep, mean_sep = self._compute_separation(centroids)
        mean_spread = self._compute_spread(centroids)
        session = self.session_snapshot()
        calibration = self.calibration_snapshot()
        coach = self._build_coach_snapshot(
            current_label=sample.label,
            predicted_class=int(result.predicted_class),
            probabilities=result.probabilities,
            calibration=calibration,
            session=session,
        )
        zone = self._zone_snapshot(centroids)

        return {
            "sample_idx": self.sample_count,
            "source_sample_idx": self.last_source_sample_idx,
            "timestamp": sample.timestamp,
            "stream": self.stream_snapshot(),
            "current_class": sample.label,
            "current_class_name": self._class_name(sample.label),
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
            "confidences": list(self._confs),
            "rewards": list(self._rewards),
            "agreements": list(self._agreements),
            "accuracies": list(self._accuracies),
            "centroids": {str(k): v.tolist() for k, v in centroids.items()},
            "centroid_window": self.centroid_window,
            "min_separation": min_sep,
            "mean_separation": mean_sep,
            "mean_spread": mean_spread,
            "session": session,
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
        }

    def _build_waiting_payload(self) -> dict[str, Any]:
        return {
            "sample_idx": self.sample_count,
            "source_sample_idx": self.last_source_sample_idx,
            "timestamp": self.last_stream_timestamp,
            "stream": self.stream_snapshot(),
            "current_class": self.current_class,
            "current_class_name": self._class_name(self.current_class),
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
            "confidences": [],
            "rewards": [],
            "agreements": [],
            "accuracies": [],
            "centroids": {},
            "centroid_window": self.centroid_window,
            "min_separation": 0.0,
            "mean_separation": 0.0,
            "mean_spread": 0.0,
            "session": self.session_snapshot(),
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
        payload["session"] = self.session_snapshot()
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
        if not self._points:
            return {}

        window = min(self.centroid_window, len(self._points))
        points_arr = np.asarray(list(self._points)[-window:], dtype=float)
        labels_arr = self._plot_labels_array()[-window:]

        centroids: dict[int, np.ndarray] = {}
        for cls in range(4):
            mask = labels_arr == cls
            if int(np.sum(mask)) >= 3:
                centroids[cls] = np.mean(points_arr[mask], axis=0)
        return centroids

    def _compute_true_label_centroids(self) -> dict[int, np.ndarray]:
        if not self._points:
            return {}

        window = min(self.centroid_window, len(self._points))
        points_arr = np.asarray(list(self._points)[-window:], dtype=float)
        labels = list(self._labels)[-window:]

        centroids: dict[int, np.ndarray] = {}
        for cls in range(4):
            indices = [idx for idx, label in enumerate(labels) if label == cls]
            if len(indices) < 3:
                continue
            centroids[cls] = np.mean(points_arr[indices], axis=0)
        return centroids

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
        if not self._points or not centroids:
            return 0.0

        window = min(self.centroid_window, len(self._points))
        points_arr = np.asarray(list(self._points)[-window:], dtype=float)
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
        if not self._neural_points:
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
