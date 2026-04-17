"""Tests for the stream-driven webapp bridge and HTTP API."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from model.types import StreamSample


class FakeReceiver:
    def __init__(self, samples: list[StreamSample] | None = None) -> None:
        self.samples = deque(samples or [])
        self.started = False
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.started = True
        self.start_calls += 1

    def stop(self) -> None:
        self.started = False
        self.stop_calls += 1

    def get(self, timeout: float = 0.2) -> StreamSample | None:  # noqa: ARG002 - protocol compatibility
        if not self.samples:
            return None
        return self.samples.popleft()


def make_sample(sample_idx: int, label: int | None, dim: int = 16) -> StreamSample:
    embedding = np.zeros(dim, dtype=np.float32)
    if label is not None:
        start = (label % 4) * (dim // 4)
        stop = start + (dim // 4)
        embedding[start:stop] = 1.0
    embedding += np.linspace(0.0, 0.02, dim, dtype=np.float32)

    return StreamSample(
        sample_idx=sample_idx,
        timestamp=float(sample_idx) * 0.1,
        embedding=embedding,
        label=label,
        label_name=None if label is None else f"class_{label}",
        class_scale=0.85 if label is not None else 0.1,
        strategy_quality=0.80 if label is not None else 0.1,
        difficulty="d2",
        raw={},
    )


def make_training_samples(n_per_class: int = 16, dim: int = 16) -> list[StreamSample]:
    samples: list[StreamSample] = []
    idx = 0
    for _ in range(n_per_class):
        for label in range(4):
            samples.append(make_sample(idx, label, dim=dim))
            idx += 1
    return samples


def _checkpoint_dir() -> Path:
    path = Path(__file__).resolve().parents[1] / "test_artifacts" / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_emulator_bridge_waits_for_stream():
    from webapp.emulator_bridge import EmulatorBridge

    receiver = FakeReceiver()
    bridge = EmulatorBridge(receiver=receiver)

    result = bridge.step(timeout=0.0)

    assert receiver.started is True
    assert result["stream"]["state"] == "waiting"
    assert result["sample_idx"] == 0
    assert result["predicted_class"] is None
    assert bridge.n_dims is None


def test_emulator_bridge_processes_stream_sample():
    from webapp.emulator_bridge import EmulatorBridge

    receiver = FakeReceiver([make_sample(0, 2)])
    bridge = EmulatorBridge(receiver=receiver)

    result = bridge.step(timeout=0.0)

    assert result["sample_idx"] == 1
    assert result["current_class"] == 2
    assert result["current_class_name"] == "Left Leg"
    assert result["predicted_class"] in (0, 1, 2, 3)
    assert len(result["probabilities"]) == 4
    assert abs(sum(result["probabilities"]) - 1.0) < 1e-5
    assert result["difficulty"] == "d2"
    assert result["stream"]["state"] == "live"
    assert result["zone"]["class_name"]
    assert bridge.n_dims == 16


def test_emulator_bridge_calibration_counts_and_training_progress():
    from webapp.emulator_bridge import EmulatorBridge

    receiver = FakeReceiver(make_training_samples(n_per_class=18))
    bridge = EmulatorBridge(receiver=receiver, viz_method="pca")

    last = None
    for _ in range(72):
        last = bridge.step(timeout=0.0)

    assert last is not None
    assert bridge.trainer is not None
    assert bridge.trainer.labeled_seen >= 72
    assert last["training"]["num_updates"] > 0
    assert last["calibration"]["classes_ready"] == 4
    assert last["calibration"]["label_counts"]["0"] >= 12
    assert last["viz_method"] == "pca"
    assert len(last["points"]) == bridge.sample_count


def test_emulator_bridge_can_switch_model_and_viz_before_samples():
    from webapp.emulator_bridge import EmulatorBridge

    bridge = EmulatorBridge(receiver=FakeReceiver())
    bridge.set_model("cnn")
    bridge.set_viz_method("lda")

    assert bridge.model_type == "cnn"
    assert bridge.viz_method == "lda"
    assert bridge.n_dims is None


def test_emulator_bridge_switching_model_resets_visual_history():
    from webapp.emulator_bridge import EmulatorBridge

    receiver = FakeReceiver([make_sample(0, 0), make_sample(1, 1)])
    bridge = EmulatorBridge(receiver=receiver)

    bridge.step(timeout=0.0)
    assert len(bridge._points) == 1

    bridge.set_model("cebra")

    assert bridge.model_type == "cebra"
    assert len(bridge._points) == 0


def test_emulator_bridge_can_save_model_snapshot(monkeypatch):
    import webapp.emulator_bridge as bridge_module
    from webapp.emulator_bridge import EmulatorBridge

    monkeypatch.setattr(bridge_module, "CHECKPOINT_DIR", _checkpoint_dir())
    receiver = FakeReceiver([make_sample(0, 1)])
    bridge = EmulatorBridge(receiver=receiver)
    bridge.step(timeout=0.0)

    path = bridge.save_model_snapshot("custom_test_snapshot")

    assert path.exists()
    assert path.name == "custom_test_snapshot.pt"
    assert path.parent == _checkpoint_dir()


def test_flask_app_creates_expected_routes():
    from webapp.app import app

    rules = {rule.rule for rule in app.url_map.iter_rules()}

    assert "/" in rules
    assert "/api/status" in rules
    assert "/api/set_training_phase" in rules
    assert "/api/set_centroid_window" in rules
    assert "/api/set_model" in rules
    assert "/api/set_viz_method" in rules
    assert "/api/save_model" in rules
    assert "/api/reset" in rules


def test_status_route_reports_stream_configuration():
    import webapp.app as webapp_module
    from webapp.emulator_bridge import EmulatorBridge

    webapp_module.bridge = EmulatorBridge(receiver=FakeReceiver())
    client = webapp_module.app.test_client()

    response = client.get("/api/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["stream"]["host"] == "localhost"
    assert payload["stream"]["port"] == 5555
    assert payload["training_phase"] == "calibration"
    assert payload["stream"]["state"] == "idle"


def test_set_training_phase_route_updates_status():
    import webapp.app as webapp_module
    from webapp.emulator_bridge import EmulatorBridge

    webapp_module.bridge = EmulatorBridge(receiver=FakeReceiver())
    client = webapp_module.app.test_client()

    response = client.post("/api/set_training_phase", json={"training_phase": "feedback"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["training_phase"] == "feedback"
    assert payload["training_phase_name"] == "Neurofeedback Coach"


def test_set_model_and_viz_routes_update_status():
    import webapp.app as webapp_module
    from webapp.emulator_bridge import EmulatorBridge

    webapp_module.bridge = EmulatorBridge(receiver=FakeReceiver())
    client = webapp_module.app.test_client()

    response_model = client.post("/api/set_model", json={"model_type": "cebra"})
    response_viz = client.post("/api/set_viz_method", json={"viz_method": "lda"})
    status = client.get("/api/status")

    assert response_model.status_code == 200
    assert response_viz.status_code == 200
    assert status.status_code == 200
    payload = status.get_json()
    assert payload["model_type"] == "cebra"
    assert payload["viz_method"] == "lda"
    assert "cnn" in payload["available_models"]
    assert "tsne" in payload["available_viz_methods"]


def test_save_model_route_writes_checkpoint(monkeypatch):
    import webapp.app as webapp_module
    import webapp.emulator_bridge as bridge_module
    from webapp.emulator_bridge import EmulatorBridge

    monkeypatch.setattr(bridge_module, "CHECKPOINT_DIR", _checkpoint_dir())
    webapp_module.bridge = EmulatorBridge(receiver=FakeReceiver([make_sample(0, 3)]))
    webapp_module.bridge.step(timeout=0.0)
    client = webapp_module.app.test_client()

    response = client.post("/api/save_model", json={"name": "named_snapshot"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["filename"] == "named_snapshot.pt"
    assert Path(payload["path"]).exists()


def test_reset_route_recreates_bridge():
    import webapp.app as webapp_module
    from webapp.emulator_bridge import EmulatorBridge

    original_bridge = EmulatorBridge(receiver=FakeReceiver(), model_type="cnn", viz_method="pca", training_phase="feedback")
    webapp_module.bridge = original_bridge
    client = webapp_module.app.test_client()

    response = client.post("/api/reset")

    assert response.status_code == 200
    assert response.get_json()["success"] is True
    assert webapp_module.bridge is not original_bridge
    assert webapp_module.bridge.model_type == "cnn"
    assert webapp_module.bridge.viz_method == "pca"
    assert webapp_module.bridge.training_phase == "feedback"
