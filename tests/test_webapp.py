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


def make_training_samples(
    n_per_class: int = 16,
    dim: int = 16,
    contiguous: bool = False,
) -> list[StreamSample]:
    samples: list[StreamSample] = []
    idx = 0
    if contiguous:
        for label in range(4):
            for _ in range(n_per_class):
                samples.append(make_sample(idx, label, dim=dim))
                idx += 1
        return samples

    for _ in range(n_per_class):
        for label in range(4):
            samples.append(make_sample(idx, label, dim=dim))
            idx += 1
    return samples


def make_skewed_training_samples(dim: int = 16) -> list[StreamSample]:
    samples: list[StreamSample] = []
    idx = 0

    for label in range(4):
        for _ in range(60):
            samples.append(make_sample(idx, label, dim=dim))
            idx += 1

    for _ in range(96):
        samples.append(make_sample(idx, 0, dim=dim))
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

    receiver = FakeReceiver(make_training_samples(n_per_class=60, contiguous=True))
    bridge = EmulatorBridge(
        receiver=receiver,
        viz_method="pca",
        calibration_samples_per_class=12,
        transition_ignore_samples=40,
    )

    last = None
    for _ in range(240):
        last = bridge.step(timeout=0.0)

    assert last is not None
    assert bridge.trainer is not None
    assert bridge.trainer.labeled_seen >= 80
    assert last["training"]["num_updates"] > 0
    assert last["calibration"]["classes_ready"] == 4
    assert last["calibration"]["label_counts"]["0"] >= 12
    assert last["calibration"]["target_per_class"] == 12
    assert last["viz_method"] == "pca"
    assert len(last["points"]) == bridge.sample_count


def test_emulator_bridge_keeps_other_classes_visible_during_skewed_calibration():
    from webapp.emulator_bridge import EmulatorBridge

    receiver = FakeReceiver(make_skewed_training_samples())
    bridge = EmulatorBridge(
        receiver=receiver,
        calibration_samples_per_class=12,
        transition_ignore_samples=40,
        centroid_window=40,
        centroid_min_samples_per_class=12,
        display_window=80,
        display_min_samples_per_class=12,
    )

    last = None
    for _ in range(336):
        last = bridge.step(timeout=0.0)

    assert last is not None
    display_labels = last["cluster_labels"]

    for cls in range(4):
        assert display_labels.count(cls) >= 12

    assert len(last["centroids"]) == 4


def test_emulator_bridge_ignores_first_40_samples_after_label_change():
    from webapp.emulator_bridge import EmulatorBridge

    receiver = FakeReceiver(make_training_samples(n_per_class=55, contiguous=True))
    bridge = EmulatorBridge(
        receiver=receiver,
        calibration_samples_per_class=20,
        transition_ignore_samples=40,
    )

    first_label_counts = []
    for _ in range(45):
        last = bridge.step(timeout=0.0)
        first_label_counts.append(last["calibration"]["label_counts"]["0"])

    assert first_label_counts[38] == 0
    assert first_label_counts[39] == 0
    assert first_label_counts[40] == 1
    assert last["transition_ignored"] is False


def test_emulator_bridge_freezes_model_and_graph_outside_calibration():
    """Feedback mode continues training (co-adaptation); only exploration freezes."""
    from webapp.emulator_bridge import EmulatorBridge

    calibration_samples = make_training_samples(n_per_class=55, contiguous=True)
    feedback_samples = [make_sample(220 + i, 0) for i in range(50)]
    exploration_samples = [make_sample(270 + i, 0) for i in range(50)]
    receiver = FakeReceiver(calibration_samples + feedback_samples + exploration_samples)
    bridge = EmulatorBridge(
        receiver=receiver,
        calibration_samples_per_class=12,
        transition_ignore_samples=40,
    )

    last = None
    for _ in range(220):
        last = bridge.step(timeout=0.0)

    assert last is not None
    updates_before = last["training"]["num_updates"]

    # Feedback mode: model keeps training (co-adaptation)
    bridge.set_training_phase("feedback")
    for _ in range(50):
        last = bridge.step(timeout=0.0)

    assert last is not None
    assert last["training"]["num_updates"] > updates_before  # training continues

    # Exploration mode: model freezes
    updates_after_feedback = last["training"]["num_updates"]
    bridge.set_training_phase("exploration")
    for _ in range(50):
        last = bridge.step(timeout=0.0)

    assert last is not None
    assert last["graph_frozen"] is True
    assert last["training"]["num_updates"] == updates_after_feedback  # frozen


def test_emulator_bridge_switches_reward_provider_with_game_phases():
    from game.rewards import GameRewardProvider
    from model.reward import ProgrammaticReward
    from webapp.emulator_bridge import EmulatorBridge

    bridge = EmulatorBridge(receiver=FakeReceiver())

    assert isinstance(bridge.reward_provider, ProgrammaticReward)

    bridge.set_training_phase("feedback")
    assert isinstance(bridge.reward_provider, GameRewardProvider)

    bridge.set_training_phase("exploration")
    assert isinstance(bridge.reward_provider, ProgrammaticReward)

    bridge.set_training_phase("calibration")
    assert isinstance(bridge.reward_provider, ProgrammaticReward)


def test_feedback_phase_uses_game_prompt_for_ui_target():
    from webapp.emulator_bridge import EmulatorBridge

    receiver = FakeReceiver(make_training_samples(n_per_class=12))
    bridge = EmulatorBridge(receiver=receiver, transition_ignore_samples=0)
    bridge.set_training_phase("feedback")

    last = None
    for _ in range(8):
        last = bridge.step(timeout=0.0)

    assert last is not None
    assert last["game"] is not None
    assert last["intended_class"] == last["game"]["target_class"]
    assert last["intended_class_name"] == last["game"]["target_class_name"]
    assert last["coach"]["score_label"] == "Prompt Match"
    assert "game" in last["session"]


def test_exploration_phase_uses_selected_class_not_game_prompt():
    from webapp.emulator_bridge import EmulatorBridge

    receiver = FakeReceiver(make_training_samples(n_per_class=12))
    bridge = EmulatorBridge(receiver=receiver, training_phase="exploration", transition_ignore_samples=0)
    bridge.set_exploration_class(3)

    last = None
    for _ in range(12):
        last = bridge.step(timeout=0.0)

    assert last is not None
    assert last["game"] is None
    assert last["intended_class"] == 3
    assert last["intended_class_name"] == "Right Leg"
    assert last["coach"]["score_label"] == "Frozen Readout"


def test_emulator_bridge_exploration_scores_points_with_frozen_model():
    from webapp.emulator_bridge import EmulatorBridge

    exploration_samples = []
    for i in range(65):
        sample = make_sample(i, 0)
        sample.embedding = sample.embedding + np.roll(
            np.linspace(0.0, 0.03, sample.embedding.shape[0], dtype=np.float32),
            i % sample.embedding.shape[0],
        )
        exploration_samples.append(sample)
    receiver = FakeReceiver(exploration_samples)
    bridge = EmulatorBridge(
        receiver=receiver,
        training_phase="exploration",
        transition_ignore_samples=40,
    )
    bridge.set_exploration_class(0)

    last = None
    for _ in range(65):
        last = bridge.step(timeout=0.0)

    assert last is not None
    exploration = last["exploration"]
    assert exploration is not None
    analysis = exploration["analysis"]
    assert analysis is not None
    assert len(analysis["point_target_probabilities"]) == len(analysis["points_2d"])
    assert len(analysis["point_predicted_classes"]) == len(analysis["points_2d"])
    assert 0.0 <= analysis["mean_target_probability"] <= 1.0
    assert analysis["clusters"]
    assert "confidence_max" in analysis["clusters"][0]


def test_emulator_bridge_exploration_keeps_rolling_3000_points():
    from webapp.emulator_bridge import EmulatorBridge

    receiver = FakeReceiver([make_sample(i, 0) for i in range(3050)])
    bridge = EmulatorBridge(
        receiver=receiver,
        training_phase="exploration",
        transition_ignore_samples=0,
    )
    bridge.set_exploration_class(0)

    for _ in range(3050):
        bridge.step(timeout=0.0)

    assert len(bridge._exploration_penultimate) == 3000


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


def test_index_route_contains_core_dashboard_anchors():
    import webapp.app as webapp_module
    from webapp.emulator_bridge import EmulatorBridge

    webapp_module.bridge = EmulatorBridge(receiver=FakeReceiver())
    client = webapp_module.app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Neurofeedback Decoder Lab" in html
    assert "â" not in html
    assert 'id="btn-start"' in html
    assert 'id="btn-stop"' in html
    assert 'id="guidance-card"' in html
    assert 'id="overview-phase"' in html
    assert 'id="overview-model"' in html
    assert 'id="manifold-plot"' in html
    assert 'id="exploration-plot"' in html
    assert 'id="save-model-status"' in html


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
    assert payload["training_phase_name"] == "Practice"
    assert "game" in payload


def test_set_training_phase_route_rejects_invalid_value():
    import webapp.app as webapp_module
    from webapp.emulator_bridge import EmulatorBridge

    webapp_module.bridge = EmulatorBridge(receiver=FakeReceiver())
    client = webapp_module.app.test_client()

    response = client.post("/api/set_training_phase", json={"training_phase": "bogus"})

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["success"] is False
    assert "Unknown training phase" in payload["error"]


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


def test_set_model_and_viz_routes_reject_invalid_values():
    import webapp.app as webapp_module
    from webapp.emulator_bridge import EmulatorBridge

    webapp_module.bridge = EmulatorBridge(receiver=FakeReceiver())
    client = webapp_module.app.test_client()

    response_model = client.post("/api/set_model", json={"model_type": "bogus"})
    response_viz = client.post("/api/set_viz_method", json={"viz_method": "bogus"})

    assert response_model.status_code == 400
    assert response_model.get_json()["success"] is False
    assert "Unknown model type" in response_model.get_json()["error"]

    assert response_viz.status_code == 400
    assert response_viz.get_json()["success"] is False
    assert "Unknown visualization method" in response_viz.get_json()["error"]


def test_set_centroid_window_route_clamps_range():
    import webapp.app as webapp_module
    from webapp.emulator_bridge import EmulatorBridge

    webapp_module.bridge = EmulatorBridge(receiver=FakeReceiver())
    client = webapp_module.app.test_client()

    low = client.post("/api/set_centroid_window", json={"window": 1})
    high = client.post("/api/set_centroid_window", json={"window": 999})

    assert low.status_code == 200
    assert low.get_json()["window"] == 10

    assert high.status_code == 200
    assert high.get_json()["window"] == 300


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
