"""Tests for the webapp module."""

from pathlib import Path

import pytest
import numpy as np


def test_emulator_bridge_initialization():
    """Test that the emulator bridge initializes correctly."""
    from webapp.emulator_bridge import EmulatorBridge

    bridge = EmulatorBridge(difficulty="d1", n_dims=256)

    assert bridge.difficulty == "d1"
    assert bridge.n_dims == 256
    assert bridge.current_class is None
    assert bridge.auto_tracking is False
    assert bridge.sample_count == 0
    assert bridge.model_type == "dnn"
    assert bridge.viz_method == "neural"
    assert bridge.control_mode == "buttons"
    assert bridge.training_phase == "calibration"
    assert bridge.pressed_arrows() == []


def test_emulator_bridge_step_returns_valid_data():
    """Test that step() returns all expected fields."""
    from webapp.emulator_bridge import EmulatorBridge

    bridge = EmulatorBridge()
    result = bridge.step()

    # Check required fields exist
    assert "sample_idx" in result
    assert "predicted_class" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "projection" in result
    assert "centroids" in result
    assert "training" in result

    # Check value ranges
    assert 0 <= result["predicted_class"] <= 3
    assert 0 <= result["confidence"] <= 1
    assert len(result["probabilities"]) == 4
    assert abs(sum(result["probabilities"]) - 1.0) < 0.01
    assert len(result["projection"]) == 2
    assert result["viz_method"] == "neural"
    assert result["training_phase"] == "calibration"
    assert result["prompt"]["guided"] is True
    assert 0 <= result["prompt"]["target_class"] <= 3


def test_emulator_bridge_set_class():
    """Test manual class selection."""
    from webapp.emulator_bridge import EmulatorBridge

    bridge = EmulatorBridge()

    for cls in [0, 1, 2, 3, None]:
        bridge.set_class(cls)
        assert bridge.training_phase == "manual"
        assert bridge.current_class == cls
        result = bridge.step()
        assert result["current_class"] == cls
        assert result["prompt"]["guided"] is False


def test_emulator_bridge_keyboard_mode_applies_held_arrow_inputs():
    """Test that keyboard emulator mode feeds strategy updates into the bridge."""
    from webapp.emulator_bridge import EmulatorBridge

    bridge = EmulatorBridge()
    bridge.set_class(0)
    bridge.set_control_mode("keyboard")

    before = bridge.dynamics.z_strategy.copy()
    bridge.set_arrow_pressed("ArrowRight", True)
    result = bridge.step()

    assert result["control_mode"] == "keyboard"
    assert result["pressed_arrows"] == ["ArrowRight"]
    assert bridge.dynamics.z_strategy[0] > before[0]


def test_emulator_bridge_switching_control_modes_clears_held_keys():
    """Test that leaving keyboard mode resets sticky arrow state."""
    from webapp.emulator_bridge import EmulatorBridge

    bridge = EmulatorBridge()
    bridge.auto_tracking = True
    bridge.set_control_mode("keyboard")
    bridge.set_arrow_pressed("ArrowLeft", True)
    assert bridge.auto_tracking is False
    assert bridge.pressed_arrows() == ["ArrowLeft"]

    bridge.set_control_mode("buttons")

    assert bridge.control_mode == "buttons"
    assert bridge.pressed_arrows() == []


def test_emulator_bridge_auto_tracking():
    """Test auto-tracking mode cycles through classes."""
    from webapp.emulator_bridge import EmulatorBridge

    bridge = EmulatorBridge()
    bridge.auto_tracking = True

    # Run for enough steps to cycle through classes
    classes_seen = set()
    for _ in range(100):
        result = bridge.step()
        classes_seen.add(result["current_class"])

    # Should have seen multiple classes
    assert len(classes_seen) >= 2


def test_emulator_bridge_centroid_window():
    """Test centroid window affects computation."""
    from webapp.emulator_bridge import EmulatorBridge

    bridge = EmulatorBridge(centroid_window=20)
    bridge.auto_tracking = True

    # Generate enough samples
    for _ in range(50):
        bridge.step()

    # Change window and verify it's used
    bridge.centroid_window = 10
    result = bridge.step()
    assert result["centroid_window"] == 10


def test_emulator_bridge_learning_improves():
    """Test that model accuracy improves over time."""
    from webapp.emulator_bridge import EmulatorBridge

    bridge = EmulatorBridge()
    bridge.auto_tracking = True

    # Run for many steps
    for _ in range(300):
        result = bridge.step()

    # Check that training has occurred
    assert result["training"]["num_updates"] > 0
    assert result["training"]["labeled_seen"] > 0


def test_emulator_bridge_can_switch_visualization_modes():
    """Test that the bridge supports alternate visualization backends."""
    from webapp.emulator_bridge import EmulatorBridge

    bridge = EmulatorBridge(viz_method="pca")
    bridge.auto_tracking = True

    for _ in range(12):
        result = bridge.step()

    assert bridge.viz_method == "pca"
    assert result["viz_method"] == "pca"
    assert result["viz_name"] == "PCA Projection"
    assert len(result["projection"]) == 2
    assert len(result["probabilities"]) == 4
    assert len(result["points"]) == bridge.sample_count


def test_emulator_bridge_supports_deep_learning_decoder_variants():
    """Test that the bridge can swap between DNN, CNN, and CEBRA-style decoders."""
    from webapp.emulator_bridge import EmulatorBridge

    for model_type in ("dnn", "cnn", "cebra"):
        bridge = EmulatorBridge(model_type=model_type)
        bridge.auto_tracking = True
        result = bridge.step()

        assert result["model_type"] == model_type
        assert len(result["probabilities"]) == 4
        assert len(result["projection"]) == 2


def test_emulator_bridge_switching_model_resets_visual_history():
    """Test that changing models clears stale visualization history."""
    from webapp.emulator_bridge import EmulatorBridge

    bridge = EmulatorBridge()
    for _ in range(5):
        bridge.step()

    assert len(bridge._points) > 0
    bridge.set_model("cnn")

    assert len(bridge._points) == 0
    assert bridge.model_type == "cnn"


def test_emulator_bridge_switching_viz_method_reprojects_history():
    """Test that visualization method changes keep the history available."""
    from webapp.emulator_bridge import EmulatorBridge

    bridge = EmulatorBridge()
    bridge.auto_tracking = True
    for _ in range(12):
        bridge.step()

    before = len(bridge._points)
    bridge.set_viz_method("lda")

    assert bridge.viz_method == "lda"
    assert len(bridge._points) == before


def test_flask_app_creates_routes():
    """Test that Flask app has expected routes."""
    from webapp.app import app

    rules = [r.rule for r in app.url_map.iter_rules()]

    assert "/" in rules
    assert "/api/status" in rules
    assert "/api/set_class" in rules
    assert "/api/toggle_tracking" in rules
    assert "/api/set_training_phase" in rules
    assert "/api/set_difficulty" in rules
    assert "/api/set_control_mode" in rules
    assert "/api/control_key" in rules
    assert "/api/set_centroid_window" in rules
    assert "/api/set_model" in rules
    assert "/api/set_viz_method" in rules
    assert "/api/save_model" in rules


def test_set_control_mode_route_updates_status():
    """Test that the Flask route switches the active input mode."""
    import webapp.app as webapp_module

    webapp_module.bridge = None
    app = webapp_module.app

    client = app.test_client()

    response = client.post("/api/set_control_mode", json={"control_mode": "keyboard"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["control_mode"] == "keyboard"
    assert payload["auto_tracking"] is False

    status = client.get("/api/status")
    assert status.status_code == 200
    status_payload = status.get_json()
    assert status_payload["control_mode"] == "keyboard"
    assert "keyboard" in status_payload["available_control_modes"]


def test_set_training_phase_route_updates_status():
    """Test that the Flask route switches the active training phase."""
    import webapp.app as webapp_module

    webapp_module.bridge = None
    app = webapp_module.app

    client = app.test_client()

    response = client.post("/api/set_training_phase", json={"training_phase": "feedback"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["training_phase"] == "feedback"

    status = client.get("/api/status")
    assert status.status_code == 200
    status_payload = status.get_json()
    assert status_payload["training_phase"] == "feedback"
    assert "calibration" in status_payload["available_training_phases"]
    assert "manual" in status_payload["available_training_phases"]


def test_set_difficulty_route_updates_status():
    """Test that the Flask route switches the active emulator difficulty."""
    import webapp.app as webapp_module

    webapp_module.bridge = None
    app = webapp_module.app

    client = app.test_client()

    response = client.post("/api/set_difficulty", json={"difficulty": "d3"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["difficulty"] == "d3"
    assert payload["sample_count"] == 0

    status = client.get("/api/status")
    assert status.status_code == 200
    status_payload = status.get_json()
    assert status_payload["difficulty"] == "d3"
    assert "d5" in status_payload["available_difficulties"]


def test_set_model_route_updates_status():
    """Test that the Flask route switches the active model."""
    import webapp.app as webapp_module

    webapp_module.bridge = None
    app = webapp_module.app

    client = app.test_client()

    response = client.post("/api/set_model", json={"model_type": "cebra"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["model_type"] == "cebra"

    status = client.get("/api/status")
    assert status.status_code == 200
    status_payload = status.get_json()
    assert status_payload["model_type"] == "cebra"
    assert "cnn" in status_payload["available_models"]
    assert "cebra" in status_payload["available_models"]
    assert "pca" in status_payload["available_viz_methods"]


def test_set_viz_method_route_updates_status():
    """Test that the Flask route switches the active visualization method."""
    import webapp.app as webapp_module

    webapp_module.bridge = None
    app = webapp_module.app

    client = app.test_client()

    response = client.post("/api/set_viz_method", json={"viz_method": "lda"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["viz_method"] == "lda"

    status = client.get("/api/status")
    assert status.status_code == 200
    status_payload = status.get_json()
    assert status_payload["viz_method"] == "lda"
    assert "neural" in status_payload["available_viz_methods"]
    assert "tsne" in status_payload["available_viz_methods"]


def _checkpoint_dir() -> Path:
    path = Path(__file__).resolve().parents[1] / "test_artifacts" / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_emulator_bridge_can_save_model_snapshot(monkeypatch):
    """Test that the bridge saves checkpoints with metadata."""
    import webapp.emulator_bridge as bridge_module
    from webapp.emulator_bridge import EmulatorBridge

    checkpoint_dir = _checkpoint_dir()
    monkeypatch.setattr(bridge_module, "CHECKPOINT_DIR", checkpoint_dir)

    bridge = EmulatorBridge()
    bridge.auto_tracking = True
    for _ in range(8):
        bridge.step()

    path = bridge.save_model_snapshot("custom_test_snapshot")

    assert path.exists()
    assert path.name == "custom_test_snapshot.pt"
    assert path.parent == checkpoint_dir


def test_save_model_route_writes_checkpoint(monkeypatch):
    """Test that the Flask save route writes a checkpoint file."""
    import webapp.app as webapp_module
    import webapp.emulator_bridge as bridge_module

    checkpoint_dir = _checkpoint_dir()
    monkeypatch.setattr(bridge_module, "CHECKPOINT_DIR", checkpoint_dir)
    webapp_module.bridge = None
    app = webapp_module.app

    client = app.test_client()

    response = client.post("/api/save_model", json={"name": "named_snapshot"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["filename"] == "named_snapshot.pt"

    saved_path = Path(payload["path"])
    assert saved_path.exists()
    assert saved_path.parent == checkpoint_dir
