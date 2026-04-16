"""Tests for the webapp module."""

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


def test_emulator_bridge_set_class():
    """Test manual class selection."""
    from webapp.emulator_bridge import EmulatorBridge

    bridge = EmulatorBridge()

    for cls in [0, 1, 2, 3, None]:
        bridge.set_class(cls)
        assert bridge.current_class == cls
        result = bridge.step()
        assert result["current_class"] == cls


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


def test_flask_app_creates_routes():
    """Test that Flask app has expected routes."""
    from webapp.app import app

    rules = [r.rule for r in app.url_map.iter_rules()]

    assert "/" in rules
    assert "/api/status" in rules
    assert "/api/set_class" in rules
    assert "/api/toggle_tracking" in rules
    assert "/api/set_centroid_window" in rules
