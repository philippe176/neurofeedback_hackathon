import pytest

from game.config import LevelPolicy, RhythmGameConfig


def test_level_policy_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="hit_window_s"):
        LevelPolicy(
            hit_window_s=0.0,
            beat_interval_s=1.0,
            min_confidence=0.5,
            min_margin=0.0,
        )

    with pytest.raises(ValueError, match="min_confidence"):
        LevelPolicy(
            hit_window_s=0.5,
            beat_interval_s=1.0,
            min_confidence=1.5,
            min_margin=0.0,
        )


def test_rhythm_game_config_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="confusion_confidence_threshold"):
        RhythmGameConfig(confusion_confidence_threshold=1.2)

    with pytest.raises(ValueError, match="margin_ema_alpha"):
        RhythmGameConfig(margin_ema_alpha=-0.1)
