import numpy as np

from game.config import LevelPolicy, RhythmGameConfig
from game.rewards import GameRewardProvider
from model.config import ModelConfig
from model.types import StreamSample


def _sample(idx: int, ts: float) -> StreamSample:
    return StreamSample(
        sample_idx=idx,
        timestamp=ts,
        embedding=np.zeros(8, dtype=np.float32),
        label=None,
        label_name=None,
        class_scale=0.7,
        strategy_quality=0.8,
        difficulty="d1",
        raw={},
    )


def _prob_vector(target: int, p_target: float = 0.9) -> np.ndarray:
    probs = np.full(4, (1.0 - p_target) / 3.0, dtype=float)
    probs[target] = p_target
    return probs


def test_game_reward_prefers_correct_and_timed_prediction() -> None:
    model_cfg = ModelConfig(reward_min=0.0, reward_max=1.0)
    game_cfg = RhythmGameConfig(
        prompt_duration_s=1.2,
        base_hit_window_s=0.40,
        base_beat_interval_s=1.4,
        enable_adaptation=False,
        levels=(
            LevelPolicy(
                hit_window_s=0.40,
                beat_interval_s=1.4,
                min_confidence=0.45,
                min_margin=0.02,
            ),
        ),
        start_level=0,
    )

    provider = GameRewardProvider(model_cfg=model_cfg, game_cfg=game_cfg)

    provider.compute(_sample(0, 100.0), np.array([0.25, 0.25, 0.25, 0.25], dtype=float))
    assert provider.last_feedback is not None
    target = provider.last_feedback.target_class

    reward_good = provider.compute(_sample(1, 100.6), _prob_vector(target, p_target=0.92))

    wrong_target = (target + 1) % 4
    reward_bad = provider.compute(_sample(2, 100.62), _prob_vector(wrong_target, p_target=0.92))

    assert reward_good > reward_bad
    assert model_cfg.reward_min <= reward_good <= model_cfg.reward_max
    assert model_cfg.reward_min <= reward_bad <= model_cfg.reward_max


def test_game_feedback_includes_upcoming_prompt_metadata() -> None:
    model_cfg = ModelConfig(reward_min=0.0, reward_max=1.0)
    game_cfg = RhythmGameConfig(enable_adaptation=False)
    provider = GameRewardProvider(model_cfg=model_cfg, game_cfg=game_cfg)

    provider.compute(_sample(0, 120.0), np.array([0.25, 0.25, 0.25, 0.25], dtype=float))
    assert provider.last_feedback is not None

    feedback = provider.last_feedback
    assert 0 <= feedback.next_target_class < game_cfg.n_classes
    assert feedback.seconds_to_window_start >= 0.0
    assert feedback.seconds_to_next_prompt_start >= 0.0


def test_auto_perform_adjusts_probabilities_toward_target() -> None:
    model_cfg = ModelConfig(reward_min=0.0, reward_max=1.0)
    game_cfg = RhythmGameConfig(
        prompt_duration_s=1.2,
        base_hit_window_s=0.4,
        base_beat_interval_s=1.4,
        auto_perform=True,
        auto_perform_strength=0.95,
        auto_prewindow_strength=0.80,
        auto_blend=1.0,
        auto_anticipation_s=1.0,
        enable_adaptation=False,
    )

    provider = GameRewardProvider(model_cfg=model_cfg, game_cfg=game_cfg)
    uniform = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

    provider.compute(_sample(0, 200.0), uniform)
    assert provider.last_feedback is not None
    target = provider.last_feedback.target_class

    adjusted = provider.adjust_probabilities(_sample(1, 200.6), uniform)
    assert np.isclose(float(adjusted.sum()), 1.0)
    assert adjusted[target] > 0.9


def test_reward_provider_simulation_toggle_methods() -> None:
    model_cfg = ModelConfig(reward_min=0.0, reward_max=1.0)
    game_cfg = RhythmGameConfig(auto_perform=False, enable_adaptation=False)
    provider = GameRewardProvider(model_cfg=model_cfg, game_cfg=game_cfg)

    assert not provider.is_auto_perform_enabled()
    provider.set_auto_perform(True)
    assert provider.is_auto_perform_enabled()
    provider.set_auto_perform(False)
    assert not provider.is_auto_perform_enabled()


def test_game_reward_provider_persists_feedback_into_sample_raw() -> None:
    model_cfg = ModelConfig(reward_min=0.0, reward_max=1.0)
    game_cfg = RhythmGameConfig(enable_adaptation=False)
    provider = GameRewardProvider(model_cfg=model_cfg, game_cfg=game_cfg)
    sample = _sample(0, 300.0)

    reward = provider.compute(sample, np.array([0.25, 0.25, 0.25, 0.25], dtype=float))

    assert reward >= model_cfg.reward_min
    assert "game" in sample.raw
    assert "prompt_id" in sample.raw["game"]
    assert "component_margin" in sample.raw["game"]
