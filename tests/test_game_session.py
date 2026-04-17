import numpy as np

from game.config import LevelPolicy, RhythmGameConfig
from game.session import GameSession
from model.types import StreamSample


def _sample(idx: int, ts: float) -> StreamSample:
    return StreamSample(
        sample_idx=idx,
        timestamp=ts,
        embedding=np.zeros(8, dtype=np.float32),
        label=None,
        label_name=None,
        class_scale=0.7,
        strategy_quality=0.7,
        difficulty="d1",
        raw={},
    )


def _uniform_probs() -> np.ndarray:
    return np.full(4, 0.25, dtype=float)


def _target_probs(target: int, p_target: float = 0.92) -> np.ndarray:
    probs = np.full(4, (1.0 - p_target) / 3.0, dtype=float)
    probs[target] = p_target
    return probs


def test_game_session_advances_prompt_ids_and_resets_streak_after_miss() -> None:
    cfg = RhythmGameConfig(
        prompt_duration_s=1.2,
        base_hit_window_s=0.35,
        base_beat_interval_s=1.4,
        enable_adaptation=False,
        levels=(
            LevelPolicy(
                hit_window_s=0.35,
                beat_interval_s=1.4,
                min_confidence=0.45,
                min_margin=0.02,
            ),
        ),
        start_level=0,
    )
    session = GameSession(cfg)
    base_ts = 500.0

    session.step(_sample(0, base_ts), _uniform_probs())
    first_prompt = session.current_prompt
    assert first_prompt is not None

    hit_feedback = session.step(
        _sample(1, base_ts + first_prompt.center_s),
        _target_probs(first_prompt.target_class),
    )
    assert hit_feedback.hit is True
    assert session.streak == 1

    session.step(_sample(2, base_ts + first_prompt.end_s + 0.05), _uniform_probs())
    second_prompt = session.current_prompt
    assert second_prompt is not None
    assert second_prompt.prompt_id == first_prompt.prompt_id + 1

    session.step(_sample(3, base_ts + second_prompt.center_s), _uniform_probs())
    session.step(_sample(4, base_ts + second_prompt.end_s + 0.05), _uniform_probs())

    assert session.total_prompts >= 2
    assert session.streak == 0


def test_adaptive_difficulty_demotes_level_on_poor_performance() -> None:
    cfg = RhythmGameConfig(
        prompt_duration_s=1.2,
        base_hit_window_s=0.35,
        base_beat_interval_s=1.4,
        enable_adaptation=True,
        adaptation_eval_prompts=3,
        start_level=1,
        levels=(
            LevelPolicy(hit_window_s=0.42, beat_interval_s=1.5, min_confidence=0.40, min_margin=0.00),
            LevelPolicy(hit_window_s=0.35, beat_interval_s=1.4, min_confidence=0.45, min_margin=0.02),
            LevelPolicy(hit_window_s=0.28, beat_interval_s=1.2, min_confidence=0.50, min_margin=0.05),
        ),
    )
    session = GameSession(cfg)
    base_ts = 800.0

    session.step(_sample(0, base_ts), _uniform_probs())

    sample_idx = 1
    for _ in range(3):
        prompt = session.current_prompt
        assert prompt is not None
        session.step(_sample(sample_idx, base_ts + prompt.center_s), _uniform_probs())
        sample_idx += 1
        session.step(_sample(sample_idx, base_ts + prompt.end_s + 0.05), _uniform_probs())
        sample_idx += 1

    assert session.current_level == 0
