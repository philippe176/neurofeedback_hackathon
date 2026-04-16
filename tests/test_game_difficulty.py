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
        class_scale=0.8,
        strategy_quality=0.8,
        difficulty="d1",
        raw={},
    )


def _good_probs(target: int) -> np.ndarray:
    probs = np.full(4, 0.02, dtype=float)
    probs[target] = 0.94
    return probs / probs.sum()


def test_adaptive_difficulty_promotes_level_on_good_performance() -> None:
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

    base_ts = 200.0
    session.step(_sample(0, base_ts), np.array([0.25, 0.25, 0.25, 0.25], dtype=float))

    sample_idx = 1
    for _ in range(3):
        assert session.current_prompt is not None
        prompt = session.current_prompt

        hit_ts = base_ts + prompt.center_s
        target = prompt.target_class
        session.step(_sample(sample_idx, hit_ts), _good_probs(target))
        sample_idx += 1

        advance_ts = base_ts + prompt.end_s + 0.02
        session.step(_sample(sample_idx, advance_ts), np.array([0.25, 0.25, 0.25, 0.25], dtype=float))
        sample_idx += 1

    assert session.current_level == 2
