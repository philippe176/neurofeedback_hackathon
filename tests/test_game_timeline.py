import numpy as np

from game.config import LevelPolicy, RhythmGameConfig
from game.timeline import PromptTimeline


def test_prompt_timeline_is_deterministic_and_valid() -> None:
    cfg = RhythmGameConfig(
        seed=123,
        levels=(
            LevelPolicy(
                hit_window_s=0.35,
                beat_interval_s=1.4,
                min_confidence=0.45,
                min_margin=0.0,
            ),
        ),
        start_level=0,
        enable_adaptation=False,
    )

    tl_a = PromptTimeline(cfg)
    tl_b = PromptTimeline(cfg)
    policy = cfg.levels[0]

    seq_a = [tl_a.make_prompt(i * policy.beat_interval_s, 0, policy).target_class for i in range(16)]
    seq_b = [tl_b.make_prompt(i * policy.beat_interval_s, 0, policy).target_class for i in range(16)]

    assert seq_a == seq_b
    assert all(0 <= target < cfg.n_classes for target in seq_a)
    assert all(seq_a[i] != seq_a[i - 1] for i in range(1, len(seq_a)))


def test_peek_next_target_is_non_destructive() -> None:
    cfg = RhythmGameConfig(
        seed=77,
        levels=(
            LevelPolicy(
                hit_window_s=0.5,
                beat_interval_s=2.2,
                min_confidence=0.3,
                min_margin=0.0,
            ),
        ),
        start_level=0,
        enable_adaptation=False,
    )
    tl = PromptTimeline(cfg)

    first = tl.next_target_class()
    peeked = tl.peek_next_target(last_target_override=first)
    second = tl.next_target_class()

    assert peeked == second
