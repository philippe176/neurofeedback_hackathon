import numpy as np

from model.config import ModelConfig
from model.reward import ProgrammaticReward


def test_programmatic_reward_is_bounded(make_stream_sample) -> None:
    cfg = ModelConfig()
    reward_fn = ProgrammaticReward(cfg)

    sample = make_stream_sample(label=2, class_scale=1.2, strategy_quality=2.0)
    probs = np.array([0.01, 0.01, 0.97, 0.01], dtype=float)

    reward = reward_fn.compute(sample, probs)
    assert cfg.reward_min <= reward <= cfg.reward_max


def test_programmatic_reward_prefers_correct_and_confident_prediction(make_stream_sample) -> None:
    cfg = ModelConfig()
    reward_fn = ProgrammaticReward(cfg)

    sample = make_stream_sample(label=1, class_scale=0.9, strategy_quality=0.9)

    good = reward_fn.compute(sample, np.array([0.02, 0.95, 0.02, 0.01], dtype=float))
    bad = reward_fn.compute(sample, np.array([0.95, 0.02, 0.02, 0.01], dtype=float))

    assert good > bad
