import numpy as np
import torch

from game.config import LevelPolicy, RhythmGameConfig
from game.rewards import GameRewardProvider
from model.config import ModelConfig
from model.network import MovementDecoder
from model.trainer import OnlineTrainer
from model.types import StreamSample


def _sample(idx: int, ts: float, dim: int) -> StreamSample:
    return StreamSample(
        sample_idx=idx,
        timestamp=ts,
        embedding=np.linspace(0.0, 1.0, dim, dtype=np.float32),
        label=idx % 4,
        label_name=None,
        class_scale=0.6,
        strategy_quality=0.7,
        difficulty="d1",
        raw={},
    )


def test_online_trainer_populates_game_feedback_fields() -> None:
    dim = 16
    cfg = ModelConfig(
        input_dim=dim,
        hidden_dim=32,
        embedding_dim=8,
        projection_dim=2,
        batch_size=2,
        buffer_size=32,
        update_every=1,
        min_buffer_before_updates=2,
        warmup_labeled_samples=1,
        device="cpu",
    )

    game_cfg = RhythmGameConfig(
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

    model = MovementDecoder(
        input_dim=dim,
        hidden_dim=cfg.hidden_dim,
        embedding_dim=cfg.embedding_dim,
        n_classes=cfg.n_classes,
        projection_dim=cfg.projection_dim,
    )
    reward_provider = GameRewardProvider(model_cfg=cfg, game_cfg=game_cfg)
    trainer = OnlineTrainer(model=model, cfg=cfg, reward_provider=reward_provider, device=torch.device("cpu"))

    step_0 = trainer.process_sample(_sample(0, 300.0, dim=dim))
    step_1 = trainer.process_sample(_sample(1, 300.6, dim=dim))

    assert step_0.game_prompt_id is not None
    assert step_1.game_target_class in (0, 1, 2, 3)
    assert step_1.game_level is not None
    assert step_1.game_prompt_progress is not None
    assert step_1.game_margin is not None
    assert step_1.game_seconds_to_window_start is not None
    assert step_1.game_next_target_class in (0, 1, 2, 3)
    assert step_1.game_seconds_to_next_prompt_start is not None
    assert isinstance(step_1.game_label_correct, bool)
    assert isinstance(step_1.game_timing_hit, bool)
    assert step_1.game_reward_components is not None
    assert np.isfinite(step_1.reward)
