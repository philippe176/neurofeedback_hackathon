import numpy as np
import torch

from model.config import ModelConfig
from model.network import MovementDecoder
from model.reward import ProgrammaticReward
from model.trainer import OnlineTrainer


def _make_trainer(input_dim: int = 16, warmup_labeled: int = 2) -> OnlineTrainer:
    cfg = ModelConfig(
        input_dim=input_dim,
        hidden_dim=32,
        embedding_dim=8,
        projection_dim=2,
        batch_size=2,
        buffer_size=32,
        update_every=1,
        min_buffer_before_updates=2,
        warmup_labeled_samples=warmup_labeled,
        device="cpu",
    )
    model = MovementDecoder(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        embedding_dim=cfg.embedding_dim,
        n_classes=cfg.n_classes,
        projection_dim=cfg.projection_dim,
    )
    reward = ProgrammaticReward(cfg)
    return OnlineTrainer(model=model, cfg=cfg, reward_provider=reward, device=torch.device("cpu"))


def test_trainer_process_sample_outputs_expected_shapes(make_stream_sample) -> None:
    trainer = _make_trainer(input_dim=16)
    sample = make_stream_sample(dim=16, label=1)

    step = trainer.process_sample(sample)

    assert step.predicted_class in (0, 1, 2, 3)
    assert 0.0 <= step.confidence <= 1.0
    assert step.probabilities.shape == (4,)
    assert step.penultimate.shape == (8,)
    assert step.projection.shape == (2,)


def test_trainer_runs_update_and_enables_rl_after_warmup(make_stream_sample) -> None:
    trainer = _make_trainer(input_dim=16, warmup_labeled=2)

    s1 = make_stream_sample(idx=1, dim=16, label=0)
    s2 = make_stream_sample(idx=2, dim=16, label=1)

    out1 = trainer.process_sample(s1)
    assert out1.training is None or not out1.training.update_applied

    out2 = trainer.process_sample(s2)
    assert out2.training is not None
    assert out2.training.update_applied
    assert out2.training.rl_enabled
    assert trainer.num_updates >= 1


def test_trainer_skips_update_when_unlabeled_and_rl_disabled(make_stream_sample) -> None:
    trainer = _make_trainer(input_dim=16, warmup_labeled=100)

    s1 = make_stream_sample(idx=1, dim=16, label=None)
    s2 = make_stream_sample(idx=2, dim=16, label=None)

    trainer.process_sample(s1)
    out2 = trainer.process_sample(s2)

    assert out2.training is not None
    assert not out2.training.update_applied
