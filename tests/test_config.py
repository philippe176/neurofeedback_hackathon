import torch
import pytest

from model.config import ModelConfig


def test_model_config_validation_projection_dim() -> None:
    with pytest.raises(ValueError, match="projection_dim"):
        ModelConfig(projection_dim=4)


def test_model_config_validation_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        ModelConfig(batch_size=1)


def test_model_config_validation_update_every() -> None:
    with pytest.raises(ValueError, match="update_every"):
        ModelConfig(update_every=0)


def test_model_config_resolve_device_cpu() -> None:
    cfg = ModelConfig(device="cpu")
    device = cfg.resolve_device()
    assert isinstance(device, torch.device)
    assert str(device) == "cpu"
