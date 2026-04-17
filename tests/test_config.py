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


def test_model_config_rejects_invalid_class_weights_length() -> None:
    with pytest.raises(ValueError, match="class_weights"):
        ModelConfig(n_classes=4, class_weights=(1.0, 1.0))


def test_model_config_rejects_negative_focal_gamma() -> None:
    with pytest.raises(ValueError, match="classification_focal_gamma"):
        ModelConfig(classification_focal_gamma=-0.1)


def test_model_config_rejects_negative_contrastive_weight() -> None:
    with pytest.raises(ValueError, match="contrastive_weight"):
        ModelConfig(contrastive_weight=-0.1)


def test_model_config_rejects_non_positive_contrastive_temperature() -> None:
    with pytest.raises(ValueError, match="contrastive_temperature"):
        ModelConfig(contrastive_temperature=0.0)


def test_model_config_rejects_invalid_viz_method() -> None:
    with pytest.raises(ValueError, match="viz_method"):
        ModelConfig(viz_method="bogus")


def test_model_config_rejects_invalid_viz_refit_every() -> None:
    with pytest.raises(ValueError, match="viz_refit_every"):
        ModelConfig(viz_refit_every=0)
