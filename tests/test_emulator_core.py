import socket
import time

import numpy as np
import pytest

from emulator.config import DIFFICULTIES
from emulator.dynamics import LatentDynamics
from emulator.emulator import BrainEmulator
from emulator.generative import GenerativeModel
from model.stream import ZMQEmbeddingReceiver


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


def test_latent_dynamics_outputs_valid_ranges() -> None:
    dyn = LatentDynamics(DIFFICULTIES["d1"], sample_rate=10.0)
    dyn.set_class(0)
    state = dyn.step()

    assert state["current_class"] == 0
    assert 0.0 <= state["strategy_quality"] <= 1.0
    assert 0.0 <= state["class_scale"] <= 1.0
    assert dyn.z_full.shape == (8,)


def test_generative_model_observation_shape() -> None:
    gm = GenerativeModel(n_obs=64, n_latent=8)
    z = np.zeros(8, dtype=float)
    R = np.eye(8)
    x = gm.observe(z, R, noise_std=0.1, class_scale=0.5)

    assert x.shape == (64,)


def test_generative_model_noise_is_seed_reproducible() -> None:
    z = np.linspace(-1.0, 1.0, 8, dtype=float)
    R = np.eye(8)
    gm_a = GenerativeModel(n_obs=32, n_latent=8, seed=7)
    gm_b = GenerativeModel(n_obs=32, n_latent=8, seed=7)

    x_a = gm_a.observe(z, R, noise_std=0.2, class_scale=0.75)
    x_b = gm_b.observe(z, R, noise_std=0.2, class_scale=0.75)

    assert np.allclose(x_a, x_b)


def test_brain_emulator_step_message_schema() -> None:
    port = _free_port()
    emu = BrainEmulator(difficulty="d1", n_dims=32, port=port, sample_rate=10.0)
    try:
        emu.set_class(1)
        msg = emu.step()
    finally:
        emu.close()

    assert isinstance(msg["sample_idx"], int)
    assert msg["label"] == 1
    assert msg["label_name"] == "right_hand"
    assert len(msg["data"]) == 32
    assert "class_scale" in msg
    assert "strategy_quality" in msg


def test_latent_dynamics_rejects_invalid_inputs() -> None:
    dyn = LatentDynamics(DIFFICULTIES["d1"], sample_rate=10.0)

    with pytest.raises(ValueError, match="class_idx"):
        dyn.set_class(99)

    with pytest.raises(ValueError, match="shape"):
        dyn.update_strategy(np.array([1.0, 0.0, -1.0], dtype=float))


def test_brain_emulator_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="difficulty"):
        BrainEmulator(difficulty="bogus")

    with pytest.raises(ValueError, match="n_dims"):
        BrainEmulator(n_dims=0)

    with pytest.raises(ValueError, match="sample_rate"):
        BrainEmulator(sample_rate=0.0)

    with pytest.raises(ValueError, match="port"):
        BrainEmulator(port=70000)


def test_generative_model_rejects_invalid_observation_inputs() -> None:
    gm = GenerativeModel(n_obs=16, n_latent=8)

    with pytest.raises(ValueError, match="z must have shape"):
        gm.observe(np.zeros(7, dtype=float), np.eye(8), noise_std=0.1)

    with pytest.raises(ValueError, match="R must have shape"):
        gm.observe(np.zeros(8, dtype=float), np.eye(7), noise_std=0.1)

    with pytest.raises(ValueError, match="noise_std"):
        gm.observe(np.zeros(8, dtype=float), np.eye(8), noise_std=-0.1)


def test_brain_emulator_close_is_idempotent_and_prevents_future_steps() -> None:
    port = _free_port()
    emu = BrainEmulator(difficulty="d1", n_dims=8, port=port, sample_rate=10.0)

    emu.close()
    emu.close()

    with pytest.raises(RuntimeError, match="closed"):
        emu.step()


def test_emulator_receiver_stitching_roundtrip() -> None:
    port = _free_port()
    emu = BrainEmulator(difficulty="d1", n_dims=16, port=port, sample_rate=10.0)
    receiver = ZMQEmbeddingReceiver(host="localhost", port=port, embedding_key="data")

    try:
        receiver.start()
        emu.set_class(0)

        # ZMQ SUB sockets can miss early messages; publish several with retries.
        got = None
        for _ in range(30):
            emu.step()
            got = receiver.get(timeout=0.1)
            if got is not None:
                break
            time.sleep(0.01)

        assert got is not None
        assert got.embedding.shape == (16,)
        assert got.label in (0, None)
    finally:
        receiver.stop()
        emu.close()
