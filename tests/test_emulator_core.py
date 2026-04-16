import socket
import time

import numpy as np

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
