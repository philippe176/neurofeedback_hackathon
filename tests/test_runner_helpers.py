import io
import time

import numpy as np

from model.realtime_runner import _print_heartbeat, _wait_for_first_sample, build_arg_parser
from model.types import InferenceStep


class _DummyReceiver:
    def __init__(self, sample=None, return_after=0):
        self.sample = sample
        self.calls = 0
        self.return_after = return_after

    def get(self, timeout=0.25):
        self.calls += 1
        if self.calls > self.return_after:
            return self.sample
        return None


class _DummyTrainer:
    reward_baseline = 0.42
    labeled_seen = 12
    num_updates = 3


def test_wait_for_first_sample_returns_when_available(make_stream_sample) -> None:
    sample = make_stream_sample(idx=7)
    receiver = _DummyReceiver(sample=sample, return_after=2)

    out = _wait_for_first_sample(receiver, timeout=1.0)
    assert out is not None
    assert out.sample_idx == 7


def test_wait_for_first_sample_times_out_when_no_data() -> None:
    receiver = _DummyReceiver(sample=None, return_after=10_000)
    start = time.time()
    out = _wait_for_first_sample(receiver, timeout=0.3)
    elapsed = time.time() - start

    assert out is None
    assert elapsed >= 0.25


def test_print_heartbeat_formats_output(capsys) -> None:
    step = InferenceStep(
        sample_idx=11,
        label=1,
        predicted_class=2,
        confidence=0.88,
        reward=0.67,
        probabilities=np.array([0.01, 0.10, 0.88, 0.01], dtype=float),
        penultimate=np.array([0.1, 0.2], dtype=float),
        projection=np.array([0.3, 0.4], dtype=float),
        training=None,
    )

    _print_heartbeat(step, _DummyTrainer())
    out = capsys.readouterr().out

    assert "pred=2" in out
    assert "reward=0.67" in out
    assert "updates=3" in out


def test_arg_parser_defaults() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.host == "localhost"
    assert args.port == 5555
    assert args.embedding_key == "data"
    assert args.viz_method == "neural"
