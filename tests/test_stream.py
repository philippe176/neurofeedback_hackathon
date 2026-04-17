import numpy as np
import pytest

from model.stream import ExperienceReplayBuffer, ZMQEmbeddingReceiver
from model.types import Experience


def test_experience_replay_buffer_capacity_and_recent_order() -> None:
    buf = ExperienceReplayBuffer(capacity=3)
    for i in range(5):
        buf.append(
            Experience(
                sample_idx=i,
                timestamp=float(i),
                embedding=np.array([i], dtype=np.float32),
                label=i % 4,
                action=i % 4,
                reward=0.1 * i,
            )
        )

    assert len(buf) == 3
    recent = buf.sample_recent(10)
    assert [x.sample_idx for x in recent] == [2, 3, 4]


def test_receiver_parse_message_success() -> None:
    receiver = ZMQEmbeddingReceiver(host="localhost", port=5555, embedding_key="emb")
    msg = {
        "sample_idx": 10,
        "timestamp": 100.0,
        "emb": [1.0, 2.0, 3.0],
        "label": 1,
        "label_name": "right_hand",
        "class_scale": 0.8,
        "strategy_quality": 0.9,
        "difficulty": "d1",
    }

    sample = receiver._parse_message(msg)
    assert sample.sample_idx == 10
    assert sample.label == 1
    assert sample.embedding.shape == (3,)
    assert sample.class_scale == 0.8


def test_receiver_parse_message_missing_embedding_key_raises() -> None:
    receiver = ZMQEmbeddingReceiver(host="localhost", port=5555, embedding_key="emb")
    with pytest.raises(KeyError):
        receiver._parse_message({"sample_idx": 1, "data": [0.0, 1.0]})


def test_receiver_queue_keeps_latest_when_full(make_stream_sample) -> None:
    receiver = ZMQEmbeddingReceiver(host="localhost", port=5555, queue_capacity=1)
    first = make_stream_sample(idx=1)
    second = make_stream_sample(idx=2)

    receiver._put_latest(first)
    receiver._put_latest(second)

    out = receiver.get(timeout=0.01)
    assert out is not None
    assert out.sample_idx == 2


def test_experience_replay_buffer_stratified_sampling_balances_classes() -> None:
    buf = ExperienceReplayBuffer(capacity=32)
    for idx in range(12):
        label = idx % 2
        buf.append(
            Experience(
                sample_idx=idx,
                timestamp=float(idx),
                embedding=np.array([idx], dtype=np.float32),
                label=label,
                action=label,
                reward=1.0,
                class_scale=0.2 + 0.1 * label,
            )
        )

    batch = buf.sample_stratified(8, recency_bias=0.5)
    labels = [exp.label for exp in batch]

    assert len(batch) == 8
    assert labels.count(0) >= 3
    assert labels.count(1) >= 3
    assert abs(labels.count(0) - labels.count(1)) <= 2


def test_experience_replay_buffer_stratified_sampling_handles_unlabeled_only() -> None:
    buf = ExperienceReplayBuffer(capacity=8)
    for idx in range(5):
        buf.append(
            Experience(
                sample_idx=idx,
                timestamp=float(idx),
                embedding=np.array([idx], dtype=np.float32),
                label=-1,
                action=0,
                reward=0.0,
            )
        )

    batch = buf.sample_stratified(3)

    assert len(batch) == 3
    assert all(exp.label == -1 for exp in batch)


def test_receiver_parse_message_flattens_nested_embedding_and_defaults() -> None:
    receiver = ZMQEmbeddingReceiver(host="localhost", port=5555, embedding_key="emb")
    sample = receiver._parse_message({"emb": [[1.0, 2.0], [3.0, 4.0]], "label": None})

    assert sample.sample_idx == -1
    assert sample.label is None
    assert sample.embedding.shape == (4,)
    assert sample.class_scale is None


def test_receiver_start_stop_are_idempotent_without_server() -> None:
    receiver = ZMQEmbeddingReceiver(
        host="localhost",
        port=6553,
        queue_capacity=4,
        receiver_timeout_ms=20,
    )

    receiver.start()
    first_thread = receiver._thread
    receiver.start()
    receiver.stop()
    receiver.stop()

    assert first_thread is not None
    assert receiver._thread is None
    assert receiver._running is False
