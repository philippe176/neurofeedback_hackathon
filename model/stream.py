from __future__ import annotations

import collections
import json
import queue
import threading
import time

import numpy as np
import zmq

from .types import Experience, StreamSample


class ExperienceReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._data: collections.deque[Experience] = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()

    def append(self, exp: Experience) -> None:
        with self._lock:
            self._data.append(exp)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def sample_recent(self, n: int) -> list[Experience]:
        with self._lock:
            if not self._data:
                return []
            n = min(n, len(self._data))
            return list(self._data)[-n:]

    def labeled_count(self) -> int:
        with self._lock:
            return sum(1 for x in self._data if x.label >= 0)


class ZMQEmbeddingReceiver:
    """Receive JSON messages from emulator and convert them to StreamSample."""

    def __init__(
        self,
        host: str,
        port: int,
        embedding_key: str = "data",
        queue_capacity: int = 2048,
        receiver_timeout_ms: int = 500,
    ) -> None:
        self.host = host
        self.port = port
        self.embedding_key = embedding_key
        self.receiver_timeout_ms = receiver_timeout_ms

        self._queue: queue.Queue[StreamSample] = queue.Queue(maxsize=queue_capacity)
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get(self, timeout: float = 0.2) -> StreamSample | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self) -> None:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.connect(f"tcp://{self.host}:{self.port}")
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.RCVTIMEO, self.receiver_timeout_ms)

        try:
            while self._running:
                try:
                    raw = sock.recv_string()
                    msg = json.loads(raw)
                    sample = self._parse_message(msg)
                    self._put_latest(sample)
                except zmq.Again:
                    continue
                except Exception:
                    continue
        finally:
            sock.close()
            ctx.term()

    def _parse_message(self, msg: dict) -> StreamSample:
        if self.embedding_key not in msg:
            raise KeyError(f"Missing embedding key '{self.embedding_key}' in stream message")

        embedding = np.asarray(msg[self.embedding_key], dtype=np.float32).reshape(-1)
        label_raw = msg.get("label", None)
        label = int(label_raw) if label_raw is not None else None

        return StreamSample(
            sample_idx=int(msg.get("sample_idx", -1)),
            timestamp=float(msg.get("timestamp", time.time())),
            embedding=embedding,
            label=label,
            label_name=msg.get("label_name", None),
            class_scale=_to_float_or_none(msg.get("class_scale", None)),
            strategy_quality=_to_float_or_none(msg.get("strategy_quality", None)),
            difficulty=msg.get("difficulty", None),
            raw=msg,
        )

    def _put_latest(self, sample: StreamSample) -> None:
        try:
            self._queue.put_nowait(sample)
            return
        except queue.Full:
            pass

        try:
            _ = self._queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            pass


def _to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
