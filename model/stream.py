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
        self._rng = np.random.default_rng(42)

    def append(self, exp: Experience) -> None:
        with self._lock:
            self._data.append(exp)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def sample_recent(self, n: int) -> list[Experience]:
        """Return the last *n* items sequentially (legacy fallback)."""
        with self._lock:
            if not self._data:
                return []
            n = min(n, len(self._data))
            return list(self._data)[-n:]

    def sample_stratified(self, n: int, recency_bias: float = 0.7) -> list[Experience]:
        """Class-balanced sampling with recency bias.

        *recency_bias* controls the share drawn from the most recent quarter
        of the buffer; the rest is drawn uniformly.  Within each pool the
        method tries to draw equal counts per class so that the model never
        trains on a single class just because the operator held one key.
        """
        with self._lock:
            total = len(self._data)
            if total == 0:
                return []
            n = min(n, total)
            data_list = list(self._data)

            recent_cutoff = max(1, total // 4)
            recent_pool = data_list[-recent_cutoff:]
            full_pool = data_list

            n_recent = min(int(n * recency_bias), len(recent_pool))
            n_uniform = n - n_recent

            recent_batch = self._balanced_draw(recent_pool, n_recent)
            uniform_batch = self._balanced_draw(full_pool, n_uniform)

            combined = recent_batch + uniform_batch
            self._rng.shuffle(combined)
            return combined[:n]

    def _balanced_draw(self, pool: list[Experience], n: int) -> list[Experience]:
        """Draw *n* items from *pool* balancing across classes.

        Within each class, samples with higher class_scale are preferred so
        that training prioritises high-quality fine-signal observations.
        """
        if n <= 0 or not pool:
            return []

        by_class: dict[int, list[int]] = {}
        unlabeled: list[int] = []
        for idx, exp in enumerate(pool):
            if exp.label >= 0:
                by_class.setdefault(exp.label, []).append(idx)
            else:
                unlabeled.append(idx)

        selected_indices: list[int] = []

        if by_class:
            classes = sorted(by_class.keys())
            per_class = max(1, n // len(classes))
            remainder = n - per_class * len(classes)

            for cls in classes:
                indices = by_class[cls]
                take = min(per_class, len(indices))
                weights = self._scale_weights(pool, indices)
                chosen = self._rng.choice(indices, size=take, replace=False, p=weights).tolist()
                selected_indices.extend(chosen)

            # Distribute remainder across classes with most samples
            if remainder > 0:
                all_remaining = []
                used = set(selected_indices)
                for cls in classes:
                    all_remaining.extend(i for i in by_class[cls] if i not in used)
                if all_remaining:
                    take = min(remainder, len(all_remaining))
                    weights = self._scale_weights(pool, all_remaining)
                    extra = self._rng.choice(all_remaining, size=take, replace=False, p=weights).tolist()
                    selected_indices.extend(extra)
        else:
            # No labeled data — draw uniformly from unlabeled
            take = min(n, len(unlabeled))
            if take:
                selected_indices = self._rng.choice(unlabeled, size=take, replace=False).tolist()

        # Fill any shortfall from the full pool
        shortfall = n - len(selected_indices)
        if shortfall > 0:
            used = set(selected_indices)
            remaining = [i for i in range(len(pool)) if i not in used]
            if remaining:
                take = min(shortfall, len(remaining))
                extra = self._rng.choice(remaining, size=take, replace=False).tolist()
                selected_indices.extend(extra)

        return [pool[i] for i in selected_indices]

    def _scale_weights(self, pool: list[Experience], indices: list[int]) -> np.ndarray:
        """Compute sampling weights from class_scale. Higher scale → more likely."""
        scales = np.array([max(0.05, pool[i].class_scale) for i in indices], dtype=np.float64)
        total = scales.sum()
        if total <= 0:
            return np.ones(len(indices), dtype=np.float64) / len(indices)
        return scales / total

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
