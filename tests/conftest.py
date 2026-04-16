import os
import sys
from pathlib import Path

# Force a non-interactive matplotlib backend for CI/headless test runs.
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time

import numpy as np
import pytest

from model.types import StreamSample


@pytest.fixture
def make_stream_sample():
    def _make(
        idx: int = 0,
        dim: int = 16,
        label: int | None = 0,
        class_scale: float | None = 0.5,
        strategy_quality: float | None = 0.5,
    ) -> StreamSample:
        return StreamSample(
            sample_idx=idx,
            timestamp=time.time(),
            embedding=np.linspace(0.0, 1.0, dim, dtype=np.float32),
            label=label,
            label_name=None,
            class_scale=class_scale,
            strategy_quality=strategy_quality,
            difficulty="d1",
            raw={},
        )

    return _make
