import numpy as np

from model.types import InferenceStep
from model.viz import RealtimeManifoldVisualizer


def _step(proj: np.ndarray) -> InferenceStep:
    return InferenceStep(
        sample_idx=1,
        label=0,
        predicted_class=0,
        confidence=0.9,
        reward=0.7,
        probabilities=np.array([0.9, 0.05, 0.03, 0.02], dtype=float),
        penultimate=np.array([0.1, 0.2, 0.3], dtype=float),
        projection=proj,
        training=None,
    )


def test_visualizer_update_2d_does_not_crash() -> None:
    viz = RealtimeManifoldVisualizer(projection_dim=2, history_len=20, draw_every=1)
    try:
        viz.update(_step(np.array([0.1, 0.2], dtype=float)))
        viz.update(_step(np.array([0.2, 0.3], dtype=float)))
    finally:
        viz.close()


def test_visualizer_update_3d_does_not_crash() -> None:
    viz = RealtimeManifoldVisualizer(projection_dim=3, history_len=20, draw_every=1)
    try:
        viz.update(_step(np.array([0.1, 0.2, 0.3], dtype=float)))
        viz.update(_step(np.array([0.2, 0.3, 0.4], dtype=float)))
    finally:
        viz.close()


def test_visualizer_supports_pca_and_lda_backends() -> None:
    for method in ("pca", "lda"):
        viz = RealtimeManifoldVisualizer(
            projection_dim=2,
            history_len=24,
            draw_every=1,
            viz_method=method,
            fit_window=20,
            refit_every=2,
        )
        try:
            for idx in range(8):
                step = InferenceStep(
                    sample_idx=idx,
                    label=idx % 4,
                    predicted_class=idx % 4,
                    confidence=0.8,
                    reward=0.6,
                    probabilities=np.array([0.7, 0.1, 0.1, 0.1], dtype=float),
                    penultimate=np.array([idx, idx + 1, idx + 2, idx + 3], dtype=float),
                    projection=np.array([0.1 * idx, 0.2 * idx], dtype=float),
                    training=None,
                )
                viz.update(step)
        finally:
            viz.close()
