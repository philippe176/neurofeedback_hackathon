import numpy as np

from game.ui import RhythmGameDashboard
from model.types import InferenceStep


def _step(idx: int, target: int, pred: int) -> InferenceStep:
    return InferenceStep(
        sample_idx=idx,
        label=None,
        predicted_class=pred,
        confidence=0.7,
        reward=0.6,
        probabilities=np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
        penultimate=np.array([0.1, 0.2, 0.3], dtype=float),
        projection=np.array([0.2 * idx, 0.1 * idx], dtype=float),
        training=None,
        game_prompt_id=idx,
        game_target_class=target,
        game_in_window=True,
        game_hit=(pred == target),
        game_label_correct=(pred == target),
        game_timing_hit=(pred == target),
        game_timing_error_s=0.1,
        game_prompt_progress=0.6,
        game_margin=0.3,
        game_level=1,
        game_streak=3,
        game_reward_components={"correctness": 1.0, "margin": 0.8},
    )


def test_dashboard_update_and_close_do_not_crash() -> None:
    dash = RhythmGameDashboard(history_len=50, draw_every=1)
    try:
        dash.update(_step(1, target=1, pred=1))
        dash.update(_step(2, target=2, pred=1))
    finally:
        dash.close()
