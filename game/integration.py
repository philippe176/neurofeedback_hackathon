from __future__ import annotations

from dataclasses import dataclass

from model.config import ModelConfig

from .config import RhythmGameConfig
from .rewards import GameRewardProvider
from .ui import RhythmConsoleHUD, RhythmGameDashboard


@dataclass(slots=True)
class GameRuntimeContext:
    reward_provider: GameRewardProvider
    hud: RhythmConsoleHUD
    dashboard: RhythmGameDashboard | None


def build_game_runtime(
    model_cfg: ModelConfig,
    game_cfg: RhythmGameConfig,
    print_every: int = 10,
    enable_dashboard: bool = True,
    dashboard_history: int = 320,
    dashboard_draw_every: int = 2,
) -> GameRuntimeContext:
    provider = GameRewardProvider(model_cfg=model_cfg, game_cfg=game_cfg)
    hud = RhythmConsoleHUD(print_every=print_every)
    dashboard = (
        RhythmGameDashboard(
            history_len=dashboard_history,
            draw_every=dashboard_draw_every,
            on_toggle_simulation=provider.set_auto_perform,
            get_simulation_enabled=provider.is_auto_perform_enabled,
        )
        if enable_dashboard
        else None
    )
    return GameRuntimeContext(reward_provider=provider, hud=hud, dashboard=dashboard)
