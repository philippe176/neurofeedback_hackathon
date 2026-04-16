"""
game_maze.py — Pygame maze game driven by BCI probability output.

Public interface
----------------
    update(probs: np.ndarray) -> None

        probs : length-4 array of softmax probabilities summing to 1.0
                index 0 = left hand  → move left
                index 1 = right hand → move right
                index 2 = left leg   → move up
                index 3 = right leg  → move down

        Internally:
            predicted_class = np.argmax(probs)
            confidence      = probs[predicted_class]   # max probability

        Movement speed scales linearly with confidence (0 = stuck, 1 = full speed).
        Player colour is a smooth red → green gradient driven by confidence.

This file has zero knowledge of ZMQ, LDA, or raw neural data.
"""

from __future__ import annotations

import threading

import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE             = 40       # pixels per maze tile
FPS              = 60
MAX_SPEED        = 160      # pixels / second at confidence = 1.0
PLAYER_R         = 12       # player radius in pixels
TUTORIAL_DURATION = 60.0   # seconds of free-roam before the maze starts

# Class index → movement direction unit vector (dx, dy)
DIR_VEC: dict[int, tuple[int, int]] = {
    0: (-1,  0),   # left hand  → left
    1: ( 1,  0),   # right hand → right
    2: ( 0, -1),   # left leg   → up
    3: ( 0,  1),   # right leg  → down
}
CLASS_NAMES = {0: "Left hand", 1: "Right hand", 2: "Left leg", 3: "Right leg"}

# ---------------------------------------------------------------------------
# Maze layout
# 1 = wall, 0 = open corridor
# Grid is 21 columns wide × 13 rows tall.
# ---------------------------------------------------------------------------
#
#  Verified path from START (col 1, row 1) to GOAL (col 19, row 9):
#  (1,1)→(1,5) via downward corridor  →  right along row-5 open corridor  →
#  (19,5)→(19,9) via downward corridor.

MAZE_GRID = [
    #  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
    [  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],  # row  0
    [  1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1 ],  # row  1  ← START (1,1)
    [  1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1 ],  # row  2
    [  1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1 ],  # row  3
    [  1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1 ],  # row  4
    [  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ],  # row  5  (open corridor)
    [  1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1 ],  # row  6
    [  1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1 ],  # row  7
    [  1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1 ],  # row  8
    [  1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1 ],  # row  9  ← GOAL (19,9)
    [  1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1 ],  # row 10
    [  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ],  # row 11
    [  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],  # row 12
]

MAZE_COLS = len(MAZE_GRID[0])   # 21
MAZE_ROWS = len(MAZE_GRID)      # 13

START_TILE = (1,  1)    # (col, row)
GOAL_TILE  = (19, 9)    # (col, row)

MAZE_W = MAZE_COLS * TILE   # 840
MAZE_H = MAZE_ROWS * TILE   # 520
HUD_H  = 80
WIN_W  = MAZE_W             # 840
WIN_H  = MAZE_H + HUD_H     # 600

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

COL_WALL      = ( 30,  30,  50)
COL_WALL_EDGE = ( 55,  55,  85)
COL_FLOOR     = ( 15,  15,  25)
COL_GOAL      = (255, 215,   0)
COL_GOAL_RIM  = (255, 240, 120)
COL_START     = ( 50,  50,  90)
COL_BG        = ( 10,  10,  18)
COL_HUD_BG    = ( 18,  18,  30)
COL_HUD_LINE  = ( 50,  50,  80)
COL_TEXT      = (200, 200, 220)
COL_DIMTEXT   = (110, 110, 140)

# ---------------------------------------------------------------------------
# Module-level singleton and lock
# ---------------------------------------------------------------------------

_game: MazeGame | None = None
_lock = threading.Lock()


def update(probs: np.ndarray) -> None:
    """
    Push a new probability estimate into the running game.

    Called by the external pipeline (may be from a non-main thread).

    Parameters
    ----------
    probs : np.ndarray, shape (4,)
        Softmax probability distribution over 4 movement classes.
        Must sum to 1.0.  Elements are in order:
            [p_left_hand, p_right_hand, p_left_leg, p_right_leg]
    """
    if _game is not None:
        with _lock:
            _game.probs = np.asarray(probs, dtype=float).copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _overlaps_wall(cx: float, cy: float, r: float) -> bool:
    """
    Return True if a circle of radius *r* centred at *(cx, cy)* overlaps
    any wall tile (or falls out of bounds).

    Uses the exact circle-vs-AABB test: nearest point on the tile rectangle
    to the circle centre, then distance check.
    """
    tx_lo = int((cx - r) // TILE)
    tx_hi = int((cx + r) // TILE)
    ty_lo = int((cy - r) // TILE)
    ty_hi = int((cy + r) // TILE)

    for ty in range(ty_lo, ty_hi + 1):
        for tx in range(tx_lo, tx_hi + 1):
            if ty < 0 or ty >= MAZE_ROWS or tx < 0 or tx >= MAZE_COLS:
                return True   # out of bounds counts as wall
            if MAZE_GRID[ty][tx] == 1:
                tile_left = tx * TILE
                tile_top  = ty * TILE
                near_x = max(tile_left, min(cx, tile_left + TILE))
                near_y = max(tile_top,  min(cy, tile_top  + TILE))
                if (cx - near_x) ** 2 + (cy - near_y) ** 2 < r * r:
                    return True
    return False


def _confidence_color(c: float) -> tuple[int, int, int]:
    """Linear red → green gradient.  c=0 → (255,0,0), c=1 → (0,255,0)."""
    c = float(np.clip(c, 0.0, 1.0))
    return (int(255 * (1.0 - c)), int(255 * c), 0)


def _draw_arrow(surf: pygame.Surface, color: tuple, cx: int, cy: int,
                direction: int, size: int) -> None:
    """Draw a solid directional triangle arrow on *surf*."""
    dx, dy = DIR_VEC[direction]
    tip    = (cx + dx * size, cy + dy * size)
    perp   = (-dy, dx)
    hw     = size * 0.55
    b1 = (cx - dx * size * 0.3 + perp[0] * hw, cy - dy * size * 0.3 + perp[1] * hw)
    b2 = (cx - dx * size * 0.3 - perp[0] * hw, cy - dy * size * 0.3 - perp[1] * hw)
    pygame.draw.polygon(surf, color, [tip, b1, b2])


def _fmt_time(seconds: float) -> str:
    """Format elapsed time as  mm:ss.xx  (e.g. '01:23.45')."""
    m  = int(seconds) // 60
    s  = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class Player:
    def __init__(self, tile_col: int, tile_row: int) -> None:
        self.x = tile_col * TILE + TILE / 2.0
        self.y = tile_row * TILE + TILE / 2.0

    def reset(self, tile_col: int, tile_row: int) -> None:
        self.x = tile_col * TILE + TILE / 2.0
        self.y = tile_row * TILE + TILE / 2.0

    def try_move(self, dx: float, dy: float) -> None:
        """Slide along walls: try X then Y independently."""
        new_x = self.x + dx
        if not _overlaps_wall(new_x, self.y, PLAYER_R):
            self.x = new_x

        new_y = self.y + dy
        if not _overlaps_wall(self.x, new_y, PLAYER_R):
            self.y = new_y

    def try_move_free(self, dx: float, dy: float) -> None:
        """Move without wall checks — clamp only to the open-area bounds."""
        x_min = TILE + PLAYER_R
        x_max = (MAZE_COLS - 1) * TILE - PLAYER_R
        y_min = TILE + PLAYER_R
        y_max = (MAZE_ROWS - 1) * TILE - PLAYER_R
        self.x = float(np.clip(self.x + dx, x_min, x_max))
        self.y = float(np.clip(self.y + dy, y_min, y_max))

    @property
    def tile(self) -> tuple[int, int]:
        return (int(self.x // TILE), int(self.y // TILE))


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class MazeGame:
    def __init__(self) -> None:
        self.probs: np.ndarray = np.full(4, 0.25)
        # Start player at the centre of the open field for the tutorial
        self.player    = Player(MAZE_COLS // 2, MAZE_ROWS // 2)
        self.running   = True
        self.won       = False
        self._win_timer = 0.0

        # Tutorial phase
        self._phase          = "tutorial"   # "tutorial" | "maze"
        self._tutorial_time  = TUTORIAL_DURATION

        # Maze timer
        self._elapsed:   float       = 0.0
        self._best_time: float | None = None
        self._final_time: float | None = None   # time frozen at goal
        self._timer_active = True

        # Preload fonts once (pygame must be initialised first)
        self._font_big   = pygame.font.SysFont("monospace", 20, bold=True)
        self._font_small = pygame.font.SysFont("monospace", 13)
        self._font_win   = pygame.font.SysFont("monospace", 52, bold=True)
        self._font_sub   = pygame.font.SysFont("monospace", 22)
        self._font_timer = pygame.font.SysFont("monospace", 18, bold=True)

        # Pulsing goal animation state
        self._goal_pulse = 0.0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Tutorial helpers
    # ------------------------------------------------------------------

    def _tick_tutorial(self, dt: float) -> None:
        self._goal_pulse = (self._goal_pulse + dt * 3.0) % (2 * np.pi)
        self._tutorial_time -= dt
        if self._tutorial_time <= 0.0:
            self._start_maze()
            return

        with _lock:
            probs = self.probs.copy()

        if np.allclose(probs, probs[0]):
            predicted_class = None
            confidence = 0.0
        else:
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])

        if predicted_class is not None and confidence > 0.0:
            ddx, ddy = DIR_VEC[predicted_class]
            speed = MAX_SPEED * confidence
            self.player.try_move_free(ddx * speed * dt, ddy * speed * dt)

    def _start_maze(self) -> None:
        self._phase = "maze"
        self.player.reset(*START_TILE)

    def skip_tutorial(self) -> None:
        """Called externally when the player presses S."""
        if self._phase == "tutorial":
            self._start_maze()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        if self._phase == "tutorial":
            self._tick_tutorial(dt)
            return

        self._goal_pulse = (self._goal_pulse + dt * 3.0) % (2 * np.pi)

        # Advance timer while playing
        if self._timer_active and not self.won:
            self._elapsed += dt

        if self.won:
            self._win_timer -= dt
            if self._win_timer <= 0.0:
                self._reset()
            return

        with _lock:
            probs = self.probs.copy()

        if np.allclose(probs, probs[0]):
            predicted_class = None
            confidence = 0.0
        else:
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])

        if predicted_class is not None and confidence > 0.0:
            ddx, ddy = DIR_VEC[predicted_class]
            speed = MAX_SPEED * confidence
            self.player.try_move(ddx * speed * dt, ddy * speed * dt)

        if self.player.tile == GOAL_TILE:
            self.won           = True
            self._win_timer    = 3.0
            self._timer_active = False
            self._final_time   = self._elapsed
            if self._best_time is None or self._elapsed < self._best_time:
                self._best_time = self._elapsed

    def _reset(self) -> None:
        self.player.reset(*START_TILE)
        self.won           = False
        self._win_timer    = 0.0
        self._elapsed      = 0.0
        self._timer_active = True
        self._final_time   = None

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self, screen: pygame.Surface) -> None:
        if self._phase == "tutorial":
            self._draw_tutorial(screen)
        else:
            screen.fill(COL_BG)
            self._draw_maze(screen)
            self._draw_player(screen)
            self._draw_hud(screen)
            if self.won:
                self._draw_win(screen)

    def _draw_tutorial(self, screen: pygame.Surface) -> None:
        screen.fill(COL_BG)
        # Open floor — outer ring stays as wall, everything inside is walkable
        for row in range(MAZE_ROWS):
            for col in range(MAZE_COLS):
                rect = pygame.Rect(col * TILE, row * TILE, TILE, TILE)
                if row == 0 or row == MAZE_ROWS - 1 or col == 0 or col == MAZE_COLS - 1:
                    pygame.draw.rect(screen, COL_WALL, rect)
                    pygame.draw.rect(screen, COL_WALL_EDGE, rect, 1)
                else:
                    pygame.draw.rect(screen, COL_FLOOR, rect)
        self._draw_player(screen)
        self._draw_hud(screen)
        self._draw_tutorial_overlay(screen)

    def _draw_tutorial_overlay(self, screen: pygame.Surface) -> None:
        # Semi-transparent banner at the top of the play area
        banner = pygame.Surface((WIN_W, 58), pygame.SRCALPHA)
        banner.fill((0, 0, 0, 155))
        screen.blit(banner, (0, 0))

        title = self._font_sub.render("FAMILIARIZATION  —  open field", True, (160, 190, 255))
        screen.blit(title, title.get_rect(centerx=WIN_W // 2, y=6))

        hint = self._font_small.render(
            "Move freely to get used to your brain signals.   S = skip", True, COL_DIMTEXT)
        screen.blit(hint, hint.get_rect(centerx=WIN_W // 2, y=34))

        # Countdown (turns gold in the last 10 s)
        remaining = max(0.0, self._tutorial_time)
        cdown_col = COL_GOAL if remaining < 10.0 else COL_TEXT
        cdown = self._font_timer.render(f"{int(remaining) + 1:2d} s", True, cdown_col)
        screen.blit(cdown, cdown.get_rect(right=WIN_W - 16, y=10))

    def _draw_maze(self, screen: pygame.Surface) -> None:
        for row in range(MAZE_ROWS):
            for col in range(MAZE_COLS):
                rect = pygame.Rect(col * TILE, row * TILE, TILE, TILE)
                if MAZE_GRID[row][col] == 1:
                    pygame.draw.rect(screen, COL_WALL, rect)
                    pygame.draw.rect(screen, COL_WALL_EDGE, rect, 1)
                else:
                    pygame.draw.rect(screen, COL_FLOOR, rect)

        # Goal tile — pulsing gold square
        gx, gy = GOAL_TILE
        pulse  = 0.5 + 0.5 * np.sin(self._goal_pulse)
        margin = int(6 + pulse * 3)
        goal_rect = pygame.Rect(
            gx * TILE + margin, gy * TILE + margin,
            TILE - margin * 2,  TILE - margin * 2,
        )
        pygame.draw.rect(screen, COL_GOAL,     goal_rect, border_radius=5)
        pygame.draw.rect(screen, COL_GOAL_RIM, goal_rect, 2, border_radius=5)

        # Start tile — subtle marker
        sx, sy = START_TILE
        s_rect = pygame.Rect(sx * TILE + 10, sy * TILE + 10, TILE - 20, TILE - 20)
        pygame.draw.rect(screen, COL_START, s_rect, border_radius=3)

    def _draw_player(self, screen: pygame.Surface) -> None:
        with _lock:
            probs = self.probs.copy()

        if np.allclose(probs, probs[0]):
            predicted_class = None
            confidence = 0.0
        else:
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])
        color = _confidence_color(confidence)

        px = int(self.player.x)
        py = int(self.player.y)

        # Soft glow halo
        glow_r    = PLAYER_R + 8
        glow_surf = pygame.Surface((glow_r * 2 + 2, glow_r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, max(0, int(90 * confidence))),
                           (glow_r + 1, glow_r + 1), glow_r)
        screen.blit(glow_surf, (px - glow_r - 1, py - glow_r - 1))

        # Main body
        pygame.draw.circle(screen, color, (px, py), PLAYER_R)

        # Specular highlight
        pygame.draw.circle(screen, (255, 255, 255), (px - 3, py - 4),
                           max(2, PLAYER_R // 4))

        # Direction arrow growing with confidence
        arrow_color = (255, 255, 255) if confidence > 0.5 else (180, 180, 180)
        if predicted_class is not None:
            _draw_arrow(screen, arrow_color, px, py, predicted_class,
                        size=int(6 + confidence * 12))

    def _draw_hud(self, screen: pygame.Surface) -> None:
        with _lock:
            probs = self.probs.copy()

        if np.allclose(probs, probs[0]):
            predicted_class = None
            confidence = 0.0
        else:
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])
        color = _confidence_color(confidence)

        # Background panel
        pygame.draw.rect(screen, COL_HUD_BG, (0, MAZE_H, WIN_W, HUD_H))
        pygame.draw.line(screen, COL_HUD_LINE, (0, MAZE_H), (WIN_W, MAZE_H), 1)

        y0 = MAZE_H + 12

        # --- Left: class label + arrow icon ---
        label = "Idle" if predicted_class is None else CLASS_NAMES[predicted_class]
        label_surf = self._font_big.render(label, True, color)
        screen.blit(label_surf, (18, y0))
        if predicted_class is not None:
            _draw_arrow(screen, color,
                        18 + label_surf.get_width() + 30, y0 + 10,
                        predicted_class, size=12)

        # --- Centre: confidence bar ---
        bar_x, bar_y = 240, y0 + 2
        bar_w, bar_h  = 260, 16
        filled_w = int(bar_w * confidence)

        pygame.draw.rect(screen, (30, 30, 50),   (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        if filled_w > 0:
            pygame.draw.rect(screen, color,      (bar_x, bar_y, filled_w, bar_h), border_radius=3)
        pygame.draw.rect(screen, (70, 70, 110),  (bar_x, bar_y, bar_w, bar_h), 1, border_radius=3)

        screen.blit(
            self._font_small.render(f"confidence  {confidence:.2f}", True, COL_DIMTEXT),
            (bar_x, bar_y + bar_h + 5),
        )

        # --- Mini prob bars (4 classes) ---
        mini_w, mini_h, gap = 26, 10, 5
        mx0 = bar_x + bar_w + 20
        my0 = y0 + 2

        for i, (p, name) in enumerate(zip(probs, ["LH", "RH", "LL", "RL"])):
            bx  = mx0 + i * (mini_w + gap)
            hi = (predicted_class is not None and i == predicted_class)
            pygame.draw.rect(screen, (35, 40, 65) if hi else (30, 30, 50),
                             (bx, my0, mini_w, mini_h), border_radius=2)
            if int(mini_w * p) > 0:
                pygame.draw.rect(screen, (200, 220, 255) if hi else (80, 100, 160),
                                 (bx, my0, int(mini_w * p), mini_h), border_radius=2)
            pygame.draw.rect(screen, (65, 65, 105), (bx, my0, mini_w, mini_h), 1, border_radius=2)
            lbl = self._font_small.render(name, True, COL_TEXT if hi else COL_DIMTEXT)
            screen.blit(lbl, (bx + (mini_w - lbl.get_width()) // 2, my0 + mini_h + 3))

        # --- Right: timer ---
        timer_x = WIN_W - 190
        elapsed = self._final_time if self._final_time is not None else self._elapsed
        timer_col = COL_GOAL if self.won else COL_TEXT
        screen.blit(
            self._font_timer.render(f"TIME  {_fmt_time(elapsed)}", True, timer_col),
            (timer_x, MAZE_H + 10),
        )
        if self._best_time is not None:
            screen.blit(
                self._font_small.render(f"BEST  {_fmt_time(self._best_time)}", True, COL_DIMTEXT),
                (timer_x, MAZE_H + 34),
            )

        # --- Bottom row: prob values ---
        prob_str = "  ".join(
            f"{'>' if predicted_class is not None and i == predicted_class else ' '}{p:.2f}"
            for i, p in enumerate(probs)
        )
        screen.blit(
            self._font_small.render(prob_str, True, COL_DIMTEXT),
            (18, MAZE_H + HUD_H - 20),
        )

    def _draw_win(self, screen: pygame.Surface) -> None:
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        screen.blit(overlay, (0, 0))

        # "YOU WIN!"
        txt  = self._font_win.render("YOU WIN!", True, COL_GOAL)
        screen.blit(txt, txt.get_rect(center=(WIN_W // 2, WIN_H // 2 - 50)))

        # Final time
        if self._final_time is not None:
            time_surf = self._font_sub.render(
                f"Time:  {_fmt_time(self._final_time)}", True, COL_TEXT)
            screen.blit(time_surf, time_surf.get_rect(center=(WIN_W // 2, WIN_H // 2 + 5)))

        # Best time
        if self._best_time is not None:
            best_surf = self._font_sub.render(
                f"Best:  {_fmt_time(self._best_time)}", True, COL_DIMTEXT)
            screen.blit(best_surf, best_surf.get_rect(center=(WIN_W // 2, WIN_H // 2 + 38)))

        # "Resetting…"
        sub = self._font_small.render("Resetting maze…", True, COL_DIMTEXT)
        screen.blit(sub, sub.get_rect(center=(WIN_W // 2, WIN_H // 2 + 72)))


# ---------------------------------------------------------------------------
# Run — starts the game loop on the calling (main) thread
# ---------------------------------------------------------------------------

def run(on_event=None, on_frame=None) -> None:
    """
    Initialise pygame, create the game, and run the 60 fps loop.
    Blocks until the window is closed.

    Parameters
    ----------
    on_event : callable(pygame.event.Event) | None
        Called for every pygame event, after the built-in quit/ESC handling.
        Use this in __main__ to react to keypresses (N, M, arrow keys, …).
    on_frame : callable() | None
        Called once per frame before game.tick().
        Use this to read pygame.key.get_pressed() and push probs via update().
    """
    global _game

    pygame.init()
    pygame.display.set_caption("BCI Maze")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()

    _game = MazeGame()

    while _game.running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _game.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    _game.running = False
                elif event.key == pygame.K_s:
                    _game.skip_tutorial()
            if on_event is not None:
                on_event(event)

        if on_frame is not None:
            on_frame()

        _game.tick(dt)
        _game.draw(screen)
        pygame.display.flip()

    pygame.quit()


# ---------------------------------------------------------------------------
# Standalone keyboard test — no BCI pipeline needed
#
# Controls
# --------
#   Arrow keys  move the player (confidence is fixed)
#   N           decrease confidence by 0.05
#   M           increase confidence by 0.05
#   Q / Escape  quit
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _confidence = [0.85]   # mutable so the closures below can write to it

    def _on_event(event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_n:
            _confidence[0] = round(max(0.0, _confidence[0] - 0.05), 2)
            print(f"confidence → {_confidence[0]:.2f}")
        elif event.key == pygame.K_m:
            _confidence[0] = round(min(1.0, _confidence[0] + 0.05), 2)
            print(f"confidence → {_confidence[0]:.2f}")

    def _on_frame() -> None:
        keys = pygame.key.get_pressed()

        if   keys[pygame.K_LEFT]:  direction = 0
        elif keys[pygame.K_RIGHT]: direction = 1
        elif keys[pygame.K_UP]:    direction = 2
        elif keys[pygame.K_DOWN]:  direction = 3
        else:                      direction = None

        c = _confidence[0]
        if direction is not None:
            if c <= 0.25:
                probs = np.zeros(4, dtype=float)
                others = [i for i in range(4) if i != direction]
                rand_vals = np.random.random(3)
                rand_vals = (1.0 - c) * rand_vals / rand_vals.sum()
                probs[direction] = c
                for i, v in zip(others, rand_vals):
                    probs[i] = v
            else:
                # Put confidence mass on the chosen class, spread the rest evenly
                probs = np.full(4, (1.0 - c) / 3.0)
                probs[direction] = c
        else:
            # No key held → idle
            probs = np.zeros(4, dtype=float)

        update(probs)

    print("Arrow keys to move  |  N = lower confidence  |  M = raise confidence")
    print(f"Starting confidence: {_confidence[0]:.2f}")

    run(on_event=_on_event, on_frame=_on_frame)
