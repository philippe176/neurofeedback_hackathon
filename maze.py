"""
game.py — Pygame maze game driven by BCI probability output.

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
import time

import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE      = 40       # pixels per maze tile
FPS       = 60
MAX_SPEED = 160      # pixels / second at confidence = 1.0
PLAYER_R  = 12       # player radius in pixels

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
                tile_left  = tx * TILE
                tile_top   = ty * TILE
                near_x = max(tile_left,        min(cx, tile_left + TILE))
                near_y = max(tile_top,         min(cy, tile_top  + TILE))
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
    # Tip of the arrow
    tip  = (cx + dx * size,      cy + dy * size)
    # Two base corners (perpendicular to direction)
    perp = (-dy, dx)
    base_hw = size * 0.55
    b1 = (cx - dx * size * 0.3 + perp[0] * base_hw,
          cy - dy * size * 0.3 + perp[1] * base_hw)
    b2 = (cx - dx * size * 0.3 - perp[0] * base_hw,
          cy - dy * size * 0.3 - perp[1] * base_hw)
    pygame.draw.polygon(surf, color, [tip, b1, b2])


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

    @property
    def tile(self) -> tuple[int, int]:
        """Tile coordinates of the player's centre."""
        return (int(self.x // TILE), int(self.y // TILE))


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class MazeGame:
    def __init__(self) -> None:
        self.probs: np.ndarray = np.full(4, 0.25)
        self.player  = Player(*START_TILE)
        self.running = True
        self.won     = False
        self._win_timer = 0.0

        # Preload fonts once (pygame must be initialised first)
        self._font_big   = pygame.font.SysFont("monospace", 20, bold=True)
        self._font_small = pygame.font.SysFont("monospace", 13)
        self._font_win   = pygame.font.SysFont("monospace", 58, bold=True)
        self._font_sub   = pygame.font.SysFont("monospace", 22)

        # Pulsing goal animation state
        self._goal_pulse = 0.0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        self._goal_pulse = (self._goal_pulse + dt * 3.0) % (2 * np.pi)

        if self.won:
            self._win_timer -= dt
            if self._win_timer <= 0.0:
                self._reset()
            return

        with _lock:
            probs = self.probs.copy()

        predicted_class = int(np.argmax(probs))
        confidence      = float(probs[predicted_class])

        if confidence > 0.0:
            ddx, ddy = DIR_VEC[predicted_class]
            speed = MAX_SPEED * confidence
            self.player.try_move(ddx * speed * dt, ddy * speed * dt)

        if self.player.tile == GOAL_TILE:
            self.won       = True
            self._win_timer = 2.8

    def _reset(self) -> None:
        self.player.reset(*START_TILE)
        self.won        = False
        self._win_timer = 0.0

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(COL_BG)
        self._draw_maze(screen)
        self._draw_player(screen)
        self._draw_hud(screen)
        if self.won:
            self._draw_win(screen)

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

        predicted_class = int(np.argmax(probs))
        confidence      = float(probs[predicted_class])
        color           = _confidence_color(confidence)

        px = int(self.player.x)
        py = int(self.player.y)

        # Soft glow halo
        glow_r   = PLAYER_R + 8
        glow_surf = pygame.Surface((glow_r * 2 + 2, glow_r * 2 + 2), pygame.SRCALPHA)
        glow_col  = (*color, max(0, int(90 * confidence)))
        pygame.draw.circle(glow_surf, glow_col, (glow_r + 1, glow_r + 1), glow_r)
        screen.blit(glow_surf, (px - glow_r - 1, py - glow_r - 1))

        # Main body
        pygame.draw.circle(screen, color, (px, py), PLAYER_R)

        # Specular highlight
        hi_r = max(2, PLAYER_R // 4)
        pygame.draw.circle(screen, (255, 255, 255), (px - 3, py - 4), hi_r)

        # Direction arrow growing with confidence
        arrow_size = int(6 + confidence * 12)
        arrow_color = (255, 255, 255) if confidence > 0.5 else (180, 180, 180)
        _draw_arrow(screen, arrow_color, px, py, predicted_class, arrow_size)

    def _draw_hud(self, screen: pygame.Surface) -> None:
        with _lock:
            probs = self.probs.copy()

        predicted_class = int(np.argmax(probs))
        confidence      = float(probs[predicted_class])
        color           = _confidence_color(confidence)

        # Background panel
        hud_rect = pygame.Rect(0, MAZE_H, WIN_W, HUD_H)
        pygame.draw.rect(screen, COL_HUD_BG, hud_rect)
        pygame.draw.line(screen, COL_HUD_LINE, (0, MAZE_H), (WIN_W, MAZE_H), 1)

        y0 = MAZE_H + 12

        # --- Left section: class label + direction arrow icon ---
        label_surf = self._font_big.render(CLASS_NAMES[predicted_class], True, color)
        screen.blit(label_surf, (18, y0))

        arrow_icon_cx = 18 + label_surf.get_width() + 30
        arrow_icon_cy = y0 + 10
        _draw_arrow(screen, color, arrow_icon_cx, arrow_icon_cy,
                    predicted_class, size=12)

        # --- Centre section: confidence bar ---
        bar_x, bar_y = 240, y0 + 2
        bar_w, bar_h  = 330, 16
        filled_w = int(bar_w * confidence)

        pygame.draw.rect(screen, (30, 30, 50), (bar_x, bar_y, bar_w, bar_h),
                         border_radius=3)
        if filled_w > 0:
            pygame.draw.rect(screen, color, (bar_x, bar_y, filled_w, bar_h),
                             border_radius=3)
        pygame.draw.rect(screen, (70, 70, 110), (bar_x, bar_y, bar_w, bar_h),
                         1, border_radius=3)

        conf_surf = self._font_small.render(
            f"confidence  {confidence:.2f}", True, COL_DIMTEXT)
        screen.blit(conf_surf, (bar_x, bar_y + bar_h + 5))

        # --- Right section: all four probability mini-bars ---
        short = ["LH", "RH", "LL", "RL"]
        mini_w, mini_h, gap = 28, 10, 6
        mx0 = bar_x + bar_w + 28
        my0 = y0 + 2

        for i, (p, name) in enumerate(zip(probs, short)):
            bx = mx0 + i * (mini_w + gap)
            filled = int(mini_w * p)
            hi = (i == predicted_class)
            bg_col   = (30, 30, 50) if not hi else (35, 40, 65)
            bar_col  = (80, 100, 160) if not hi else (200, 220, 255)
            pygame.draw.rect(screen, bg_col,  (bx, my0, mini_w, mini_h),
                             border_radius=2)
            if filled > 0:
                pygame.draw.rect(screen, bar_col, (bx, my0, filled, mini_h),
                                 border_radius=2)
            pygame.draw.rect(screen, (65, 65, 105), (bx, my0, mini_w, mini_h),
                             1, border_radius=2)
            lbl = self._font_small.render(name, True,
                                          COL_TEXT if hi else COL_DIMTEXT)
            screen.blit(lbl, (bx + (mini_w - lbl.get_width()) // 2,
                               my0 + mini_h + 3))

        # Second line HUD: prob values as text
        prob_str = "  ".join(
            f"{'>' if i == predicted_class else ' '}{p:.2f}" for i, p in enumerate(probs)
        )
        prob_surf = self._font_small.render(prob_str, True, COL_DIMTEXT)
        screen.blit(prob_surf, (18, MAZE_H + HUD_H - 20))

    def _draw_win(self, screen: pygame.Surface) -> None:
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        screen.blit(overlay, (0, 0))

        txt  = self._font_win.render("YOU WIN!", True, COL_GOAL)
        rect = txt.get_rect(center=(WIN_W // 2, WIN_H // 2 - 25))
        screen.blit(txt, rect)

        sub  = self._font_sub.render("Resetting maze…", True, COL_TEXT)
        rect2 = sub.get_rect(center=(WIN_W // 2, WIN_H // 2 + 45))
        screen.blit(sub, rect2)


# ---------------------------------------------------------------------------
# Run — starts the game loop on the calling (main) thread
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Initialise pygame, create the game, and run the 60 fps loop.
    Blocks until the window is closed.

    The external pipeline should call ``update(probs)`` from a background
    thread before or after calling this function.
    """
    global _game

    pygame.init()
    pygame.display.set_caption("BCI Maze")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()

    _game = MazeGame()

    while _game.running:
        dt = clock.tick(FPS) / 1000.0   # seconds elapsed since last frame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _game.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    _game.running = False

        _game.tick(dt)
        _game.draw(screen)
        pygame.display.flip()

    pygame.quit()


# ---------------------------------------------------------------------------
# Standalone test  — random probs, no BCI pipeline needed
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def _random_feeder() -> None:
        """Feed random softmax probs at 10 Hz — same rate as the real pipeline."""
        rng = np.random.default_rng()
        while True:
            logits = rng.normal(0.0, 1.8, size=4)
            e      = np.exp(logits - logits.max())
            probs  = e / e.sum()
            update(probs)
            time.sleep(0.1)

    feeder = threading.Thread(target=_random_feeder, daemon=True)
    feeder.start()

    run()   # blocks until the window is closed
