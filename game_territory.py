"""
game_territory.py — "Capture the Territory" BCI game.

Public interface
----------------
    update(projection: np.ndarray, label: int | None) -> None

        projection : 2-element array [x, y] in LDA projection space
        label      : int 0-3 or None

When run standalone, uses live LDA on ZMQ samples (same approach as
receiver_gui_v2.py) as a stand-in for the real model pipeline projection.

Run the emulator first:
    python -m emulator -d easy

Then:
    python game_territory.py
"""

from __future__ import annotations

import collections
import json
import math
import threading
import time

import numpy as np
import pygame
import zmq

# ─── Config ───────────────────────────────────────────────────────────────────

HOST              = "localhost"
PORT              = 5555
PER_CLASS_FIT_BUF = 40      # rolling raw buffer per class for LDA fitting
CENTROID_BUF_LEN  = 80      # rolling raw buffer per class for centroid
REPROJECT_EVERY   = 10      # refit LDA every N game frames (~6 Hz at 60 fps)
MIN_PER_CLASS     = 8       # min samples per class before LDA is valid

FPS          = 60
WIN_W        = 1000
WIN_H        = 720
HUD_H        = 80
VIEWPORT_H   = WIN_H - HUD_H   # 640

WORLD_SCALE  = 180          # pixels per LDA unit
TERRITORY_R  = 0.7          # LDA units — fixed territory circle radius (+40 %)
SMOOTH_ALPHA = 0.15         # EMA weight for projection smoothing (lower = smoother)

DWELL_REQ    = 3.0          # seconds of stable dwell needed to advance a level
VAR_WINDOW   = 1.5          # seconds of history used for variance computation
MIN_DWELL_PTS = 4           # minimum points before variance is considered valid
# Variance thresholds for level 0→1, 1→2, 2→3
VAR_THRESH   = [0.05, 0.015, 0.004]

TRAIL_MAXLEN = 120
TRAIL_FADE   = 8.0          # seconds for trail to fade completely

EXPLORE_R    = 0.3          # LDA units — dwell radius for exploration detection
EXPLORE_DIST = 1.5          # min distance from all centroids to flag exploration
EXPLORE_DWELL = 4.0         # seconds to flag as candidate cluster
CUE_DURATION  = 8.0         # seconds per class in auto-cue mode

GRID_SPACING = 1.0          # LDA units between grid lines

FOG_CELL    = 0.35          # LDA units — fog grid cell size
FOG_CLEAR_R = 0.30          # radius around each trail point that clears fog

# ─── Colors ───────────────────────────────────────────────────────────────────

# Matches receiver_gui_v2.py CLASS_COLORS (converted to RGB tuples)
CLASS_COLORS: dict[int | None, tuple[int, int, int]] = {
    0:    (100, 149, 237),   # cornflower blue  — left_hand
    1:    (255, 165,   0),   # orange           — right_hand
    2:    ( 50, 205,  50),   # lime green       — left_leg
    3:    (220,  20,  60),   # crimson          — right_leg
    None: (120, 120, 120),   # grey             — rest / unlabeled
}
CLASS_NAMES = {0: "left_hand", 1: "right_hand", 2: "left_leg", 3: "right_leg", None: "rest"}

COL_BG       = ( 10,  10,  18)
COL_GRID     = ( 22,  22,  38)
COL_HUD_BG   = ( 15,  15,  28)
COL_HUD_LINE = ( 45,  45,  75)
COL_TEXT     = (200, 200, 220)
COL_DIMTEXT  = (100, 100, 130)
COL_GOLD     = (255, 215,   0)
COL_WHITE    = (255, 255, 255)
COL_EXPLORE  = (160, 160, 255)

# ─── ZMQ receiver ─────────────────────────────────────────────────────────────

_buffer:       collections.deque = collections.deque(maxlen=300)
_fit_buf:      dict[int, collections.deque] = {c: collections.deque(maxlen=PER_CLASS_FIT_BUF) for c in range(4)}
_centroid_buf: dict[int, collections.deque] = {c: collections.deque(maxlen=CENTROID_BUF_LEN)  for c in range(4)}
_zmq_lock  = threading.Lock()
_zmq_running = True


def _receiver_thread() -> None:
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{HOST}:{PORT}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, 500)
    print(f"[territory] Connected to tcp://{HOST}:{PORT}")
    while _zmq_running:
        try:
            msg = json.loads(sock.recv_string())
            entry = {
                "data":  np.array(msg["data"], dtype=float),
                "label": msg["label"],
                "idx":   msg["sample_idx"],
            }
            with _zmq_lock:
                _buffer.append(entry)
                if msg["label"] is not None:
                    _fit_buf[msg["label"]].append(entry)
                    _centroid_buf[msg["label"]].append(entry)
        except zmq.Again:
            pass
    sock.close()
    ctx.term()

# ─── LDA Projection ───────────────────────────────────────────────────────────

class LDAProjection:
    """
    2-component LDA projection, same algorithm as receiver_gui_v2.py.
    Refitted every REPROJECT_EVERY frames; sign-aligned to prevent flipping.
    Falls back to PCA when fewer than MIN_PER_CLASS samples per class.
    """

    def __init__(self) -> None:
        self.components: np.ndarray | None = None   # (2, D)
        self._mean:      np.ndarray | None = None
        self.method = "waiting"

    def refit(self, fit_buf: dict) -> None:
        Xs, ys = [], []
        for cls, entries in fit_buf.items():
            if len(entries) >= MIN_PER_CLASS:
                data = np.array([e["data"] for e in entries])
                Xs.append(data)
                ys.extend([cls] * len(data))
        if len(Xs) < 2:
            return
        X    = np.vstack(Xs)
        y    = np.array(ys)
        mean = X.mean(axis=0)
        Xc   = X - mean
        try:
            new_comp  = self._lda_components(Xc, y)
            self.method = "LDA"
        except Exception:
            _, _, Vt  = np.linalg.svd(Xc, full_matrices=False)
            new_comp  = Vt[:2]
            self.method = "PCA"
        if self.components is not None:
            for i in range(2):
                if np.dot(new_comp[i], self.components[i]) < 0:
                    new_comp[i] *= -1
        self.components = new_comp
        self._mean      = mean

    def project(self, X: np.ndarray) -> np.ndarray:
        """(N, D) → (N, 2).  Returns zeros if not yet fitted."""
        if self.components is None or self._mean is None:
            return np.zeros((len(X), 2))
        return (X - self._mean) @ self.components.T

    @staticmethod
    def _lda_components(Xc: np.ndarray, y: np.ndarray) -> np.ndarray:
        classes  = np.unique(y)
        n_total  = len(Xc)
        n_dims   = Xc.shape[1]
        S_W = np.zeros((n_dims, n_dims))
        for c in classes:
            d = Xc[y == c]; d = d - d.mean(axis=0)
            S_W += d.T @ d
        grand_mean = Xc.mean(axis=0)
        S_B = np.zeros((n_dims, n_dims))
        for c in classes:
            n_c = (y == c).sum()
            diff = (Xc[y == c].mean(axis=0) - grand_mean).reshape(-1, 1)
            S_B += n_c * (diff @ diff.T)
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        keep     = min(len(classes) * 4, len(s), n_total - 1)
        W        = Vt[:keep].T / (s[:keep] + 1e-8)
        S_W_w    = W.T @ S_W @ W + np.eye(keep) * 1e-6
        S_B_w    = W.T @ S_B @ W
        evals, evecs = np.linalg.eig(np.linalg.solve(S_W_w, S_B_w))
        evals, evecs = evals.real, evecs.real
        top2     = evecs[:, np.argsort(evals)[::-1][:2]].T
        comp     = top2 @ W.T
        return comp / (np.linalg.norm(comp, axis=1, keepdims=True) + 1e-12)

# ─── Module-level singleton + public interface ────────────────────────────────

_game: "TerritoryGame | None" = None
_game_lock = threading.Lock()


def update(projection: np.ndarray, label: int | None) -> None:
    """
    Push a new 2D projection coordinate into the running game.
    Safe to call from any thread.
    """
    if _game is not None:
        with _game_lock:
            _game._push(np.asarray(projection, dtype=float), label)

# ─── Drawing helpers ──────────────────────────────────────────────────────────

def _draw_glow(screen: pygame.Surface, color: tuple,
               center: tuple, radius: int, intensity: float,
               layers: int = 5) -> None:
    """Soft circular glow.  intensity ∈ [0, 1]."""
    if intensity <= 0:
        return
    gc = layers * 14 + radius + 2
    surf_size = gc * 2
    glow = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
    for i in range(layers, 0, -1):
        r = radius + i * 14
        a = int(intensity * 55 / i)
        pygame.draw.circle(glow, (*color, a), (gc, gc), r)
    screen.blit(glow, (int(center[0]) - gc, int(center[1]) - gc))


def _draw_arc_progress(screen: pygame.Surface, color: tuple,
                       center: tuple, radius: int,
                       progress: float, width: int = 3) -> None:
    """Clockwise arc from 12 o'clock, progress ∈ [0, 1]."""
    if progress < 0.01:
        return
    steps  = max(2, int(72 * progress))
    start  = -math.pi / 2                      # 12 o'clock
    end    = start - progress * 2 * math.pi    # clockwise = decreasing angle
    cx, cy = center
    pts = [
        (cx + radius * math.cos(start + (end - start) * i / steps),
         cy + radius * math.sin(start + (end - start) * i / steps))
        for i in range(steps + 1)
    ]
    if len(pts) >= 2:
        pygame.draw.lines(screen, color, False, pts, width)

def _fog_blobs(ix: int, iy: int) -> list[tuple[float, float, float, int]]:
    """
    Deterministic fog blob parameters for cell (ix, iy).
    Returns list of (offset_x, offset_y, radius_world, alpha) tuples.
    Uses a simple LCG so there are zero per-call allocations.
    """
    s = ((ix * 73856093) ^ (iy * 19349663)) & 0xFFFFFFFF

    def _rnd(s: int) -> tuple[int, float]:
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        return s, s / 0xFFFFFFFF

    s, v   = _rnd(s)
    n      = 2 + int(v * 3)          # 2–4 blobs per cell
    blobs: list[tuple[float, float, float, int]] = []
    for _ in range(n):
        s, ox_f = _rnd(s);  ox = (ox_f - 0.5) * 0.9 * FOG_CELL
        s, oy_f = _rnd(s);  oy = (oy_f - 0.5) * 0.9 * FOG_CELL
        s, r_f  = _rnd(s);  r  = 0.10 + r_f * 0.14
        s, a_f  = _rnd(s);  a  = int(80 + a_f * 80)
        blobs.append((ox, oy, r, a))
    return blobs

# ─── Territory state ──────────────────────────────────────────────────────────

class TerritoryState:
    def __init__(self, cls: int) -> None:
        self.cls        = cls
        self.level      = 0
        self.pulse      = 0.0                           # animation phase
        # (game_time, np.ndarray[2]) — positions while player is inside circle
        self._dwell: list[tuple[float, np.ndarray]] = []
        self._good_since: float | None = None          # when variance first met thresh

    # -- called every frame -----------------------------------------------

    def tick_anim(self, dt: float) -> None:
        self.pulse = (self.pulse + dt * 2.5) % (2 * math.pi)

    # -- called when new sample arrives (10 Hz) ---------------------------

    def on_sample(self, player_pos: np.ndarray, centroid: np.ndarray,
                  now: float, label: int | None = None) -> None:
        if self.level >= 3:
            return

        # Label gate: wrong class resets dwell; None is neutral (no advance, no reset)
        if isinstance(label, int) and label != self.cls:
            self._dwell.clear()
            self._good_since = None
            return

        inside = float(np.linalg.norm(player_pos - centroid)) < TERRITORY_R

        if not inside:
            self._dwell.clear()
            self._good_since = None
            return

        # Maintain sliding dwell window (keep last DWELL_REQ + 1 s)
        cutoff = now - (DWELL_REQ + 1.0)
        self._dwell = [(t, p) for t, p in self._dwell if t >= cutoff]
        self._dwell.append((now, player_pos.copy()))

        # Variance over last VAR_WINDOW seconds
        var_cutoff = now - VAR_WINDOW
        var_pts    = [p for t, p in self._dwell if t >= var_cutoff]
        if len(var_pts) >= MIN_DWELL_PTS:
            variance = float(np.mean(np.var(np.array(var_pts), axis=0)))
        else:
            variance = float("inf")

        thresh = VAR_THRESH[self.level]
        if variance < thresh:
            if self._good_since is None:
                self._good_since = now
            elif now - self._good_since >= DWELL_REQ:
                self.level       = min(self.level + 1, 3)
                self._dwell.clear()
                self._good_since = None
        else:
            self._good_since = None   # variance spike resets streak

    def progress(self, now: float) -> float:
        """0..1 — fraction of required dwell time accumulated."""
        if self._good_since is None:
            return 0.0
        return min((now - self._good_since) / DWELL_REQ, 1.0)

# ─── Exploration tracker ──────────────────────────────────────────────────────

class ExplorationTracker:
    def __init__(self) -> None:
        self._ref:         np.ndarray | None = None
        self._dwell_start: float | None      = None
        self.candidates:   list[np.ndarray]  = []
        self.active:       np.ndarray | None = None  # most-recent candidate

    def on_sample(self, player_pos: np.ndarray,
                  centroids: dict[int, np.ndarray], now: float) -> None:
        # Must be far from all known centroids
        if centroids:
            min_dist = min(np.linalg.norm(player_pos - c)
                           for c in centroids.values())
            far = min_dist > EXPLORE_DIST
        else:
            far = False

        if not far:
            self._ref         = None
            self._dwell_start = None
            return

        if self._ref is None:
            self._ref         = player_pos.copy()
            self._dwell_start = now
            return

        if np.linalg.norm(player_pos - self._ref) > EXPLORE_R:
            # Drifted out — reset reference
            self._ref         = player_pos.copy()
            self._dwell_start = now
            return

        # Still inside the dwell ball
        if now - self._dwell_start >= EXPLORE_DWELL:
            center = self._ref.copy()
            is_new = all(np.linalg.norm(center - c) > EXPLORE_R * 2
                         for c in self.candidates)
            if is_new:
                self.candidates.append(center)
                self.active = center
                print(f"[explore] Candidate cluster at "
                      f"({center[0]:.2f}, {center[1]:.2f})  "
                      f"total={len(self.candidates)}")
            self._dwell_start = now   # reset so we don't spam

    @property
    def dwell_progress(self) -> tuple[np.ndarray | None, float]:
        """(reference_pos, progress 0..1) for current dwell."""
        if self._ref is None or self._dwell_start is None:
            return None, 0.0
        p = min((time.monotonic() - self._dwell_start) / EXPLORE_DWELL, 1.0)
        return self._ref, p

# ─── TerritoryGame ────────────────────────────────────────────────────────────

class TerritoryGame:
    def __init__(self) -> None:
        self.running = True

        # Injected by update() / _push()
        self._proj:     np.ndarray | None = None
        self._label:    int | None        = None
        self._new_data  = False

        # Game state
        self._player_pos:    np.ndarray | None = None
        self._raw_proj:      np.ndarray | None = None   # unsmoothed projection
        self._smoothed_proj: np.ndarray | None = None   # EMA-smoothed projection
        self._camera:        np.ndarray        = np.zeros(2)
        self._game_time:     float             = 0.0

        # Colored ink trail: (smoothed pos, label, timestamp)
        self._trail: collections.deque[tuple[np.ndarray, int | None, float]] = \
            collections.deque(maxlen=TRAIL_MAXLEN)

        # Fog of war
        self._fog_cleared:    set[tuple[int, int]]              = set()
        self._fog_blob_cache: dict[tuple[int, int], list]       = {}
        self._fog_surf:       pygame.Surface                    = pygame.Surface(
            (WIN_W, VIEWPORT_H), pygame.SRCALPHA)

        # Per-class centroid positions in 2D (updated by main loop)
        self._centroids: dict[int, np.ndarray | None] = {c: None for c in range(4)}

        self._territories  = {c: TerritoryState(c) for c in range(4)}
        self._explore      = ExplorationTracker()

        self._active_label: int | None = None   # cued intent (manual or auto)
        self._auto_mode:    bool        = False
        self._cue_elapsed:  float       = 0.0

        self._lda_method = "waiting"
        self._fonts_ready = False

    # ── Public: called from main loop ─────────────────────────────────────

    def update_centroids(self, centroids: dict[int, np.ndarray | None]) -> None:
        self._centroids = centroids

    # ── Called by module-level update() (possibly from another thread) ────

    def _push(self, projection: np.ndarray, label: int | None) -> None:
        self._proj    = projection
        self._label   = label
        self._new_data = True

    # ── Per-frame tick ────────────────────────────────────────────────────

    def tick(self, dt: float) -> None:
        self._game_time += dt

        # Auto-cue cycling
        if self._auto_mode:
            self._cue_elapsed += dt
            if self._cue_elapsed >= CUE_DURATION:
                others = [c for c in range(4) if c != self._active_label]
                self._active_label = int(np.random.choice(others))
                self._cue_elapsed  = 0.0

        if not self._fonts_ready:
            self._font_big   = pygame.font.SysFont("monospace", 18, bold=True)
            self._font_small = pygame.font.SysFont("monospace", 13)
            self._font_cue   = pygame.font.SysFont("monospace", 32, bold=True)
            self._fonts_ready = True

        # Animation ticks (60 Hz)
        for terr in self._territories.values():
            terr.tick_anim(dt)

        if not self._new_data:
            return
        self._new_data  = False
        proj            = self._proj
        label           = self._label

        if proj is None:
            return

        # EMA smoothing — raw stored separately, smoothed drives everything
        self._raw_proj = proj.copy()
        if self._smoothed_proj is None:
            self._smoothed_proj = proj.copy()
        else:
            self._smoothed_proj = (SMOOTH_ALPHA * proj
                                   + (1.0 - SMOOTH_ALPHA) * self._smoothed_proj)
        smooth = self._smoothed_proj

        self._player_pos = smooth.copy()
        self._camera     = smooth.copy()          # camera follows player
        self._trail.append((smooth.copy(), label, self._game_time))

        # Permanently clear fog cells within FOG_CLEAR_R of this trail point
        ix0 = math.floor((smooth[0] - FOG_CLEAR_R) / FOG_CELL)
        ix1 = math.ceil( (smooth[0] + FOG_CLEAR_R) / FOG_CELL)
        iy0 = math.floor((smooth[1] - FOG_CLEAR_R) / FOG_CELL)
        iy1 = math.ceil( (smooth[1] + FOG_CLEAR_R) / FOG_CELL)
        for _ix in range(ix0, ix1 + 1):
            for _iy in range(iy0, iy1 + 1):
                if ((smooth[0] - _ix * FOG_CELL) ** 2
                        + (smooth[1] - _iy * FOG_CELL) ** 2 <= FOG_CLEAR_R ** 2):
                    self._fog_cleared.add((_ix, _iy))

        now = self._game_time
        valid_centroids = {c: v for c, v in self._centroids.items()
                           if v is not None}

        for cls, terr in self._territories.items():
            cen = self._centroids.get(cls)
            if cen is not None:
                terr.on_sample(smooth, cen, now, self._active_label)

        self._explore.on_sample(smooth, valid_centroids, now)

    # ── Draw ──────────────────────────────────────────────────────────────

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(COL_BG)
        cam = self._camera

        self._draw_grid(screen, cam)
        self._draw_fog(screen, cam)
        self._draw_exploration(screen, cam)
        self._draw_territories(screen, cam)
        self._draw_trail(screen, cam)
        self._draw_player(screen, cam)
        self._draw_cue(screen)
        self._draw_hud(screen)

    # -- coordinate helper -------------------------------------------------

    def _w2s(self, pos: np.ndarray, cam: np.ndarray) -> tuple[int, int]:
        """World LDA coords → screen pixels (y-axis flipped)."""
        sx = int((pos[0] - cam[0]) * WORLD_SCALE + WIN_W / 2)
        sy = int(-(pos[1] - cam[1]) * WORLD_SCALE + VIEWPORT_H / 2)
        return sx, sy

    # -- grid --------------------------------------------------------------

    def _draw_grid(self, screen: pygame.Surface, cam: np.ndarray) -> None:
        hw = WIN_W      / 2 / WORLD_SCALE + GRID_SPACING
        hh = VIEWPORT_H / 2 / WORLD_SCALE + GRID_SPACING

        x = math.floor((cam[0] - hw) / GRID_SPACING) * GRID_SPACING
        while x <= cam[0] + hw:
            sx, _ = self._w2s(np.array([x, cam[1]]), cam)
            pygame.draw.line(screen, COL_GRID, (sx, 0), (sx, VIEWPORT_H))
            x += GRID_SPACING

        y = math.floor((cam[1] - hh) / GRID_SPACING) * GRID_SPACING
        while y <= cam[1] + hh:
            _, sy = self._w2s(np.array([cam[0], y]), cam)
            pygame.draw.line(screen, COL_GRID, (0, sy), (WIN_W, sy))
            y += GRID_SPACING

    # -- fog of war --------------------------------------------------------

    def _draw_fog(self, screen: pygame.Surface, cam: np.ndarray) -> None:
        """
        Procedural fog-of-war drawn as overlapping semi-transparent blobs.
        Cells within FOG_CLEAR_R of any trail point are permanently cleared.
        Blob parameters are cached per cell so the LCG runs at most once per cell.
        """
        hw = WIN_W      / 2 / WORLD_SCALE + FOG_CELL * 2
        hh = VIEWPORT_H / 2 / WORLD_SCALE + FOG_CELL * 2

        ix_min = math.floor((cam[0] - hw) / FOG_CELL)
        ix_max = math.ceil( (cam[0] + hw) / FOG_CELL)
        iy_min = math.floor((cam[1] - hh) / FOG_CELL)
        iy_max = math.ceil( (cam[1] + hh) / FOG_CELL)

        self._fog_surf.fill((0, 0, 0, 0))

        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                if (ix, iy) in self._fog_cleared:
                    continue
                if (ix, iy) not in self._fog_blob_cache:
                    self._fog_blob_cache[(ix, iy)] = _fog_blobs(ix, iy)
                for ox, oy, r_world, alpha in self._fog_blob_cache[(ix, iy)]:
                    wp     = np.array([ix * FOG_CELL + ox, iy * FOG_CELL + oy])
                    sx, sy = self._w2s(wp, cam)
                    r_px   = max(1, int(r_world * WORLD_SCALE))
                    pygame.draw.circle(self._fog_surf, (90, 95, 130, alpha),
                                       (sx, sy), r_px)

        screen.blit(self._fog_surf, (0, 0))

    # -- territory circles -------------------------------------------------

    def _draw_territories(self, screen: pygame.Surface, cam: np.ndarray) -> None:
        r_px = int(TERRITORY_R * WORLD_SCALE)

        for cls in range(4):
            cen = self._centroids.get(cls)
            if cen is None:
                continue
            terr  = self._territories[cls]
            color = CLASS_COLORS[cls]
            level = terr.level
            cx, cy = self._w2s(cen, cam)

            # Glow — intensity scales with capture level
            _draw_glow(screen, color, (cx, cy), r_px,
                       intensity=0.35 + level * 0.22, layers=4)

            # Interior fill (more opaque at higher levels)
            if level >= 1:
                fill_a = 18 + level * 18
                fs = pygame.Surface((r_px * 2, r_px * 2), pygame.SRCALPHA)
                pygame.draw.circle(fs, (*color, fill_a), (r_px, r_px), r_px)
                screen.blit(fs, (cx - r_px, cy - r_px))

            # Circle outline
            out_color = COL_GOLD if level >= 3 else color
            out_width = 3 if level >= 3 else 2
            pygame.draw.circle(screen, out_color, (cx, cy), r_px, out_width)

            # Level-3 pulsing ring
            if level >= 3:
                pulse = 0.5 + 0.5 * math.sin(terr.pulse)
                pr    = r_px + int(8 + pulse * 12)
                pa    = int(170 * pulse)
                ps    = pygame.Surface(((pr + 4) * 2, (pr + 4) * 2), pygame.SRCALPHA)
                pygame.draw.circle(ps, (*COL_GOLD, pa), (pr + 4, pr + 4), pr, 3)
                screen.blit(ps, (cx - pr - 4, cy - pr - 4))

            # Progress arc (stability indicator)
            if level < 3:
                prog = terr.progress(self._game_time)
                if prog > 0:
                    arc_col = COL_GOLD if level == 2 else COL_WHITE
                    _draw_arc_progress(screen, arc_col, (cx, cy),
                                       r_px + 6, prog, width=4)

            # Level badge in circle center
            if self._fonts_ready:
                badge_col = COL_GOLD if level == 3 else COL_DIMTEXT
                badge = self._font_small.render(f"L{level}", True, badge_col)
                screen.blit(badge, (cx - badge.get_width() // 2,
                                    cy - badge.get_height() // 2))
                # Class name below circle
                name_s = self._font_small.render(CLASS_NAMES[cls], True, color)
                screen.blit(name_s, (cx - name_s.get_width() // 2,
                                     cy + r_px + 5))

    # -- ink trail ---------------------------------------------------------

    def _draw_trail(self, screen: pygame.Surface, cam: np.ndarray) -> None:
        now = self._game_time
        for pos, label, t in self._trail:
            age = now - t
            if age >= TRAIL_FADE:
                continue
            alpha_f = 1.0 - age / TRAIL_FADE
            color   = CLASS_COLORS.get(label, CLASS_COLORS[None])
            sx, sy  = self._w2s(pos, cam)
            r       = max(2, int(4 * alpha_f))
            ds      = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(ds, (*color, int(200 * alpha_f)), (r + 1, r + 1), r)
            screen.blit(ds, (sx - r - 1, sy - r - 1))

    # -- player dot --------------------------------------------------------

    def _draw_player(self, screen: pygame.Surface, cam: np.ndarray) -> None:
        if self._player_pos is None:
            return
        sx, sy = self._w2s(self._player_pos, cam)
        color  = CLASS_COLORS.get(self._label, CLASS_COLORS[None])

        _draw_glow(screen, color, (sx, sy), 10, 0.85, layers=3)
        pygame.draw.circle(screen, color, (sx, sy), 10)
        pygame.draw.circle(screen, COL_WHITE, (sx - 3, sy - 3), 3)   # specular

    # -- cue overlay -------------------------------------------------------

    def _draw_cue(self, screen: pygame.Surface) -> None:
        """'NOW: CLASS NAME' banner at top of viewport; countdown bar in auto mode."""
        if not self._fonts_ready or self._active_label is None:
            return
        cls   = self._active_label
        color = CLASS_COLORS[cls]
        label = CLASS_NAMES[cls].upper().replace("_", " ")
        surf  = self._font_cue.render(f"NOW:  {label}", True, color)
        screen.blit(surf, ((WIN_W - surf.get_width()) // 2, 12))

        if self._auto_mode:
            progress = max(0.0, (CUE_DURATION - self._cue_elapsed) / CUE_DURATION)
            bar_w, bar_h = 220, 6
            bx = (WIN_W - bar_w) // 2
            by = 12 + surf.get_height() + 5
            pygame.draw.rect(screen, (35, 35, 55), (bx, by, bar_w, bar_h),
                             border_radius=3)
            filled = int(bar_w * progress)
            if filled > 0:
                pygame.draw.rect(screen, color, (bx, by, filled, bar_h),
                                 border_radius=3)

    # -- exploration -------------------------------------------------------

    def _draw_exploration(self, screen: pygame.Surface, cam: np.ndarray) -> None:
        now = self._game_time
        r_px = int(EXPLORE_R * WORLD_SCALE)

        # All logged candidates — faint permanent ring
        for cand in self._explore.candidates:
            sx, sy = self._w2s(cand, cam)
            cs = pygame.Surface(((r_px + 4) * 2, (r_px + 4) * 2), pygame.SRCALPHA)
            pygame.draw.circle(cs, (*COL_EXPLORE, 45),
                               (r_px + 4, r_px + 4), r_px, 2)
            screen.blit(cs, (sx - r_px - 4, sy - r_px - 4))

        # Active candidate — pulsing ring
        if self._explore.active is not None:
            sx, sy  = self._w2s(self._explore.active, cam)
            pulse   = 0.5 + 0.5 * math.sin(now * 4.0)
            pr      = r_px + int(10 * pulse)
            pa      = int(200 * pulse)
            ps      = pygame.Surface(((pr + 4) * 2, (pr + 4) * 2), pygame.SRCALPHA)
            pygame.draw.circle(ps, (*COL_EXPLORE, pa), (pr + 4, pr + 4), pr, 3)
            screen.blit(ps, (sx - pr - 4, sy - pr - 4))

        # In-progress dwell arc
        ref_pos, dwell_prog = self._explore.dwell_progress
        if ref_pos is not None and dwell_prog > 0.05:
            sx, sy = self._w2s(ref_pos, cam)
            _draw_arc_progress(screen, COL_EXPLORE, (sx, sy),
                               r_px + 10, dwell_prog, width=2)

    # -- HUD ---------------------------------------------------------------

    def _draw_hud(self, screen: pygame.Surface) -> None:
        if not self._fonts_ready:
            return

        pygame.draw.rect(screen, COL_HUD_BG, (0, VIEWPORT_H, WIN_W, HUD_H))
        pygame.draw.line(screen, COL_HUD_LINE,
                         (0, VIEWPORT_H), (WIN_W, VIEWPORT_H), 1)

        # Territory level boxes — centered
        box_w, box_h = 90, 52
        gap          = 16
        total_w      = 4 * box_w + 3 * gap
        x0           = (WIN_W - total_w) // 2
        y0           = VIEWPORT_H + (HUD_H - box_h) // 2

        for cls in range(4):
            bx    = x0 + cls * (box_w + gap)
            terr  = self._territories[cls]
            color = CLASS_COLORS[cls]
            level = terr.level

            # Box border
            border_col = COL_GOLD if level == 3 else (50, 50, 80)
            pygame.draw.rect(screen, border_col,
                             (bx, y0, box_w, box_h), 2, border_radius=4)

            # Filled background proportional to level
            if level > 0:
                fh = int((box_h - 4) * level / 3)
                fs = pygame.Surface((box_w - 4, fh), pygame.SRCALPHA)
                fs.fill((*color, 55))
                screen.blit(fs, (bx + 2, y0 + box_h - 2 - fh))

            # Class name
            ns = self._font_small.render(CLASS_NAMES[cls], True, color)
            screen.blit(ns, (bx + (box_w - ns.get_width()) // 2, y0 + 4))

            # Level number
            lv_col = COL_GOLD if level == 3 else COL_TEXT
            ls = self._font_big.render(f"Lv {level}", True, lv_col)
            screen.blit(ls, (bx + (box_w - ls.get_width()) // 2, y0 + 24))

        # Bottom-left: position + LDA method + mode
        info = (f"{self._lda_method}  "
                f"pos ({self._player_pos[0]:.2f}, {self._player_pos[1]:.2f})"
                if self._player_pos is not None
                else self._lda_method)
        screen.blit(self._font_small.render(info, True, COL_DIMTEXT),
                    (10, VIEWPORT_H + 8))
        mode_str = "Mode: AUTO" if self._auto_mode else "Mode: MANUAL"
        screen.blit(self._font_small.render(mode_str, True, COL_DIMTEXT),
                    (10, VIEWPORT_H + 24))

        # Bottom-right: exploration candidates count
        n = len(self._explore.candidates)
        if n > 0:
            cs = self._font_small.render(
                f"Clusters found: {n}", True, COL_EXPLORE)
            screen.blit(cs, (WIN_W - cs.get_width() - 10, VIEWPORT_H + 8))

        # Exploration progress (bottom-right below)
        _, dp = self._explore.dwell_progress
        if dp > 0.05:
            ep = self._font_small.render(
                f"Explore dwell: {dp * 100:.0f}%", True, COL_EXPLORE)
            screen.blit(ep, (WIN_W - ep.get_width() - 10, VIEWPORT_H + 24))

# ─── Run ─────────────────────────────────────────────────────────────────────

def run() -> None:
    global _game, _zmq_running

    pygame.init()
    pygame.display.set_caption("Capture the Territory — BCI")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()

    _game  = TerritoryGame()
    proj   = LDAProjection()
    frame  = 0

    threading.Thread(target=_receiver_thread, daemon=True).start()

    while _game.running:
        dt    = clock.tick(FPS) / 1000.0
        frame += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _game.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    _game.running = False
                elif event.key == pygame.K_0:
                    _game._active_label = None
                    _game._auto_mode    = False
                elif event.key == pygame.K_1:
                    _game._active_label = 0
                    _game._auto_mode    = False
                elif event.key == pygame.K_2:
                    _game._active_label = 1
                    _game._auto_mode    = False
                elif event.key == pygame.K_3:
                    _game._active_label = 2
                    _game._auto_mode    = False
                elif event.key == pygame.K_4:
                    _game._active_label = 3
                    _game._auto_mode    = False
                elif event.key == pygame.K_r:
                    _game._auto_mode = not _game._auto_mode
                    if _game._auto_mode:
                        _game._active_label = int(np.random.randint(0, 4))
                        _game._cue_elapsed  = 0.0
                    else:
                        _game._active_label = None

        # ── Snapshot ZMQ buffers ─────────────────────────────────────────
        with _zmq_lock:
            if not _buffer:
                current_entry = None
                fit_snap  = {c: list(b) for c, b in _fit_buf.items()}
                cent_snap = {c: list(b) for c, b in _centroid_buf.items()}
            else:
                current_entry = _buffer[-1]
                fit_snap  = {c: list(b) for c, b in _fit_buf.items()}
                cent_snap = {c: list(b) for c, b in _centroid_buf.items()}

        # ── Refit LDA every REPROJECT_EVERY frames ───────────────────────
        if frame % REPROJECT_EVERY == 0:
            proj.refit(fit_snap)
            _game._lda_method = proj.method

        # ── Compute 2D centroids from raw centroid buffers ────────────────
        centroids_2d: dict[int, np.ndarray | None] = {}
        for cls in range(4):
            entries = cent_snap.get(cls, [])
            if len(entries) >= 2:
                X = np.array([e["data"] for e in entries])
                centroids_2d[cls] = proj.project(X).mean(axis=0)
            else:
                centroids_2d[cls] = None
        _game.update_centroids(centroids_2d)

        # ── Project current sample → push to game ────────────────────────
        if current_entry is not None:
            X2 = proj.project(current_entry["data"].reshape(1, -1))[0]
            update(X2, current_entry["label"])

        # ── Game tick + draw ──────────────────────────────────────────────
        _game.tick(dt)
        _game.draw(screen)
        pygame.display.flip()

    _zmq_running = False
    pygame.quit()


# ─── Standalone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("Capture the Territory — BCI Game")
    print("Waiting for emulator on tcp://localhost:5555 …")
    print("Keys: 1-4 = cue class  |  0 = clear cue  |  R = toggle auto-cue  |  Q = quit\n")
    run()
