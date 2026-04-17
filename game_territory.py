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

FPS        = 60
WIN_W      = 1000
WIN_H      = 720
HUD_H      = 80
VIEWPORT_H = WIN_H - HUD_H   # 640

WORLD_SCALE    = 140          # pixels per LDA unit
TERRITORY_R    = 0.7          # LDA units — fixed territory circle radius
SMOOTH_ALPHA   = 0.15         # EMA weight for projection smoothing (lower = smoother)
CAMERA_MARGIN  = 180          # px from viewport edge before camera scrolls
CENTROID_EMA   = 0.04         # per-frame smoothing for centroid positions during explore phase

N_SUCCESSES          = 5    # holds needed to trigger centroid recalculation
DWELL_REQ            = 3.0  # seconds of continuous hold for one success
GRACE_PERIOD         = 0.5  # seconds outside circle before hold progress resets
CENTROID_MOVE_THRESH = 0.3  # fraction of TERRITORY_R: min displacement to move centroid
EXPLORE_PHASE        = 60.0 # seconds of free exploration before convergence locks in

TRAIL_MAXLEN = 120
TRAIL_FADE   = 8.0          # seconds for trail to fade completely

CUE_DURATION  = 10.0         # seconds per class in auto-cue mode

GRID_SPACING = 1.0          # LDA units between grid lines

FOG_CELL    = 0.22          # LDA units — fog grid cell size (smaller = denser coverage)
FOG_CLEAR_R = 0.12          # radius around each trail point that clears fog

# ─── Colors ───────────────────────────────────────────────────────────────────

CLASS_COLORS: dict[int | None, tuple[int, int, int]] = {
    0:    (100, 149, 237),   # cornflower blue  — left_hand
    1:    (255, 165,   0),   # orange           — right_hand
    2:    ( 50, 205,  50),   # lime green       — left_leg
    3:    (220,  20,  60),   # crimson          — right_leg
    None: (120, 120, 120),   # grey             — rest / unlabeled
}
CLASS_NAMES    = {0: "left_hand", 1: "right_hand", 2: "left_leg", 3: "right_leg", None: "rest"}
CLASS_INITIALS = {0: "L", 1: "R", 2: "l", 3: "r"}

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
_zmq_lock    = threading.Lock()
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
    """
    s = ((ix * 73856093) ^ (iy * 19349663)) & 0xFFFFFFFF

    def _rnd(s: int) -> tuple[int, float]:
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        return s, s / 0xFFFFFFFF

    s, v   = _rnd(s)
    n      = 4 + int(v * 4)   # 4–7 blobs per cell
    blobs: list[tuple[float, float, float, int]] = []
    for _ in range(n):
        s, ox_f = _rnd(s);  ox = (ox_f - 0.5) * 0.9 * FOG_CELL
        s, oy_f = _rnd(s);  oy = (oy_f - 0.5) * 0.9 * FOG_CELL
        s, r_f  = _rnd(s);  r  = 0.05 + r_f * 0.08
        s, a_f  = _rnd(s);  a  = int(80 + a_f * 80)
        blobs.append((ox, oy, r, a))
    return blobs

# ─── Territory state ──────────────────────────────────────────────────────────

class TerritoryState:
    def __init__(self, cls: int) -> None:
        self.cls                     = cls
        self.success_count:    int   = 0
        self._hold_start:      float | None = None
        self._outside_since:   float | None = None
        self._dwell_positions: list[np.ndarray] = []
        self.pulse:            float = 0.0
        self.discovery_pulse:  float | None = None
        self._hold_progress_cached: float = 0.0

    @property
    def hold_progress(self) -> float:
        """0..1 — fraction of DWELL_REQ completed in current hold."""
        return self._hold_progress_cached

    def tick_anim(self, dt: float, game_time: float) -> None:
        self.pulse = (self.pulse + dt * 2.5) % (2 * math.pi)
        if self._hold_start is not None:
            self._hold_progress_cached = min(
                (game_time - self._hold_start) / DWELL_REQ, 1.0)
        else:
            self._hold_progress_cached = 0.0

    def on_sample(self, player_pos: np.ndarray, centroid: np.ndarray,
                  now: float, label) -> bool:
        """
        Called on each new sample.  Returns True when N_SUCCESSES holds are
        completed and the caller should trigger centroid recalculation.
        """
        inside = float(np.linalg.norm(player_pos - centroid)) < TERRITORY_R

        if not inside:
            if self._outside_since is None:
                self._outside_since = now
            elif now - self._outside_since > GRACE_PERIOD:
                self._hold_start = None   # reset hold progress
            return False

        # Player is inside
        self._outside_since = None

        if label == "BLOCK" or (isinstance(label, int) and label != self.cls):
            self._hold_start = None
            return False

        if self._hold_start is None:
            self._hold_start = now

        if now - self._hold_start >= DWELL_REQ:
            self._dwell_positions.append(player_pos.copy())
            self.success_count += 1
            self._hold_start = now   # reset for next hold immediately
            if self.success_count >= N_SUCCESSES:
                return True

        return False

# ─── TerritoryGame ────────────────────────────────────────────────────────────

class TerritoryGame:
    def __init__(self) -> None:
        self.running = True

        self._proj:     np.ndarray | None = None
        self._label:    int | None        = None
        self._new_data  = False

        self._player_pos:    np.ndarray | None = None
        self._raw_proj:      np.ndarray | None = None
        self._smoothed_proj: np.ndarray | None = None
        self._camera:        np.ndarray        = np.zeros(2)
        self._game_time:     float             = 0.0

        # Colored ink trail: (smoothed pos, label, timestamp, separability_score)
        self._trail: collections.deque[tuple[np.ndarray, int | None, float, float]] = \
            collections.deque(maxlen=TRAIL_MAXLEN)

        self._fog_cleared:    set[tuple[int, int]]              = set()
        self._fog_blob_cache: dict[tuple[int, int], list]       = {}
        self._fog_surf:       pygame.Surface                    = pygame.Surface(
            (WIN_W, VIEWPORT_H), pygame.SRCALPHA)

        self._centroids: dict[int, np.ndarray | None] = {c: None for c in range(4)}

        self._territories  = {c: TerritoryState(c) for c in range(4)}

        self._active_label: int | None = None
        self._auto_mode:    bool        = False
        self._cue_elapsed:  float       = 0.0
        self._ambiguous:    bool        = False

        self._lda_method  = "waiting"
        self._fonts_ready = False

    # ── Public ────────────────────────────────────────────────────────────

    def update_centroids(self, centroids: dict[int, np.ndarray | None]) -> None:
        if self._game_time < EXPLORE_PHASE:
            # Exploration phase: EMA-drift each circle toward its buffer centroid.
            # Each class moves independently, so LDA refit jitter (which shifts
            # all projected positions at once) becomes a slow, smooth drift
            # rather than a simultaneous jump.
            for cls, c in centroids.items():
                if c is None:
                    continue
                if self._centroids.get(cls) is None:
                    self._centroids[cls] = c.copy()   # first appearance: snap
                else:
                    self._centroids[cls] = (CENTROID_EMA * c
                                            + (1.0 - CENTROID_EMA) * self._centroids[cls])
        else:
            # Convergence phase: centroids are frozen — only updated via
            # the N_SUCCESSES recalculation inside tick().
            # Seed any class that still has no centroid (late arrivals).
            for cls, c in centroids.items():
                if self._centroids.get(cls) is None and c is not None:
                    self._centroids[cls] = c

    def _push(self, projection: np.ndarray, label: int | None) -> None:
        self._proj     = projection
        self._label    = label
        self._new_data = True

    # ── Helpers ───────────────────────────────────────────────────────────

    def _separability_score(self, pos: np.ndarray) -> float:
        """Min distance to other centroids, normalized by TERRITORY_R. Clamped 0..1."""
        known = [c for c in self._centroids.values() if c is not None]
        if len(known) < 2:
            return 0.0
        dists = sorted(float(np.linalg.norm(pos - c)) for c in known)
        # second smallest = nearest *other* centroid (smallest might be own)
        return float(np.clip(dists[1] / (TERRITORY_R * 3), 0.0, 1.0))

    def _clarity_score(self, cls: int) -> float:
        own = self._centroids.get(cls)
        if own is None:
            return 0.0
        others = [c for k, c in self._centroids.items() if k != cls and c is not None]
        if not others:
            return 0.0
        min_dist = min(float(np.linalg.norm(own - c)) for c in others)
        return float(np.clip(min_dist / (TERRITORY_R * 2), 0.0, 1.0))

    # ── Per-frame tick ────────────────────────────────────────────────────

    def tick(self, dt: float) -> None:
        self._game_time += dt

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

        # Animation ticks (60 Hz) — updates hold_progress each frame
        for terr in self._territories.values():
            terr.tick_anim(dt, self._game_time)

        if not self._new_data:
            return
        self._new_data = False
        proj           = self._proj
        label          = self._label

        if proj is None:
            return

        self._raw_proj = proj.copy()
        if self._smoothed_proj is None:
            self._smoothed_proj = proj.copy()
            self._camera        = proj.copy()   # center camera on first sample
        else:
            self._smoothed_proj = (SMOOTH_ALPHA * proj
                                   + (1.0 - SMOOTH_ALPHA) * self._smoothed_proj)
        smooth = self._smoothed_proj

        self._player_pos = smooth.copy()

        # Deadzone camera: only scroll when player nears the viewport edge
        sx = (smooth[0] - self._camera[0]) * WORLD_SCALE + WIN_W / 2
        sy = -(smooth[1] - self._camera[1]) * WORLD_SCALE + VIEWPORT_H / 2
        if sx < CAMERA_MARGIN:
            self._camera[0] = smooth[0] + (WIN_W / 2 - CAMERA_MARGIN) / WORLD_SCALE
        elif sx > WIN_W - CAMERA_MARGIN:
            self._camera[0] = smooth[0] - (WIN_W / 2 - CAMERA_MARGIN) / WORLD_SCALE
        if sy < CAMERA_MARGIN:
            self._camera[1] = smooth[1] - (VIEWPORT_H / 2 - CAMERA_MARGIN) / WORLD_SCALE
        elif sy > VIEWPORT_H - CAMERA_MARGIN:
            self._camera[1] = smooth[1] + (VIEWPORT_H / 2 - CAMERA_MARGIN) / WORLD_SCALE

        sep_score = self._separability_score(smooth)
        self._trail.append((smooth.copy(), label, self._game_time, sep_score))

        # Fog clearing
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

        self._ambiguous = sum(
            1 for cen in self._centroids.values()
            if cen is not None and float(np.linalg.norm(smooth - cen)) < TERRITORY_R
        ) > 1

        in_explore_phase = self._game_time < EXPLORE_PHASE

        for cls, terr in self._territories.items():
            cen = self._centroids.get(cls)
            if cen is not None:
                if in_explore_phase or self._ambiguous:
                    cue_for_capture = "BLOCK"   # no captures during exploration
                else:
                    cue_for_capture = self._active_label if self._auto_mode else "BLOCK"
                should_recalc = terr.on_sample(smooth, cen, now, cue_for_capture)
                if should_recalc:
                    new_centroid = np.mean(terr._dwell_positions, axis=0)
                    old_centroid = self._centroids[cls]
                    displacement = (float(np.linalg.norm(new_centroid - old_centroid))
                                    if old_centroid is not None else float("inf"))
                    if displacement > CENTROID_MOVE_THRESH * TERRITORY_R:
                        self._centroids[cls] = new_centroid
                        terr.discovery_pulse = self._game_time
                    terr._dwell_positions.clear()
                    terr.success_count = 0


    # ── Draw ──────────────────────────────────────────────────────────────

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(COL_BG)
        cam = self._camera

        self._draw_grid(screen, cam)
        self._draw_fog(screen, cam)
        self._draw_territories(screen, cam)
        self._draw_trail(screen, cam)
        self._draw_player(screen, cam)
        self._draw_cue(screen)
        self._draw_hud(screen)
        self._draw_minimap(screen)

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
            terr    = self._territories[cls]
            color   = CLASS_COLORS[cls]
            cx, cy  = self._w2s(cen, cam)
            is_cued = (cls == self._active_label)

            # Glow
            glow_intensity = 0.75 if is_cued else 0.35
            _draw_glow(screen, color, (cx, cy), r_px,
                       intensity=glow_intensity, layers=4)

            # Interior fill
            fill_a = 28 if is_cued else 18
            fs = pygame.Surface((r_px * 2, r_px * 2), pygame.SRCALPHA)
            pygame.draw.circle(fs, (*color, fill_a), (r_px, r_px), r_px)
            screen.blit(fs, (cx - r_px, cy - r_px))

            # Circle outline
            if is_cued:
                pygame.draw.circle(screen, COL_WHITE, (cx, cy), r_px, 4)
                # Outer halo at r_px + 6, class color, width 2, alpha ~120
                halo_r    = r_px + 6
                halo_pad  = 4
                hs_size   = (halo_r + halo_pad) * 2
                hs = pygame.Surface((hs_size, hs_size), pygame.SRCALPHA)
                pygame.draw.circle(hs, (*color, 120),
                                   (halo_r + halo_pad, halo_r + halo_pad), halo_r, 2)
                screen.blit(hs, (cx - halo_r - halo_pad, cy - halo_r - halo_pad))
            else:
                pygame.draw.circle(screen, color, (cx, cy), r_px, 2)

            # Hold progress arc — always in class color, at r_px + 6
            prog = terr.hold_progress
            if prog > 0:
                _draw_arc_progress(screen, color, (cx, cy), r_px + 6, prog, width=3)

            # Success counter (replaces level badge)
            if self._fonts_ready:
                badge = self._font_small.render(
                    f"{terr.success_count}/{N_SUCCESSES}", True, COL_WHITE)
                screen.blit(badge, (cx - badge.get_width() // 2,
                                    cy - badge.get_height() // 2))
                name_s = self._font_small.render(CLASS_NAMES[cls], True, color)
                screen.blit(name_s, (cx - name_s.get_width() // 2,
                                     cy + r_px + 5))

            # Discovery pulse ripple
            if terr.discovery_pulse is not None:
                age = self._game_time - terr.discovery_pulse
                if age >= 1.5:
                    terr.discovery_pulse = None
                else:
                    frac     = age / 1.5
                    # Expanding ring: radius grows from r_px to r_px * 2.5
                    ripple_r = int(r_px + (r_px * 1.5) * frac)
                    ripple_a = int(200 * (1 - frac))
                    rs_pad   = 6
                    rs_size  = (ripple_r + rs_pad) * 2
                    rs = pygame.Surface((rs_size, rs_size), pygame.SRCALPHA)
                    pygame.draw.circle(rs, (*color, ripple_a),
                                       (ripple_r + rs_pad, ripple_r + rs_pad),
                                       ripple_r, 3)
                    screen.blit(rs, (cx - ripple_r - rs_pad, cy - ripple_r - rs_pad))

                    # Connecting lines to all other known centroids
                    line_alpha = int(150 * (1 - frac))
                    if line_alpha > 0:
                        line_surf = pygame.Surface((WIN_W, VIEWPORT_H), pygame.SRCALPHA)
                        for other_cls, other_cen in self._centroids.items():
                            if other_cls == cls or other_cen is None:
                                continue
                            ox, oy     = self._w2s(other_cen, cam)
                            dist       = float(np.linalg.norm(cen - other_cen))
                            thickness  = max(1, int(dist / TERRITORY_R))
                            other_col  = CLASS_COLORS[other_cls]
                            pygame.draw.line(line_surf,
                                             (*other_col, line_alpha),
                                             (cx, cy), (ox, oy), thickness)
                        screen.blit(line_surf, (0, 0))

    # -- ink trail ---------------------------------------------------------

    def _draw_trail(self, screen: pygame.Surface, cam: np.ndarray) -> None:
        now = self._game_time
        for pos, label, t, sep_score in self._trail:
            age = now - t
            if age >= TRAIL_FADE:
                continue
            alpha_f    = 1.0 - age / TRAIL_FADE
            color      = CLASS_COLORS.get(label, CLASS_COLORS[None])
            sx, sy     = self._w2s(pos, cam)
            r          = max(2, int(4 * alpha_f))
            ds         = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
            base_alpha = 200 * (0.3 + 0.7 * sep_score)
            pygame.draw.circle(ds, (*color, int(base_alpha * alpha_f)),
                               (r + 1, r + 1), r)
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

    # -- HUD ---------------------------------------------------------------

    def _draw_hud(self, screen: pygame.Surface) -> None:
        if not self._fonts_ready:
            return

        pygame.draw.rect(screen, COL_HUD_BG, (0, VIEWPORT_H, WIN_W, HUD_H))
        pygame.draw.line(screen, COL_HUD_LINE,
                         (0, VIEWPORT_H), (WIN_W, VIEWPORT_H), 1)

        box_w, box_h = 90, 52
        gap          = 16
        total_w      = 4 * box_w + 3 * gap
        x0           = (WIN_W - total_w) // 2
        y0           = VIEWPORT_H + (HUD_H - box_h) // 2

        for cls in range(4):
            bx    = x0 + cls * (box_w + gap)
            terr  = self._territories[cls]
            color = CLASS_COLORS[cls]

            # Box border
            pygame.draw.rect(screen, (50, 50, 80),
                             (bx, y0, box_w, box_h), 2, border_radius=4)

            # Class name
            ns = self._font_small.render(CLASS_NAMES[cls], True, color)
            screen.blit(ns, (bx + (box_w - ns.get_width()) // 2, y0 + 4))

            # Success fraction
            frac_s = self._font_big.render(
                f"{terr.success_count}/{N_SUCCESSES}", True, COL_TEXT)
            screen.blit(frac_s, (bx + (box_w - frac_s.get_width()) // 2, y0 + 20))

            # "clarity" label at bottom of box
            cl_label = self._font_small.render("clarity", True, COL_DIMTEXT)
            screen.blit(cl_label, (
                bx + (box_w - cl_label.get_width()) // 2,
                y0 + box_h - cl_label.get_height() - 2))

            # Clarity bar below box
            clarity = self._clarity_score(cls)
            bar_y   = y0 + box_h + 4
            pygame.draw.rect(screen, (30, 30, 50), (bx, bar_y, box_w, 5))
            fill_w = int(box_w * clarity)
            if fill_w > 0:
                pygame.draw.rect(screen, color, (bx, bar_y, fill_w, 5))

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

        # Explore-phase countdown banner
        if self._game_time < EXPLORE_PHASE:
            remaining = EXPLORE_PHASE - self._game_time
            phase_str = f"EXPLORING  {remaining:.0f}s  (S to skip)"
            ps = self._font_big.render(phase_str, True, COL_EXPLORE)
            screen.blit(ps, ((WIN_W - ps.get_width()) // 2, VIEWPORT_H + 8))
            # thin progress bar showing how much of explore phase is done
            bar_w = 300
            bx    = (WIN_W - bar_w) // 2
            by    = VIEWPORT_H + 8 + ps.get_height() + 2
            frac  = self._game_time / EXPLORE_PHASE
            pygame.draw.rect(screen, (30, 30, 60), (bx, by, bar_w, 4))
            pygame.draw.rect(screen, COL_EXPLORE, (bx, by, int(bar_w * frac), 4))

    # -- constellation minimap ---------------------------------------------

    def _draw_minimap(self, screen: pygame.Surface) -> None:
        MAP_W, MAP_H = 160, 160
        MAP_X        = WIN_W - 170
        MAP_Y        = 10
        PAD          = 10

        map_surf = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
        map_surf.fill((10, 10, 25, 180))

        known = {cls: c for cls, c in self._centroids.items() if c is not None}

        if len(known) < 2:
            if self._fonts_ready:
                ws = self._font_small.render("waiting...", True, COL_DIMTEXT)
                map_surf.blit(ws, ((MAP_W - ws.get_width()) // 2,
                                   (MAP_H - ws.get_height()) // 2))
        else:
            positions = list(known.values())
            xs = [float(p[0]) for p in positions]
            ys = [float(p[1]) for p in positions]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            if max_x == min_x:
                max_x += 0.5; min_x -= 0.5
            if max_y == min_y:
                max_y += 0.5; min_y -= 0.5

            usable_w = MAP_W - 2 * PAD
            usable_h = MAP_H - 2 * PAD

            def world_to_map(p: np.ndarray) -> tuple[int, int]:
                mx = PAD + (p[0] - min_x) / (max_x - min_x) * usable_w
                my = PAD + (1.0 - (p[1] - min_y) / (max_y - min_y)) * usable_h
                return int(mx), int(my)

            # Lines between all pairs
            cls_list = list(known.keys())
            for i in range(len(cls_list)):
                for j in range(i + 1, len(cls_list)):
                    ca, cb   = cls_list[i], cls_list[j]
                    pa       = world_to_map(known[ca])
                    pb       = world_to_map(known[cb])
                    sep      = float(np.linalg.norm(known[ca] - known[cb]))
                    thickness = max(1, int(sep / TERRITORY_R))
                    col_a    = CLASS_COLORS[ca]
                    col_b    = CLASS_COLORS[cb]
                    avg_col  = tuple(int((col_a[k] + col_b[k]) // 2) for k in range(3))
                    pygame.draw.line(map_surf, (*avg_col, 180), pa, pb, thickness)

            # Centroid dots with initials
            for cls, centroid in known.items():
                mx, my = world_to_map(centroid)
                color  = CLASS_COLORS[cls]
                pygame.draw.circle(map_surf, (*color, 220), (mx, my), 6)
                if self._fonts_ready:
                    initial = CLASS_INITIALS.get(cls, str(cls))
                    init_s  = self._font_small.render(initial, True, COL_WHITE)
                    map_surf.blit(init_s, (mx - init_s.get_width() // 2,
                                           my - init_s.get_height() // 2))

        # Border
        pygame.draw.rect(map_surf, (60, 60, 100), (0, 0, MAP_W, MAP_H), 1)
        screen.blit(map_surf, (MAP_X, MAP_Y))

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
                elif event.key == pygame.K_s:
                    if _game._game_time < EXPLORE_PHASE:
                        _game._game_time = EXPLORE_PHASE

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
    print("Keys: 1-4 = cue class  |  0 = clear cue  |  R = toggle auto-cue  |  S = skip explore phase  |  Q = quit\n")
    run()
