"""
platform.py — 3D tilting-platform neurofeedback game.

Public interface
----------------
    update(probs: np.ndarray) -> None
        probs: length-4 float array, probabilities summing to 1.0
               index 0 = left hand, 1 = right hand, 2 = right leg, 3 = left leg

The first call to update() initialises the pygame/OpenGL window.
Every subsequent call advances physics and renders one frame.

Standalone test (keyboard mode)
--------------------------------
    python platform.py

Keys
----
    ← left arrow  : left hand   (class 0)
    → right arrow : right hand  (class 1)
    ↑ up arrow    : right leg   (class 2)
    ↓ down arrow  : left leg    (class 3)
    no key        : equal probs → zero net torque
    P             : toggle perturbation
    Q / Esc       : quit
"""

from __future__ import annotations

import math
import random
import sys
import time

import numpy as np
import pygame
from pygame.locals import (
    DOUBLEBUF, OPENGL, QUIT, KEYDOWN,
    K_ESCAPE, K_q, K_p, K_LEFT, K_RIGHT, K_UP, K_DOWN,
)
from OpenGL.GL import (
    GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
    GL_LINEAR, GL_LINE_LOOP, GL_LINES, GL_MODELVIEW, GL_ONE_MINUS_SRC_ALPHA,
    GL_POINTS, GL_PROJECTION, GL_QUADS, GL_RGBA, GL_SRC_ALPHA, GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_UNSIGNED_BYTE,
    glBegin, glBindTexture, glBlendFunc, glClear, glClearColor, glColor3f,
    glColor4f, glDeleteTextures, glDisable, glEnable, glEnd, glGenTextures,
    glLineWidth, glLoadIdentity, glMatrixMode, glMultMatrixf, glOrtho,
    glPointSize, glPopMatrix, glPushMatrix, glTexCoord2f, glTexImage2D,
    glTexParameteri, glVertex2f, glVertex3f,
)
from OpenGL.GLU import gluLookAt, gluPerspective

# ── window / display constants ─────────────────────────────────────────────────

WIN_W, WIN_H = 960, 720
TARGET_FPS   = 60

# ── physics constants ──────────────────────────────────────────────────────────

PLATFORM_HALF = 1.0           # corners at (±1, ±1, 0) on the flat platform

# Torque added per frame = confidence × TORQUE_SCALE
TORQUE_SCALE   = 0.035        # rad / frame at confidence=1.0
MAX_ANGLE      = math.radians(40)

# Stability threshold: both |θ| below this → "stable"
STABLE_EPS     = math.radians(3.5)
IDLE_BEFORE_PERTURB = 3.0     # seconds of stability before a perturbation fires
PERTURB_KICK   = math.radians(18)

# Tilt axes (unit vectors in 3-D)
# θ1 tilts around y = -x  →  axis direction (1, -1, 0)/√2
# θ2 tilts around y =  x  →  axis direction (1,  1, 0)/√2
_AXIS1 = np.array([ 1.0, -1.0, 0.0]) / math.sqrt(2.0)
_AXIS2 = np.array([ 1.0,  1.0, 0.0]) / math.sqrt(2.0)

# class → (axis_index, sign)
# Class 0 left_hand  : front-left ↑  → +θ2
# Class 1 right_hand : front-right ↑ → +θ1
# Class 2 right_leg  : back-right ↑  → −θ2
# Class 3 left_leg   : back-left ↑   → −θ1
_TORQUE_MAP = [
    (1, +1.0),
    (0, +1.0),
    (1, -1.0),
    (0, -1.0),
]

# corner names and their (x, y) positions on the flat platform
_CORNERS = {
    "fr": ( 1.0,  1.0),   # front-right
    "fl": (-1.0,  1.0),   # front-left
    "br": ( 1.0, -1.0),   # back-right
    "bl": (-1.0, -1.0),   # back-left
}

# which corner each class pushes up
_CLASS_CORNER = {0: "fl", 1: "fr", 2: "br", 3: "bl"}

_CLASS_LABELS = ["Left Hand", "Right Hand", "Right Leg", "Left Leg"]
_CORNER_LABEL = {
    "fr": "front-right ↑",
    "fl": "front-left ↑",
    "br": "back-right ↑",
    "bl": "back-left ↑",
}

# ── module-level state ─────────────────────────────────────────────────────────

class _State:
    # physics
    theta:    list[float]         = [0.0, 0.0]   # [θ1, θ2]
    omega:    list[float]         = [0.0, 0.0]   # angular velocity (rad/frame)

    # perturbation
    perturb_enabled: bool         = True
    stable_since:    float | None = None
    active_event:    dict | None  = None          # {start, kick}

    # scoring
    events: list[dict]            = None          # type: ignore[assignment]

    # bookkeeping
    last_probs: np.ndarray | None = None
    last_time:  float             = 0.0

    def __init__(self):
        self.theta          = [0.0, 0.0]
        self.omega          = [0.0, 0.0]
        self.perturb_enabled = True
        self.stable_since   = None
        self.active_event   = None
        self.events         = []
        self.last_probs     = np.full(4, 0.25)
        self.last_time      = time.monotonic()


_st = _State()

# pygame / GL handles
_initialized  = False
_screen:  pygame.Surface | None = None
_clock:   pygame.time.Clock | None = None
_hud_surf: pygame.Surface | None = None
_font_lg: pygame.font.Font | None = None
_font_md: pygame.font.Font | None = None
_font_sm: pygame.font.Font | None = None


# ── public API ─────────────────────────────────────────────────────────────────

def update(probs: np.ndarray) -> None:
    """
    Advance physics by one step and render one frame.

    Parameters
    ----------
    probs : np.ndarray, shape (4,)
        Class probabilities [left_hand, right_hand, right_leg, left_leg],
        summing to 1.0.
    """
    _ensure_init()
    _st.last_probs = np.asarray(probs, dtype=float)

    # physics
    now = time.monotonic()
    _st.last_time = now

    predicted = int(np.argmax(probs))
    confidence = float(probs[predicted])

    axis_idx, sign = _TORQUE_MAP[predicted]
    _st.omega[axis_idx] += sign * confidence * TORQUE_SCALE

    for i in range(2):
        _st.theta[i] += _st.omega[i]
        _st.theta[i]  = float(np.clip(_st.theta[i], -MAX_ANGLE, MAX_ANGLE))

    _tick_perturbation(now)
    _tick_scoring(now)

    # render
    _handle_events()
    _render_frame()


# ── physics helpers ────────────────────────────────────────────────────────────

def _is_stable() -> bool:
    return abs(_st.theta[0]) < STABLE_EPS and abs(_st.theta[1]) < STABLE_EPS


def _tick_perturbation(now: float) -> None:
    if _is_stable():
        if _st.stable_since is None:
            _st.stable_since = now
    else:
        _st.stable_since = None

    if (
        _st.perturb_enabled
        and _st.active_event is None
        and _st.stable_since is not None
        and now - _st.stable_since >= IDLE_BEFORE_PERTURB
    ):
        kick0 = random.choice([-1, 0, 1]) * PERTURB_KICK
        kick1 = random.choice([-1, 0, 1]) * PERTURB_KICK
        if kick0 == 0 and kick1 == 0:
            kick0 = PERTURB_KICK * random.choice([-1, 1])
        _st.theta[0] = float(np.clip(_st.theta[0] + kick0, -MAX_ANGLE, MAX_ANGLE))
        _st.theta[1] = float(np.clip(_st.theta[1] + kick1, -MAX_ANGLE, MAX_ANGLE))
        _st.stable_since  = None
        _st.active_event  = {"start": now, "kick": (kick0, kick1)}


def _tick_scoring(now: float) -> None:
    if _st.active_event is None:
        return
    if _is_stable():
        elapsed = now - _st.active_event["start"]
        score   = max(5.0, 100.0 * math.exp(-0.25 * elapsed))
        _st.events.append({"elapsed": elapsed, "score": score})
        _st.active_event = None
        _st.stable_since = now


def avg_score() -> float:
    if not _st.events:
        return 0.0
    return sum(e["score"] for e in _st.events) / len(_st.events)


# ── OpenGL initialisation ──────────────────────────────────────────────────────

def _ensure_init() -> None:
    global _initialized, _screen, _clock, _hud_surf
    global _font_lg, _font_md, _font_sm
    if _initialized:
        return
    pygame.init()
    pygame.font.init()
    _screen   = pygame.display.set_mode((WIN_W, WIN_H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Neurofeedback Platform")
    _clock    = pygame.time.Clock()
    _hud_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    _font_lg  = pygame.font.SysFont("monospace", 22, bold=True)
    _font_md  = pygame.font.SysFont("monospace", 16)
    _font_sm  = pygame.font.SysFont("monospace", 13)

    glClearColor(0.06, 0.07, 0.13, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, WIN_W / WIN_H, 0.1, 50.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0.0, -5.0, 3.5,
              0.0,  0.0, 0.0,
              0.0,  0.0, 1.0)

    _initialized = True


# ── event handling ─────────────────────────────────────────────────────────────

def _handle_events() -> None:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN:
            if event.key in (K_ESCAPE, K_q):
                pygame.quit()
                sys.exit()
            elif event.key == K_p:
                _st.perturb_enabled = not _st.perturb_enabled
                _st.stable_since    = None


# ── 3-D rendering ──────────────────────────────────────────────────────────────

def _rot_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """4×4 row-major rotation matrix for the given axis (unit) and angle."""
    c, s = math.cos(angle), math.sin(angle)
    t    = 1.0 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y,  0.0],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x,  0.0],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c,    0.0],
        [0.0,          0.0,           0.0,           1.0],
    ], dtype=np.float32)


def _push_platform_transform() -> None:
    R1 = _rot_matrix(_AXIS1, _st.theta[0])
    R2 = _rot_matrix(_AXIS2, _st.theta[1])
    R  = (R2 @ R1).T          # OpenGL column-major = transpose of row-major
    glMultMatrixf(R)


def _draw_platform() -> None:
    tilt = math.sqrt(_st.theta[0]**2 + _st.theta[1]**2)
    t    = min(1.0, tilt / MAX_ANGLE)
    r    = min(1.0, t * 2.2)
    g    = max(0.0, 1.0 - t * 1.8)
    b    = 0.15

    sz = PLATFORM_HALF

    glPushMatrix()
    _push_platform_transform()

    # filled quad
    glBegin(GL_QUADS)
    glColor4f(r, g, b, 0.80)
    glVertex3f(-sz, -sz, 0.0)
    glVertex3f( sz, -sz, 0.0)
    glVertex3f( sz,  sz, 0.0)
    glVertex3f(-sz,  sz, 0.0)
    glEnd()

    # subdivision grid on the surface
    glLineWidth(1.0)
    glBegin(GL_LINES)
    glColor4f(1.0, 1.0, 1.0, 0.20)
    N = 5
    for i in range(1, N):
        v = -sz + 2.0 * sz * i / N
        glVertex3f(v,  -sz, 0.003); glVertex3f(v,   sz, 0.003)
        glVertex3f(-sz, v,  0.003); glVertex3f( sz,  v,  0.003)
    glEnd()

    # border
    glLineWidth(2.5)
    glBegin(GL_LINE_LOOP)
    glColor4f(1.0, 1.0, 1.0, 0.9)
    glVertex3f(-sz, -sz, 0.004)
    glVertex3f( sz, -sz, 0.004)
    glVertex3f( sz,  sz, 0.004)
    glVertex3f(-sz,  sz, 0.004)
    glEnd()

    # tilt-axis lines (diagonal cross)
    glLineWidth(1.5)
    glBegin(GL_LINES)
    glColor4f(0.6, 0.6, 1.0, 0.5)   # θ1 axis: y = -x
    glVertex3f(-sz,  sz, 0.005); glVertex3f( sz, -sz, 0.005)
    glColor4f(1.0, 0.6, 0.6, 0.5)   # θ2 axis: y = x
    glVertex3f(-sz, -sz, 0.005); glVertex3f( sz,  sz, 0.005)
    glEnd()

    # corner dots — highlight active corner
    probs     = _st.last_probs if _st.last_probs is not None else np.full(4, 0.25)
    predicted = int(np.argmax(probs))
    active_cn = _CLASS_CORNER[predicted]

    for cname, (cx, cy) in _CORNERS.items():
        is_active = (cname == active_cn)
        glPointSize(14.0 if is_active else 7.0)
        glBegin(GL_POINTS)
        if is_active:
            glColor4f(1.0, 1.0, 0.0, 1.0)
        else:
            glColor4f(0.75, 0.75, 0.75, 0.65)
        glVertex3f(cx * sz, cy * sz, 0.006)
        glEnd()

    glPopMatrix()


def _draw_ground_grid() -> None:
    sz = 3.0
    N  = 12
    glLineWidth(1.0)
    glBegin(GL_LINES)
    glColor4f(0.25, 0.27, 0.38, 0.55)
    for i in range(N + 1):
        v = -sz + 2.0 * sz * i / N
        glVertex3f(v,  -sz, -0.02); glVertex3f(v,   sz, -0.02)
        glVertex3f(-sz, v,  -0.02); glVertex3f( sz,  v,  -0.02)
    glEnd()


def _draw_world_axes() -> None:
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glColor3f(0.9, 0.25, 0.25); glVertex3f(0,0,0); glVertex3f(0.5, 0.0, 0.0)
    glColor3f(0.25, 0.9, 0.25); glVertex3f(0,0,0); glVertex3f(0.0, 0.5, 0.0)
    glColor3f(0.25, 0.4,  0.9); glVertex3f(0,0,0); glVertex3f(0.0, 0.0, 0.5)
    glEnd()


# ── HUD helpers ────────────────────────────────────────────────────────────────

def _txt(surf: pygame.Surface, text: str, pos: tuple,
         font: pygame.font.Font, color: tuple = (220, 220, 220)) -> None:
    surf.blit(font.render(text, True, color), pos)


def _draw_hud(surf: pygame.Surface) -> None:
    surf.fill((0, 0, 0, 0))
    W, H = surf.get_size()
    now  = time.monotonic()
    probs = _st.last_probs if _st.last_probs is not None else np.full(4, 0.25)

    # ── tilt angles ──
    t1 = math.degrees(_st.theta[0])
    t2 = math.degrees(_st.theta[1])
    _txt(surf, f"\u03b81: {t1:+6.1f}\u00b0   \u03b82: {t2:+6.1f}\u00b0", (14, 14), _font_md)

    # ── status ──
    stable = _is_stable()
    s_col  = (70, 255, 100) if stable else (255, 165, 50)
    s_lbl  = "STABLE" if stable else "TILTED"
    _txt(surf, s_lbl, (14, 38), _font_lg, s_col)

    # ── score ──
    n   = len(_st.events)
    avg = avg_score()
    _txt(surf, f"Score: {avg:5.1f}  ({n} recoveries)", (14, 68), _font_md)

    # ── perturbation toggle ──
    p_on  = _st.perturb_enabled
    p_col = (90, 240, 90) if p_on else (220, 80, 80)
    _txt(surf, f"Perturbation: {'ON ' if p_on else 'OFF'}  [P]",
         (W - 270, 14), _font_md, p_col)

    # ── active perturbation alert ──
    if _st.active_event is not None:
        elapsed = now - _st.active_event["start"]
        _txt(surf, f"!  RECOVER  !   {elapsed:.1f} s",
             (W // 2 - 110, 14), _font_lg, (255, 60, 60))

    # ── idle / next-perturbation progress bar ──
    if _st.stable_since is not None and _st.active_event is None and p_on:
        prog  = min(1.0, (now - _st.stable_since) / IDLE_BEFORE_PERTURB)
        bx, by, bw, bh = W // 2 - 110, H - 34, 220, 11
        pygame.draw.rect(surf, (35, 35, 50),     (bx, by, bw, bh), border_radius=4)
        pygame.draw.rect(surf, (55, 190, 210),   (bx, by, int(bw * prog), bh), border_radius=4)
        _txt(surf, "next perturbation", (bx + 10, by - 17), _font_sm, (100, 190, 210))

    # ── probability bars (top-right) ──
    bar_colors = [
        (80,  120, 255),   # left hand  — blue
        (255, 140,  40),   # right hand — orange
        (60,  210,  80),   # right leg  — green
        (200,  60, 200),   # left leg   — purple
    ]
    bx0, by0, bw, bh = W - 220, 50, 190, 14
    for i, (p, col) in enumerate(zip(probs, bar_colors)):
        oy = by0 + i * (bh + 5)
        pygame.draw.rect(surf, (45, 45, 58), (bx0, oy, bw, bh), border_radius=3)
        pygame.draw.rect(surf, col,           (bx0, oy, int(bw * p), bh), border_radius=3)
        _txt(surf, f"{_CLASS_LABELS[i][:4]} {p:.2f}", (bx0 - 75, oy), _font_sm)

    # ── control mapping (bottom-left) ──
    mx, my = 14, H - 120
    _txt(surf, "Key mapping:", (mx, my), _font_md, (170, 170, 255))
    key_map = [
        ("\u2190 Left ",  0, "Left Hand",  "front-left \u2191"),
        ("\u2192 Right",  1, "Right Hand", "front-right \u2191"),
        ("\u2191 Up   ",  2, "Right Leg",  "back-right \u2191"),
        ("\u2193 Down ",  3, "Left Leg",   "back-left \u2191"),
    ]
    for row, (key, cls, name, corner) in enumerate(key_map):
        y = my + 22 + row * 22
        predicted_cls = int(np.argmax(probs))
        hl = (255, 240, 80) if (predicted_cls == cls and float(probs[cls]) > 0.4) else (190, 190, 190)
        _txt(surf, f"  {key} : {name:<12}  {corner}", (mx, y), _font_sm, hl)


def _blit_hud_as_gl_texture(surf: pygame.Surface) -> None:
    """Upload surf as an OpenGL texture and draw it fullscreen over the 3D scene."""
    tex_bytes = pygame.image.tostring(surf, "RGBA", True)
    w, h      = surf.get_size()

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_bytes)

    # switch to 2-D ortho
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, w, 0, h, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glEnable(GL_TEXTURE_2D)
    glDisable(GL_DEPTH_TEST)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glBegin(GL_QUADS)
    glColor4f(1, 1, 1, 1)
    glTexCoord2f(0, 0); glVertex2f(0, 0)
    glTexCoord2f(1, 0); glVertex2f(w, 0)
    glTexCoord2f(1, 1); glVertex2f(w, h)
    glTexCoord2f(0, 1); glVertex2f(0, h)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

    glDeleteTextures(1, [tex_id])


def _render_frame() -> None:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    _draw_ground_grid()
    _draw_world_axes()
    _draw_platform()

    assert _hud_surf is not None
    _draw_hud(_hud_surf)
    _blit_hud_as_gl_texture(_hud_surf)

    pygame.display.flip()
    if _clock is not None:
        _clock.tick(TARGET_FPS)


# ── standalone keyboard test ───────────────────────────────────────────────────

if __name__ == "__main__":
    _ensure_init()

    while True:
        _handle_events()

        keys = pygame.key.get_pressed()
        if   keys[K_LEFT]:  probs = np.array([1.0, 0.0, 0.0, 0.0])
        elif keys[K_RIGHT]: probs = np.array([0.0, 1.0, 0.0, 0.0])
        elif keys[K_UP]:    probs = np.array([0.0, 0.0, 1.0, 0.0])
        elif keys[K_DOWN]:  probs = np.array([0.0, 0.0, 0.0, 1.0])
        else:               probs = np.array([0.25, 0.25, 0.25, 0.25])

        update(probs)
