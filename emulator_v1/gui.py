"""
Tkinter-based keyboard control window for the Brain Emulator.
Uses only stdlib (tkinter) — no pygame font dependency.

Controls
--------
  1 / 2 / 3 / 4   — set intention to left_hand / right_hand / left_leg / right_leg
  0                — rest (no intention)
  Arrow keys       — navigate strategy space (hold for continuous movement)
  ESC / Q          — quit
"""

import sys
import termios
import tty
import tkinter as tk
from tkinter import font as tkfont

import numpy as np

from .config import CLASS_COLORS, CLASS_NAMES, DIFFICULTIES
from .emulator import BrainEmulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb(r, g, b) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _lerp_color(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def run_emulator_gui(
    difficulty: str = "medium",
    n_dims: int = 256,
    port: int = 5555,
) -> None:
    emulator = BrainEmulator(difficulty=difficulty, n_dims=n_dims, port=port)

    print(f"\n{'='*55}")
    print(f"  Brain Emulator started")
    print(f"  ZMQ PUB  →  tcp://localhost:{port}")
    print(f"  Difficulty : {difficulty}   |   Dims : {n_dims}   |   Rate : {emulator.sample_rate} Hz")
    print(f"{'='*55}")
    print("  Keys : 1=left_hand  2=right_hand  3=left_leg  4=right_leg  0=rest")
    print("  Arrows : strategy navigation   ESC/Q : quit\n")

    root = tk.Tk()
    root.title(f"Brain Emulator  [{difficulty.upper()}]")
    root.resizable(False, False)
    root.configure(bg="#0c0c14")

    W, H = 660, 540
    canvas = tk.Canvas(root, width=W, height=H, bg="#0c0c14", highlightthickness=0)
    canvas.pack()

    # ---- Font setup ----
    try:
        f_title = tkfont.Font(family="Courier", size=13, weight="bold")
        f_large = tkfont.Font(family="Courier", size=22, weight="bold")
        f_med   = tkfont.Font(family="Courier", size=14)
        f_small = tkfont.Font(family="Courier", size=12)
    except Exception:
        f_title = f_large = f_med = f_small = tkfont.Font(size=12)

    # ---- State ----
    pressed_arrows: set[str] = set()
    ARROW_MAP = {
        "Left":  np.array([-1.0,  0.0]),
        "Right": np.array([ 1.0,  0.0]),
        "Up":    np.array([ 0.0,  1.0]),
        "Down":  np.array([ 0.0, -1.0]),
    }
    CLASS_KEY_MAP = {"1": 0, "2": 1, "3": 2, "4": 3, "0": None}

    strategy_trail: list[np.ndarray] = []
    TRAIL_LEN = 40

    # ---- Key bindings ----
    def on_key_press(event):
        k = event.keysym
        if k in ("Escape", "q", "Q"):
            emulator.close()
            root.destroy()
            sys.exit()
        if k in CLASS_KEY_MAP:
            emulator.set_class(CLASS_KEY_MAP[k])
        if k in ARROW_MAP:
            pressed_arrows.add(k)

    def on_key_release(event):
        pressed_arrows.discard(event.keysym)

    root.bind("<KeyPress>",   on_key_press)
    root.bind("<KeyRelease>", on_key_release)
    root.focus_set()

    # ---- Layout constants ----
    PAD      = 20
    PAD_SIZE = 220        # strategy pad square
    PAD_TOP  = 140
    HALF     = PAD_SIZE // 2 - 14
    PCX      = PAD + PAD_SIZE // 2
    PCY      = PAD_TOP + PAD_SIZE // 2
    STX      = PAD + PAD_SIZE + 20   # stats panel x
    STW      = W - STX - PAD

    # ---- Draw loop (60 fps) ----
    def draw():
        canvas.delete("dynamic")   # clear all dynamic elements each frame

        dyn = emulator.dynamics
        cur = dyn.current_class
        sq  = dyn.strategy_quality
        zc  = dyn.z_class
        zs  = dyn.z_strategy

        # -- Header --
        canvas.create_text(
            PAD, 14, anchor="w",
            text=f"BRAIN EMULATOR  [{difficulty.upper()}]   {n_dims} ch @ {int(emulator.sample_rate)} Hz",
            font=f_title, fill="#a0a0b4", tags="dynamic",
        )

        # -- Intention label --
        label_str = CLASS_NAMES[cur].upper().replace("_", " ") if cur is not None else "REST"
        label_col = _rgb(*CLASS_COLORS[cur])
        canvas.create_text(
            PAD, 45, anchor="w",
            text=f"INTENTION:  {label_str}",
            font=f_large, fill=label_col, tags="dynamic",
        )

        # -- Class buttons row --
        btns = [(0, "1", "Left Hand"), (1, "2", "Right Hand"),
                (2, "3", "Left Leg"),  (3, "4", "Right Leg")]
        bw, bh = 138, 30
        for idx, key_str, name in btns:
            active = (cur == idx)
            x0 = PAD + idx * (bw + 6)
            y0 = 88
            bg  = _rgb(*CLASS_COLORS[idx]) if active else "#232332"
            tc  = "#0a0a0a"               if active else "#646478"
            canvas.create_rectangle(x0, y0, x0+bw, y0+bh, fill=bg, outline="#46466a",
                                    width=1, tags="dynamic")
            canvas.create_text(x0 + bw//2, y0 + bh//2,
                               text=f"[{key_str}] {name}", font=f_small, fill=tc,
                               tags="dynamic")
        # rest button
        rx0 = W - PAD - 72
        rest_bg = "#646478" if cur is None else "#232332"
        canvas.create_rectangle(rx0, 88, rx0+72, 118, fill=rest_bg, outline="#46466a",
                                 width=1, tags="dynamic")
        canvas.create_text(rx0+36, 103, text="[0] Rest", font=f_small,
                           fill="#c8c8c8" if cur is None else "#646478", tags="dynamic")

        # -- Strategy pad --
        canvas.create_rectangle(PAD, PAD_TOP, PAD+PAD_SIZE, PAD_TOP+PAD_SIZE,
                                 fill="#191928", outline="#373760", width=2, tags="dynamic")
        canvas.create_text(PAD, PAD_TOP - 16, anchor="w",
                           text="STRATEGY  (arrow keys)  — target: (0, 0)",
                           font=f_small, fill="#64648c", tags="dynamic")
        # crosshair
        canvas.create_line(PAD+10, PCY, PAD+PAD_SIZE-10, PCY, fill="#282840", tags="dynamic")
        canvas.create_line(PCX, PAD_TOP+10, PCX, PAD_TOP+PAD_SIZE-10, fill="#282840", tags="dynamic")

        # Target ring at (0,0)
        tr = 12
        canvas.create_oval(PCX-tr, PCY-tr, PCX+tr, PCY+tr,
                           outline="#3a3a5a", width=2, tags="dynamic")

        # Disturbance arrow (shows what's pushing z_strategy right now)
        dist = dyn.current_disturbance
        dist_mag = np.linalg.norm(dist)
        if dist_mag > 0.001:
            arrow_scale = HALF * 0.8
            ex = int(PCX + dist[0] * arrow_scale)
            ey = int(PCY - dist[1] * arrow_scale)
            canvas.create_line(PCX, PCY, ex, ey,
                               fill="#c84040", width=3, arrow="last",
                               arrowshape=(10, 12, 4), tags="dynamic")

        # Trail
        for i, s in enumerate(strategy_trail):
            frac = (i + 1) / max(len(strategy_trail), 1)
            col  = _rgb(*_lerp_color((30, 50, 100), (100, 160, 255), frac))
            r    = max(2, int(5 * frac))
            sx   = int(PCX + s[0] * HALF)
            sy   = int(PCY - s[1] * HALF)
            canvas.create_oval(sx-r, sy-r, sx+r, sy+r, fill=col, outline="", tags="dynamic")

        # Current dot
        sx = int(PCX + zs[0] * HALF)
        sy = int(PCY - zs[1] * HALF)
        canvas.create_oval(sx-9, sy-9, sx+9, sy+9, fill="#5090ff", outline="", tags="dynamic")
        canvas.create_oval(sx-4, sy-4, sx+4, sy+4, fill="#c8dcff", outline="", tags="dynamic")

        # -- Stats panel --
        canvas.create_rectangle(STX, PAD_TOP, STX+STW, PAD_TOP+PAD_SIZE,
                                 fill="#191928", outline="#373760", width=2, tags="dynamic")
        canvas.create_text(STX+8, PAD_TOP-16, anchor="w",
                           text="STATE INFO", font=f_small, fill="#64648c", tags="dynamic")

        def _bar(label, value, y_off, color):
            canvas.create_text(STX+10, PAD_TOP+y_off, anchor="w",
                               text=label, font=f_small, fill="#a0a0b4", tags="dynamic")
            bx, by = STX+10, PAD_TOP+y_off+18
            bw = STW - 20
            canvas.create_rectangle(bx, by, bx+bw, by+12, fill="#2d2d3c",
                                    outline="", tags="dynamic")
            fw = max(1, int(bw * np.clip(value, 0, 1)))
            canvas.create_rectangle(bx, by, bx+fw, by+12, fill=color,
                                    outline="", tags="dynamic")
            canvas.create_text(bx + bw + 6, by + 6, anchor="w",
                               text=f"{value:.2f}", font=f_small, fill=color,
                               tags="dynamic")

        sq_col = _rgb(*_lerp_color((200, 60, 60), (60, 200, 60), sq))
        sc     = dyn.class_scale
        sc_col = _rgb(*_lerp_color((200, 60, 60), (60, 200, 60), sc))

        _bar("Strategy quality  (stay near centre)", sq, 14,  sq_col)
        _bar("Signal strength   (hold quality)",     sc, 56,  sc_col)

        # z values
        canvas.create_text(STX+10, PAD_TOP+102, anchor="w",
                           text=f"z_class  [{zc[0]:+.2f}  {zc[1]:+.2f}  {zc[2]:+.2f}]",
                           font=f_small, fill="#8282a0", tags="dynamic")
        canvas.create_text(STX+10, PAD_TOP+122, anchor="w",
                           text=f"z_strat  [{zs[0]:+.2f}  {zs[1]:+.2f}]",
                           font=f_small, fill="#8282a0", tags="dynamic")
        canvas.create_text(STX+10, PAD_TOP+142, anchor="w",
                           text=f"t = {dyn.t:.1f} s",
                           font=f_small, fill="#646482", tags="dynamic")

        # Disturbance magnitude indicator
        dist_mag = float(np.linalg.norm(dyn.current_disturbance))
        dcol = "#dc6432" if dist_mag > 0.01 else "#3a3a5a"
        canvas.create_text(STX+10, PAD_TOP+165, anchor="w",
                           text=f"disturbance  {'ON' if dist_mag > 0.01 else 'off'}  "
                                f"({dyn.current_disturbance[0]:+.2f}, "
                                f"{dyn.current_disturbance[1]:+.2f})",
                           font=f_small, fill=dcol, tags="dynamic")

        # -- Footer --
        fy = PAD_TOP + PAD_SIZE + 16
        canvas.create_text(PAD, fy, anchor="w",
                           text=f"ZMQ PUB  →  tcp://localhost:{emulator.port}   samples: {emulator.sample_count}",
                           font=f_small, fill="#3ca03c", tags="dynamic")
        canvas.create_text(PAD, fy+22, anchor="w",
                           text="[ESC/Q] quit",
                           font=f_small, fill="#464650", tags="dynamic")

        root.after(16, draw)   # ~60 fps

    # ---- Sample loop (10 Hz) ----
    def sample():
        # Apply held arrow keys
        for k in list(pressed_arrows):
            if k in ARROW_MAP:
                emulator.update_strategy(ARROW_MAP[k])

        emulator.step()
        strategy_trail.append(emulator.dynamics.z_strategy.copy())
        if len(strategy_trail) > TRAIL_LEN:
            strategy_trail.pop(0)

        root.after(100, sample)   # 10 Hz

    draw()
    sample()

    # Suppress terminal echo so arrow/number keys don't print garbage
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        root.mainloop()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
