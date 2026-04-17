"""
main.py — BCI Neurofeedback super-menu.

Launches a pygame game-selection screen, starts the Brain Emulator as a
subprocess, wires up a ZMQ → model → game pipeline, then hands off to
whichever game the user chose.

Pipeline
--------
  [emulator subprocess]  ──ZMQ PUB──>  [subscriber thread]
                                              │
                                        ModelInterface.predict(raw_data)
                                              │
                                        shared probs array
                                              │
                                        game.update(probs)   (main thread)
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time

import numpy as np
import pygame
import zmq

# ---------------------------------------------------------------------------
# Model stub — replace predict() with your real model
# ---------------------------------------------------------------------------

class ModelInterface:
    """Slot your trained model in here.  predict() must be thread-safe."""

    def predict(self, raw_data: list[float]) -> np.ndarray:
        """Return length-4 softmax probabilities [left_hand, right_hand, left_leg, right_leg]."""
        return np.full(4, 0.25)


# ---------------------------------------------------------------------------
# Shared state between the ZMQ thread and the game loop
# ---------------------------------------------------------------------------

_probs: np.ndarray = np.full(4, 0.25)
_probs_lock = threading.Lock()


def _zmq_subscriber_thread(model: ModelInterface, port: int = 5555) -> None:
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://localhost:{port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, 500)

    global _probs
    while True:
        try:
            msg  = sock.recv_string()
            data = json.loads(msg)
            raw  = data["data"]
            p    = model.predict(raw)
            with _probs_lock:
                _probs = np.asarray(p, dtype=float)
        except zmq.Again:
            pass
        except Exception:
            pass


def _get_probs() -> np.ndarray:
    with _probs_lock:
        return _probs.copy()


# ---------------------------------------------------------------------------
# Emulator subprocess
# ---------------------------------------------------------------------------

def _start_emulator(port: int = 5555) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "emulator", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)   # give the emulator a moment to bind the socket
    return proc


# ---------------------------------------------------------------------------
# Game launchers
# ---------------------------------------------------------------------------

def _run_maze(model: ModelInterface, port: int) -> None:
    import game_maze

    def on_frame() -> None:
        game_maze.update(_get_probs())

    game_maze.run(on_frame=on_frame)


def _run_platform(model: ModelInterface, port: int) -> None:
    import game_platform

    # Platform swaps indices 2 and 3 vs maze: [lh, rh, right_leg, left_leg]
    try:
        while True:
            p = _get_probs()
            game_platform.update(p[[0, 1, 3, 2]])
    except SystemExit:
        pass


# Fixed 2D positions for the four classes in territory projection space.
# Probabilities are used as weights to interpolate between these corners,
# so a confident prediction pulls the point toward the matching corner.
_TERRITORY_CORNERS = np.array([
    [-1.0,  0.0],   # 0: left hand
    [ 1.0,  0.0],   # 1: right hand
    [ 0.0, -1.0],   # 2: left leg
    [ 0.0,  1.0],   # 3: right leg
], dtype=float)


def _run_territory(model: ModelInterface, port: int) -> None:
    import game_territory

    pygame.init()
    pygame.display.set_caption("Capture the Territory — BCI")
    screen = pygame.display.set_mode((game_territory.WIN_W, game_territory.WIN_H))
    clock  = pygame.time.Clock()

    game_territory._game = game_territory.TerritoryGame()
    game_territory._game.update_centroids(
        {c: _TERRITORY_CORNERS[c] for c in range(4)}
    )

    while game_territory._game.running:
        dt = clock.tick(game_territory.FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_territory._game.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    game_territory._game.running = False

        probs      = _get_probs()
        label      = int(np.argmax(probs))
        projection = probs @ _TERRITORY_CORNERS   # weighted sum of corner positions

        game_territory.update(projection, label)
        game_territory._game.tick(dt)
        game_territory._game.draw(screen)
        pygame.display.flip()

    pygame.quit()


GAMES = {
    "Maze":      _run_maze,
    "Platform":  _run_platform,
    "Territory": _run_territory,
}

# ---------------------------------------------------------------------------
# Pygame menu
# ---------------------------------------------------------------------------

MENU_W, MENU_H = 640, 400
BG_COLOR      = (15,  15,  30)
TITLE_COLOR   = (220, 220, 255)
BTN_COLOR     = (40,  40,  80)
BTN_HOVER     = (70,  70, 140)
BTN_TEXT      = (200, 220, 255)
BTN_W, BTN_H  = 300, 70
BTN_GAP       = 24


def _menu() -> str | None:
    """Show the game-selection screen.  Returns the chosen game name or None on quit."""
    pygame.init()
    screen = pygame.display.set_mode((MENU_W, MENU_H))
    pygame.display.set_caption("BCI Neurofeedback — Select Game")
    clock  = pygame.font.SysFont(None, 28)  # reuse var name below

    font_title = pygame.font.SysFont(None, 52)
    font_btn   = pygame.font.SysFont(None, 36)

    names  = list(GAMES.keys())
    total  = len(names) * (BTN_H + BTN_GAP) - BTN_GAP
    y0     = (MENU_H - total) // 2 + 60   # leave room for title

    buttons: list[tuple[pygame.Rect, str]] = []
    for i, name in enumerate(names):
        rect = pygame.Rect(
            (MENU_W - BTN_W) // 2,
            y0 + i * (BTN_H + BTN_GAP),
            BTN_W,
            BTN_H,
        )
        buttons.append((rect, name))

    chosen = None
    running = True
    while running:
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for rect, name in buttons:
                    if rect.collidepoint(mx, my):
                        chosen = name
                        running = False

        screen.fill(BG_COLOR)

        title_surf = font_title.render("BCI Neurofeedback", True, TITLE_COLOR)
        screen.blit(title_surf, title_surf.get_rect(center=(MENU_W // 2, 60)))

        sub_surf = pygame.font.SysFont(None, 24).render(
            "Choose a game", True, (140, 140, 180))
        screen.blit(sub_surf, sub_surf.get_rect(center=(MENU_W // 2, 105)))

        for rect, name in buttons:
            color = BTN_HOVER if rect.collidepoint(mx, my) else BTN_COLOR
            pygame.draw.rect(screen, color, rect, border_radius=10)
            label = font_btn.render(name, True, BTN_TEXT)
            screen.blit(label, label.get_rect(center=rect.center))

        pygame.display.flip()
        pygame.time.Clock().tick(60)

    pygame.quit()
    return chosen


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    chosen = _menu()
    if chosen is None:
        return

    model     = ModelInterface()
    emulator  = _start_emulator()

    try:
        t = threading.Thread(
            target=_zmq_subscriber_thread,
            args=(model,),
            daemon=True,
        )
        t.start()

        print(f"[menu] launching {chosen}…")
        GAMES[chosen](model, port=5555)
    finally:
        emulator.terminate()
        emulator.wait()
        print("[menu] emulator stopped.")


if __name__ == "__main__":
    main()
