"""
Starter visualization GUI for the Brain Emulator.

Connects to the emulator via ZMQ and shows:
  Left panel  — 2D projection of brain state over time.
                Points fade out with age so you can follow the trajectory.
                Projection axes are recomputed every few seconds and
                sign-aligned with the previous axes so the plot stays stable.
  Right panel — Class prediction over time: each projected point is compared
                to class centroids (same markers as the left plot); probabilities
                are softmax(logits) with logits = −‖x − μ_c‖² so nearer classes win.

Run the emulator first:
    python -m emulator -d easy

Then in a separate terminal:
    python receiver_gui.py
"""

import json
import os
import threading
import collections
import time
from pathlib import Path

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider
import zmq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOST = "localhost"
PORT = 5555

HISTORY_LEN = 300  # samples kept for display (30 s at 10 Hz)
PER_CLASS_FIT_BUF = 40  # last N samples kept PER CLASS for fitting
CENTROID_HISTORY_LEN = (
    80  # per-class history for centroid markers (forgetting)
)
CENTROID_HISTORY_MAX = 600  # hard cap for stored class history (~60 s @ 10 Hz)
PHASE2_AVG_N = 5  # representative point: average of last N samples per class
PRED_DISPLAY_LEN = 100  # samples shown on class-prediction panel
TRAIL_LEN = 40  # length of the bright trajectory line
REPROJECT_EVERY = 10  # recompute projection every N new samples (~1 s)
UPDATE_MS = 150  # plot refresh interval in ms
MIN_PER_CLASS = 8  # need at least this many samples per class before LDA

CLASS_COLORS = {
    0: "#6495ed",  # cornflower blue  — left_hand
    1: "#ffa500",  # orange           — right_hand
    2: "#32cd32",  # lime green       — left_leg
    3: "#dc143c",  # crimson          — right_leg
    None: "#787878",
}
CLASS_NAMES = {
    0: "left_hand",
    1: "right_hand",
    2: "left_leg",
    3: "right_leg",
    None: "rest",
}

# ---------------------------------------------------------------------------
# ZMQ receiver thread
# ---------------------------------------------------------------------------

# Display history — all samples, used for the faded scatter trail
_buffer: collections.deque = collections.deque(maxlen=HISTORY_LEN)
# Per-class fit buffer — last PER_CLASS_FIT_BUF samples per labeled class
# Used to fit the projection so all 4 classes are always represented
_fit_buf: dict[int, collections.deque] = {
    c: collections.deque(maxlen=PER_CLASS_FIT_BUF) for c in range(4)
}
# Per-class centroid buffer — last CENTROID_HISTORY_LEN samples per class
# Used only for drawing class center markers in projected 2D space
_centroid_buf: dict[int, collections.deque] = {
    c: collections.deque(maxlen=CENTROID_HISTORY_MAX) for c in range(4)
}
_lock = threading.Lock()
_meta: dict = {}
_running = True
_proj_mode = "lda"  # "lda" or "pca" — toggled with L / P keys
_centroid_hist_len = CENTROID_HISTORY_LEN
_show_last_hist = False
_phase2_active = False
_phase2_frozen_centroids: dict[int, np.ndarray] | None = None
_phase2_frozen_proj = None
_phase2_pending_freeze = False
_phase2_refit_requested = False
_phase2_data: list[dict] = []
# Last linear projection matching centroid_project (for export); cleared when no affine proj.
_last_proj_mean: np.ndarray | None = None
_last_proj_W: np.ndarray | None = None
_export_centroids: dict[int, np.ndarray] = {}
_export_method: str = ""


def _record_proj(mean: np.ndarray, W: np.ndarray) -> None:
    global _last_proj_mean, _last_proj_W
    _last_proj_mean = np.asarray(mean, dtype=float).copy()
    _last_proj_W = np.asarray(W, dtype=float).copy()


def _clear_proj_record() -> None:
    global _last_proj_mean, _last_proj_W
    _last_proj_mean = None
    _last_proj_W = None


def _receiver_thread():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{HOST}:{PORT}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, 500)
    print(f"Receiver connected to tcp://{HOST}:{PORT}")
    while _running:
        try:
            msg = json.loads(sock.recv_string())
            with _lock:
                entry = {
                    "data": np.array(msg["data"], dtype=float),
                    "label": msg["label"],
                    "idx": msg["sample_idx"],
                }
                _buffer.append(entry)
                if msg["label"] is not None:
                    _fit_buf[msg["label"]].append(entry)
                    _centroid_buf[msg["label"]].append(entry)
                if _phase2_active:
                    _phase2_data.append(entry.copy())
                _meta.update(
                    {
                        k: msg[k]
                        for k in (
                            "sample_idx",
                            "difficulty",
                            "sample_rate",
                            "class_scale",
                            "strategy_quality",
                        )
                        if k in msg
                    }
                )
        except zmq.Again:
            pass
    sock.close()
    ctx.term()


threading.Thread(target=_receiver_thread, daemon=True).start()

# ---------------------------------------------------------------------------
# Stable PCA projection
# ---------------------------------------------------------------------------


class LDAProjection:
    """
    Fits a 2-component LDA projection using the per-class fit buffers.

    LDA uses class labels to find the directions that maximally separate
    classes — unlike PCA which finds directions of maximum total variance
    and ignores labels entirely.

    The fit buffer keeps the last PER_CLASS_FIT_BUF samples per class, so
    all 4 classes are always represented regardless of what the player has
    been pressing recently.

    Falls back to PCA when fewer than MIN_PER_CLASS samples exist per class.
    Signs are aligned with previous axes on each refit to prevent flipping.
    """

    def __init__(self):
        self.components: np.ndarray | None = None  # shape (2, n_dims)
        self._mean: np.ndarray | None = None
        self._since_update = 0
        self.method = "waiting"

    def update(self, fit_buf: dict, X_all: np.ndarray) -> np.ndarray:
        """
        fit_buf: {class_idx: deque of entries} — per-class sample buffers
        X_all:   (N, n_dims) — all history for display
        Returns  (N, 2) projected coordinates
        """
        self._since_update += 1
        if len(X_all) == 0:
            return np.zeros((0, 2))

        if self.components is None or self._since_update >= REPROJECT_EVERY:
            self._refit(fit_buf)
            self._since_update = 0

        if self.components is None:
            # Not enough labeled data yet — fall back to PCA on whatever we have
            self.method = "PCA (warmup)"
            mean = X_all.mean(axis=0)
            try:
                _, _, Vt = np.linalg.svd(X_all - mean, full_matrices=False)
                return (X_all - mean) @ Vt[:2].T
            except Exception:
                return np.zeros((len(X_all), 2))

        return (X_all - self._mean) @ self.components.T

    def _refit(self, fit_buf: dict):
        # Collect fit data per class
        Xs, ys = [], []
        for cls, buf in fit_buf.items():
            if len(buf) >= MIN_PER_CLASS:
                data = np.array([e["data"] for e in buf])
                Xs.append(data)
                ys.extend([cls] * len(data))

        if len(Xs) < 2:
            self.method = "waiting"
            return  # not enough classes yet

        X = np.vstack(Xs)
        y = np.array(ys)
        mean = X.mean(axis=0)
        Xc = X - mean

        try:
            new_comp = self._lda_components(Xc, y)
            self.method = "LDA"
        except Exception:
            # LDA can fail (singular matrices); fall back to PCA
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            new_comp = Vt[:2]
            self.method = "PCA"

        if self.components is not None:
            for i in range(2):
                if np.dot(new_comp[i], self.components[i]) < 0:
                    new_comp[i] *= -1

        self.components = new_comp
        self._mean = mean

    @staticmethod
    def _lda_components(Xc: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return top-2 LDA directions (Fisher's linear discriminant)."""
        classes = np.unique(y)
        n_total = len(Xc)
        n_dims = Xc.shape[1]

        # Within-class scatter S_W
        S_W = np.zeros((n_dims, n_dims))
        for c in classes:
            Xc_c = Xc[y == c]
            diff = Xc_c - Xc_c.mean(axis=0)
            S_W += diff.T @ diff

        # Between-class scatter S_B
        grand_mean = Xc.mean(axis=0)
        S_B = np.zeros((n_dims, n_dims))
        for c in classes:
            n_c = (y == c).sum()
            mu_c = Xc[y == c].mean(axis=0)
            diff = (mu_c - grand_mean).reshape(-1, 1)
            S_B += n_c * (diff @ diff.T)

        # Solve generalised eigenproblem via PCA whitening trick
        # (avoids inverting the full n_dims × n_dims S_W matrix)
        # 1. PCA-whiten to reduce to manageable size
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        keep = min(len(classes) * 4, len(s), n_total - 1)
        Vt_r = Vt[:keep]
        s_r = s[:keep]
        W = Vt_r.T / (s_r + 1e-8)  # whitening matrix

        S_W_w = W.T @ S_W @ W
        S_B_w = W.T @ S_B @ W

        # Eigenvectors of S_W_w^{-1} S_B_w
        S_W_w += np.eye(keep) * 1e-6  # regularise
        evals, evecs = np.linalg.eig(np.linalg.solve(S_W_w, S_B_w))
        evals = evals.real
        evecs = evecs.real
        idx = np.argsort(evals)[::-1]
        top2 = evecs[:, idx[:2]].T  # (2, keep)

        # Map back to original space
        components = top2 @ W.T  # (2, n_dims)
        norms = np.linalg.norm(components, axis=1, keepdims=True)
        return components / (norms + 1e-12)


_proj = LDAProjection()

# ---------------------------------------------------------------------------
# Fisher separability score
# ---------------------------------------------------------------------------


def softmax_from_centroid_distances(
    points: np.ndarray, centroids: dict[int, np.ndarray]
) -> np.ndarray:
    """
    points: (T, 2) projected coordinates
    centroids: class_idx -> (2,) mean in the same 2D space
    Returns (T, 4) row-wise softmax over classes 0..3.
    Logits are negative squared Euclidean distance (missing centroids → −∞).
    """
    T = len(points)
    out = np.zeros((T, 4), dtype=float)
    if T == 0:
        return out

    centers = np.full((4, 2), np.nan, dtype=float)
    have = np.zeros(4, dtype=bool)
    for c in range(4):
        if c in centroids:
            centers[c] = centroids[c]
            have[c] = True

    if not have.any():
        out[:, :] = 0.25
        return out

    d2 = np.full((T, 4), np.inf, dtype=float)
    for c in range(4):
        if not have[c]:
            continue
        diff = points[:, :] - centers[c]
        d2[:, c] = np.sum(diff * diff, axis=1)

    logits = np.where(have, -d2, -1e9)
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / (ex.sum(axis=1, keepdims=True) + 1e-12)


def fisher_score(X2: np.ndarray, labels: np.ndarray) -> float:
    classes = [c for c in np.unique(labels) if c >= 0]
    if len(classes) < 2:
        return 0.0
    mu = X2.mean(axis=0)
    between = sum(
        (labels == c).sum()
        * np.linalg.norm(X2[labels == c].mean(axis=0) - mu) ** 2
        for c in classes
    )
    within = sum(
        np.sum((X2[labels == c] - X2[labels == c].mean(axis=0)) ** 2)
        for c in classes
    )
    return float(between / (within + 1e-9))


# ---------------------------------------------------------------------------
# Plot setup
# ---------------------------------------------------------------------------

fig, (ax_proj, ax_raw) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#0e0e1a")

for ax in (ax_proj, ax_raw):
    ax.set_facecolor("#141422")
    ax.tick_params(colors="#888", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")

ax_proj.set_title("PCA projection", color="#aaa", fontsize=11)
ax_proj.set_xlabel("PC 1", color="#777", fontsize=9)
ax_proj.set_ylabel("PC 2", color="#777", fontsize=9)

ax_raw.set_title(
    "Class prediction (softmax of −dist² to centroids)", color="#aaa", fontsize=11
)
ax_raw.set_xlabel("sample index", color="#777", fontsize=9)
ax_raw.set_ylabel("probability", color="#777", fontsize=9)

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=CLASS_COLORS[i],
        markersize=8,
        label=CLASS_NAMES[i],
        linewidth=0,
    )
    for i in range(4)
] + [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=CLASS_COLORS[None],
        markersize=6,
        label="rest",
        linewidth=0,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#888",
        markeredgecolor="white",
        markersize=12,
        label="centroid",
        linewidth=0,
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        markerfacecolor="none",
        markeredgecolor="white",
        markersize=10,
        label="frozen centroid",
        linewidth=0,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#bbbbbb",
        markeredgecolor="black",
        markersize=9,
        label="phase2 representative",
        linewidth=0,
    ),
    Line2D(
        [0],
        [0],
        marker="x",
        color="white",
        markersize=8,
        label="last points",
        linewidth=0,
    ),
]
ax_proj.legend(
    handles=legend_handles,
    loc="upper right",
    facecolor="#1e1e2e",
    edgecolor="#444",
    labelcolor="#ccc",
    fontsize=8,
)

# tight_layout runs once after all control axes exist (see below). Do not call
# tight_layout inside the animation loop — it breaks Button hit-testing.

# ---------------------------------------------------------------------------
# PCA / LDA toggle button
# ---------------------------------------------------------------------------

ax_btn_save = fig.add_axes([0.02, 0.01, 0.10, 0.055])
ax_btn_save.set_facecolor("#1e1e2e")
_btn_save = Button(
    ax_btn_save, "Save model", color="#1e1e2e", hovercolor="#2e2e4e"
)
_btn_save.label.set_color("#cccccc")
_btn_save.label.set_fontsize(10)
_btn_save.label.set_fontweight("bold")

ax_btn = fig.add_axes([0.13, 0.01, 0.12, 0.055])
ax_btn.set_facecolor("#1e1e2e")
_btn_toggle = Button(
    ax_btn, "Mode: LDA", color="#1e1e2e", hovercolor="#2e2e4e"
)
_btn_toggle.label.set_color("#cccccc")
_btn_toggle.label.set_fontsize(10)
_btn_toggle.label.set_fontweight("bold")


def _on_btn_click(_event):
    global _proj_mode
    if _proj_mode == "lda":
        _proj_mode = "pca"
        _btn_toggle.label.set_text("Mode: PCA")
        print("Projection: PCA")
    else:
        _proj_mode = "lda"
        _btn_toggle.label.set_text("Mode: LDA")
        print("Projection: LDA")
    fig.canvas.draw_idle()


_btn_toggle.on_clicked(_on_btn_click)

# Show last-points overlay toggle button
ax_btn_hist = fig.add_axes([0.26, 0.01, 0.14, 0.055])
ax_btn_hist.set_facecolor("#1e1e2e")
_btn_hist = Button(
    ax_btn_hist, "LastPts: OFF", color="#1e1e2e", hovercolor="#2e2e4e"
)
_btn_hist.label.set_color("#cccccc")
_btn_hist.label.set_fontsize(10)
_btn_hist.label.set_fontweight("bold")


def _on_hist_btn_click(_event):
    global _show_last_hist
    _show_last_hist = not _show_last_hist
    _btn_hist.label.set_text(f"LastPts: {'ON' if _show_last_hist else 'OFF'}")
    fig.canvas.draw_idle()


_btn_hist.on_clicked(_on_hist_btn_click)

# Phase 2 toggle button
ax_btn_phase2 = fig.add_axes([0.41, 0.01, 0.15, 0.055])
ax_btn_phase2.set_facecolor("#1e1e2e")
_btn_phase2 = Button(
    ax_btn_phase2, "Phase 2: OFF", color="#1e1e2e", hovercolor="#2e2e4e"
)
_btn_phase2.label.set_color("#cccccc")
_btn_phase2.label.set_fontsize(10)
_btn_phase2.label.set_fontweight("bold")


def _on_phase2_btn_click(_event):
    global _phase2_active, _phase2_frozen_centroids, _phase2_frozen_proj, _phase2_pending_freeze
    global _phase2_refit_requested, _phase2_data
    _phase2_active = not _phase2_active
    if _phase2_active:
        _phase2_pending_freeze = True
        _phase2_refit_requested = False
        _phase2_data = []
        _btn_phase2.label.set_text("Phase 2: ON")
        print(
            "Phase 2 enabled: collecting phase-2 data + freezing on next frame"
        )
    else:
        _phase2_pending_freeze = False
        _phase2_refit_requested = False
        _phase2_frozen_centroids = None
        _phase2_frozen_proj = None
        _phase2_data = []
        _btn_phase2.label.set_text("Phase 2: OFF")
        print("Phase 2 disabled: returned to live projection/centroids")
    fig.canvas.draw_idle()


_btn_phase2.on_clicked(_on_phase2_btn_click)

# Phase 2 projection refit button (uses all data gathered since Phase 2 ON)
ax_btn_phase2_refit = fig.add_axes([0.57, 0.01, 0.08, 0.055])
ax_btn_phase2_refit.set_facecolor("#1e1e2e")
_btn_phase2_refit = Button(
    ax_btn_phase2_refit, "P2 Refit", color="#1e1e2e", hovercolor="#2e2e4e"
)
_btn_phase2_refit.label.set_color("#cccccc")
_btn_phase2_refit.label.set_fontsize(10)
_btn_phase2_refit.label.set_fontweight("bold")


def _on_phase2_refit_click(_event):
    global _phase2_refit_requested
    if not _phase2_active:
        print("Phase 2 refit ignored: enable Phase 2 first")
        return
    _phase2_refit_requested = True
    print(
        "Phase 2 refit requested: recomputing frozen projection from collected phase-2 data"
    )


_btn_phase2_refit.on_clicked(_on_phase2_refit_click)


def _default_save_path() -> str:
    """Unique path under cwd (works with macosx/Qt backends; no Tk dialog)."""
    return str(
        Path.cwd()
        / f"classic_brain_model_{int(time.time() * 1000) % 1_000_000_000}.npz"
    )


def _pick_save_path_interactive() -> str:
    """Tk file dialog; only reliable when Matplotlib uses the TkAgg backend."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass
    root.lift()
    root.update_idletasks()
    root.update()
    path = filedialog.asksaveasfilename(
        parent=root,
        title="Save classic model",
        defaultextension=".npz",
        filetypes=[("NumPy archive", "*.npz"), ("All files", "*.*")],
        initialfile="classic_brain_model.npz",
    )
    root.destroy()
    return path if isinstance(path, str) else ""


def _on_save_btn_click(_event):
    import traceback

    print("[save] clicked", flush=True)
    if _last_proj_mean is None or _last_proj_W is None:
        print(
            "Save model: no affine projection yet (need LDA/PCA with fit data, "
            "or Phase 2 freeze/refit).",
            flush=True,
        )
        return
    if not _export_centroids:
        print(
            "Save model: no class centroids yet — collect labeled samples first.",
            flush=True,
        )
        return

    path = ""
    env_path = os.environ.get("CLASSIC_MODEL_SAVE_PATH", "").strip()
    if env_path:
        path = env_path
    else:
        backend = matplotlib.get_backend().lower()
        want_dialog = os.environ.get("CLASSIC_MODEL_SAVE_DIALOG", "").strip() in (
            "1",
            "true",
            "yes",
        )
        # macosx / qt backends: a second Tk root often blocks or shows no dialog.
        use_tk_dialog = want_dialog or (backend == "tkagg")
        if use_tk_dialog:
            try:
                path = _pick_save_path_interactive()
            except Exception as exc:
                print(
                    f"Save model: file dialog failed ({exc}); using cwd.",
                    flush=True,
                )
                traceback.print_exc()
                path = _default_save_path()
        else:
            path = _default_save_path()
            print(
                f"[save] backend={backend!r}: writing to {path} "
                "(set CLASSIC_MODEL_SAVE_PATH to choose a path, or "
                "CLASSIC_MODEL_SAVE_DIALOG=1 to try Tk picker on non-Tk backends)",
                flush=True,
            )

    if not path:
        print("Save model cancelled.", flush=True)
        return

    C = np.full((4, 2), np.nan, dtype=float)
    for c in range(4):
        if c in _export_centroids:
            C[c] = _export_centroids[c]

    meta = json.dumps(
        {
            "version": 1,
            "method": _export_method,
            "class_names": [CLASS_NAMES[i] for i in range(4)],
        }
    ).encode("utf-8")

    try:
        np.savez_compressed(
            path,
            mean=_last_proj_mean,
            W=_last_proj_W,
            centroids=C,
            meta_json=np.frombuffer(meta, dtype=np.uint8),
        )
    except Exception:
        print("Save model: writing file failed:")
        traceback.print_exc()
        return
    print(f"Saved classic model to {Path(path).resolve()}", flush=True)
    try:
        fig.canvas.manager.set_window_title(
            f"Saved model → {Path(path).name}"
        )
    except Exception:
        pass


_btn_save.on_clicked(_on_save_btn_click)

# Slider: live control of centroid history length (forgetting)
ax_hist_slider = fig.add_axes([0.15, 0.075, 0.70, 0.03])
ax_hist_slider.set_facecolor("#1e1e2e")
_slider_cent_hist = Slider(
    ax=ax_hist_slider,
    label="Centroid history",
    valmin=5,
    valmax=CENTROID_HISTORY_MAX,
    valinit=CENTROID_HISTORY_LEN,
    valstep=1,
    color="#3f7cac",
)
_slider_cent_hist.label.set_color("#cccccc")
_slider_cent_hist.valtext.set_color("#cccccc")


def _on_cent_hist_change(val):
    global _centroid_hist_len
    _centroid_hist_len = int(val)


_slider_cent_hist.on_changed(_on_cent_hist_change)

# Manual margins: tight_layout fights extra axes and breaks widgets on some backends.
for _ctl_ax in (
    ax_btn_save,
    ax_btn,
    ax_btn_hist,
    ax_btn_phase2,
    ax_btn_phase2_refit,
    ax_hist_slider,
):
    _ctl_ax.set_in_layout(False)
fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.22, wspace=0.14)
fig.canvas.draw_idle()

# ---------------------------------------------------------------------------
# Key handler — L = LDA, P = PCA
# ---------------------------------------------------------------------------


def _on_key(event):
    global _proj_mode, _show_last_hist
    global _phase2_active, _phase2_frozen_centroids, _phase2_frozen_proj, _phase2_pending_freeze
    global _phase2_refit_requested, _phase2_data
    if event.key in ("l", "L"):
        _proj_mode = "lda"
        _btn_toggle.label.set_text("Mode: LDA")
        print("Projection: LDA")
    elif event.key in ("p", "P"):
        _proj_mode = "pca"
        _btn_toggle.label.set_text("Mode: PCA")
        print("Projection: PCA")
    elif event.key in ("h", "H"):
        _show_last_hist = not _show_last_hist
        _btn_hist.label.set_text(
            f"LastPts: {'ON' if _show_last_hist else 'OFF'}"
        )
        print(f"Last points overlay: {'ON' if _show_last_hist else 'OFF'}")
    elif event.key == "2":
        _phase2_active = not _phase2_active
        if _phase2_active:
            _phase2_pending_freeze = True
            _phase2_refit_requested = False
            _phase2_data = []
            _btn_phase2.label.set_text("Phase 2: ON")
            print(
                "Phase 2 enabled: collecting phase-2 data + freezing on next frame"
            )
        else:
            _phase2_pending_freeze = False
            _phase2_refit_requested = False
            _phase2_frozen_centroids = None
            _phase2_frozen_proj = None
            _phase2_data = []
            _btn_phase2.label.set_text("Phase 2: OFF")
            print("Phase 2 disabled: returned to live projection/centroids")
    elif event.key in ("r", "R"):
        if not _phase2_active:
            print("Phase 2 refit ignored: enable Phase 2 first")
        else:
            _phase2_refit_requested = True
            print(
                "Phase 2 refit requested: recomputing frozen projection from collected phase-2 data"
            )
    elif event.key in ("s", "S"):
        _on_save_btn_click(None)


fig.canvas.mpl_connect("key_press_event", _on_key)

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def update(_frame):
    try:
        _update_inner()
    except Exception:
        import traceback

        traceback.print_exc()


def _update_inner():
    global _phase2_pending_freeze, _phase2_frozen_centroids, _phase2_frozen_proj, _phase2_refit_requested
    global _export_centroids, _export_method
    with _lock:
        if len(_buffer) < 2:
            return
        snapshot = list(_buffer)
        fit_buf_snap = {c: list(buf) for c, buf in _fit_buf.items()}
        centroid_buf_snap = {c: list(buf) for c, buf in _centroid_buf.items()}
        phase2_data_snap = list(_phase2_data)

    data = np.array([s["data"] for s in snapshot])
    labels = np.array(
        [s["label"] if s["label"] is not None else -1 for s in snapshot]
    )
    idxs = np.array([s["idx"] for s in snapshot])
    N = len(snapshot)

    # Project: LDA mode uses per-class fit buffers; PCA mode uses recent history.
    # Keep the active projection transform so centroid buffers can use the exact same 2D space.
    centroid_project = None
    if _phase2_active and _phase2_frozen_proj is not None:
        coords = _phase2_frozen_proj(data)
        centroid_project = _phase2_frozen_proj
    elif _proj_mode == "lda":
        coords = _proj.update(fit_buf_snap, data)
        if _proj.components is not None and _proj._mean is not None:
            comp = _proj.components.copy()
            mean = _proj._mean.copy()
            centroid_project = lambda X, c=comp, m=mean: (X - m) @ c.T
            _record_proj(mean, comp)
    else:
        # Plain PCA — recompute on the most recent 100 samples
        n_fit = min(100, N)
        mean = data[-n_fit:].mean(axis=0)
        _, _, Vt = np.linalg.svd(data[-n_fit:] - mean, full_matrices=False)
        coords = (data - mean) @ Vt[:2].T
        _proj.method = "PCA"
        pc = Vt[:2].copy()
        pm = mean.copy()
        centroid_project = lambda X, p=pc, m=pm: (X - m) @ p.T
        _record_proj(pm, pc)

    if centroid_project is None:
        _clear_proj_record()

    # Centroids + Phase 2 refit before drawing so left/right panels match.
    live_centroids: dict[int, np.ndarray] = {}
    for cls in range(4):
        cbuf = centroid_buf_snap.get(cls, [])
        if len(cbuf) == 0:
            continue
        hist_len = max(1, _centroid_hist_len)
        Xc = np.array([e["data"] for e in cbuf[-hist_len:]], dtype=float)
        if centroid_project is not None:
            ccoords = centroid_project(Xc)
        else:
            mask = labels == cls
            if not mask.any():
                continue
            ccoords = coords[mask]
        live_centroids[cls] = ccoords.mean(axis=0)

    if (
        _phase2_active
        and _phase2_pending_freeze
        and centroid_project is not None
    ):
        _phase2_frozen_proj = centroid_project
        _phase2_frozen_centroids = {
            cls: pos.copy() for cls, pos in live_centroids.items()
        }
        _phase2_pending_freeze = False

    if (
        _phase2_active
        and _phase2_refit_requested
        and len(phase2_data_snap) >= 2
    ):
        X_all = np.array([e["data"] for e in phase2_data_snap], dtype=float)
        y_all = np.array(
            [
                e["label"] if e["label"] is not None else -1
                for e in phase2_data_snap
            ],
            dtype=int,
        )
        new_proj = None

        if _proj_mode == "lda":
            fit_data = []
            fit_labels = []
            for cls in range(4):
                cls_mask = y_all == cls
                if cls_mask.sum() >= MIN_PER_CLASS:
                    fit_data.append(X_all[cls_mask])
                    fit_labels.extend([cls] * int(cls_mask.sum()))
            if len(fit_data) >= 2:
                X_fit = np.vstack(fit_data)
                y_fit = np.array(fit_labels)
                fit_mean = X_fit.mean(axis=0)
                X_fit_c = X_fit - fit_mean
                try:
                    comp = LDAProjection._lda_components(X_fit_c, y_fit)
                    new_proj = lambda X, c=comp, m=fit_mean: (X - m) @ c.T
                    _proj.method = "LDA (P2 frozen)"
                    _record_proj(fit_mean, comp)
                except Exception:
                    _, _, Vt = np.linalg.svd(
                        X_all - X_all.mean(axis=0), full_matrices=False
                    )
                    pmean = X_all.mean(axis=0)
                    pcs = Vt[:2]
                    new_proj = lambda X, p=pcs, m=pmean: (X - m) @ p.T
                    _proj.method = "PCA (P2 fallback)"
                    _record_proj(pmean, pcs)
            else:
                _, _, Vt = np.linalg.svd(
                    X_all - X_all.mean(axis=0), full_matrices=False
                )
                pmean = X_all.mean(axis=0)
                pcs = Vt[:2]
                new_proj = lambda X, p=pcs, m=pmean: (X - m) @ p.T
                _proj.method = "PCA (P2 warmup)"
                _record_proj(pmean, pcs)
        else:
            pmean = X_all.mean(axis=0)
            _, _, Vt = np.linalg.svd(X_all - pmean, full_matrices=False)
            pcs = Vt[:2]
            new_proj = lambda X, p=pcs, m=pmean: (X - m) @ p.T
            _proj.method = "PCA (P2 frozen)"
            _record_proj(pmean, pcs)

        if new_proj is not None:
            _phase2_frozen_proj = new_proj
            frozen_centroids: dict[int, np.ndarray] = {}
            for cls in range(4):
                cls_mask = y_all == cls
                if cls_mask.any():
                    ccoords = _phase2_frozen_proj(X_all[cls_mask])
                    frozen_centroids[cls] = ccoords.mean(axis=0)
            _phase2_frozen_centroids = frozen_centroids
            centroid_project = _phase2_frozen_proj
            coords = _phase2_frozen_proj(data)
            print(
                f"Phase 2 refit applied using {len(phase2_data_snap)} gathered samples"
            )

        _phase2_refit_requested = False

    if _phase2_active and _phase2_frozen_centroids:
        centroids_for_pred = _phase2_frozen_centroids
    else:
        centroids_for_pred = live_centroids

    _export_centroids = {
        c: pos.copy() for c, pos in centroids_for_pred.items()
    }
    _export_method = _proj.method

    # ----------------------------------------------------------------
    # Left: stable PCA scatter with fade trail
    # ----------------------------------------------------------------

    ax_proj.cla()
    ax_proj.set_facecolor("#141422")
    for sp in ax_proj.spines.values():
        sp.set_edgecolor("#333355")

    # Age-based alpha: newest sample = 1.0, oldest = 0.05
    ages = np.linspace(0.05, 1.0, N)

    # Draw points per class, alpha encodes age
    for cls_key, color in CLASS_COLORS.items():
        if cls_key is None:
            mask = labels == -1
            size, zorder = 10, 1
        else:
            mask = labels == cls_key
            size, zorder = 18, 2
        if not mask.any():
            continue
        rgba = np.zeros((mask.sum(), 4))
        c = plt.matplotlib.colors.to_rgb(color)
        rgba[:, :3] = c
        rgba[:, 3] = ages[mask]
        ax_proj.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=rgba,
            s=size,
            linewidths=0,
            zorder=zorder,
        )

    # Bright trajectory line for the most recent TRAIL_LEN points
    if N >= 2:
        trail_n = min(TRAIL_LEN, N)
        tx, ty = coords[-trail_n:, 0], coords[-trail_n:, 1]
        ax_proj.plot(
            tx, ty, color="#ffffff", linewidth=0.8, alpha=0.35, zorder=3
        )

    # Current point: large white dot
    current_label = snapshot[-1]["label"]
    current_color = (
        CLASS_COLORS[current_label]
        if current_label in CLASS_COLORS
        else CLASS_COLORS[None]
    )
    dot_color = current_color if _phase2_active else "white"
    ax_proj.scatter(
        coords[-1, 0],
        coords[-1, 1],
        c=dot_color,
        s=70,
        zorder=8 if _phase2_active else 5,
        linewidths=0,
    )

    # Optional overlay: all labeled history points currently in the display buffer.
    if _show_last_hist:
        for cls in range(4):
            cls_idx = np.where(labels == cls)[0]
            if len(cls_idx) == 0:
                continue
            ax_proj.scatter(
                coords[cls_idx, 0],
                coords[cls_idx, 1],
                marker="x",
                c="white",
                s=28,
                linewidths=0.9,
                alpha=0.9,
                zorder=5,
            )

    if _phase2_active and _phase2_frozen_centroids:
        for cls, cmean in _phase2_frozen_centroids.items():
            ax_proj.scatter(
                cmean[0],
                cmean[1],
                c=CLASS_COLORS[cls],
                s=240,
                marker="D",
                edgecolors="white",
                linewidths=2.3,
                zorder=6,
            )
    else:
        for cls, cmean in live_centroids.items():
            ax_proj.scatter(
                cmean[0],
                cmean[1],
                c=CLASS_COLORS[cls],
                s=220,
                edgecolors="white",
                linewidths=2,
                zorder=6,
            )

    # Phase 2 representative points: average last PHASE2_AVG_N same-class samples.
    if _phase2_active and _phase2_frozen_proj is not None:
        for cls in range(4):
            cls_idx = np.where(labels == cls)[0]
            if len(cls_idx) == 0:
                continue
            take = cls_idx[-PHASE2_AVG_N:]
            rep_coords = _phase2_frozen_proj(data[take])
            rep_mean = rep_coords.mean(axis=0)
            ax_proj.scatter(
                rep_mean[0],
                rep_mean[1],
                c=CLASS_COLORS[cls],
                s=120,
                marker="o",
                edgecolors="black",
                linewidths=1.5,
                zorder=7,
            )

    # Separability score on recent labeled samples (reflects current strategy quality)
    n_score = min(100, N)
    score_labels = labels[-n_score:]
    score_coords = coords[-n_score:]
    labeled_mask = score_labels >= 0
    score = (
        fisher_score(score_coords[labeled_mask], score_labels[labeled_mask])
        if labeled_mask.sum() > 8
        else 0.0
    )

    diff = _meta.get("difficulty", "?")
    scale = _meta.get("class_scale", None)
    scale_str = f"   signal={scale:.2f}" if scale is not None else ""
    overlay_str = "   lastPts=ON" if _show_last_hist else ""
    phase2_str = "   [Phase 2]" if _phase2_active else ""
    ax_proj.set_title(
        f"{_proj.method}   sep={score:.2f}{scale_str}   cent_hist={_centroid_hist_len}{overlay_str}{phase2_str}   [{diff}]",
        color="#aaa",
        fontsize=10,
    )
    ax_proj.set_xlabel("PC 1", color="#777", fontsize=9)
    ax_proj.set_ylabel("PC 2", color="#777", fontsize=9)
    ax_proj.tick_params(colors="#888", labelsize=9)
    ax_proj.legend(
        handles=legend_handles,
        loc="upper right",
        facecolor="#1e1e2e",
        edgecolor="#444",
        labelcolor="#ccc",
        fontsize=8,
    )

    # ----------------------------------------------------------------
    # Right: centroid-distance softmax (same 2D coords + centroids as left)
    # ----------------------------------------------------------------
    ax_raw.cla()
    ax_raw.set_facecolor("#141422")
    for sp in ax_raw.spines.values():
        sp.set_edgecolor("#333355")

    n_pred = min(PRED_DISPLAY_LEN, N)
    wcoords = coords[-n_pred:]
    ridxs = idxs[-n_pred:]
    rlabels = labels[-n_pred:]
    probs = softmax_from_centroid_distances(wcoords, centroids_for_pred)

    for i in range(len(ridxs) - 1):
        lbl = rlabels[i]
        if lbl >= 0:
            ax_raw.axvspan(
                ridxs[i],
                ridxs[i + 1],
                color=CLASS_COLORS[lbl],
                alpha=0.06,
                linewidth=0,
            )

    stack_colors = [CLASS_COLORS[c] for c in range(4)]
    ax_raw.stackplot(
        ridxs,
        probs[:, 0],
        probs[:, 1],
        probs[:, 2],
        probs[:, 3],
        colors=stack_colors,
        alpha=0.88,
        linewidths=0,
    )

    ax_raw.set_ylim(0.0, 1.0)
    ax_raw.set_title(
        "Class prediction (softmax of −dist² to centroids)",
        color="#aaa",
        fontsize=11,
    )
    ax_raw.set_xlabel("sample index", color="#777", fontsize=9)
    ax_raw.set_ylabel("stacked P(class)", color="#777", fontsize=9)
    ax_raw.tick_params(colors="#888", labelsize=9)


ani = animation.FuncAnimation(
    fig, update, interval=UPDATE_MS, cache_frame_data=False
)

print(
    f"Visualization running.  History: {HISTORY_LEN} samples  |  Refresh: {UPDATE_MS} ms"
)
print(
    "Keys (click the plot window first):  L = LDA projection   P = PCA projection   2 = Phase 2   R = P2 refit"
)
print(
    f"Centroid history slider range: 5..{CENTROID_HISTORY_MAX} (default {CENTROID_HISTORY_LEN})"
)
print(
    "Last-points overlay: button or H key (shows all labeled history in buffer)"
)
print(
    "Save model: button or S key → .npz in cwd (macosx backend); "
    "CLASSIC_MODEL_SAVE_PATH=/path/file.npz to set path; "
    "CLASSIC_MODEL_SAVE_DIALOG=1 to try Tk file picker."
)
print("Close the window to quit.\n")

try:
    plt.show()
finally:
    _running = False
