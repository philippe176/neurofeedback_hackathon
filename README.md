# Brain Emulator — BCI Neurofeedback Hackathon

A synthetic neural data streamer that emulates what a brain implant would broadcast
during a spinal cord injury rehabilitation session.  Students use it as the data source
for building a real-time neurofeedback GUI.

---

## Background: what is this simulating?

In real BCI sessions with spinal cord injury patients, a multi-electrode array implanted
in motor cortex records the activity of hundreds of neurons simultaneously.  The goal is
to **decode the patient's movement intentions** from this high-dimensional signal so the
patient can receive feedback and learn to produce more discriminable brain states.

The core clinical challenge:

> Some movement intentions (e.g. left hand vs right hand) produce neural signals that
> **overlap heavily** in the raw recording space.  The patient needs neurofeedback —
> a real-time visualisation of their own brain state — to discover mental strategies
> that make the signals separable.  Without feedback, patients cannot find these
> strategies on their own.

This emulator reproduces that challenge without requiring a real patient or implant.

---

## How the emulator works

### The latent state (what the "brain" is doing)

The brain state is represented as an **8-dimensional latent vector** `z`:

```
z = [ z_class    (dims 0-2) ]   — encodes which movement is intended
    [ z_strategy (dims 3-4) ]   — controlled by the arrow keys
    [ z_noise    (dims 5-7) ]   — slow random walk / physiological drift
```

**z_class** is pulled toward one of four class centroids depending on which number key
is held:

| Key | Intention    | Centroid in z_class space     |
|-----|-------------|-------------------------------|
| `1` | Left hand   | `[ 2.0,  0.9,  0.0]`          |
| `2` | Right hand  | `[ 2.0, -0.9,  0.0]`          |
| `3` | Left leg    | `[-2.0,  0.9,  0.0]`          |
| `4` | Right leg   | `[-2.0, -0.9,  0.0]`          |
| `0` | Rest        | decays to origin               |

This structure is intentional: **hand vs leg is easy** (large separation on dim 0),
**left vs right is hard** (small separation on dim 1) — mirroring real clinical data.

### The optimal strategy

The optimal strategy position is always `(0, 0)` in strategy space — the same for
every class.

Each difficulty level has a **per-class temporal disturbance function** that continuously
pushes `z_strategy` away from `(0, 0)`.  The operator must learn to counter-balance
these disturbances with arrow keys to keep `z_strategy` near the origin.

Different difficulty levels use different disturbance patterns (pulses, diagonal kicks,
double-tap rhythms, rotating forces, dual-frequency drives) — see below.

### Intention vs strategy — why they are separate

This distinction is the core of the simulation:

- **Intention** (number key) = *what* movement the patient is trying to perform.
  This sets the class label attached to each data sample.
- **Strategy** (arrow keys) = *how* well the patient is maintaining focus.
  This determines whether the neural signal actually reflects the intention.

In a real session the clinician announces the intention ("now: left hand") — that is
the label.  Whether the patient's brain signal *looks like* left hand depends entirely
on the mental strategy they use.  The neurofeedback teaches them to maintain better
strategies.

Crucially, using the arrow keys to fight disturbances does **not** corrupt the label,
because the label comes only from the number key, not from how many arrows are pressed.

### The observation model (latent → 256 dims)

The observed 256-dimensional signal `x` is generated as:

```
x = A  @  R(z_strategy)  @  diag(scale)  @  z  +  noise
```

- **A** — a fixed random 256×8 mixing matrix (unknown to students).
  Represents the electrode array geometry scrambling the underlying signal.

- **R(z_strategy)** — a strategy-dependent rotation matrix built from three
  [Givens rotations](https://en.wikipedia.org/wiki/Givens_rotation) in planes
  `(0,5)`, `(1,6)`, `(2,7)`.  When `z_strategy` is far from `(0, 0)`, this rotation
  mixes the class signal into the noise dimensions, making classes indistinguishable.
  The projection that recovers the classes must be recomputed as strategy shifts.

- **scale** — suppresses the class signal when strategy is poor.  Implemented as a
  **leaky integrator** with time constant ~3 s: arriving at `(0, 0)` is not enough —
  the signal builds up gradually and requires the strategy to be *held*.
  Leaving too soon causes decay.

- **noise** — Gaussian observation noise added on top.

The result: finding the right 2D projection is not a one-time exercise.  As the
strategy shifts, the subspace in which classes are visible *rotates*, so the
projection must be continuously adapted.  This is exactly what happens in real BCI
co-adaptation.

### Temporal dynamics of the class signal

`class_scale` is a leaky integrator:

```
scale(t+1) = scale(t) + (dt / τ) * (strategy_quality³ - scale(t))
```

where τ = 3 seconds and `strategy_quality = exp(-2.5 * ||z_strategy||)`.

Consequences:

- **Build-up**: arriving at `(0, 0)` gives ~16% scale after 0.5 s,
  ~50% after 2 s, ~82% after 5 s.
- **Decay**: drifting away causes scale to fall back toward zero.
- **Switching classes**: the disturbance pattern changes immediately, so the operator
  must learn the new counter-rhythm AND hold it before the signal builds up.

### Why classes are hard to separate by default

When `z_strategy` is far from `(0, 0)` (arrows not compensating):

- `strategy_quality` is near zero → `class_scale` stays near zero
- The rotation `R(z_strategy)` mixes class signal into noise dims
- PCA / LDA on the observed signal finds noise, not classes

When the operator keeps `z_strategy` near `(0, 0)`:

- `class_scale` rises toward 1.0
- `R(z_strategy) ≈ I` — class signal is preserved
- Projection methods can clearly separate all four classes

### Difficulty levels and disturbance patterns

| Level | Disturbance pattern | Description |
|-------|---------------------|-------------|
| `d1`  | Cardinal pulses     | 2 Hz on/off pulses, each class pushed in a different cardinal direction (R/U/L/D). Counter: tap the opposing arrow ~2×/s in sync. |
| `d2`  | Diagonal pulses     | 1.5 Hz pulses at diagonal directions with 2:1 axis ratio. Counter: hold diagonal arrows at the right rhythm. |
| `d3`  | Double-tap rhythm   | Two 120 ms bursts per second with a 750 ms rest — then repeat. Counter: two quick taps, then release. Holding continuously over-corrects. |
| `d4`  | Rotating force      | Continuously rotating disturbance (1 full rotation per 2.5 s). Counter: smoothly rotate through arrow combinations. |
| `d5`  | Dual-frequency      | Independent x and y axes driven at different square-wave frequencies. Counter: tap L/R and U/D independently at their own rhythms. |

Noise and signal parameters also increase across levels:

| Parameter | d1 | d2 | d3 | d4 | d5 |
|-----------|----|----|----|----|-----|
| Observation noise std | 0.35 | 0.50 | 0.65 | 0.85 | 1.10 |
| Latent noise std | 0.04 | 0.06 | 0.08 | 0.10 | 0.12 |
| Disturbance amplitude | 0.065 | 0.075 | 0.080 | 0.090 | 0.095 |

---

## Installation

Requires Python ≥ 3.10.

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pyzmq`, `matplotlib`, `torch`.

---

## Running the emulator

```bash
# default: d1 difficulty, 256 dimensions, ZMQ port 5555
python -m emulator

# harder levels
python -m emulator --difficulty d2
python -m emulator --difficulty d5

# custom dims and port
python -m emulator --difficulty d3 --dims 128 --port 5556
```

### Controls

| Key | Action |
|-----|--------|
| `1` | Set intention → left hand |
| `2` | Set intention → right hand |
| `3` | Set intention → left leg |
| `4` | Set intention → right leg |
| `0` | Rest (no intention) |
| `←` `→` `↑` `↓` | Navigate strategy space (hold for continuous movement) |
| `ESC` / `Q` | Quit |

The **strategy quality** bar in the GUI shows how close `z_strategy` is to `(0, 0)`.
This is visible to the operator but would not be shown to a real patient — it is there
so you can verify the emulator is behaving correctly.

---

## Receiving data in your GUI

The emulator publishes one JSON message per sample over a **ZMQ PUB** socket at 10 Hz.

### Message format

```json
{
    "timestamp":        1713000000.123,
    "sample_idx":       42,
    "data":             [0.31, -1.22, 0.07, ...],
    "label":            0,
    "label_name":       "left_hand",
    "n_dims":           256,
    "sample_rate":      10,
    "difficulty":       "d1",
    "class_scale":      0.74,
    "strategy_quality": 0.92
}
```

| Field | Type | Description |
|---|---|---|
| `timestamp` | float | Unix time of sample generation |
| `sample_idx` | int | Monotonically increasing counter |
| `data` | list[float] | The 256-dim neural signal — this is your model input |
| `label` | int \| null | Intended class (0–3) or null for rest |
| `label_name` | str | `"left_hand"`, `"right_hand"`, `"left_leg"`, `"right_leg"`, `"rest"` |
| `n_dims` | int | Length of `data` |
| `sample_rate` | int | Samples per second (10 Hz) |
| `difficulty` | str | Active difficulty level |
| `class_scale` | float | Current signal strength 0–1 (leaky-integrated) |
| `strategy_quality` | float | How close operator is to optimal (0–1) |

### Starter template

`starter_template.py` — a minimal, well-commented receiver that accumulates a sliding
window of labeled samples and has clearly marked `TODO` sections for plugging in your
projection and visualisation code.  Start here.

```bash
# Run emulator in one terminal
python -m emulator

# Run starter template in another
python starter_template.py
```

### Real-time movement model (new)

A first implementation of an online decoder and manifold visualizer now lives in
`model/`. It expects one embedding vector per timestep and trains in the loop
with a hybrid objective:

- supervised manifold objective (ENZO spec):
  - latent classification loss
  - latent compactness + centroid separation
  - class-conditional temporal consistency
  - direct projection classification loss
  - projection compactness + centroid separation + temporal consistency
- reward-weighted policy loss as a secondary online adaptation signal

The default stream key is `data` (compatible with the current emulator), but
you can switch to a dedicated embedding field from an upstream encoder.

```bash
# terminal 1: emulator
python -m emulator --difficulty d1

# terminal 2: realtime decoder + manifold visualization
python -m model --embedding-key data --projection-dim 2
```

Optional arguments:

- `--no-viz` to disable matplotlib rendering
- `--projection-dim 3` for 3D manifold view
- `--warmup-labeled 200` to control when reward-weighted updates start
- `--update-every 10` to control optimizer cadence
- `--device cuda` to force GPU when available

### Thought rhythm game mode (Just-Dance style)

The realtime runner now supports a prompt-timing mode where the model is
rewarded for producing the correct class with reliable confidence/separability
within a timing window.

The current reward policy is distinction-first: label correctness, confidence
margin, and separability are weighted more strongly than timing.

```bash
# terminal 1: emulator
python -m emulator --difficulty d1

# terminal 2: model + rhythm prompt rewards (slow/permissive defaults)
python -m model --game-mode
```

Adaptive difficulty scaffolding is available and can be enabled with:

```bash
python -m model --game-mode --game-adaptive-difficulty
```

Useful game-specific flags:

- `--game-beat-interval` to control prompt spacing
- `--game-prompt-duration` / `--game-hit-window` for slower vs stricter timing
- `--game-print-every` to control console HUD cadence
- `--game-seed` to make prompt sequences reproducible
- `--game-dashboard-history` to control embedding history size in the dashboard
- `--game-dashboard-draw-every` to control dashboard redraw cadence

Automatic high-performance simulation (for demos/warm starts):

```bash
python -m model --game-mode --game-auto-perform
```

Fine-grain control is available with:

- `--game-auto-strength`
- `--game-auto-prewindow-strength`
- `--game-auto-blend`
- `--game-auto-anticipation`

When visualization is enabled, game mode opens a richer dashboard with:

- a live embedding-space panel showing class cluster separation progress
- a coaching panel showing NOW vs UP NEXT intentions and countdowns
- a decoder confidence panel (target-highlighted vs predicted class)
- a reward-breakdown panel for per-sample feedback components
- distinction metrics (per-class correctness, dominant confusions)
- trend charts for reward, margin, and correctness over time

### Minimal Python receiver

```python
import json
import numpy as np
import zmq

ctx    = zmq.Context()
socket = ctx.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")   # subscribe to all messages

while True:
    msg   = json.loads(socket.recv_string())
    data  = np.array(msg["data"])      # shape: (256,)
    label = msg["label"]               # int 0-3, or None
    name  = msg["label_name"]          # e.g. "left_hand"
    # your processing here
```

### Receiving from other languages

Any language with a ZMQ SUB binding can connect.  The message is plain JSON so no
special deserialiser is needed.

**JavaScript (Node.js)**
```js
const zmq = require("zeromq");
const sock = new zmq.Subscriber();
sock.connect("tcp://localhost:5555");
sock.subscribe("");
for await (const [msg] of sock) {
    const { data, label, label_name } = JSON.parse(msg.toString());
}
```

**Julia**
```julia
using ZMQ, JSON
ctx  = Context()
sock = Socket(ctx, SUB)
connect(sock, "tcp://localhost:5555")
subscribe(sock, "")
while true
    msg = JSON.parse(String(recv(sock)))
    data, label = msg["data"], msg["label"]
end
```

---

## Provided scripts

| File | Purpose |
|------|---------|
| `starter_template.py` | **Start here.** Minimal receiver with TODO placeholders for your model and visualisation. |
| `receiver_example.py` | Console printer — shows raw numbers arriving from the emulator. Useful for sanity-checking the connection. |
| `receiver_gui.py` | Reference visualisation — live LDA/PCA scatter + raw signal plot. Shows what a finished GUI might look like; read it for inspiration but build your own. |

---

## What students should build

The task is to build a GUI that:

1. **Receives** the data stream and accumulates a sliding window of labeled samples.
2. **Computes a projection** (e.g. PCA, LDA, UMAP) from the 256-dim signal to 2D/3D.
3. **Displays** the projected brain state in real time, coloured by label.
4. **Recomputes** the projection regularly — as the operator's strategy drifts, the
   subspace in which classes are visible rotates, so a static projection will go stale.
5. **Provides feedback** that helps the operator maintain good strategy:
   e.g. a separability score, a confidence bar, or a cursor the operator tries to
   keep centered.

The key insight students should discover: *there is no single projection that works
forever*.  As the operator's strategy shifts, the projection must continuously adapt —
exactly what happens in real neurofeedback co-adaptation.
