"""
Minimal ZMQ receiver — shows students how to connect to the emulator.

Run the emulator first:
    python -m emulator

Then in a separate terminal:
    python receiver_example.py

The emulator publishes one JSON message per sample at 10 Hz.
Each message looks like:
{
    "timestamp":   1713000000.123,
    "sample_idx":  42,
    "data":        [0.31, -1.22, ...],   # 256 floats — the raw brain signal
    "label":       0,                    # int 0-3, or null for rest
    "label_name":  "left_hand",          # human-readable
    "n_dims":      256,
    "sample_rate": 10,
    "difficulty":  "d1"
}
"""

import json
import time

import numpy as np
import zmq


HOST = "localhost"
PORT = 5555


def main() -> None:
    ctx    = zmq.Context()
    socket = ctx.socket(zmq.SUB)
    socket.connect(f"tcp://{HOST}:{PORT}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")   # subscribe to all messages

    print(f"Connected to tcp://{HOST}:{PORT}  — waiting for data …")
    print("Press Ctrl-C to stop.\n")
    print(f"{'time':>7}  {'idx':>6}  {'label':^14}  {'mean':>8}  {'std':>8}")
    print("-" * 55)

    count      = 0
    start_time = time.time()

    try:
        while True:
            raw  = socket.recv_string()
            msg  = json.loads(raw)

            data        = np.array(msg["data"], dtype=float)   # shape: (n_dims,)
            label       = msg["label"]                          # int or None
            label_name  = msg["label_name"]                     # str
            sample_idx  = msg["sample_idx"]
            count      += 1

            elapsed = time.time() - start_time
            if count % 10 == 0:   # print ~once per second
                print(
                    f"{elapsed:7.1f}  {sample_idx:6d}  {label_name:^14}  "
                    f"{data.mean():+8.3f}  {data.std():8.3f}"
                )

    except KeyboardInterrupt:
        print(f"\nReceived {count} samples in {time.time() - start_time:.1f} s")
    finally:
        socket.close()
        ctx.term()


if __name__ == "__main__":
    main()
