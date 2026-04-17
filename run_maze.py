# Rewritten run_maze.py
import json, threading
import numpy as np
import zmq
import game_maze

GUI_HOST, GUI_PORT = "localhost", 5556

def bci_thread():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{GUI_HOST}:{GUI_PORT}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        try:
            msg = json.loads(sock.recv_string())
            if "probs" in msg:
                probs = np.array(msg["probs"], dtype=float)
                game_maze.update(probs)
        except Exception:
            pass

threading.Thread(target=bci_thread, daemon=True).start()
game_maze.run()