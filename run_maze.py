import json, threading, collections
import numpy as np
import zmq
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import game_maze

HOST, PORT = "localhost", 5555
PER_CLASS_BUF = 40
MIN_PER_CLASS = 8
FIT_EVERY = 10

class_buffers = {c: collections.deque(maxlen=PER_CLASS_BUF) for c in range(4)}
lda = LinearDiscriminantAnalysis()
fitted = False
sample_count = 0

def bci_thread():
    global fitted, sample_count
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{HOST}:{PORT}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        msg = json.loads(sock.recv_string())
        data  = np.array(msg["data"], dtype=float)
        label = msg["label"]

        if label is not None:
            class_buffers[label].append(data)

        sample_count += 1

        # Refit every FIT_EVERY samples once all 4 classes have MIN_PER_CLASS samples
        if sample_count % FIT_EVERY == 0:
            classes_ready = [c for c, buf in class_buffers.items() if len(buf) >= MIN_PER_CLASS]
            if len(classes_ready) >= 2:
                X = np.vstack([class_buffers[c] for c in classes_ready])
                y = np.concatenate([[c] * len(class_buffers[c]) for c in classes_ready])
                lda.fit(X, y)
                fitted = True

        if fitted:
            probs_raw = lda.predict_proba(data.reshape(1, -1))[0]
            # lda.classes_ may not be [0,1,2,3] if not all classes seen yet — pad
            probs = np.full(4, 0.25)
            for i, cls in enumerate(lda.classes_):
                probs[int(cls)] = probs_raw[i]
            probs /= probs.sum()
            game_maze.update(probs)

threading.Thread(target=bci_thread, daemon=True).start()

# maze run() must be on the main thread (pygame requirement)
game_maze.run()