"""
Flask application with Socket.IO support for real-time neurofeedback feedback.
"""

from __future__ import annotations

import threading
import time

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

from .emulator_bridge import EmulatorBridge

app = Flask(__name__)
app.config["SECRET_KEY"] = "neurofeedback-secret-key"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

bridge: EmulatorBridge | None = None
streaming_thread: threading.Thread | None = None
is_streaming = False


def get_bridge() -> EmulatorBridge:
    global bridge
    if bridge is None:
        bridge = EmulatorBridge()
    return bridge


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    b = get_bridge()
    snapshot = b.status_snapshot()
    snapshot["streaming"] = is_streaming
    snapshot["model_updates"] = b.trainer.num_updates if b.trainer else 0
    return jsonify(snapshot)


@app.route("/api/set_training_phase", methods=["POST"])
def set_training_phase():
    data = request.get_json() or {}
    phase = data.get("training_phase", "calibration")
    b = get_bridge()
    try:
        b.set_training_phase(phase)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    return jsonify(
        {
            "success": True,
            "training_phase": b.training_phase,
            "training_phase_name": b.training_phase_name,
            "training_phase_description": b.training_phase_description,
            "calibration": b.calibration_snapshot(),
            "coach": b.coach_snapshot(),
            "session": b.session_snapshot(),
        }
    )


@app.route("/api/set_centroid_window", methods=["POST"])
def set_centroid_window():
    data = request.get_json() or {}
    window = int(data.get("window", 50))
    b = get_bridge()
    b.centroid_window = max(10, min(window, 300))
    return jsonify({"success": True, "window": b.centroid_window})


@app.route("/api/set_model", methods=["POST"])
def set_model():
    data = request.get_json() or {}
    model_type = data.get("model_type", "dnn")
    b = get_bridge()
    try:
        b.set_model(model_type)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    return jsonify({"success": True, "model_type": b.model_type, "model_name": b.model_name})


@app.route("/api/set_viz_method", methods=["POST"])
def set_viz_method():
    data = request.get_json() or {}
    viz_method = data.get("viz_method", "neural")
    b = get_bridge()
    try:
        b.set_viz_method(viz_method)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    return jsonify({"success": True, "viz_method": b.viz_method, "viz_name": b.viz_name})


@app.route("/api/save_model", methods=["POST"])
def save_model():
    data = request.get_json() or {}
    requested_name = data.get("name")
    b = get_bridge()
    try:
        path = b.save_model_snapshot(requested_name)
    except Exception as exc:  # pragma: no cover - defensive API boundary
        return jsonify({"success": False, "error": str(exc)}), 500
    return jsonify(
        {
            "success": True,
            "path": str(path),
            "filename": path.name,
            "default_name": b.default_checkpoint_name(),
        }
    )


@app.route("/api/reset", methods=["POST"])
def reset():
    global bridge
    previous = bridge
    current_settings = {}
    if previous is not None:
        current_settings = {
            "stream_host": previous.stream_host,
            "stream_port": previous.stream_port,
            "embedding_key": previous.embedding_key,
            "model_type": previous.model_type,
            "viz_method": previous.viz_method,
            "training_phase": previous.training_phase,
        }
        previous.close()

    bridge = EmulatorBridge(**current_settings)
    return jsonify({"success": True})


@socketio.on("connect")
def handle_connect():
    emit(
        "status",
        {
            "connected": True,
            "streaming": is_streaming,
            "stream": get_bridge().stream_snapshot(),
        },
    )


@socketio.on("disconnect")
def handle_disconnect():
    return None


@socketio.on("start_streaming")
def handle_start_streaming():
    global is_streaming, streaming_thread
    if is_streaming:
        return

    bridge = get_bridge()
    bridge.start_stream()
    is_streaming = True
    streaming_thread = threading.Thread(target=streaming_loop, daemon=True)
    streaming_thread.start()
    emit(
        "streaming_status",
        {
            "streaming": True,
            "stream": bridge.stream_snapshot(),
        },
    )


@socketio.on("stop_streaming")
def handle_stop_streaming():
    global is_streaming
    is_streaming = False
    get_bridge().stop_stream()
    emit(
        "streaming_status",
        {
            "streaming": False,
            "stream": get_bridge().stream_snapshot(),
        },
    )


@socketio.on("set_training_phase")
def handle_set_training_phase(data):
    phase = (data or {}).get("training_phase", "calibration")
    b = get_bridge()
    try:
        b.set_training_phase(phase)
    except ValueError as exc:
        emit("error", {"message": str(exc)})
        return

    emit(
        "training_phase_changed",
        {
            "training_phase": b.training_phase,
            "training_phase_name": b.training_phase_name,
            "training_phase_description": b.training_phase_description,
            "calibration": b.calibration_snapshot(),
            "coach": b.coach_snapshot(),
            "session": b.session_snapshot(),
        },
        broadcast=True,
    )


@socketio.on("set_centroid_window")
def handle_set_centroid_window(data):
    b = get_bridge()
    window = int((data or {}).get("window", 50))
    b.centroid_window = max(10, min(window, 300))
    emit("centroid_window_changed", {"window": b.centroid_window})


@socketio.on("set_model")
def handle_set_model(data):
    model_type = (data or {}).get("model_type", "dnn")
    b = get_bridge()
    try:
        b.set_model(model_type)
    except ValueError as exc:
        emit("error", {"message": str(exc)})
        return
    emit(
        "model_changed",
        {
            "model_type": b.model_type,
            "model_name": b.model_name,
            "available_models": b.available_models(),
        },
        broadcast=True,
    )


@socketio.on("set_viz_method")
def handle_set_viz_method(data):
    viz_method = (data or {}).get("viz_method", "neural")
    b = get_bridge()
    try:
        b.set_viz_method(viz_method)
    except ValueError as exc:
        emit("error", {"message": str(exc)})
        return
    emit(
        "viz_method_changed",
        {
            "viz_method": b.viz_method,
            "viz_name": b.viz_name,
            "available_viz_methods": b.available_viz_methods(),
        },
        broadcast=True,
    )


def streaming_loop():
    global is_streaming

    bridge = get_bridge()
    interval = 1.0 / 10.0

    while is_streaming:
        started = time.time()
        try:
            update = bridge.step(timeout=0.08)
            socketio.emit("update", update)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            socketio.emit("error", {"message": str(exc)})

        elapsed = time.time() - started
        time.sleep(max(0.0, interval - elapsed))


def run_app(host: str = "0.0.0.0", port: int = 8080, debug: bool = False):
    print(f"Starting Neurofeedback Web App at http://{host}:{port}")
    print("1. Run `python -m emulator` in a separate terminal.")
    print("2. Open the browser UI and click Start Listening.")
    print("3. Use 1-4 and the arrow keys in the emulator window.")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    run_app(debug=True)
