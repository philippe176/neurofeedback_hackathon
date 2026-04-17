"""
Flask application with WebSocket support for real-time neurofeedback visualization.
"""

from __future__ import annotations

import threading
import time

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from .emulator_bridge import EmulatorBridge

app = Flask(__name__)
app.config["SECRET_KEY"] = "neurofeedback-secret-key"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Global bridge instance
bridge: EmulatorBridge | None = None
streaming_thread: threading.Thread | None = None
is_streaming = False


def get_bridge() -> EmulatorBridge:
    """Get or create the emulator bridge."""
    global bridge
    if bridge is None:
        bridge = EmulatorBridge()
    return bridge


@app.route("/")
def index():
    """Render the main application page."""
    return render_template("index.html")


@app.route("/api/status")
def status():
    """Get current system status."""
    b = get_bridge()
    snapshot = b.status_snapshot()
    snapshot["streaming"] = is_streaming
    snapshot["model_updates"] = b.trainer.num_updates if b.trainer else 0
    return jsonify(snapshot)


@app.route("/api/set_class", methods=["POST"])
def set_class():
    """Set the current mental state class."""
    data = request.get_json()
    class_idx = data.get("class_idx")
    b = get_bridge()
    b.set_class(class_idx)
    return jsonify({"success": True, "class_idx": class_idx})


@app.route("/api/toggle_tracking", methods=["POST"])
def toggle_tracking():
    """Toggle auto-tracking mode."""
    b = get_bridge()
    b.auto_tracking = not b.auto_tracking
    if b.auto_tracking:
        b.set_control_mode("buttons")
    return jsonify({
        "success": True,
        "auto_tracking": b.auto_tracking,
        "control_mode": b.control_mode,
        "pressed_arrows": b.pressed_arrows(),
    })


@app.route("/api/set_training_phase", methods=["POST"])
def set_training_phase():
    """Switch between calibration, neurofeedback, and manual sandbox phases."""
    data = request.get_json() or {}
    training_phase = data.get("training_phase", "calibration")
    b = get_bridge()
    try:
        b.set_training_phase(training_phase)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    return jsonify({
        "success": True,
        "training_phase": b.training_phase,
        "training_phase_name": b.training_phase_name,
        "training_phase_description": b.status_snapshot()["training_phase_description"],
        "prompt": b.preview_prompt_state(),
        "session": b.session_snapshot(),
        "calibration": b.calibration_snapshot(),
        "coach": b.coach_snapshot(),
    })


@app.route("/api/set_difficulty", methods=["POST"])
def set_difficulty():
    """Select which disturbance pattern the embedded emulator should use."""
    data = request.get_json() or {}
    difficulty = data.get("difficulty", "d1")
    b = get_bridge()
    try:
        b.set_difficulty(difficulty)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    return jsonify({
        "success": True,
        "difficulty": b.difficulty,
        "difficulty_name": b.difficulty_name,
        "sample_count": b.sample_count,
        "calibration": b.calibration_snapshot(),
        "coach": b.coach_snapshot(),
    })


@app.route("/api/set_control_mode", methods=["POST"])
def set_control_mode():
    """Select the web input mode used to drive the emulator."""
    data = request.get_json() or {}
    control_mode = data.get("control_mode", "buttons")
    b = get_bridge()
    try:
        b.set_control_mode(control_mode)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    return jsonify({
        "success": True,
        "control_mode": b.control_mode,
        "control_mode_name": b.available_control_modes()[b.control_mode],
        "auto_tracking": b.auto_tracking,
        "pressed_arrows": b.pressed_arrows(),
    })


@app.route("/api/control_key", methods=["POST"])
def control_key():
    """Track held emulator arrow keys coming from the browser."""
    data = request.get_json() or {}
    key = data.get("key")
    pressed = bool(data.get("pressed"))
    b = get_bridge()
    try:
        b.set_arrow_pressed(key, pressed)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    return jsonify({
        "success": True,
        "pressed_arrows": b.pressed_arrows(),
        "auto_tracking": b.auto_tracking,
    })


@app.route("/api/set_centroid_window", methods=["POST"])
def set_centroid_window():
    """Set the centroid window size."""
    data = request.get_json()
    window = int(data.get("window", 50))
    b = get_bridge()
    b.centroid_window = max(10, min(window, 300))
    return jsonify({"success": True, "window": b.centroid_window})


@app.route("/api/set_model", methods=["POST"])
def set_model():
    """Set the active model implementation."""
    data = request.get_json() or {}
    model_type = data.get("model_type", "dnn")
    b = get_bridge()
    try:
        b.set_model(model_type)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    return jsonify({
        "success": True,
        "model_type": b.model_type,
        "model_name": b.model_name,
    })


@app.route("/api/set_viz_method", methods=["POST"])
def set_viz_method():
    """Set the active visualization backend."""
    data = request.get_json() or {}
    viz_method = data.get("viz_method", "neural")
    b = get_bridge()
    try:
        b.set_viz_method(viz_method)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    return jsonify({
        "success": True,
        "viz_method": b.viz_method,
        "viz_name": b.viz_name,
    })


@app.route("/api/save_model", methods=["POST"])
def save_model():
    """Save the active decoder checkpoint with an optional custom name."""
    data = request.get_json() or {}
    requested_name = data.get("name")
    b = get_bridge()
    try:
        path = b.save_model_snapshot(requested_name)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500
    return jsonify({
        "success": True,
        "path": str(path),
        "filename": path.name,
        "default_name": b.default_checkpoint_name(),
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset the model and emulator."""
    global bridge
    bridge = EmulatorBridge()
    return jsonify({"success": True})


@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    print("Client connected")
    emit("status", {"connected": True})


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    print("Client disconnected")


@socketio.on("start_streaming")
def handle_start_streaming():
    """Start the real-time streaming loop."""
    global is_streaming, streaming_thread

    if is_streaming:
        return

    is_streaming = True
    streaming_thread = threading.Thread(target=streaming_loop, daemon=True)
    streaming_thread.start()
    emit("streaming_status", {"streaming": True})


@socketio.on("stop_streaming")
def handle_stop_streaming():
    """Stop the streaming loop."""
    global is_streaming
    is_streaming = False
    emit("streaming_status", {"streaming": False})


@socketio.on("set_class")
def handle_set_class(data):
    """Handle class selection from client."""
    class_idx = data.get("class_idx")
    b = get_bridge()
    b.set_class(class_idx)
    emit("class_changed", {"class_idx": class_idx})


@socketio.on("toggle_tracking")
def handle_toggle_tracking():
    """Handle tracking toggle from client."""
    b = get_bridge()
    b.auto_tracking = not b.auto_tracking
    if b.auto_tracking:
        b.set_control_mode("buttons")
    emit(
        "tracking_changed",
        {
            "auto_tracking": b.auto_tracking,
            "control_mode": b.control_mode,
            "pressed_arrows": b.pressed_arrows(),
        },
        broadcast=True,
    )


@socketio.on("set_training_phase")
def handle_set_training_phase(data):
    """Handle guided pipeline phase changes from the client."""
    training_phase = (data or {}).get("training_phase", "calibration")
    b = get_bridge()
    try:
        b.set_training_phase(training_phase)
    except ValueError as exc:
        emit("error", {"message": str(exc)})
        return
    emit(
        "training_phase_changed",
        {
            "training_phase": b.training_phase,
            "training_phase_name": b.training_phase_name,
            "training_phase_description": b.status_snapshot()["training_phase_description"],
            "prompt": b.preview_prompt_state(),
            "session": b.session_snapshot(),
            "calibration": b.calibration_snapshot(),
            "coach": b.coach_snapshot(),
        },
        broadcast=True,
    )


@socketio.on("set_difficulty")
def handle_set_difficulty(data):
    """Handle emulator difficulty changes from the client."""
    difficulty = (data or {}).get("difficulty", "d1")
    b = get_bridge()
    try:
        b.set_difficulty(difficulty)
    except ValueError as exc:
        emit("error", {"message": str(exc)})
        return
    emit(
        "difficulty_changed",
        {
            "difficulty": b.difficulty,
            "difficulty_name": b.difficulty_name,
            "sample_count": b.sample_count,
            "calibration": b.calibration_snapshot(),
            "coach": b.coach_snapshot(),
        },
        broadcast=True,
    )


@socketio.on("set_control_mode")
def handle_set_control_mode(data):
    """Handle control mode selection from client."""
    control_mode = (data or {}).get("control_mode", "buttons")
    b = get_bridge()
    try:
        b.set_control_mode(control_mode)
    except ValueError as exc:
        emit("error", {"message": str(exc)})
        return
    emit(
        "control_mode_changed",
        {
            "control_mode": b.control_mode,
            "control_mode_name": b.available_control_modes()[b.control_mode],
            "auto_tracking": b.auto_tracking,
            "pressed_arrows": b.pressed_arrows(),
        },
        broadcast=True,
    )


@socketio.on("control_key")
def handle_control_key(data):
    """Handle held arrow-key state from the browser."""
    payload = data or {}
    key = payload.get("key")
    pressed = bool(payload.get("pressed"))
    b = get_bridge()
    try:
        b.set_arrow_pressed(key, pressed)
    except ValueError as exc:
        emit("error", {"message": str(exc)})
        return
    emit(
        "control_state_changed",
        {
            "pressed_arrows": b.pressed_arrows(),
            "auto_tracking": b.auto_tracking,
        },
        broadcast=True,
    )


@socketio.on("set_centroid_window")
def handle_set_centroid_window(data):
    """Handle centroid window change from client."""
    window = int(data.get("window", 50))
    b = get_bridge()
    b.centroid_window = max(10, min(window, 300))
    emit("centroid_window_changed", {"window": b.centroid_window})


@socketio.on("set_model")
def handle_set_model(data):
    """Handle model selection from client."""
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
    """Handle visualization selection from client."""
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
    """Main streaming loop that generates samples and broadcasts updates."""
    global is_streaming

    b = get_bridge()
    sample_rate = 10.0  # 10 Hz
    interval = 1.0 / sample_rate

    while is_streaming:
        start_time = time.time()

        try:
            # Generate a sample and process it
            update = b.step()

            # Broadcast to all connected clients
            socketio.emit("update", update)

        except Exception as e:
            print(f"Error in streaming loop: {e}")
            socketio.emit("error", {"message": str(e)})

        # Maintain consistent sample rate
        elapsed = time.time() - start_time
        sleep_time = max(0, interval - elapsed)
        time.sleep(sleep_time)


def run_app(host: str = "0.0.0.0", port: int = 8080, debug: bool = False):
    """Run the Flask application."""
    print(f"Starting Neurofeedback Web App at http://{host}:{port}")
    print("Open your browser to http://localhost:8080")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    run_app(debug=True)
