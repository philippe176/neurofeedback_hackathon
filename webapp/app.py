"""
Flask application with WebSocket support for real-time neurofeedback visualization.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from typing import Any

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
    return jsonify({
        "streaming": is_streaming,
        "current_class": b.current_class,
        "auto_tracking": b.auto_tracking,
        "sample_count": b.sample_count,
        "model_updates": b.trainer.num_updates if b.trainer else 0,
        "difficulty": b.difficulty,
    })


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
    return jsonify({"success": True, "auto_tracking": b.auto_tracking})


@app.route("/api/set_centroid_window", methods=["POST"])
def set_centroid_window():
    """Set the centroid window size."""
    data = request.get_json()
    window = int(data.get("window", 50))
    b = get_bridge()
    b.centroid_window = max(10, min(window, 300))
    return jsonify({"success": True, "window": b.centroid_window})


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
    emit("tracking_changed", {"auto_tracking": b.auto_tracking})


@socketio.on("set_centroid_window")
def handle_set_centroid_window(data):
    """Handle centroid window change from client."""
    window = int(data.get("window", 50))
    b = get_bridge()
    b.centroid_window = max(10, min(window, 300))
    emit("centroid_window_changed", {"window": b.centroid_window})


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
