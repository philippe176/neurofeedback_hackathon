"""
Entry point for the Neurofeedback Web Application.

Run the emulator first in a separate terminal:

    python -m emulator

Then start the dashboard:

    python -m webapp
"""

from __future__ import annotations

import argparse

from .app import run_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Neurofeedback BCI Web Application")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port to serve on")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    print("=" * 60)
    print("  Neurofeedback Decoder Lab")
    print("=" * 60)
    print("1. Start the emulator in another terminal: python -m emulator")
    print(f"2. Open http://localhost:{args.port}")
    print("3. Click Start Listening in the browser")
    print("4. Use 1-4 and the arrow keys inside the emulator window")
    print("=" * 60)

    run_app(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
