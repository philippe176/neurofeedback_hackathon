"""
Entry point for the Neurofeedback Web Application.

Usage:
    python -m webapp                    # Start on port 8080
    python -m webapp --port 3000        # Custom port
    python -m webapp --debug            # Debug mode

Description:
    This web application provides a real-time visualization interface for
    the BCI neurofeedback system. It includes:

    - Live manifold visualization showing the latent space projection
    - Class probability bar charts
    - Training progress metrics over time
    - Interactive controls for mental state simulation
    - Auto-tracking mode for demonstration

Controls:
    1. Start Streaming: Begin real-time data generation and processing
    2. Stop Streaming: Pause the data stream
    3. Auto-Tracking: Automatically cycles through mental states with
       optimal strategy (keeps z_strategy near origin)
    4. Mental State Buttons: Manually select which mental state to simulate
    5. Centroid Window Slider: Adjust how many recent samples are used
       to compute cluster centroids

The system emulates brain signals for 4 movement classes:
    - Left Hand (LH) - Blue
    - Right Hand (RH) - Orange
    - Left Leg (LL) - Green
    - Right Leg (RL) - Red/Coral
"""

import argparse

from .app import run_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neurofeedback BCI Web Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Port to run the server on (default: 8080)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with auto-reload",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Neurofeedback BCI - Web Visualization")
    print("=" * 60)
    print()
    print("Instructions:")
    print(f"  1. Open your browser to http://localhost:{args.port}")
    print("  2. Click 'Start Streaming' to begin")
    print("  3. Click 'Enable Auto-Tracking' for automatic demo")
    print("  4. Or manually select mental states with the buttons")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()

    run_app(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
