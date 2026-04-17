"""
Entry point for the Brain Emulator.

Usage:
    python -m emulator                        # medium difficulty, 256 dims, port 5555
    python -m emulator --difficulty easy
    python -m emulator --difficulty hard --dims 128 --port 5556
"""

import argparse
from .config import DIFFICULTIES
from .gui import run_emulator_gui


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brain Emulator — BCI Neurofeedback Hackathon"
    )
    parser.add_argument(
        "--difficulty", "-d",
        choices=list(DIFFICULTIES.keys()),
        default="d1",
        help="Difficulty level  (default: d1)",
    )
    parser.add_argument(
        "--dims", "-n",
        type=int,
        default=256,
        help="Number of observation dimensions  (default: 256)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5555,
        help="ZMQ publisher port  (default: 5555)",
    )
    args = parser.parse_args()
    run_emulator_gui(difficulty=args.difficulty, n_dims=args.dims, port=args.port)


if __name__ == "__main__":
    main()
