"""Application entrypoint with UI mode selection."""

import argparse

from flight_booking_agent import run_cli
from gradio_app import run_gradio


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for selecting the interaction mode."""
    parser = argparse.ArgumentParser(description="Flight ticket assistant")
    parser.add_argument(
        "--ui",
        choices=["cli", "gradio"],
        default="cli",
        help="Choose interaction mode: 'cli' for terminal, 'gradio' for web UI.",
    )
    return parser.parse_args()


def main() -> None:
    """Start the application in CLI or Gradio mode based on flags."""
    args = _parse_args()
    if args.ui == "gradio":
        run_gradio()
        return
    run_cli()


if __name__ == "__main__":
    main()
