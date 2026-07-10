"""CLI entry point for highperfasr."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="highperfasr",
        description="Production ASR serving for NeMo models",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Start the ASR server")
    serve_parser.add_argument("--config", default=None, help="Path to config YAML")
    serve_parser.add_argument("--mode", choices=["batch", "stream", "both"], default=None)
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)
    serve_parser.add_argument("--model", default=None, help="Model name (overrides config)")

    version_parser = subparsers.add_parser("version", help="Show version")  # noqa: F841

    args = parser.parse_args()

    if args.command == "version":
        from highperfasr import __version__

        print(f"highperfasr {__version__}")
        return

    if args.command == "serve":
        _serve(args)
        return

    parser.print_help()
    sys.exit(1)


def _serve(args):
    from highperfasr.server import run_server

    run_server(
        config_path=args.config,
        mode=args.mode,
        host=args.host,
        port=args.port,
        model=args.model,
    )
