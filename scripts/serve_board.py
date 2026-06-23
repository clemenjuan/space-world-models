#!/usr/bin/env python3
"""Serve the active EventSat experiment board on port 8801."""
from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


DEFAULT_PORT = 8801
DEFAULT_DIRECTORY = Path("data/figures")
DEFAULT_BOARD = "index.html"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--directory", default=str(DEFAULT_DIRECTORY))
    parser.add_argument("--board", default=DEFAULT_BOARD)
    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    board = directory / args.board
    if not board.exists():
        raise FileNotFoundError(f"board not found: {board}")

    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer((args.host, int(args.port)), handler)
    print(f"serving {board} at http://127.0.0.1:{args.port}/{args.board}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
