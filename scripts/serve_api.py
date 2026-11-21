#!/usr/bin/env python3
"""Démarre l'API FastAPI (GET /health, POST /triage) pour le backend Spring Boot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faadjou_ai.api import create_app  # noqa: E402
from faadjou_ai.config_utils import load_config, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="API Faadjou AI.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    host = args.host or config["api"].get("host", "0.0.0.0")
    port = int(args.port or config["api"].get("port", 8000))
    uvicorn.run(create_app(config), host=host, port=port)


if __name__ == "__main__":
    main()
