"""Command-line interface for P3C-Orch."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .config import load_config
from .predictor import train_predictor
from .simulation import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="p3c-orch",
        description="Predictive 3C orchestration reference implementation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="run configured synthetic experiments")
    simulate.add_argument("--config", type=Path, required=True)
    simulate.add_argument("--output", type=Path, required=True)
    simulate.add_argument("--slots", type=int, help="override the configured slot count")
    simulate.add_argument("--seeds", type=int, nargs="+", help="override configured seeds")

    inspect = subparsers.add_parser("inspect-config", help="validate and print configuration")
    inspect.add_argument("--config", type=Path, required=True)

    train = subparsers.add_parser("train-predictor", help="train the optional MLP predictor")
    train.add_argument("--csv", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect-config":
        config = load_config(args.config)
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "simulate":
        config = load_config(args.config).with_overrides(
            slots=args.slots,
            seeds=tuple(args.seeds) if args.seeds else None,
        )
        paths = run_experiment(config, args.output)
        print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
        return 0
    if args.command == "train-predictor":
        metrics = train_predictor(
            args.csv,
            args.output,
            random_state=args.random_state,
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
