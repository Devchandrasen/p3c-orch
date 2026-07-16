"""Command-line interface for the complete P3C-Orch implementation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .analysis import analyze_results
from .config import ProjectConfig, load_config
from .constants import ABLATION_METHODS
from .dataset import generate_predictor_dataset
from .predictor import train_predictor
from .simulation import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="p3c-orch",
        description="Predictive 3C orchestration implementation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser(
        "simulate", help="run configured methods, regimes, and seeds"
    )
    _add_experiment_arguments(simulate)

    protocol = subparsers.add_parser(
        "run-protocol", help="run the complete seven-regime blind protocol"
    )
    _add_experiment_arguments(protocol)
    protocol.add_argument(
        "--analyze",
        action="store_true",
        help="compute objective, paired tests, and plots after simulation",
    )

    ablate = subparsers.add_parser(
        "ablate", help="run the combined-stress component ablations"
    )
    _add_experiment_arguments(ablate)

    inspect = subparsers.add_parser(
        "inspect-config", help="validate and print configuration"
    )
    inspect.add_argument("--config", type=Path, required=True)

    generate = subparsers.add_parser(
        "generate-predictor-data",
        help="generate current/next-slot channel samples for the ANN",
    )
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--samples", type=int, default=6000)
    generate.add_argument("--seed", type=int, default=2026)

    prepare = subparsers.add_parser(
        "prepare-predictor",
        help="generate data, train the 64x64x64 MLP, and save calibration",
    )
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--samples", type=int, default=6000)
    prepare.add_argument("--seed", type=int, default=2026)

    train = subparsers.add_parser(
        "train-predictor", help="train the optional 64x64x64 MLP predictor"
    )
    train.add_argument("--csv", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--random-state", type=int, default=42)

    analyze = subparsers.add_parser(
        "analyze", help="compute objective, Holm-Wilcoxon tests, and plots"
    )
    analyze.add_argument("--runs", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--reference", default="p3c-lr")
    analyze.add_argument("--comparator", default="reactive-3c")
    analyze.add_argument("--no-plots", action="store_true")
    return parser


def _add_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slots", type=int, help="override configured slot count")
    parser.add_argument("--seeds", type=int, nargs="+", help="override blind seeds")
    parser.add_argument("--methods", nargs="+", help="override evaluated methods")
    parser.add_argument("--regimes", nargs="+", help="override operating regimes")
    parser.add_argument("--model", type=Path, help="override the predictor NPZ path")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect-config":
        config = load_config(args.config)
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command in {"simulate", "run-protocol", "ablate"}:
        config = _experiment_config(args)
        paths = run_experiment(config, args.output)
        if args.command == "run-protocol" and args.analyze:
            paths.update(
                {
                    f"analysis_{name}": path
                    for name, path in analyze_results(
                        paths["run_metrics"], Path(args.output) / "analysis"
                    ).items()
                }
            )
        print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
        return 0
    if args.command == "generate-predictor-data":
        path = generate_predictor_dataset(
            load_config(args.config),
            args.output,
            samples=args.samples,
            seed=args.seed,
        )
        print(json.dumps({"dataset": str(path)}, indent=2))
        return 0
    if args.command == "prepare-predictor":
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        dataset_path = output / "predictor_samples.csv"
        model_path = output / "link_predictor.npz"
        generate_predictor_dataset(
            load_config(args.config),
            dataset_path,
            samples=args.samples,
            seed=args.seed,
        )
        metrics = train_predictor(dataset_path, model_path, random_state=args.seed)
        metrics_path = output / "predictor_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "dataset": str(dataset_path),
                    "model": str(model_path),
                    "metrics": str(metrics_path),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "train-predictor":
        metrics = train_predictor(
            args.csv,
            args.output,
            random_state=args.random_state,
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if args.command == "analyze":
        paths = analyze_results(
            args.runs,
            args.output,
            reference_method=args.reference,
            comparator_method=args.comparator,
            make_plots=not args.no_plots,
        )
        print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def _experiment_config(args: argparse.Namespace) -> ProjectConfig:
    config = load_config(args.config)
    methods = tuple(args.methods) if args.methods else None
    regimes = tuple(args.regimes) if args.regimes else None
    if args.command == "ablate":
        methods = (
            "p3c-lr",
            *sorted(ABLATION_METHODS),
            "p3c-sr",
            "et-p3c",
        )
        regimes = ("r7-combined-stress",)
    config = config.with_overrides(
        slots=args.slots,
        seeds=tuple(args.seeds) if args.seeds else None,
        methods=methods,
        regimes=regimes,
    )
    if args.model:
        config = replace(
            config,
            predictor=replace(config.predictor, model_path=args.model.resolve()),
        )
        config.validate()
    return config
