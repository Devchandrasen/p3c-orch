"""Aggregate objective, paired significance tests, and fresh-result plots."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .constants import OBJECTIVE_WEIGHTS, PRACTICAL_METHODS, REGIMES


def analyze_results(
    run_metrics_path: str | Path,
    output_dir: str | Path,
    *,
    reference_method: str = "p3c-lr",
    comparator_method: str = "reactive-3c",
    make_plots: bool = True,
) -> dict[str, Path]:
    """Analyze freshly generated paired runs."""

    rows = _read_metric_rows(run_metrics_path)
    if not rows:
        raise ValueError("run metrics file is empty")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    objective_rows = compute_objective(rows)
    objective_path = output / "aggregate_objective.csv"
    _write_rows(objective_path, objective_rows)
    significance_rows = paired_significance(
        rows,
        reference_method=reference_method,
        comparator_method=comparator_method,
    )
    significance_path = output / "paired_significance.csv"
    _write_rows(significance_path, significance_rows)
    paths = {
        "aggregate_objective": objective_path,
        "paired_significance": significance_path,
    }
    if make_plots:
        paths.update(generate_plots(rows, objective_rows, output / "figures"))
    return paths


def compute_objective(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute the normalized lower-is-better protocol objective."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["regime"]), str(row["method"]))].append(row)
    means = {
        key: {
            metric: float(np.mean([float(row[metric]) for row in group_rows]))
            for metric in OBJECTIVE_WEIGHTS
        }
        for key, group_rows in grouped.items()
    }
    objective_rows: list[dict[str, Any]] = []
    for regime in sorted({key[0] for key in means}):
        methods = sorted(key[1] for key in means if key[0] == regime)
        normalized: dict[str, dict[str, float]] = defaultdict(dict)
        for metric in OBJECTIVE_WEIGHTS:
            values = {method: means[(regime, method)][metric] for method in methods}
            low, high = min(values.values()), max(values.values())
            if math.isclose(low, high):
                normalized[metric] = {method: 0.5 for method in methods}
            else:
                normalized[metric] = {
                    method: (value - low) / (high - low)
                    for method, value in values.items()
                }
        for method in methods:
            objective = sum(
                weight * normalized[metric][method]
                for metric, weight in OBJECTIVE_WEIGHTS.items()
            )
            objective_rows.append(
                {
                    "regime": regime,
                    "method": method,
                    "objective": objective,
                    "rank": 0,
                }
            )
        regime_rows = [row for row in objective_rows if row["regime"] == regime]
        for rank, row in enumerate(
            sorted(regime_rows, key=lambda item: float(item["objective"])), start=1
        ):
            row["rank"] = rank
    return objective_rows


def paired_significance(
    rows: list[dict[str, Any]],
    *,
    reference_method: str,
    comparator_method: str,
) -> list[dict[str, Any]]:
    """Run paired two-sided Wilcoxon tests and Holm-correct within each regime."""

    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:  # pragma: no cover - optional analysis extra
        raise RuntimeError(
            'Install the analysis extra with: pip install -e ".[analysis]"'
        ) from exc
    by_key = {
        (str(row["regime"]), str(row["method"]), int(row["seed"])): row
        for row in rows
    }
    result: list[dict[str, Any]] = []
    for regime in sorted({str(row["regime"]) for row in rows}):
        seeds = sorted(
            {
                int(row["seed"])
                for row in rows
                if row["regime"] == regime and row["method"] == reference_method
            }
            & {
                int(row["seed"])
                for row in rows
                if row["regime"] == regime and row["method"] == comparator_method
            }
        )
        if not seeds:
            continue
        regime_rows: list[dict[str, Any]] = []
        for metric, weight in OBJECTIVE_WEIGHTS.items():
            reference = np.asarray(
                [float(by_key[(regime, reference_method, seed)][metric]) for seed in seeds]
            )
            comparator = np.asarray(
                [float(by_key[(regime, comparator_method, seed)][metric]) for seed in seeds]
            )
            differences = reference - comparator
            if np.allclose(differences, 0.0):
                p_value = 1.0
            else:
                p_value = float(
                    wilcoxon(
                        reference,
                        comparator,
                        alternative="two-sided",
                        zero_method="wilcox",
                    ).pvalue
                )
            reference_mean = float(np.mean(reference))
            comparator_mean = float(np.mean(comparator))
            better = (
                reference_mean < comparator_mean
                if weight > 0
                else reference_mean > comparator_mean
            )
            regime_rows.append(
                {
                    "regime": regime,
                    "metric": metric,
                    "reference": reference_method,
                    "comparator": comparator_method,
                    "paired_runs": len(seeds),
                    "reference_mean": reference_mean,
                    "comparator_mean": comparator_mean,
                    "direction": "better" if better else "worse",
                    "raw_p": p_value,
                    "corrected_p": 1.0,
                    "reliable": False,
                }
            )
        corrected = holm_correction([float(row["raw_p"]) for row in regime_rows])
        for row, corrected_p in zip(regime_rows, corrected, strict=True):
            row["corrected_p"] = corrected_p
            row["reliable"] = corrected_p < 0.05
        result.extend(regime_rows)
    return result


def holm_correction(p_values: list[float]) -> list[float]:
    """Return monotone Holm-Bonferroni adjusted p-values in original order."""

    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    corrected = [1.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running_max = max(running_max, candidate)
        corrected[index] = running_max
    return corrected


def generate_plots(
    rows: list[dict[str, Any]],
    objective_rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Generate evaluation plots from current run outputs only."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional analysis extra
        raise RuntimeError(
            'Install the analysis extra with: pip install -e ".[analysis]"'
        ) from exc
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    methods = [
        method
        for method in PRACTICAL_METHODS
        if any(str(row["method"]) == method for row in rows)
    ]
    regime_order = [regime for regime in REGIMES if any(row["regime"] == regime for row in rows)]
    paths: dict[str, Path] = {}

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    combined = "r7-combined-stress" if "r7-combined-stress" in regime_order else regime_order[-1]
    for method in methods:
        values = sorted(
            float(row["average_delay"])
            for row in rows
            if row["regime"] == combined and row["method"] == method
        )
        if values:
            axis.step(values, np.arange(1, len(values) + 1) / len(values), label=method)
    axis.set(xlabel="Run-average delay (slots)", ylabel="Empirical CDF", ylim=(0, 1.02))
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    paths["delay_cdf"] = _save_figure(figure, output / "delay_cdf.png")

    paths["energy_by_regime"] = _grouped_bar_plot(
        plt,
        rows,
        methods,
        regime_order,
        "energy_kj",
        "Energy (kJ)",
        output / "energy_by_regime.png",
    )
    paths["handovers_by_regime"] = _grouped_bar_plot(
        plt,
        rows,
        methods,
        regime_order,
        "handovers_per_100",
        "Handovers per 100 active cluster-slots",
        output / "handovers_by_regime.png",
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    local = [_mean(rows, combined, method, "local_cache_hit_ratio") for method in methods]
    neighbor = [
        _mean(rows, combined, method, "neighbor_cache_hit_ratio") for method in methods
    ]
    x = np.arange(len(methods))
    axis.bar(x, local, label="local")
    axis.bar(x, neighbor, bottom=local, label="neighbor")
    axis.set_xticks(x, methods, rotation=30, ha="right")
    axis.set_ylabel("Useful cache-hit ratio (%)")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    paths["cache_breakdown"] = _save_figure(figure, output / "cache_breakdown.png")

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    weather_metrics = (
        ("clear", "clear_outage_probability"),
        ("rainy", "rainy_outage_probability"),
        ("rain-hot", "rain_hot_outage_probability"),
    )
    width = 0.8 / max(len(methods), 1)
    for index, method in enumerate(methods):
        values = [_mean(rows, combined, method, metric) for _, metric in weather_metrics]
        axis.bar(
            np.arange(len(weather_metrics)) + index * width,
            values,
            width,
            label=method,
        )
    axis.set_xticks(
        np.arange(len(weather_metrics)) + width * (len(methods) - 1) / 2,
        [name for name, _ in weather_metrics],
    )
    axis.set_ylabel("Outage probability (%)")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(axis="y", alpha=0.3)
    paths["outage_by_weather"] = _save_figure(
        figure, output / "outage_by_weather.png"
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    combined_objective = [
        row for row in objective_rows if row["regime"] == combined
    ]
    combined_objective.sort(key=lambda row: int(row["rank"]))
    axis.bar(
        [str(row["method"]) for row in combined_objective],
        [float(row["objective"]) for row in combined_objective],
    )
    axis.set_ylabel("Normalized aggregate objective (lower is better)")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.3)
    paths["aggregate_objective_plot"] = _save_figure(
        figure, output / "aggregate_objective.png"
    )
    return paths


def _grouped_bar_plot(
    plt: Any,
    rows: list[dict[str, Any]],
    methods: list[str],
    regimes: list[str],
    metric: str,
    ylabel: str,
    output: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    width = 0.8 / max(len(methods), 1)
    for index, method in enumerate(methods):
        values = [_mean(rows, regime, method, metric) for regime in regimes]
        axis.bar(np.arange(len(regimes)) + index * width, values, width, label=method)
    axis.set_xticks(
        np.arange(len(regimes)) + width * (len(methods) - 1) / 2,
        [REGIMES[regime].regime_id for regime in regimes],
    )
    axis.set_ylabel(ylabel)
    axis.legend(fontsize=7, ncol=2)
    axis.grid(axis="y", alpha=0.3)
    return _save_figure(figure, output)


def _mean(rows: list[dict[str, Any]], regime: str, method: str, metric: str) -> float:
    values = [
        float(row[metric])
        for row in rows
        if row["regime"] == regime and row["method"] == method
    ]
    return float(np.mean(values)) if values else 0.0


def _save_figure(figure: Any, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path


def _read_metric_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        for key in row.keys() - {"regime", "method", "seed"}:
            row[key] = float(row[key])
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
