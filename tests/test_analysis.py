from __future__ import annotations

from typing import Any

import pytest

from p3c_orch.analysis import (
    analyze_results,
    compute_objective,
    holm_correction,
    paired_significance,
)
from p3c_orch.constants import OBJECTIVE_WEIGHTS
from p3c_orch.simulation import run_experiment

from .test_simulation import small_config

pytest.importorskip("scipy")


def objective_row(method: str, seed: int, *, good: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "regime": "r7-combined-stress",
        "method": method,
        "seed": seed,
    }
    for metric, weight in OBJECTIVE_WEIGHTS.items():
        base = float(seed)
        row[metric] = base if (good and weight > 0) or (not good and weight < 0) else base + 10.0
    return row


def test_normalized_objective_rewards_lower_costs_and_higher_benefits() -> None:
    rows = [objective_row("good", 1, good=True), objective_row("bad", 1, good=False)]
    objective = compute_objective(rows)
    ranked = {row["method"]: row for row in objective}
    assert ranked["good"]["rank"] == 1
    assert ranked["good"]["objective"] < ranked["bad"]["objective"]


def test_holm_correction_is_monotone_and_bounded() -> None:
    corrected = holm_correction([0.01, 0.04, 0.03, 0.5])
    assert all(0.0 <= value <= 1.0 for value in corrected)
    assert corrected[0] == pytest.approx(0.04)
    assert corrected[3] == pytest.approx(0.5)


def test_paired_significance_uses_matching_seeds() -> None:
    rows = [
        row
        for seed in range(1, 7)
        for row in (
            objective_row("p3c-lr", seed, good=True),
            objective_row("reactive-3c", seed, good=False),
        )
    ]
    result = paired_significance(
        rows,
        reference_method="p3c-lr",
        comparator_method="reactive-3c",
    )
    assert len(result) == len(OBJECTIVE_WEIGHTS)
    assert {row["paired_runs"] for row in result} == {6}
    assert all(float(row["corrected_p"]) >= float(row["raw_p"]) for row in result)


def test_analysis_writes_outputs_from_a_fresh_run(tmp_path) -> None:
    paths = run_experiment(small_config(), tmp_path / "run")
    analysis = analyze_results(
        paths["run_metrics"],
        tmp_path / "analysis",
        reference_method="p3c-lr",
        comparator_method="p3c-sr",
        make_plots=False,
    )
    assert set(analysis) == {"aggregate_objective", "paired_significance"}
    assert all(path.exists() for path in analysis.values())


def test_plot_manifest_keeps_csv_and_plot_paths_distinct(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    paths = run_experiment(small_config(), tmp_path / "run")
    analysis = analyze_results(
        paths["run_metrics"],
        tmp_path / "analysis",
        reference_method="p3c-lr",
        comparator_method="p3c-sr",
    )
    assert analysis["aggregate_objective"].suffix == ".csv"
    assert analysis["aggregate_objective_plot"].suffix == ".png"
    assert analysis["aggregate_objective"] != analysis["aggregate_objective_plot"]
