import csv
import math
from dataclasses import asdict, replace

import numpy as np
import pytest

from p3c_orch.config import ProjectConfig, SimulationConfig
from p3c_orch.constants import SUPPORTED_METHODS
from p3c_orch.models import ClusterState, SwarmState
from p3c_orch.simulation import SimulationRunner, run_experiment

from .factories import make_observation


def small_config() -> ProjectConfig:
    return ProjectConfig(
        simulation=SimulationConfig(
            slots=3,
            area_size_m=500.0,
            swarms=2,
            clusters=3,
            cache_size=5,
            library_size=20,
            arrival_rate=1.0,
            deadline_min=2,
            deadline_max=4,
            seeds=(7,),
            methods=("p3c-lr", "p3c-sr"),
        )
    )


@pytest.mark.parametrize("method", sorted(SUPPORTED_METHODS))
def test_every_method_is_deterministic_and_returns_bounded_metrics(method: str) -> None:
    runner = SimulationRunner(small_config())
    first = runner.run(method, 7)
    second = runner.run(method, 7)
    assert asdict(first) == asdict(second)
    values = asdict(first)
    assert all(
        math.isfinite(float(value))
        for name, value in values.items()
        if name not in {"method", "seed"}
    )
    for name in (
        "drop_rate",
        "outage_probability",
        "useful_cache_hit_ratio",
        "handovers_per_100",
    ):
        assert 0.0 <= float(values[name]) <= 100.0


def test_experiment_writes_detailed_summary_and_metadata(tmp_path) -> None:
    config = small_config()
    paths = run_experiment(config, tmp_path)
    assert set(paths) == {"run_metrics", "summary", "metadata"}
    assert all(path.exists() for path in paths.values())
    with paths["run_metrics"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(config.simulation.methods) * len(config.simulation.seeds)


def test_project_config_slot_override() -> None:
    config = small_config().with_overrides(slots=5, seeds=(9, 10))
    assert config.simulation.slots == 5
    assert config.simulation.seeds == (9, 10)
    assert replace(config.simulation, slots=6).slots == 6


@pytest.mark.parametrize("method", ["nearest", "rate-max", "random"])
def test_baselines_reserve_energy_cumulatively(method: str) -> None:
    runner = SimulationRunner(small_config())
    swarms = [SwarmState(0, 0, 0, 150, 20, 30, 6.0, 10)]
    clusters = [
        ClusterState(0, 0, 0, 0, 0, 1, 1, 4, 1),
        ClusterState(1, 0, 0, 0, 0, 1, 1, 4, 1),
    ]
    observations = [
        make_observation(cluster_id=cluster_id, energy_cost=0.75, residual_energy_kj=6.0)
        for cluster_id in range(2)
    ]
    result = runner._baseline_assign(
        method=method,
        swarms=swarms,
        clusters=clusters,
        observations=observations,
        policy_rng=np.random.default_rng(7),
    )
    assert result.assignments == {0: 0}


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported baseline method"):
        SimulationRunner(small_config()).run("unknown", 7)
