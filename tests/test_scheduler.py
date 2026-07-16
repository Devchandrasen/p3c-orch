import numpy as np
import pytest

from p3c_orch.config import SchedulerConfig
from p3c_orch.models import ClusterState, SwarmState
from p3c_orch.predictor import CalibratedRiskEstimator, CurrentMarginPredictor
from p3c_orch.scheduler import P3CScheduler, options_for_variant

from .factories import make_observation


def pair(swarm_id: int, margin: float, distance: float):
    return make_observation(
        swarm_id=swarm_id,
        current_margin_db=margin,
        distance_m=distance,
        relative_speed_mps=2.0,
    )


def estimator(predictor=None) -> CalibratedRiskEstimator:
    return CalibratedRiskEstimator(
        predictor or CurrentMarginPredictor(),
        margin_threshold_db=0.0,
        residual_scale_db={"clear": 4.0},
    )


def test_dwell_guard_keeps_previous_feasible_swarm() -> None:
    scheduler = P3CScheduler(
        SchedulerConfig(dwell_threshold=1.0),
        estimator(),
        variant="p3c-lr",
    )
    swarms = [
        SwarmState(0, 0, 0, 150, 20, 30, 100, 10),
        SwarmState(1, 0, 0, 150, 20, 30, 100, 10),
    ]
    cluster = ClusterState(0, 0, 0, 0, 0, 1, 2, 5, 1, previous_swarm=0)
    result = scheduler.schedule(
        slot=0,
        swarms=swarms,
        clusters=[cluster],
        observations=[pair(0, 2.0, 600.0), pair(1, 8.0, 400.0)],
    )
    assert result.assignments == {0: 0}


def test_each_cluster_receives_at_most_one_assignment() -> None:
    scheduler = P3CScheduler(SchedulerConfig(), estimator(), variant="p3c-sr")
    swarms = [
        SwarmState(0, 0, 0, 150, 20, 30, 100, 10),
        SwarmState(1, 0, 0, 150, 20, 30, 100, 10),
    ]
    cluster = ClusterState(0, 0, 0, 0, 0, 1, 2, 5, 1)
    result = scheduler.schedule(
        slot=0,
        swarms=swarms,
        clusters=[cluster],
        observations=[pair(0, 2.0, 600.0), pair(1, 8.0, 400.0)],
    )
    assert len(result.assignments) == 1
    assert result.assignments[0] in {0, 1}


def test_predictions_are_batched_once_per_schedule() -> None:
    class CountingPredictor:
        def __init__(self) -> None:
            self.calls = 0

        def predict_many(self, observations):
            self.calls += 1
            return np.asarray(
                [observation.current_margin_db for observation in observations]
            )

    predictor = CountingPredictor()
    scheduler = P3CScheduler(SchedulerConfig(), estimator(predictor), variant="p3c-lr")
    swarms = [
        SwarmState(0, 0, 0, 150, 20, 30, 100, 10),
        SwarmState(1, 0, 0, 150, 20, 30, 100, 10),
    ]
    clusters = [
        ClusterState(0, 0, 0, 0, 0, 1, 2, 5, 1),
        ClusterState(1, 0, 0, 0, 0, 1, 2, 5, 1),
    ]
    observations = [
        make_observation(swarm_id=swarm_id, cluster_id=cluster_id)
        for swarm_id in range(2)
        for cluster_id in range(2)
    ]
    scheduler.schedule(
        slot=0, swarms=swarms, clusters=clusters, observations=observations
    )
    assert predictor.calls == 1


def test_no_assignment_when_energy_floor_would_be_crossed() -> None:
    scheduler = P3CScheduler(
        SchedulerConfig(minimum_energy_kj=5.0), estimator(), variant="p3c-lr"
    )
    result = scheduler.schedule(
        slot=0,
        swarms=[SwarmState(0, 0, 0, 150, 20, 30, 5.4, 10)],
        clusters=[ClusterState(0, 0, 0, 0, 0, 1, 2, 5, 1)],
        observations=[make_observation(energy_cost=0.5, residual_energy_kj=5.4)],
    )
    assert result.assignments == {}


def test_scheduler_rejects_invalid_config_and_variant() -> None:
    with pytest.raises(ValueError, match="minimum_energy_kj"):
        P3CScheduler(
            SchedulerConfig(minimum_energy_kj=-1.0), estimator(), variant="p3c-lr"
        )
    with pytest.raises(ValueError, match="unsupported P3C"):
        P3CScheduler(SchedulerConfig(), estimator(), variant="nearest")


@pytest.mark.parametrize(
    "variant, attribute",
    [
        ("no-ann-prediction", "use_prediction"),
        ("no-cache-value", "use_cache_value"),
        ("no-dwell", "use_dwell"),
        ("no-risk-calibration", "use_calibration"),
        ("no-neighbor-cache", "use_neighbor_cache"),
    ],
)
def test_component_ablation_disables_one_capability(variant: str, attribute: str) -> None:
    assert getattr(options_for_variant(variant), attribute) is False


def test_stability_and_event_variants_are_enabled() -> None:
    assert options_for_variant("p3c-sr").use_stability is True
    assert options_for_variant("et-p3c").event_triggered is True
