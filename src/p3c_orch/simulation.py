"""Deterministic synthetic environment and experiment runner."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import ProjectConfig
from .constants import BASELINE_METHODS, P3C_METHODS, WEATHER_ATTENUATION_DB
from .models import (
    AssignmentResult,
    ClusterState,
    PairObservation,
    RiskEstimate,
    SimulationMetrics,
    SwarmState,
)
from .predictor import (
    CalibratedRiskEstimator,
    CurrentMarginPredictor,
    HeuristicMarginPredictor,
    LearnedMarginPredictor,
)
from .scheduler import P3CScheduler

METRIC_FIELDS = (
    "average_delay",
    "p95_delay",
    "energy_kj",
    "handovers_per_100",
    "useful_cache_hit_ratio",
    "outage_probability",
    "throughput",
    "drop_rate",
)


@dataclass(frozen=True)
class SyntheticEnvironmentParameters:
    """Fixed assumptions for the bundled synthetic reference environment."""

    policy_seed_offset: int = 1_000_003
    altitude_range_m: tuple[float, float] = (100.0, 200.0)
    bandwidth_range_mhz: tuple[float, float] = (15.0, 25.0)
    transmit_power_range_dbm: tuple[float, float] = (28.0, 32.0)
    energy_range_kj: tuple[float, float] = (220.0, 280.0)
    service_capacity_range: tuple[float, float] = (8.0, 12.0)
    initial_velocity_std_mps: float = 6.0
    initial_backlog_max: float = 2.0
    swarm_mobility_std_m: float = 2.0
    velocity_persistence: float = 0.85
    velocity_noise_std_mps: float = 1.5
    content_change_probability: float = 0.35
    shadowing_std_db: float = 2.0
    link_budget_gain_db: float = 25.0
    path_loss_coefficient: float = 20.0
    rate_sigmoid_scale_db: float = 4.0
    minimum_rate: float = 0.1
    bandwidth_rate_factor: float = 0.5
    base_energy_cost_kj: float = 0.15
    rate_energy_factor: float = 0.04
    distance_energy_divisor: float = 10_000.0
    neighbor_fetch_energy_kj: float = 0.10
    uncached_delay_penalty: float = 0.5
    delay_service_floor: float = 0.25
    delay_cap: float = 100.0
    maximum_link_distance_m: float = 1_500.0
    content_zipf_exponent: float = 1.35


class SimulationRunner:
    """Run one method and seed in the synthetic reference environment."""

    def __init__(
        self,
        config: ProjectConfig,
        environment: SyntheticEnvironmentParameters | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.environment = environment or SyntheticEnvironmentParameters()
        self._heuristic_predictor = HeuristicMarginPredictor()
        self._reactive_predictor = CurrentMarginPredictor()
        self._learned_predictor = (
            LearnedMarginPredictor(config.predictor.model_path)
            if config.predictor.model_path
            else None
        )

    def run(self, method: str, seed: int) -> SimulationMetrics:
        simulation = self.config.simulation
        environment_rng = np.random.default_rng(seed)
        policy_rng = np.random.default_rng(seed + self.environment.policy_seed_offset)
        swarms = self._initial_swarms(environment_rng)
        clusters = self._initial_clusters(environment_rng)
        initial_energy = sum(swarm.residual_energy_kj for swarm in swarms)
        previous_outages: dict[tuple[int, int], bool] = {}

        delays: list[float] = []
        handovers = 0
        assignments_count = 0
        successful_assignments = 0
        useful_cache_hits = 0
        outages = 0
        throughput = 0.0
        dropped = 0.0
        total_offered = sum(cluster.backlog for cluster in clusters)
        active_cluster_slots = 0

        weather_names = tuple(simulation.weather_probabilities)
        weather_probabilities = tuple(simulation.weather_probabilities.values())

        for slot in range(simulation.slots):
            weather = str(
                environment_rng.choice(weather_names, p=weather_probabilities)
            )
            self._advance_environment(environment_rng, swarms, clusters, slot)
            observations = self._build_observations(
                environment_rng,
                slot,
                weather,
                swarms,
                clusters,
                previous_outages,
            )
            result = self._assign(
                method=method,
                slot=slot,
                swarms=swarms,
                clusters=clusters,
                observations=observations,
                policy_rng=policy_rng,
            )
            observation_by_key = {
                (observation.swarm_id, observation.cluster_id): observation
                for observation in observations
            }
            swarm_by_id = {swarm.swarm_id: swarm for swarm in swarms}
            service_remaining = {
                swarm.swarm_id: swarm.service_capacity for swarm in swarms
            }
            for swarm in swarms:
                swarm.load = 0.0

            for cluster in clusters:
                total_offered += cluster.arrival
                queue_before = cluster.backlog + cluster.arrival
                if queue_before <= 0:
                    continue
                active_cluster_slots += 1
                selected = result.assignments.get(cluster.cluster_id)
                service = 0.0
                if selected is not None:
                    assignments_count += 1
                    swarm = swarm_by_id[selected]
                    observation = observation_by_key[(selected, cluster.cluster_id)]
                    actual_outage = (
                        observation.current_margin_db
                        < self.config.predictor.margin_threshold_db
                    )
                    previous_outages[(selected, cluster.cluster_id)] = actual_outage
                    if actual_outage:
                        outages += 1
                    else:
                        service = min(
                            queue_before,
                            observation.raw_rate,
                            service_remaining[selected],
                        )
                        service_remaining[selected] -= service
                        if service > 0:
                            successful_assignments += 1
                            if observation.local_cache_hit or observation.neighbor_cache_hit:
                                useful_cache_hits += 1
                            self._cache_content(swarm, cluster.requested_content)
                    swarm.residual_energy_kj = max(
                        0.0, swarm.residual_energy_kj - observation.energy_cost
                    )
                    swarm.load = min(
                        1.0, swarm.load + service / max(swarm.service_capacity, 1e-9)
                    )
                    if cluster.previous_swarm is not None and cluster.previous_swarm != selected:
                        handovers += 1
                    cluster.previous_swarm = selected

                throughput += service
                delay = queue_before / max(service, self.environment.delay_service_floor)
                delays.append(min(delay, self.environment.delay_cap))
                cluster.backlog = max(queue_before - service, 0.0)
                if slot >= cluster.deadline_slot and cluster.backlog > 0:
                    dropped += cluster.backlog
                    cluster.backlog = 0.0
                    cluster.deadline_slot = slot + self._sample_deadline(environment_rng)

        final_energy = sum(swarm.residual_energy_kj for swarm in swarms)
        return SimulationMetrics(
            method=method,
            seed=seed,
            average_delay=float(np.mean(delays)) if delays else 0.0,
            p95_delay=float(np.percentile(delays, 95)) if delays else 0.0,
            energy_kj=initial_energy - final_energy,
            handovers_per_100=100.0 * handovers / max(active_cluster_slots, 1),
            useful_cache_hit_ratio=100.0
            * useful_cache_hits
            / max(successful_assignments, 1),
            outage_probability=100.0 * outages / max(assignments_count, 1),
            throughput=throughput,
            drop_rate=100.0 * dropped / max(total_offered, 1.0),
        )

    def _initial_swarms(self, rng: np.random.Generator) -> list[SwarmState]:
        simulation = self.config.simulation
        swarms: list[SwarmState] = []
        for swarm_id in range(simulation.swarms):
            cache_values = rng.choice(
                simulation.library_size,
                size=min(simulation.cache_size, simulation.library_size),
                replace=False,
            )
            swarms.append(
                SwarmState(
                    swarm_id=swarm_id,
                    x_m=float(rng.uniform(0, simulation.area_size_m)),
                    y_m=float(rng.uniform(0, simulation.area_size_m)),
                    altitude_m=float(rng.uniform(*self.environment.altitude_range_m)),
                    bandwidth_mhz=float(
                        rng.uniform(*self.environment.bandwidth_range_mhz)
                    ),
                    transmit_power_dbm=float(
                        rng.uniform(*self.environment.transmit_power_range_dbm)
                    ),
                    residual_energy_kj=float(
                        rng.uniform(*self.environment.energy_range_kj)
                    ),
                    service_capacity=float(
                        rng.uniform(*self.environment.service_capacity_range)
                    ),
                    cache={int(value) for value in cache_values},
                )
            )
        return swarms

    def _initial_clusters(self, rng: np.random.Generator) -> list[ClusterState]:
        simulation = self.config.simulation
        clusters: list[ClusterState] = []
        for cluster_id in range(simulation.clusters):
            clusters.append(
                ClusterState(
                    cluster_id=cluster_id,
                    x_m=float(rng.uniform(0, simulation.area_size_m)),
                    y_m=float(rng.uniform(0, simulation.area_size_m)),
                    velocity_x_mps=float(
                        rng.normal(0.0, self.environment.initial_velocity_std_mps)
                    ),
                    velocity_y_mps=float(
                        rng.normal(0.0, self.environment.initial_velocity_std_mps)
                    ),
                    arrival=0.0,
                    backlog=float(rng.uniform(0.0, self.environment.initial_backlog_max)),
                    deadline_slot=self._sample_deadline(rng),
                    requested_content=self._sample_content(rng),
                )
            )
        return clusters

    def _advance_environment(
        self,
        rng: np.random.Generator,
        swarms: list[SwarmState],
        clusters: list[ClusterState],
        slot: int,
    ) -> None:
        simulation = self.config.simulation
        for swarm in swarms:
            swarm.x_m = self._reflect(
                swarm.x_m
                + float(rng.normal(0.0, self.environment.swarm_mobility_std_m)),
                simulation.area_size_m,
            )
            swarm.y_m = self._reflect(
                swarm.y_m
                + float(rng.normal(0.0, self.environment.swarm_mobility_std_m)),
                simulation.area_size_m,
            )
        for cluster in clusters:
            cluster.velocity_x_mps = (
                self.environment.velocity_persistence * cluster.velocity_x_mps
                + float(rng.normal(0.0, self.environment.velocity_noise_std_mps))
            )
            cluster.velocity_y_mps = (
                self.environment.velocity_persistence * cluster.velocity_y_mps
                + float(rng.normal(0.0, self.environment.velocity_noise_std_mps))
            )
            cluster.x_m = self._reflect(
                cluster.x_m + cluster.velocity_x_mps, simulation.area_size_m
            )
            cluster.y_m = self._reflect(
                cluster.y_m + cluster.velocity_y_mps, simulation.area_size_m
            )
            cluster.arrival = float(rng.poisson(simulation.arrival_rate))
            candidate_deadline = slot + self._sample_deadline(rng)
            candidate_content = self._sample_content(rng)
            if cluster.backlog <= 1e-9 or slot > cluster.deadline_slot:
                cluster.deadline_slot = candidate_deadline
            if rng.random() < self.environment.content_change_probability:
                cluster.requested_content = candidate_content

    def _build_observations(
        self,
        rng: np.random.Generator,
        slot: int,
        weather: str,
        swarms: list[SwarmState],
        clusters: list[ClusterState],
        previous_outages: dict[tuple[int, int], bool],
    ) -> list[PairObservation]:
        observations: list[PairObservation] = []
        cache_union = set().union(*(swarm.cache for swarm in swarms))
        for swarm in swarms:
            for cluster in clusters:
                ground_distance = math.hypot(swarm.x_m - cluster.x_m, swarm.y_m - cluster.y_m)
                distance = math.hypot(ground_distance, swarm.altitude_m)
                elevation = math.degrees(
                    math.atan2(swarm.altitude_m, max(ground_distance, 1e-9))
                )
                speed = math.hypot(cluster.velocity_x_mps, cluster.velocity_y_mps)
                shadowing = float(rng.normal(0.0, self.environment.shadowing_std_db))
                margin = (
                    swarm.transmit_power_dbm
                    + self.environment.link_budget_gain_db
                    - self.environment.path_loss_coefficient
                    * math.log10(distance + 1.0)
                    - WEATHER_ATTENUATION_DB[weather]
                    - shadowing
                )
                rate_fraction = 1.0 / (
                    1.0 + math.exp(-margin / self.environment.rate_sigmoid_scale_db)
                )
                raw_rate = max(
                    self.environment.minimum_rate,
                    self.environment.bandwidth_rate_factor
                    * swarm.bandwidth_mhz
                    * rate_fraction,
                )
                local_hit = cluster.requested_content in swarm.cache
                neighbor_hit = not local_hit and cluster.requested_content in cache_union
                energy_cost = (
                    self.environment.base_energy_cost_kj
                    + self.environment.rate_energy_factor * raw_rate
                    + distance / self.environment.distance_energy_divisor
                    + (
                        self.environment.neighbor_fetch_energy_kj
                        if neighbor_hit
                        else 0.0
                    )
                )
                deadline_pressure = 1.0 / max(cluster.deadline_slot - slot + 1, 1)
                delay_cost = (
                    cluster.backlog / max(raw_rate, self.environment.delay_service_floor)
                    + deadline_pressure
                    + (
                        self.environment.uncached_delay_penalty
                        if not local_hit and not neighbor_hit
                        else 0.0
                    )
                )
                observations.append(
                    PairObservation(
                        slot=slot,
                        swarm_id=swarm.swarm_id,
                        cluster_id=cluster.cluster_id,
                        weather=weather,
                        distance_m=distance,
                        elevation_deg=elevation,
                        relative_speed_mps=speed,
                        bandwidth_mhz=swarm.bandwidth_mhz,
                        transmit_power_dbm=swarm.transmit_power_dbm,
                        current_margin_db=margin,
                        previous_outage=previous_outages.get(
                            (swarm.swarm_id, cluster.cluster_id), False
                        ),
                        local_cache_hit=local_hit,
                        neighbor_cache_hit=neighbor_hit,
                        residual_energy_kj=swarm.residual_energy_kj,
                        load=swarm.load,
                        time_fraction=slot / max(self.config.simulation.slots - 1, 1),
                        raw_rate=raw_rate,
                        energy_cost=energy_cost,
                        delay_cost=delay_cost,
                        neighbor_fetch_penalty=float(neighbor_hit),
                        feasible=distance <= self.environment.maximum_link_distance_m,
                    )
                )
        return observations

    def _assign(
        self,
        *,
        method: str,
        slot: int,
        swarms: list[SwarmState],
        clusters: list[ClusterState],
        observations: list[PairObservation],
        policy_rng: np.random.Generator,
    ) -> AssignmentResult:
        predictor_config = self.config.predictor
        if method in P3C_METHODS:
            if method == "reactive-3c":
                margin_predictor = self._reactive_predictor
            elif self._learned_predictor is not None:
                margin_predictor = self._learned_predictor
            else:
                margin_predictor = self._heuristic_predictor
            estimator = CalibratedRiskEstimator(
                margin_predictor,
                margin_threshold_db=predictor_config.margin_threshold_db,
                residual_scale_db=predictor_config.residual_scale_db,
            )
            scheduler = P3CScheduler(
                self.config.scheduler,
                estimator,
                variant=method,
            )
            return scheduler.schedule(
                slot=slot,
                swarms=swarms,
                clusters=clusters,
                observations=observations,
            )
        return self._baseline_assign(
            method=method,
            swarms=swarms,
            clusters=clusters,
            observations=observations,
            policy_rng=policy_rng,
        )

    def _baseline_assign(
        self,
        *,
        method: str,
        swarms: list[SwarmState],
        clusters: list[ClusterState],
        observations: list[PairObservation],
        policy_rng: np.random.Generator,
    ) -> AssignmentResult:
        if method not in BASELINE_METHODS:
            raise ValueError(f"unsupported baseline method: {method}")
        observations_by_cluster: dict[int, list[PairObservation]] = {
            cluster.cluster_id: [] for cluster in clusters
        }
        for observation in observations:
            if observation.feasible:
                observations_by_cluster[observation.cluster_id].append(observation)
        capacity_left = {swarm.swarm_id: swarm.service_capacity for swarm in swarms}
        energy_left = {swarm.swarm_id: swarm.residual_energy_kj for swarm in swarms}
        assignments: dict[int, int] = {}
        scores: dict[tuple[int, int], float] = {}
        estimates: dict[tuple[int, int], RiskEstimate] = {}
        service_rates: dict[tuple[int, int], float] = {}
        for observation in observations:
            key = (observation.swarm_id, observation.cluster_id)
            risk = float(
                observation.current_margin_db < self.config.predictor.margin_threshold_db
            )
            estimates[key] = RiskEstimate(observation.current_margin_db, risk)
            service_rates[key] = observation.raw_rate * (1.0 - risk)
            if method == "nearest":
                scores[key] = -observation.distance_m
            elif method == "rate-max":
                scores[key] = observation.raw_rate
            elif method == "random":
                scores[key] = 0.0

        for cluster in sorted(clusters, key=lambda item: item.cluster_id):
            if cluster.backlog + cluster.arrival <= 0:
                continue
            candidates = [
                observation
                for observation in observations_by_cluster[cluster.cluster_id]
                if capacity_left[observation.swarm_id] > 0
                and energy_left[observation.swarm_id] - observation.energy_cost
                >= self.config.scheduler.minimum_energy_kj
            ]
            if not candidates:
                continue
            if method == "random":
                selected = candidates[int(policy_rng.integers(0, len(candidates)))]
            else:
                selected = max(
                    candidates,
                    key=lambda observation: (
                        scores[(observation.swarm_id, observation.cluster_id)],
                        -observation.swarm_id,
                    ),
                )
            assignments[cluster.cluster_id] = selected.swarm_id
            capacity_left[selected.swarm_id] -= min(
                selected.raw_rate,
                cluster.backlog + cluster.arrival,
                capacity_left[selected.swarm_id],
            )
            energy_left[selected.swarm_id] -= selected.energy_cost
        return AssignmentResult(assignments, scores, estimates, service_rates)

    def _cache_content(self, swarm: SwarmState, content: int) -> None:
        swarm.cache.add(content)
        while len(swarm.cache) > self.config.simulation.cache_size:
            swarm.cache.remove(min(swarm.cache))

    def _sample_content(self, rng: np.random.Generator) -> int:
        rank = int(rng.zipf(self.environment.content_zipf_exponent)) - 1
        return rank % self.config.simulation.library_size

    def _sample_deadline(self, rng: np.random.Generator) -> int:
        simulation = self.config.simulation
        return int(rng.integers(simulation.deadline_min, simulation.deadline_max + 1))

    @staticmethod
    def _reflect(value: float, upper: float) -> float:
        while value < 0 or value > upper:
            value = -value if value < 0 else 2 * upper - value
        return value


def run_experiment(config: ProjectConfig, output_dir: str | Path) -> dict[str, Path]:
    """Run configured methods and seeds, then write detailed and summary CSV files."""

    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runner = SimulationRunner(config)
    metrics = [
        runner.run(method, seed)
        for method in config.simulation.methods
        for seed in config.simulation.seeds
    ]

    detailed_path = output / "run_metrics.csv"
    with detailed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(metrics[0])))
        writer.writeheader()
        writer.writerows(asdict(item) for item in metrics)

    summary_path = output / "summary.csv"
    summary_rows = _summarize(metrics)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata_path = output / "run_metadata.json"
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": config.to_dict(),
        "detailed_metrics": detailed_path.name,
        "summary": summary_path.name,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "run_metrics": detailed_path,
        "summary": summary_path,
        "metadata": metadata_path,
    }


def _summarize(metrics: list[SimulationMetrics]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    methods = sorted({item.method for item in metrics})
    for method in methods:
        method_rows = [item for item in metrics if item.method == method]
        row: dict[str, float | int | str] = {"method": method, "runs": len(method_rows)}
        for field_name in METRIC_FIELDS:
            values = [float(getattr(item, field_name)) for item in method_rows]
            mean = statistics.fmean(values)
            ci95 = (
                1.96 * statistics.stdev(values) / math.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
            row[f"{field_name}_mean"] = mean
            row[f"{field_name}_ci95"] = ci95
        rows.append(row)
    return rows
