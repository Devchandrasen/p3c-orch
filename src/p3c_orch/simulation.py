"""Complete deterministic simulation and experiment protocol for P3C-Orch."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .channel import AirToGroundChannel, MarkovWeather
from .config import ProjectConfig
from .constants import (
    ABLATION_METHODS,
    BASELINE_METHODS,
    P3C_METHODS,
    PROTOCOL_BASE_BURST_MULTIPLIER,
    PROTOCOL_BASE_BURST_PROBABILITY,
    REGIMES,
    Regime,
)
from .models import (
    AssignmentResult,
    ClusterState,
    ContentItem,
    PairObservation,
    RiskEstimate,
    SimulationMetrics,
    SwarmState,
    UserState,
)
from .predictor import (
    CalibratedRiskEstimator,
    CurrentMarginPredictor,
    HeuristicMarginPredictor,
    LearnedMarginPredictor,
)
from .scheduler import P3CScheduler, options_for_variant

METRIC_FIELDS = (
    "average_delay",
    "p95_delay",
    "energy_kj",
    "handovers_per_100",
    "useful_cache_hit_ratio",
    "local_cache_hit_ratio",
    "neighbor_cache_hit_ratio",
    "outage_probability",
    "clear_outage_probability",
    "rainy_outage_probability",
    "rain_hot_outage_probability",
    "throughput",
    "drop_rate",
    "average_backlog",
    "sla_violation_rate",
    "load_imbalance",
    "jain_fairness",
    "controller_operations",
)


@dataclass(frozen=True)
class EnergyModel:
    movement_base_kj: float = 0.002
    movement_speed_kj: float = 0.0002
    transmission_base_kj: float = 0.004
    transmission_rate_kj: float = 0.0007
    local_cache_kj: float = 0.001
    neighbor_cache_kj: float = 0.004
    infrastructure_fetch_kj: float = 0.010


class SimulationRunner:
    """Run one policy, blind seed, and operating regime."""

    policy_seed_offset = 1_000_003

    def __init__(self, config: ProjectConfig, energy_model: EnergyModel | None = None) -> None:
        config.validate()
        self.config = config
        self.energy_model = energy_model or EnergyModel()
        self.channel = AirToGroundChannel(config.channel)
        self._heuristic_predictor = HeuristicMarginPredictor()
        self._reactive_predictor = CurrentMarginPredictor()
        self._learned_predictor = (
            LearnedMarginPredictor(config.predictor.model_path)
            if config.predictor.model_path
            else None
        )

    def run(
        self,
        method: str,
        seed: int,
        regime_name: str | None = None,
    ) -> SimulationMetrics:
        if method not in BASELINE_METHODS | P3C_METHODS | ABLATION_METHODS:
            raise ValueError(f"unsupported method: {method}")
        regime_key = regime_name or self.config.simulation.regimes[0]
        if regime_key not in REGIMES:
            raise ValueError(f"unsupported regime: {regime_key}")
        regime = REGIMES[regime_key]
        simulation = self.config.simulation
        initialization_rng = np.random.default_rng(seed)
        policy_rng = np.random.default_rng(seed + self.policy_seed_offset)
        contents = self._initial_contents(initialization_rng)
        users = self._initial_users(initialization_rng)
        swarms = self._initial_swarms(initialization_rng, contents, regime)
        clusters = self._initial_clusters(initialization_rng, users, contents)
        weather_rng = np.random.default_rng(seed + 101)
        weather_process = MarkovWeather(
            regime.weather_probabilities,
            simulation.weather_persistence,
            weather_rng,
        )
        initial_energy = sum(swarm.residual_energy_kj for swarm in swarms)
        previous_outages: dict[tuple[int, int], bool] = {}
        delays: list[float] = []
        handovers = 0
        assignments_count = 0
        successful_assignments = 0
        useful_cache_hits = 0
        local_cache_hits = 0
        neighbor_cache_hits = 0
        outages = 0
        weather_assignments = {"clear": 0, "rainy": 0, "rain_hot": 0}
        weather_outages = {"clear": 0, "rainy": 0, "rain_hot": 0}
        throughput = 0.0
        dropped = 0.0
        total_offered = sum(cluster.backlog for cluster in clusters)
        active_cluster_slots = 0
        sla_violations = 0
        backlog_samples: list[float] = []
        load_imbalance_samples: list[float] = []
        served_by_swarm = {swarm.swarm_id: 0.0 for swarm in swarms}
        controller_operations = 0

        for slot in range(simulation.slots):
            slot_rng = np.random.default_rng(seed * 10_000_019 + slot * 1_009 + 17)
            weather = weather_process.advance()
            self._advance_swarms(slot_rng, swarms, regime)
            self._apply_movement_energy(swarms)
            self._advance_users(slot_rng, users, regime)
            self._update_clusters(slot_rng, users, clusters, contents, regime, slot)
            observations = self._build_observations(
                slot_rng,
                slot,
                weather,
                swarms,
                clusters,
                previous_outages,
                method,
            )
            controller_operations += len(observations)
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
                    weather_assignments[weather] += 1
                    if actual_outage:
                        outages += 1
                        weather_outages[weather] += 1
                    else:
                        service = min(
                            queue_before,
                            observation.raw_rate,
                            service_remaining[selected],
                        )
                        service_remaining[selected] -= service
                        if service > 0:
                            successful_assignments += 1
                            served_by_swarm[selected] += service
                            if observation.local_cache_hit or observation.neighbor_cache_hit:
                                useful_cache_hits += 1
                            if observation.local_cache_hit:
                                local_cache_hits += 1
                            elif observation.neighbor_cache_hit:
                                neighbor_cache_hits += 1
                            self._cache_content(
                                swarm,
                                contents[cluster.requested_content],
                                slot,
                            )
                    swarm.residual_energy_kj = max(
                        0.0,
                        swarm.residual_energy_kj - observation.energy_cost,
                    )
                    swarm.load = min(
                        1.0,
                        swarm.load + service / max(swarm.service_capacity, 1e-9),
                    )
                    if (
                        cluster.previous_swarm is not None
                        and cluster.previous_swarm != selected
                    ):
                        handovers += 1
                    cluster.previous_swarm = selected
                throughput += service
                delay = queue_before / max(service, 0.25)
                delays.append(min(delay, 100.0))
                cluster.backlog = max(queue_before - service, 0.0)
                backlog_samples.append(cluster.backlog)
                if slot >= cluster.deadline_slot and cluster.backlog > 0:
                    dropped += cluster.backlog
                    sla_violations += 1
                    cluster.backlog = 0.0
                    cluster.deadline_slot = slot + self._sample_deadline(slot_rng)
            loads = [swarm.load for swarm in swarms]
            load_imbalance_samples.append(float(np.std(loads)))
            self._purge_expired_cache(swarms, slot)

        final_energy = sum(swarm.residual_energy_kj for swarm in swarms)
        served_values = np.asarray(list(served_by_swarm.values()), dtype=float)
        fairness = self._jain_fairness(served_values)
        return SimulationMetrics(
            regime=regime_key,
            method=method,
            seed=seed,
            average_delay=float(np.mean(delays)) if delays else 0.0,
            p95_delay=float(np.percentile(delays, 95)) if delays else 0.0,
            energy_kj=initial_energy - final_energy,
            handovers_per_100=100.0 * handovers / max(active_cluster_slots, 1),
            useful_cache_hit_ratio=(
                100.0 * useful_cache_hits / max(successful_assignments, 1)
            ),
            local_cache_hit_ratio=(
                100.0 * local_cache_hits / max(successful_assignments, 1)
            ),
            neighbor_cache_hit_ratio=(
                100.0 * neighbor_cache_hits / max(successful_assignments, 1)
            ),
            outage_probability=100.0 * outages / max(assignments_count, 1),
            clear_outage_probability=(
                100.0 * weather_outages["clear"] / max(weather_assignments["clear"], 1)
            ),
            rainy_outage_probability=(
                100.0 * weather_outages["rainy"] / max(weather_assignments["rainy"], 1)
            ),
            rain_hot_outage_probability=(
                100.0
                * weather_outages["rain_hot"]
                / max(weather_assignments["rain_hot"], 1)
            ),
            throughput=throughput,
            drop_rate=100.0 * dropped / max(total_offered, 1.0),
            average_backlog=(
                float(np.mean(backlog_samples)) if backlog_samples else 0.0
            ),
            sla_violation_rate=(
                100.0 * sla_violations / max(active_cluster_slots, 1)
            ),
            load_imbalance=(
                float(np.mean(load_imbalance_samples))
                if load_imbalance_samples
                else 0.0
            ),
            jain_fairness=100.0 * fairness,
            controller_operations=controller_operations,
        )

    def _initial_contents(self, rng: np.random.Generator) -> dict[int, ContentItem]:
        simulation = self.config.simulation
        return {
            content_id: ContentItem(
                content_id,
                float(
                    rng.uniform(
                        simulation.file_size_min_mb,
                        simulation.file_size_max_mb,
                    )
                ),
            )
            for content_id in range(simulation.library_size)
        }

    def _initial_users(self, rng: np.random.Generator) -> list[UserState]:
        simulation = self.config.simulation
        return [
            UserState(
                user_id=user_id,
                x_m=float(rng.uniform(0, simulation.area_size_m)),
                y_m=float(rng.uniform(0, simulation.area_size_m)),
                velocity_x_mps=float(rng.normal(0.0, simulation.user_speed_std_mps)),
                velocity_y_mps=float(rng.normal(0.0, simulation.user_speed_std_mps)),
            )
            for user_id in range(simulation.users)
        ]

    def _initial_swarms(
        self,
        rng: np.random.Generator,
        contents: dict[int, ContentItem],
        regime: Regime,
    ) -> list[SwarmState]:
        simulation = self.config.simulation
        swarms: list[SwarmState] = []
        for swarm_id in range(simulation.swarms):
            uav_count = int(
                rng.integers(
                    simulation.uavs_per_swarm_min,
                    simulation.uavs_per_swarm_max + 1,
                )
            )
            swarm = SwarmState(
                swarm_id=swarm_id,
                x_m=float(rng.uniform(0, simulation.area_size_m)),
                y_m=float(rng.uniform(0, simulation.area_size_m)),
                altitude_m=float(rng.uniform(100.0, 200.0)),
                bandwidth_mhz=float(rng.uniform(15.0, 25.0)),
                transmit_power_dbm=float(rng.uniform(28.0, 32.0)),
                residual_energy_kj=float(rng.uniform(220.0, 280.0)),
                service_capacity=regime.capacity_multiplier * uav_count * 7.5,
                uav_count=uav_count,
            )
            candidates = rng.permutation(simulation.library_size)
            for content_id in candidates:
                item = contents[int(content_id)]
                if len(swarm.cache) >= simulation.cache_size:
                    break
                if swarm.cache_used_mb + item.size_mb > simulation.cache_capacity_mb:
                    continue
                swarm.cache.add(item.content_id)
                swarm.cache_sizes_mb[item.content_id] = item.size_mb
                swarm.cache_expiry[item.content_id] = simulation.content_ttl_slots
            swarms.append(swarm)
        return swarms

    def _initial_clusters(
        self,
        rng: np.random.Generator,
        users: list[UserState],
        contents: dict[int, ContentItem],
    ) -> list[ClusterState]:
        simulation = self.config.simulation
        order = np.argsort([user.x_m for user in users])
        groups = np.array_split(order, simulation.clusters)
        clusters: list[ClusterState] = []
        for cluster_id, group in enumerate(groups):
            members = [users[int(index)] for index in group]
            content = contents[self._sample_content(rng, 1.0)]
            clusters.append(
                ClusterState(
                    cluster_id=cluster_id,
                    x_m=float(np.mean([user.x_m for user in members])),
                    y_m=float(np.mean([user.y_m for user in members])),
                    velocity_x_mps=float(
                        np.mean([user.velocity_x_mps for user in members])
                    ),
                    velocity_y_mps=float(
                        np.mean([user.velocity_y_mps for user in members])
                    ),
                    arrival=0.0,
                    backlog=float(rng.uniform(0.0, 2.0)),
                    deadline_slot=self._sample_deadline(rng),
                    requested_content=content.content_id,
                    content_size_mb=content.size_mb,
                    user_ids=tuple(user.user_id for user in members),
                )
            )
        return clusters

    def _advance_swarms(
        self,
        rng: np.random.Generator,
        swarms: list[SwarmState],
        regime: Regime,
    ) -> None:
        area = self.config.simulation.area_size_m
        step_std = 2.0 * regime.mobility_multiplier
        for swarm in swarms:
            swarm.x_m = self._reflect(
                swarm.x_m + float(rng.normal(0.0, step_std)), area
            )
            swarm.y_m = self._reflect(
                swarm.y_m + float(rng.normal(0.0, step_std)), area
            )

    def _apply_movement_energy(self, swarms: list[SwarmState]) -> None:
        for swarm in swarms:
            cost = self.energy_model.movement_base_kj * swarm.uav_count
            swarm.residual_energy_kj = max(0.0, swarm.residual_energy_kj - cost)

    def _advance_users(
        self,
        rng: np.random.Generator,
        users: list[UserState],
        regime: Regime,
    ) -> None:
        simulation = self.config.simulation
        alpha = simulation.gauss_markov_alpha
        innovation_scale = (
            math.sqrt(1.0 - alpha**2)
            * simulation.mobility_noise_std_mps
            * regime.mobility_multiplier
        )
        for user in users:
            user.velocity_x_mps = alpha * user.velocity_x_mps + float(
                rng.normal(0.0, innovation_scale)
            )
            user.velocity_y_mps = alpha * user.velocity_y_mps + float(
                rng.normal(0.0, innovation_scale)
            )
            user.x_m = self._reflect(
                user.x_m + user.velocity_x_mps,
                simulation.area_size_m,
            )
            user.y_m = self._reflect(
                user.y_m + user.velocity_y_mps,
                simulation.area_size_m,
            )

    def _update_clusters(
        self,
        rng: np.random.Generator,
        users: list[UserState],
        clusters: list[ClusterState],
        contents: dict[int, ContentItem],
        regime: Regime,
        slot: int,
    ) -> None:
        simulation = self.config.simulation
        positions = np.asarray([(user.x_m, user.y_m) for user in users], dtype=float)
        centroids = np.asarray([(cluster.x_m, cluster.y_m) for cluster in clusters])
        labels = np.zeros(len(users), dtype=int)
        for _ in range(4):
            distances = np.linalg.norm(
                positions[:, None, :] - centroids[None, :, :], axis=2
            )
            labels = np.argmin(distances, axis=1)
            for cluster_id in range(len(clusters)):
                indices = np.flatnonzero(labels == cluster_id)
                if len(indices) == 0:
                    farthest = int(np.argmax(np.min(distances, axis=1)))
                    labels[farthest] = cluster_id
                    indices = np.asarray([farthest])
                centroids[cluster_id] = np.mean(positions[indices], axis=0)

        base_request_probability = min(
            0.95,
            simulation.arrival_rate
            * regime.arrival_multiplier
            * simulation.clusters
            / simulation.users,
        )
        burst_probability = min(
            1.0,
            simulation.burst_probability
            * regime.burst_probability
            / PROTOCOL_BASE_BURST_PROBABILITY,
        )
        effective_burst_multiplier = (
            simulation.burst_multiplier
            * regime.burst_multiplier
            / PROTOCOL_BASE_BURST_MULTIPLIER
        )
        burst = rng.random() < burst_probability
        request_probability = min(
            0.98,
            base_request_probability
            * (effective_burst_multiplier if burst else 1.0),
        )
        requests_by_user: dict[int, int] = {}
        for user in users:
            if rng.random() < request_probability:
                requests_by_user[user.user_id] = self._sample_content(
                    rng, regime.cache_pressure
                )

        for cluster in clusters:
            indices = np.flatnonzero(labels == cluster.cluster_id)
            members = [users[int(index)] for index in indices]
            previous_x, previous_y = cluster.x_m, cluster.y_m
            cluster.x_m = float(centroids[cluster.cluster_id, 0])
            cluster.y_m = float(centroids[cluster.cluster_id, 1])
            cluster.velocity_x_mps = cluster.x_m - previous_x
            cluster.velocity_y_mps = cluster.y_m - previous_y
            cluster.user_ids = tuple(user.user_id for user in members)
            content_requests = [
                requests_by_user[user.user_id]
                for user in members
                if user.user_id in requests_by_user
            ]
            cluster.arrival = sum(contents[item].size_mb for item in content_requests)
            if content_requests:
                requested_content = Counter(content_requests).most_common(1)[0][0]
                cluster.requested_content = requested_content
                cluster.content_size_mb = contents[requested_content].size_mb
            candidate_deadline = slot + self._sample_deadline(rng)
            if cluster.backlog <= 1e-9 or slot > cluster.deadline_slot:
                cluster.deadline_slot = candidate_deadline

    def _build_observations(
        self,
        rng: np.random.Generator,
        slot: int,
        weather: str,
        swarms: list[SwarmState],
        clusters: list[ClusterState],
        previous_outages: dict[tuple[int, int], bool],
        method: str,
    ) -> list[PairObservation]:
        observations: list[PairObservation] = []
        for swarm in swarms:
            for cluster in clusters:
                ground_distance = math.hypot(
                    swarm.x_m - cluster.x_m,
                    swarm.y_m - cluster.y_m,
                )
                relative_speed = math.hypot(
                    cluster.velocity_x_mps,
                    cluster.velocity_y_mps,
                )
                sample = self.channel.sample(
                    horizontal_distance_m=ground_distance,
                    altitude_m=swarm.altitude_m,
                    bandwidth_mhz=swarm.bandwidth_mhz,
                    transmit_power_dbm=swarm.transmit_power_dbm,
                    relative_speed_mps=relative_speed,
                    weather=weather,
                    rng=rng,
                )
                content_id = cluster.requested_content
                local_present = content_id in swarm.cache
                local_hit = (
                    local_present and swarm.cache_expiry.get(content_id, -1) > slot
                )
                stale_penalty = float(local_present and not local_hit)
                neighbor_hit = any(
                    content_id in neighbor.cache
                    and neighbor.cache_expiry.get(content_id, -1) > slot
                    for neighbor in swarms
                    if neighbor.swarm_id != swarm.swarm_id
                )
                if method in {"no-cache-value", "no-neighbor-cache"}:
                    neighbor_hit = False
                transmission_energy = (
                    self.energy_model.transmission_base_kj
                    + self.energy_model.transmission_rate_kj
                    * min(sample.rate_mb_per_slot, cluster.backlog + cluster.arrival)
                )
                movement_energy = self.energy_model.movement_speed_kj * (
                    1.0 + relative_speed / 10.0
                )
                if local_hit:
                    cache_energy = self.energy_model.local_cache_kj
                elif neighbor_hit:
                    cache_energy = self.energy_model.neighbor_cache_kj
                else:
                    cache_energy = self.energy_model.infrastructure_fetch_kj
                energy_cost = transmission_energy + movement_energy + cache_energy
                deadline_risk = 1.0 / max(cluster.deadline_slot - slot + 1, 1)
                queue_risk = cluster.backlog / max(swarm.service_capacity, 1e-9)
                cache_miss_penalty = float(not local_hit and not neighbor_hit)
                delay_cost = (
                    0.30 * deadline_risk
                    + 0.35 * queue_risk
                    + 0.15 * cache_miss_penalty
                )
                observations.append(
                    PairObservation(
                        slot=slot,
                        swarm_id=swarm.swarm_id,
                        cluster_id=cluster.cluster_id,
                        weather=weather,
                        distance_m=sample.distance_m,
                        elevation_deg=sample.elevation_deg,
                        relative_speed_mps=relative_speed,
                        bandwidth_mhz=swarm.bandwidth_mhz,
                        transmit_power_dbm=swarm.transmit_power_dbm,
                        current_margin_db=sample.margin_db,
                        previous_outage=previous_outages.get(
                            (swarm.swarm_id, cluster.cluster_id), False
                        ),
                        local_cache_hit=local_hit,
                        neighbor_cache_hit=neighbor_hit,
                        residual_energy_kj=swarm.residual_energy_kj,
                        load=swarm.load,
                        time_fraction=slot / max(self.config.simulation.slots - 1, 1),
                        raw_rate=sample.rate_mb_per_slot,
                        energy_cost=energy_cost,
                        delay_cost=delay_cost,
                        stale_penalty=stale_penalty,
                        neighbor_fetch_penalty=float(neighbor_hit),
                        deadline_risk=deadline_risk,
                        queue_risk=queue_risk,
                        cache_miss_penalty=cache_miss_penalty,
                        transmission_energy_kj=transmission_energy,
                        mobility_energy_kj=movement_energy,
                        cache_energy_kj=cache_energy,
                        feasible=(
                            sample.distance_m
                            <= self.config.channel.maximum_link_distance_m
                            and swarm.residual_energy_kj
                            >= self.config.scheduler.minimum_energy_kj
                        ),
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
        if method in P3C_METHODS | ABLATION_METHODS:
            options = options_for_variant(method)
            if not options.use_prediction:
                margin_predictor = self._reactive_predictor
            elif self._learned_predictor is not None:
                margin_predictor = self._learned_predictor
            else:
                margin_predictor = self._heuristic_predictor
            estimator = CalibratedRiskEstimator(
                margin_predictor,
                margin_threshold_db=self.config.predictor.margin_threshold_db,
                residual_scale_db=(
                    self._learned_predictor.residual_scale_db
                    if self._learned_predictor is not None
                    and self._learned_predictor.residual_scale_db
                    else self.config.predictor.residual_scale_db
                ),
                calibrated=options.use_calibration,
            )
            scheduler = P3CScheduler(
                self.config.scheduler,
                estimator,
                variant=method,
                options=options,
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
            if observation.feasible and observation.cluster_id in observations_by_cluster:
                observations_by_cluster[observation.cluster_id].append(observation)
        capacity_left = {swarm.swarm_id: swarm.service_capacity for swarm in swarms}
        energy_left = {swarm.swarm_id: swarm.residual_energy_kj for swarm in swarms}
        assignments: dict[int, int] = {}
        scores: dict[tuple[int, int], float] = {}
        estimates: dict[tuple[int, int], RiskEstimate] = {}
        service_rates: dict[tuple[int, int], float] = {}
        mucco_raw: dict[tuple[int, int], float] = {}
        for observation in observations:
            key = (observation.swarm_id, observation.cluster_id)
            risk = float(
                observation.current_margin_db
                < self.config.predictor.margin_threshold_db
            )
            estimates[key] = RiskEstimate(observation.current_margin_db, risk)
            service_rates[key] = observation.raw_rate * (1.0 - risk)
            if method == "nearest":
                scores[key] = -observation.distance_m
            elif method == "rate-max":
                scores[key] = observation.raw_rate
            elif method == "random":
                scores[key] = 0.0
            else:
                cache_value = (
                    1.0 * float(observation.local_cache_hit)
                    + 0.5 * float(observation.neighbor_cache_hit)
                )
                energy_efficiency = observation.raw_rate / max(
                    observation.energy_cost, 1e-9
                )
                fairness = 1.0 / (1.0 + max(observation.load, 0.0))
                mucco_raw[key] = 0.45 * cache_value + 0.35 * energy_efficiency + 0.20 * fairness
        if method == "mucco-like" and mucco_raw:
            values = np.asarray(list(mucco_raw.values()), dtype=float)
            low, high = float(np.min(values)), float(np.max(values))
            scale = max(high - low, 1e-12)
            scores.update({key: (value - low) / scale for key, value in mucco_raw.items()})

        for cluster in sorted(clusters, key=lambda item: item.cluster_id):
            if cluster.backlog + cluster.arrival <= 0:
                continue
            candidates = [
                observation
                for observation in observations_by_cluster[cluster.cluster_id]
                if capacity_left[observation.swarm_id] > 0
                and energy_left[observation.swarm_id] - observation.energy_cost
                >= self.config.scheduler.minimum_energy_kj
                and service_rates[(observation.swarm_id, observation.cluster_id)] > 0
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
            reserved = min(
                selected.raw_rate,
                cluster.backlog + cluster.arrival,
                capacity_left[selected.swarm_id],
            )
            capacity_left[selected.swarm_id] -= reserved
            energy_left[selected.swarm_id] -= selected.energy_cost
        return AssignmentResult(assignments, scores, estimates, service_rates)

    def _cache_content(self, swarm: SwarmState, content: ContentItem, slot: int) -> None:
        simulation = self.config.simulation
        self._purge_swarm_cache(swarm, slot)
        while (
            len(swarm.cache) >= simulation.cache_size
            or swarm.cache_used_mb + content.size_mb > simulation.cache_capacity_mb
        ):
            if not swarm.cache:
                return
            victim = min(
                swarm.cache,
                key=lambda content_id: swarm.cache_expiry.get(content_id, -1),
            )
            self._remove_cache_entry(swarm, victim)
        swarm.cache.add(content.content_id)
        swarm.cache_sizes_mb[content.content_id] = content.size_mb
        swarm.cache_expiry[content.content_id] = slot + simulation.content_ttl_slots

    def _purge_expired_cache(self, swarms: list[SwarmState], slot: int) -> None:
        for swarm in swarms:
            self._purge_swarm_cache(swarm, slot)

    def _purge_swarm_cache(self, swarm: SwarmState, slot: int) -> None:
        expired = [
            content_id
            for content_id in swarm.cache
            if swarm.cache_expiry.get(content_id, -1) <= slot
        ]
        for content_id in expired:
            self._remove_cache_entry(swarm, content_id)

    @staticmethod
    def _remove_cache_entry(swarm: SwarmState, content_id: int) -> None:
        swarm.cache.discard(content_id)
        swarm.cache_expiry.pop(content_id, None)
        swarm.cache_sizes_mb.pop(content_id, None)

    def _sample_content(self, rng: np.random.Generator, pressure: float) -> int:
        exponent = max(1.01, self.config.simulation.zipf_exponent / pressure)
        rank = int(rng.zipf(exponent)) - 1
        return rank % self.config.simulation.library_size

    def _sample_deadline(self, rng: np.random.Generator) -> int:
        simulation = self.config.simulation
        return int(rng.integers(simulation.deadline_min, simulation.deadline_max + 1))

    @staticmethod
    def _jain_fairness(values: np.ndarray) -> float:
        denominator = len(values) * float(np.sum(values**2))
        if denominator <= 0:
            return 1.0
        return float(np.sum(values) ** 2 / denominator)

    @staticmethod
    def _reflect(value: float, upper: float) -> float:
        while value < 0 or value > upper:
            value = -value if value < 0 else 2 * upper - value
        return value


def run_experiment(config: ProjectConfig, output_dir: str | Path) -> dict[str, Path]:
    """Run configured regimes, methods, and seeds and write machine-readable outputs."""

    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runner = SimulationRunner(config)
    metrics = [
        runner.run(method, seed, regime)
        for regime in config.simulation.regimes
        for method in config.simulation.methods
        for seed in config.simulation.seeds
    ]
    detailed_path = output / "run_metrics.csv"
    with detailed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(metrics[0])))
        writer.writeheader()
        writer.writerows(asdict(item) for item in metrics)
    summary_path = output / "summary.csv"
    summary_rows = summarize_metrics(metrics)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    metadata_path = output / "run_metadata.json"
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": config.to_dict(),
        "protocol": {
            "paired_seeds": True,
            "bundled_results": False,
            "results_origin": "generated_by_this_run",
        },
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


def summarize_metrics(
    metrics: list[SimulationMetrics],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    groups = sorted({(item.regime, item.method) for item in metrics})
    for regime, method in groups:
        method_rows = [
            item for item in metrics if item.regime == regime and item.method == method
        ]
        row: dict[str, float | int | str] = {
            "regime": regime,
            "method": method,
            "runs": len(method_rows),
        }
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
