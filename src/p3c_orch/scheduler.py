"""P3C-LR and P3C-SR swarm-to-cluster scheduling."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable

from .config import SchedulerConfig
from .constants import P3C_METHODS
from .models import (
    AssignmentResult,
    ClusterState,
    PairObservation,
    RiskEstimate,
    SwarmState,
)
from .predictor import CalibratedRiskEstimator

PairKey = tuple[int, int]


def _normalize(values: dict[Hashable, float]) -> dict[Hashable, float]:
    if not values:
        return {}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("scheduler terms must contain only finite values")
    low = min(values.values())
    high = max(values.values())
    if abs(high - low) < 1e-12:
        return {key: 0.5 for key in values}
    scale = high - low
    return {key: (value - low) / scale for key, value in values.items()}


class P3CScheduler:
    """Score feasible pairs and assign clusters with capacity and dwell constraints."""

    def __init__(
        self,
        config: SchedulerConfig,
        risk_estimator: CalibratedRiskEstimator,
        *,
        variant: str = "p3c-lr",
    ) -> None:
        config.validate()
        if variant not in P3C_METHODS:
            raise ValueError(f"unsupported P3C scheduler variant: {variant}")
        self.config = config
        self.risk_estimator = risk_estimator
        self.variant = variant

    def schedule(
        self,
        *,
        slot: int,
        swarms: list[SwarmState],
        clusters: list[ClusterState],
        observations: list[PairObservation],
    ) -> AssignmentResult:
        swarm_by_id = {swarm.swarm_id: swarm for swarm in swarms}
        cluster_by_id = {cluster.cluster_id: cluster for cluster in clusters}
        raw_terms: dict[str, dict[PairKey, float]] = defaultdict(dict)
        estimates: dict[PairKey, RiskEstimate] = {}
        service_rates: dict[PairKey, float] = {}

        eligible_observations: list[PairObservation] = []
        for observation in observations:
            swarm = swarm_by_id.get(observation.swarm_id)
            cluster = cluster_by_id.get(observation.cluster_id)
            if swarm is None or cluster is None or not observation.feasible:
                continue
            if swarm.residual_energy_kj < self.config.minimum_energy_kj:
                continue
            eligible_observations.append(observation)

        risk_estimates = self.risk_estimator.estimate_many(eligible_observations)
        for observation, estimate in zip(
            eligible_observations, risk_estimates, strict=True
        ):
            swarm = swarm_by_id[observation.swarm_id]
            cluster = cluster_by_id[observation.cluster_id]
            key = (observation.swarm_id, observation.cluster_id)
            estimates[key] = estimate
            service = observation.raw_rate * (1.0 - estimate.outage_probability)
            service_rates[key] = max(service, 0.0)

            cache_value = (1.0 - estimate.outage_probability) * (
                float(observation.local_cache_hit)
                + 0.5 * float(observation.neighbor_cache_hit)
                - 0.25 * observation.stale_penalty
                - 0.15 * observation.neighbor_fetch_penalty
            )
            is_previous = cluster.previous_swarm == swarm.swarm_id
            switch_cost = 0.0
            if cluster.previous_swarm is not None and not is_previous:
                switch_cost = 0.2 + observation.distance_m / 1000.0

            raw_terms["margin"][key] = estimate.predicted_margin_db
            raw_terms["cache"][key] = cache_value
            raw_terms["fairness"][key] = 1.0 / (1.0 + max(observation.load, 0.0))
            raw_terms["dwell"][key] = float(is_previous)
            raw_terms["delay"][key] = observation.delay_cost + estimate.outage_probability
            raw_terms["energy"][key] = observation.energy_cost
            raw_terms["outage"][key] = estimate.outage_probability
            raw_terms["switch"][key] = switch_cost
            raw_terms["load"][key] = max(observation.load, 0.0)
            raw_terms["stability"][key] = (
                cluster.urgency(slot) * service
                - self.config.load_regularization * observation.load * service
            )

        normalized = {name: _normalize(values) for name, values in raw_terms.items()}
        weights = self.config.weights
        scores: dict[PairKey, float] = {}
        for key in estimates:
            score = (
                weights.margin * normalized["margin"][key]
                + weights.cache * normalized["cache"][key]
                + weights.fairness * normalized["fairness"][key]
                + weights.dwell * normalized["dwell"][key]
                - weights.delay * normalized["delay"][key]
                - weights.energy * normalized["energy"][key]
                - weights.outage * normalized["outage"][key]
                - weights.switch * normalized["switch"][key]
                - weights.load * normalized["load"][key]
            )
            if self.variant == "p3c-sr":
                score += self.config.stability_weight * normalized["stability"][key]
            scores[key] = score

        assignments = self._assign(
            slot=slot,
            swarms=swarm_by_id,
            clusters=cluster_by_id,
            observations=observations,
            scores=scores,
            service_rates=service_rates,
        )
        return AssignmentResult(
            assignments=assignments,
            scores=scores,
            estimates=estimates,
            service_rates=service_rates,
        )

    def _assign(
        self,
        *,
        slot: int,
        swarms: dict[int, SwarmState],
        clusters: dict[int, ClusterState],
        observations: list[PairObservation],
        scores: dict[PairKey, float],
        service_rates: dict[PairKey, float],
    ) -> dict[int, int]:
        observation_by_key = {
            (observation.swarm_id, observation.cluster_id): observation
            for observation in observations
        }
        candidates_by_cluster: dict[int, list[int]] = defaultdict(list)
        for swarm_id, cluster_id in scores:
            candidates_by_cluster[cluster_id].append(swarm_id)

        capacity_left = {
            swarm_id: max(0.0, swarm.service_capacity * (1.0 - min(max(swarm.load, 0.0), 1.0)))
            for swarm_id, swarm in swarms.items()
        }
        energy_left = {swarm_id: swarm.residual_energy_kj for swarm_id, swarm in swarms.items()}
        ordered_clusters = sorted(
            clusters.values(),
            key=lambda cluster: (cluster.urgency(slot), cluster.backlog, -cluster.cluster_id),
            reverse=True,
        )
        assignments: dict[int, int] = {}
        for cluster in ordered_clusters:
            candidate_ids = []
            for swarm_id in candidates_by_cluster.get(cluster.cluster_id, []):
                key = (swarm_id, cluster.cluster_id)
                observation = observation_by_key[key]
                has_energy = (
                    energy_left[swarm_id] - observation.energy_cost
                    >= self.config.minimum_energy_kj
                )
                if capacity_left[swarm_id] > 0 and has_energy and service_rates[key] > 0:
                    candidate_ids.append(swarm_id)
            if not candidate_ids:
                continue

            selected = max(
                candidate_ids,
                key=lambda swarm_id: (scores[(swarm_id, cluster.cluster_id)], -swarm_id),
            )
            previous = cluster.previous_swarm
            if previous in candidate_ids:
                best_score = scores[(selected, cluster.cluster_id)]
                previous_score = scores[(previous, cluster.cluster_id)]
                if best_score - previous_score <= self.config.dwell_threshold:
                    selected = previous

            key = (selected, cluster.cluster_id)
            observation = observation_by_key[key]
            requested = max(cluster.backlog + cluster.arrival, 0.0)
            reserved = min(service_rates[key], capacity_left[selected], requested)
            if reserved <= 0:
                continue
            assignments[cluster.cluster_id] = selected
            capacity_left[selected] -= reserved
            energy_left[selected] -= observation.energy_cost

        return assignments
