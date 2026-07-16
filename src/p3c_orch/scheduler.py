"""P3C-LR, P3C-SR, ET-P3C, and component-ablation scheduling."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass

from .config import SchedulerConfig
from .constants import ABLATION_METHODS, P3C_METHODS
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


@dataclass(frozen=True)
class PolicyOptions:
    use_prediction: bool = True
    use_cache_value: bool = True
    use_neighbor_cache: bool = True
    use_dwell: bool = True
    use_calibration: bool = True
    use_stability: bool = False
    event_triggered: bool = False


def options_for_variant(variant: str) -> PolicyOptions:
    if variant == "reactive-3c":
        return PolicyOptions(use_prediction=False)
    if variant == "p3c-sr":
        return PolicyOptions(use_stability=True)
    if variant == "et-p3c":
        return PolicyOptions(event_triggered=True)
    if variant == "no-ann-prediction":
        return PolicyOptions(use_prediction=False)
    if variant == "no-cache-value":
        return PolicyOptions(use_cache_value=False, use_neighbor_cache=False)
    if variant == "no-dwell":
        return PolicyOptions(use_dwell=False)
    if variant == "no-risk-calibration":
        return PolicyOptions(use_calibration=False)
    if variant == "no-neighbor-cache":
        return PolicyOptions(use_neighbor_cache=False)
    return PolicyOptions()


class P3CScheduler:
    """Score feasible pairs and enforce capacity, energy, and dwell constraints."""

    def __init__(
        self,
        config: SchedulerConfig,
        risk_estimator: CalibratedRiskEstimator,
        *,
        variant: str = "p3c-lr",
        options: PolicyOptions | None = None,
    ) -> None:
        config.validate()
        if variant not in P3C_METHODS | ABLATION_METHODS:
            raise ValueError(f"unsupported P3C scheduler variant: {variant}")
        self.config = config
        self.risk_estimator = risk_estimator
        self.variant = variant
        self.options = options or options_for_variant(variant)

    def schedule(
        self,
        *,
        slot: int,
        swarms: list[SwarmState],
        clusters: list[ClusterState],
        observations: list[PairObservation],
    ) -> AssignmentResult:
        self._validate_identifiers(swarms, clusters, observations)
        swarm_by_id = {swarm.swarm_id: swarm for swarm in swarms}
        cluster_by_id = {cluster.cluster_id: cluster for cluster in clusters}
        raw_terms: dict[str, dict[PairKey, float]] = defaultdict(dict)
        estimates: dict[PairKey, RiskEstimate] = {}
        service_rates: dict[PairKey, float] = {}

        eligible = [
            observation
            for observation in observations
            if observation.swarm_id in swarm_by_id
            and observation.cluster_id in cluster_by_id
            and observation.feasible
            and swarm_by_id[observation.swarm_id].residual_energy_kj
            >= self.config.minimum_energy_kj
        ]
        risk_estimates = self.risk_estimator.estimate_many(eligible)
        for observation, estimate in zip(eligible, risk_estimates, strict=True):
            swarm = swarm_by_id[observation.swarm_id]
            cluster = cluster_by_id[observation.cluster_id]
            key = (observation.swarm_id, observation.cluster_id)
            estimates[key] = estimate
            service = max(
                observation.raw_rate * (1.0 - estimate.outage_probability), 0.0
            )
            service_rates[key] = service

            neighbor_hit = (
                observation.neighbor_cache_hit and self.options.use_neighbor_cache
            )
            cache_value = (1.0 - estimate.outage_probability) * (
                self.config.cache_local_weight * float(observation.local_cache_hit)
                + self.config.cache_neighbor_weight * float(neighbor_hit)
                - self.config.cache_stale_penalty * observation.stale_penalty
                - self.config.cache_fetch_penalty
                * observation.neighbor_fetch_penalty
                * float(neighbor_hit)
            )
            if not self.options.use_cache_value:
                cache_value = 0.0
            is_previous = cluster.previous_swarm == swarm.swarm_id
            switch_cost = 0.0
            if cluster.previous_swarm is not None and not is_previous:
                switch_cost = (
                    self.config.switch_fixed_cost
                    + self.config.switch_distance_cost * observation.distance_m
                )

            delay_risk = (
                self.config.delay_deadline_weight * observation.deadline_risk
                + self.config.delay_queue_weight * observation.queue_risk
                + self.config.delay_miss_weight * observation.cache_miss_penalty
                + self.config.delay_outage_weight * estimate.outage_probability
            )
            if delay_risk <= 0:
                delay_risk = observation.delay_cost + estimate.outage_probability
            raw_terms["margin"][key] = estimate.predicted_margin_db
            raw_terms["cache"][key] = cache_value
            raw_terms["fairness"][key] = 1.0 / (1.0 + max(swarm.load, 0.0))
            raw_terms["dwell"][key] = float(is_previous and self.options.use_dwell)
            raw_terms["delay"][key] = delay_risk
            raw_terms["energy"][key] = observation.energy_cost
            raw_terms["outage"][key] = estimate.outage_probability
            raw_terms["switch"][key] = switch_cost
            raw_terms["load"][key] = max(swarm.load, 0.0)
            raw_terms["stability"][key] = (
                cluster.urgency(slot) * service
                - self.config.load_regularization * swarm.load * service
            )

        normalized = {name: _normalize(values) for name, values in raw_terms.items()}
        scores = self._score_pairs(estimates, normalized)
        assignments = self._assign(
            slot=slot,
            swarms=swarm_by_id,
            clusters=cluster_by_id,
            observations=observations,
            scores=scores,
            service_rates=service_rates,
        )
        return AssignmentResult(assignments, scores, estimates, service_rates)

    def _score_pairs(
        self,
        estimates: dict[PairKey, RiskEstimate],
        normalized: dict[str, dict[PairKey, float]],
    ) -> dict[PairKey, float]:
        weights = self.config.weights
        scores: dict[PairKey, float] = {}
        for key, estimate in estimates.items():
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
            if self.options.use_stability:
                score += self.config.stability_weight * normalized["stability"][key]
            if (
                self.options.event_triggered
                and estimate.outage_probability >= self.config.event_risk_threshold
            ):
                impulse = min(
                    self.config.event_impulse_cap,
                    self.config.stability_weight * normalized["stability"][key],
                )
                score += impulse
            scores[key] = score
        return scores

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
            swarm_id: max(
                0.0,
                swarm.service_capacity
                * (1.0 - min(max(swarm.load, 0.0), 1.0)),
            )
            for swarm_id, swarm in swarms.items()
        }
        energy_left = {
            swarm_id: swarm.residual_energy_kj for swarm_id, swarm in swarms.items()
        }
        ordered_clusters = sorted(
            clusters.values(),
            key=lambda cluster: (
                cluster.urgency(slot),
                cluster.backlog,
                -cluster.cluster_id,
            ),
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
                key=lambda swarm_id: (
                    scores[(swarm_id, cluster.cluster_id)],
                    -swarm_id,
                ),
            )
            previous = cluster.previous_swarm
            if self.options.use_dwell and previous in candidate_ids:
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

    @staticmethod
    def _validate_identifiers(
        swarms: list[SwarmState],
        clusters: list[ClusterState],
        observations: list[PairObservation],
    ) -> None:
        swarm_ids = [swarm.swarm_id for swarm in swarms]
        cluster_ids = [cluster.cluster_id for cluster in clusters]
        observation_keys = [
            (observation.swarm_id, observation.cluster_id)
            for observation in observations
        ]
        if len(swarm_ids) != len(set(swarm_ids)):
            raise ValueError("swarm identifiers must be unique")
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("cluster identifiers must be unique")
        if len(observation_keys) != len(set(observation_keys)):
            raise ValueError("pair observations must have unique identifiers")
