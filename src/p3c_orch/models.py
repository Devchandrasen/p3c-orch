"""Data structures shared by the scheduler, channel model, and simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import weather_feature_values


@dataclass
class UserState:
    user_id: int
    x_m: float
    y_m: float
    velocity_x_mps: float
    velocity_y_mps: float


@dataclass(frozen=True)
class ContentItem:
    content_id: int
    size_mb: float


@dataclass
class SwarmState:
    swarm_id: int
    x_m: float
    y_m: float
    altitude_m: float
    bandwidth_mhz: float
    transmit_power_dbm: float
    residual_energy_kj: float
    service_capacity: float
    uav_count: int = 6
    load: float = 0.0
    cache: set[int] = field(default_factory=set)
    cache_expiry: dict[int, int] = field(default_factory=dict)
    cache_sizes_mb: dict[int, float] = field(default_factory=dict)

    @property
    def cache_used_mb(self) -> float:
        return sum(self.cache_sizes_mb.get(content_id, 0.0) for content_id in self.cache)


@dataclass
class ClusterState:
    cluster_id: int
    x_m: float
    y_m: float
    velocity_x_mps: float
    velocity_y_mps: float
    arrival: float
    backlog: float
    deadline_slot: int
    requested_content: int
    previous_swarm: int | None = None
    content_size_mb: float = 1.0
    user_ids: tuple[int, ...] = ()

    def urgency(self, slot: int) -> float:
        return self.backlog / max(self.deadline_slot - slot + 1, 1)


@dataclass(frozen=True)
class PairObservation:
    slot: int
    swarm_id: int
    cluster_id: int
    weather: str
    distance_m: float
    elevation_deg: float
    relative_speed_mps: float
    bandwidth_mhz: float
    transmit_power_dbm: float
    current_margin_db: float
    previous_outage: bool
    local_cache_hit: bool
    neighbor_cache_hit: bool
    residual_energy_kj: float
    load: float
    time_fraction: float
    raw_rate: float
    energy_cost: float
    delay_cost: float
    stale_penalty: float = 0.0
    neighbor_fetch_penalty: float = 0.0
    deadline_risk: float = 0.0
    queue_risk: float = 0.0
    cache_miss_penalty: float = 0.0
    transmission_energy_kj: float = 0.0
    mobility_energy_kj: float = 0.0
    cache_energy_kj: float = 0.0
    feasible: bool = True

    def predictor_features(self) -> dict[str, float]:
        weather_rainy, weather_rain_hot = weather_feature_values(self.weather)
        return {
            "distance_m": self.distance_m,
            "elevation_deg": self.elevation_deg,
            "relative_speed_mps": self.relative_speed_mps,
            "bandwidth_mhz": self.bandwidth_mhz,
            "transmit_power_dbm": self.transmit_power_dbm,
            "current_margin_db": self.current_margin_db,
            "previous_outage": float(self.previous_outage),
            "local_cache_hit": float(self.local_cache_hit),
            "residual_energy_kj": self.residual_energy_kj,
            "load": self.load,
            "time_fraction": self.time_fraction,
            "weather_rainy": weather_rainy,
            "weather_rain_hot": weather_rain_hot,
        }


@dataclass(frozen=True)
class RiskEstimate:
    predicted_margin_db: float
    outage_probability: float


@dataclass
class AssignmentResult:
    assignments: dict[int, int]
    scores: dict[tuple[int, int], float]
    estimates: dict[tuple[int, int], RiskEstimate]
    service_rates: dict[tuple[int, int], float]


@dataclass(frozen=True)
class SimulationMetrics:
    regime: str
    method: str
    seed: int
    average_delay: float
    p95_delay: float
    energy_kj: float
    handovers_per_100: float
    useful_cache_hit_ratio: float
    local_cache_hit_ratio: float
    neighbor_cache_hit_ratio: float
    outage_probability: float
    clear_outage_probability: float
    rainy_outage_probability: float
    rain_hot_outage_probability: float
    throughput: float
    drop_rate: float
    average_backlog: float
    sla_violation_rate: float
    load_imbalance: float
    jain_fairness: float
    controller_operations: int
