"""Configuration loading and validation for the complete P3C-Orch protocol."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from .constants import (
    DEFAULT_METHODS,
    DEFAULT_REGIMES,
    PROTOCOL_BASE_BURST_MULTIPLIER,
    PROTOCOL_BASE_BURST_PROBABILITY,
    REGIMES,
    SUPPORTED_METHODS,
    SUPPORTED_WEATHER_STATES,
)


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_positive(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class SimulationConfig:
    slots: int = 200
    area_size_m: float = 1000.0
    swarms: int = 4
    clusters: int = 10
    users: int = 60
    uavs_per_swarm_min: int = 6
    uavs_per_swarm_max: int = 10
    cache_size: int = 15
    cache_capacity_mb: float = 80.0
    library_size: int = 100
    file_size_min_mb: float = 1.0
    file_size_max_mb: float = 10.0
    content_ttl_slots: int = 30
    zipf_exponent: float = 1.35
    arrival_rate: float = 2.0
    burst_probability: float = PROTOCOL_BASE_BURST_PROBABILITY
    burst_multiplier: float = PROTOCOL_BASE_BURST_MULTIPLIER
    deadline_min: int = 4
    deadline_max: int = 8
    gauss_markov_alpha: float = 0.85
    user_speed_std_mps: float = 6.0
    mobility_noise_std_mps: float = 1.5
    weather_persistence: float = 0.80
    seeds: tuple[int, ...] = tuple(range(4001, 4081))
    methods: tuple[str, ...] = DEFAULT_METHODS
    regimes: tuple[str, ...] = DEFAULT_REGIMES
    weather_probabilities: dict[str, float] = field(
        default_factory=lambda: {"clear": 0.60, "rainy": 0.25, "rain_hot": 0.15}
    )

    def validate(self) -> None:
        integer_fields = {
            "slots": self.slots,
            "swarms": self.swarms,
            "clusters": self.clusters,
            "users": self.users,
            "uavs_per_swarm_min": self.uavs_per_swarm_min,
            "uavs_per_swarm_max": self.uavs_per_swarm_max,
            "cache_size": self.cache_size,
            "library_size": self.library_size,
            "content_ttl_slots": self.content_ttl_slots,
        }
        for name, value in integer_fields.items():
            if not _is_integer(value) or value <= 0:
                raise ValueError(f"simulation.{name} must be a positive integer")
        if self.uavs_per_swarm_max < self.uavs_per_swarm_min:
            raise ValueError("simulation UAV-count range is invalid")
        if self.users < self.clusters:
            raise ValueError("simulation.users must be at least simulation.clusters")
        for name in (
            "area_size_m",
            "cache_capacity_mb",
            "file_size_min_mb",
            "file_size_max_mb",
            "zipf_exponent",
            "arrival_rate",
            "burst_multiplier",
            "user_speed_std_mps",
            "mobility_noise_std_mps",
        ):
            _finite_positive(f"simulation.{name}", float(getattr(self, name)))
        if self.file_size_max_mb < self.file_size_min_mb:
            raise ValueError("simulation file-size range is invalid")
        if not 0 <= self.burst_probability <= 1:
            raise ValueError("simulation.burst_probability must be between 0 and 1")
        if not 0 <= self.gauss_markov_alpha < 1:
            raise ValueError("simulation.gauss_markov_alpha must be in [0, 1)")
        if not 0 <= self.weather_persistence < 1:
            raise ValueError("simulation.weather_persistence must be in [0, 1)")
        if not _is_integer(self.deadline_min) or not _is_integer(self.deadline_max):
            raise ValueError("simulation deadlines must be integers")
        if self.deadline_min <= 0 or self.deadline_max < self.deadline_min:
            raise ValueError("deadline range is invalid")
        if not self.seeds or any(not _is_integer(seed) for seed in self.seeds):
            raise ValueError("simulation seeds must be non-empty integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("simulation seeds must be unique")
        if not self.methods or any(not isinstance(method, str) for method in self.methods):
            raise ValueError("simulation methods must be non-empty strings")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("simulation methods must be unique")
        unknown_methods = set(self.methods) - SUPPORTED_METHODS
        if unknown_methods:
            raise ValueError(f"unsupported methods: {', '.join(sorted(unknown_methods))}")
        if not self.regimes or any(not isinstance(regime, str) for regime in self.regimes):
            raise ValueError("simulation regimes must be non-empty strings")
        unknown_regimes = set(self.regimes) - set(REGIMES)
        if unknown_regimes:
            raise ValueError(f"unsupported regimes: {', '.join(sorted(unknown_regimes))}")
        self._validate_weather_probabilities()

    def _validate_weather_probabilities(self) -> None:
        if not self.weather_probabilities:
            raise ValueError("weather probabilities cannot be empty")
        if any(not isinstance(state, str) for state in self.weather_probabilities):
            raise ValueError("weather state names must be strings")
        unsupported = set(self.weather_probabilities) - SUPPORTED_WEATHER_STATES
        if unsupported:
            raise ValueError(
                "unsupported weather states: " + ", ".join(sorted(unsupported))
            )
        try:
            values = tuple(float(value) for value in self.weather_probabilities.values())
        except (TypeError, ValueError) as exc:
            raise ValueError("weather probabilities must be numeric") from exc
        total = sum(values)
        if (
            any(not math.isfinite(value) or value < 0 for value in values)
            or not math.isfinite(total)
            or abs(total - 1.0) > 1e-6
        ):
            raise ValueError(
                "weather probabilities must be finite, non-negative, and sum to 1"
            )


@dataclass(frozen=True)
class ChannelConfig:
    carrier_frequency_hz: float = 2.4e9
    noise_density_dbm_hz: float = -174.0
    receiver_noise_figure_db: float = 7.0
    los_environment_a: float = 9.61
    los_environment_b: float = 0.16
    los_excess_loss_db: float = 1.0
    nlos_excess_loss_db: float = 20.0
    required_snr_db: float = 3.0
    maximum_link_distance_m: float = 1500.0
    spectral_efficiency_cap: float = 8.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"channel.{name} must be finite")
        for name in (
            "carrier_frequency_hz",
            "receiver_noise_figure_db",
            "los_environment_a",
            "los_environment_b",
            "maximum_link_distance_m",
            "spectral_efficiency_cap",
        ):
            _finite_positive(f"channel.{name}", float(getattr(self, name)))
        if self.nlos_excess_loss_db < self.los_excess_loss_db:
            raise ValueError("channel NLOS loss must not be lower than LOS loss")


@dataclass(frozen=True)
class PredictorConfig:
    margin_threshold_db: float = 0.0
    model_path: Path | None = None
    residual_scale_db: dict[str, float] = field(
        default_factory=lambda: {"clear": 4.711, "rainy": 4.108, "rain_hot": 4.310}
    )

    def validate(self) -> None:
        if not math.isfinite(self.margin_threshold_db):
            raise ValueError("predictor.margin_threshold_db must be finite")
        if not self.residual_scale_db:
            raise ValueError("predictor.residual_scale_db cannot be empty")
        try:
            scales = tuple(float(value) for value in self.residual_scale_db.values())
        except (TypeError, ValueError) as exc:
            raise ValueError("all residual scales must be numeric") from exc
        if any(not math.isfinite(value) or value <= 0 for value in scales):
            raise ValueError("all residual scales must be finite and positive")
        if self.model_path is not None and not self.model_path.is_file():
            raise ValueError(f"predictor model does not exist: {self.model_path}")


@dataclass(frozen=True)
class SchedulerWeights:
    margin: float = 0.20
    cache: float = 0.15
    fairness: float = 0.08
    dwell: float = 0.12
    delay: float = 0.15
    energy: float = 0.08
    outage: float = 0.10
    switch: float = 0.07
    load: float = 0.05

    def validate(self) -> None:
        values = tuple(asdict(self).values())
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("scheduler weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one scheduler weight must be positive")


@dataclass(frozen=True)
class SchedulerConfig:
    dwell_threshold: float = 0.05
    minimum_energy_kj: float = 5.0
    stability_weight: float = 0.15
    load_regularization: float = 0.10
    event_risk_threshold: float = 0.65
    event_impulse_cap: float = 0.12
    cache_local_weight: float = 1.0
    cache_neighbor_weight: float = 0.5
    cache_stale_penalty: float = 0.25
    cache_fetch_penalty: float = 0.15
    delay_deadline_weight: float = 0.30
    delay_queue_weight: float = 0.35
    delay_miss_weight: float = 0.15
    delay_outage_weight: float = 0.20
    switch_fixed_cost: float = 0.20
    switch_distance_cost: float = 0.001
    weights: SchedulerWeights = field(default_factory=SchedulerWeights)

    def validate(self) -> None:
        values = {
            name: value
            for name, value in asdict(self).items()
            if name != "weights"
        }
        for name, value in values.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"scheduler.{name} must be finite and non-negative")
        if not 0 <= self.event_risk_threshold <= 1:
            raise ValueError("scheduler.event_risk_threshold must be between 0 and 1")
        self.weights.validate()


@dataclass(frozen=True)
class ProjectConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def validate(self) -> None:
        self.simulation.validate()
        self.channel.validate()
        self.predictor.validate()
        self.scheduler.validate()
        missing_scales = set(SUPPORTED_WEATHER_STATES) - set(
            self.predictor.residual_scale_db
        )
        if missing_scales:
            raise ValueError(
                "missing residual scales for weather states: "
                + ", ".join(sorted(missing_scales))
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        model_path = payload["predictor"]["model_path"]
        payload["predictor"]["model_path"] = str(model_path) if model_path else None
        for name in ("seeds", "methods", "regimes"):
            payload["simulation"][name] = list(getattr(self.simulation, name))
        return payload

    def with_overrides(
        self,
        *,
        slots: int | None = None,
        seeds: tuple[int, ...] | None = None,
        methods: tuple[str, ...] | None = None,
        regimes: tuple[str, ...] | None = None,
    ) -> ProjectConfig:
        simulation = replace(
            self.simulation,
            slots=slots if slots is not None else self.simulation.slots,
            seeds=seeds if seeds is not None else self.simulation.seeds,
            methods=methods if methods is not None else self.simulation.methods,
            regimes=regimes if regimes is not None else self.simulation.regimes,
        )
        result = replace(self, simulation=simulation)
        result.validate()
        return result


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _reject_unknown_keys(value: dict[str, Any], cls: type[Any], name: str) -> None:
    allowed = {item.name for item in fields(cls)}
    unknown = set(value) - allowed
    if unknown:
        keys = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"unknown {name} keys: {keys}")


def _coerce_section(raw: dict[str, Any], default: Any) -> dict[str, Any]:
    values = asdict(default)
    values.update(raw)
    for name, default_value in asdict(default).items():
        value = values[name]
        if not isinstance(value, str):
            continue
        if isinstance(default_value, bool):
            continue
        if isinstance(default_value, int):
            values[name] = int(value)
        elif isinstance(default_value, float):
            values[name] = float(value)
    return values


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a project YAML configuration."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        root = _mapping(yaml.safe_load(handle) or {}, "configuration")
    allowed_root = {"simulation", "channel", "predictor", "scheduler"}
    unknown_root = set(root) - allowed_root
    if unknown_root:
        raise ValueError(
            "unknown configuration keys: " + ", ".join(sorted(unknown_root))
        )

    simulation_raw = _mapping(root.get("simulation"), "simulation")
    channel_raw = _mapping(root.get("channel"), "channel")
    predictor_raw = _mapping(root.get("predictor"), "predictor")
    scheduler_raw = _mapping(root.get("scheduler"), "scheduler")
    weights_raw = _mapping(scheduler_raw.get("weights"), "scheduler.weights")
    _reject_unknown_keys(simulation_raw, SimulationConfig, "simulation")
    _reject_unknown_keys(channel_raw, ChannelConfig, "channel")
    _reject_unknown_keys(predictor_raw, PredictorConfig, "predictor")
    _reject_unknown_keys(scheduler_raw, SchedulerConfig, "scheduler")
    _reject_unknown_keys(weights_raw, SchedulerWeights, "scheduler.weights")

    defaults = ProjectConfig()
    simulation_values = _coerce_section(simulation_raw, defaults.simulation)
    simulation_values["seeds"] = tuple(int(seed) for seed in simulation_values["seeds"])
    simulation_values["methods"] = tuple(
        str(method) for method in simulation_values["methods"]
    )
    simulation_values["regimes"] = tuple(
        str(regime) for regime in simulation_values["regimes"]
    )
    weather_raw = _mapping(
        simulation_values["weather_probabilities"],
        "simulation.weather_probabilities",
    )
    simulation_values["weather_probabilities"] = {
        str(key): float(value) for key, value in weather_raw.items()
    }

    model_path = predictor_raw.get("model_path", defaults.predictor.model_path)
    if model_path:
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = (config_path.parent / model_path).resolve()
    predictor_values = _coerce_section(predictor_raw, defaults.predictor)
    predictor_values["model_path"] = model_path
    residual_raw = _mapping(
        predictor_values["residual_scale_db"], "predictor.residual_scale_db"
    )
    predictor_values["residual_scale_db"] = {
        str(key): float(value) for key, value in residual_raw.items()
    }

    scheduler_values = _coerce_section(scheduler_raw, defaults.scheduler)
    scheduler_values["weights"] = SchedulerWeights(
        **{
            key: float(value)
            for key, value in _coerce_section(weights_raw, defaults.scheduler.weights).items()
        }
    )
    config = ProjectConfig(
        simulation=SimulationConfig(**simulation_values),
        channel=ChannelConfig(**_coerce_section(channel_raw, defaults.channel)),
        predictor=PredictorConfig(**predictor_values),
        scheduler=SchedulerConfig(**scheduler_values),
    )
    config.validate()
    return config
