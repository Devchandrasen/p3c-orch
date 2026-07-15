"""Configuration loading and validation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .constants import DEFAULT_METHODS, SUPPORTED_METHODS, SUPPORTED_WEATHER_STATES


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class SimulationConfig:
    slots: int = 30
    area_size_m: float = 1000.0
    swarms: int = 4
    clusters: int = 10
    cache_size: int = 15
    library_size: int = 100
    arrival_rate: float = 2.0
    deadline_min: int = 4
    deadline_max: int = 8
    seeds: tuple[int, ...] = (4001, 4002, 4003)
    methods: tuple[str, ...] = DEFAULT_METHODS
    weather_probabilities: dict[str, float] = field(
        default_factory=lambda: {"clear": 0.60, "rainy": 0.25, "rain_hot": 0.15}
    )

    def validate(self) -> None:
        integer_fields = {
            "slots": self.slots,
            "swarms": self.swarms,
            "clusters": self.clusters,
            "cache_size": self.cache_size,
            "library_size": self.library_size,
        }
        for name, value in integer_fields.items():
            if not _is_integer(value):
                raise ValueError(f"simulation.{name} must be an integer")
        positive = {
            "slots": self.slots,
            "area_size_m": self.area_size_m,
            "swarms": self.swarms,
            "clusters": self.clusters,
            "cache_size": self.cache_size,
            "library_size": self.library_size,
            "arrival_rate": self.arrival_rate,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"simulation.{name} must be finite and positive")
        if not _is_integer(self.deadline_min) or not _is_integer(self.deadline_max):
            raise ValueError("simulation deadlines must be integers")
        if self.deadline_min <= 0 or self.deadline_max < self.deadline_min:
            raise ValueError("deadline range is invalid")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if any(not _is_integer(seed) for seed in self.seeds):
            raise ValueError("simulation seeds must be integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("simulation seeds must be unique")
        if not self.methods:
            raise ValueError("at least one method is required")
        if any(not isinstance(method, str) for method in self.methods):
            raise ValueError("simulation methods must be strings")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("simulation methods must be unique")
        unknown = set(self.methods) - SUPPORTED_METHODS
        if unknown:
            raise ValueError(f"unsupported methods: {', '.join(sorted(unknown))}")
        if not self.weather_probabilities:
            raise ValueError("weather probabilities cannot be empty")
        if any(not isinstance(state, str) for state in self.weather_probabilities):
            raise ValueError("weather state names must be strings")
        unsupported_weather = set(self.weather_probabilities) - SUPPORTED_WEATHER_STATES
        if unsupported_weather:
            raise ValueError(
                "unsupported weather states: "
                + ", ".join(sorted(unsupported_weather))
            )
        try:
            probability_values = tuple(
                float(value) for value in self.weather_probabilities.values()
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("weather probabilities must be numeric") from exc
        total = sum(probability_values)
        invalid = any(not math.isfinite(value) or value < 0 for value in probability_values)
        if invalid or not math.isfinite(total) or abs(total - 1.0) > 1e-6:
            raise ValueError("weather probabilities must be finite, non-negative, and sum to 1")


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
    weights: SchedulerWeights = field(default_factory=SchedulerWeights)

    def validate(self) -> None:
        values = {
            "dwell_threshold": self.dwell_threshold,
            "minimum_energy_kj": self.minimum_energy_kj,
            "stability_weight": self.stability_weight,
            "load_regularization": self.load_regularization,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"scheduler.{name} must be finite and non-negative")
        self.weights.validate()


@dataclass(frozen=True)
class ProjectConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def validate(self) -> None:
        self.simulation.validate()
        self.predictor.validate()
        self.scheduler.validate()
        missing_scales = set(self.simulation.weather_probabilities) - set(
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
        payload["simulation"]["seeds"] = list(self.simulation.seeds)
        payload["simulation"]["methods"] = list(self.simulation.methods)
        return payload

    def with_overrides(
        self, *, slots: int | None = None, seeds: tuple[int, ...] | None = None
    ) -> ProjectConfig:
        simulation = replace(
            self.simulation,
            slots=slots if slots is not None else self.simulation.slots,
            seeds=seeds if seeds is not None else self.simulation.seeds,
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


def _reject_unknown_keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        keys = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"unknown {name} keys: {keys}")


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a project YAML configuration."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    root = _mapping(raw, "configuration")
    simulation_raw = _mapping(root.get("simulation"), "simulation")
    predictor_raw = _mapping(root.get("predictor"), "predictor")
    scheduler_raw = _mapping(root.get("scheduler"), "scheduler")
    weights_raw = _mapping(scheduler_raw.get("weights"), "scheduler.weights")
    _reject_unknown_keys(root, {"simulation", "predictor", "scheduler"}, "configuration")
    _reject_unknown_keys(
        simulation_raw,
        {
            "slots",
            "area_size_m",
            "swarms",
            "clusters",
            "cache_size",
            "library_size",
            "arrival_rate",
            "deadline_min",
            "deadline_max",
            "seeds",
            "methods",
            "weather_probabilities",
        },
        "simulation",
    )
    _reject_unknown_keys(
        predictor_raw,
        {"margin_threshold_db", "model_path", "residual_scale_db"},
        "predictor",
    )
    _reject_unknown_keys(
        scheduler_raw,
        {
            "dwell_threshold",
            "minimum_energy_kj",
            "stability_weight",
            "load_regularization",
            "weights",
        },
        "scheduler",
    )
    _reject_unknown_keys(weights_raw, set(asdict(SchedulerWeights())), "scheduler.weights")

    defaults = ProjectConfig()
    simulation_defaults = defaults.simulation
    predictor_defaults = defaults.predictor
    scheduler_defaults = defaults.scheduler
    weather_probabilities_raw = _mapping(
        simulation_raw.get(
            "weather_probabilities", simulation_defaults.weather_probabilities
        ),
        "simulation.weather_probabilities",
    )
    residual_scale_raw = _mapping(
        predictor_raw.get("residual_scale_db", predictor_defaults.residual_scale_db),
        "predictor.residual_scale_db",
    )

    model_path = predictor_raw.get("model_path")
    if model_path:
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = (config_path.parent / model_path).resolve()

    simulation = SimulationConfig(
        slots=int(simulation_raw.get("slots", simulation_defaults.slots)),
        area_size_m=float(
            simulation_raw.get("area_size_m", simulation_defaults.area_size_m)
        ),
        swarms=int(simulation_raw.get("swarms", simulation_defaults.swarms)),
        clusters=int(simulation_raw.get("clusters", simulation_defaults.clusters)),
        cache_size=int(simulation_raw.get("cache_size", simulation_defaults.cache_size)),
        library_size=int(
            simulation_raw.get("library_size", simulation_defaults.library_size)
        ),
        arrival_rate=float(
            simulation_raw.get("arrival_rate", simulation_defaults.arrival_rate)
        ),
        deadline_min=int(
            simulation_raw.get("deadline_min", simulation_defaults.deadline_min)
        ),
        deadline_max=int(
            simulation_raw.get("deadline_max", simulation_defaults.deadline_max)
        ),
        seeds=tuple(
            int(seed) for seed in simulation_raw.get("seeds", simulation_defaults.seeds)
        ),
        methods=tuple(
            str(method)
            for method in simulation_raw.get("methods", simulation_defaults.methods)
        ),
        weather_probabilities={
            str(key): float(value)
            for key, value in weather_probabilities_raw.items()
        },
    )
    predictor = PredictorConfig(
        margin_threshold_db=float(
            predictor_raw.get(
                "margin_threshold_db", predictor_defaults.margin_threshold_db
            )
        ),
        model_path=model_path,
        residual_scale_db={
            str(key): float(value)
            for key, value in residual_scale_raw.items()
        },
    )
    scheduler = SchedulerConfig(
        dwell_threshold=float(
            scheduler_raw.get("dwell_threshold", scheduler_defaults.dwell_threshold)
        ),
        minimum_energy_kj=float(
            scheduler_raw.get("minimum_energy_kj", scheduler_defaults.minimum_energy_kj)
        ),
        stability_weight=float(
            scheduler_raw.get("stability_weight", scheduler_defaults.stability_weight)
        ),
        load_regularization=float(
            scheduler_raw.get(
                "load_regularization", scheduler_defaults.load_regularization
            )
        ),
        weights=SchedulerWeights(
            **{
                field_name: float(weights_raw.get(field_name, default_value))
                for field_name, default_value in asdict(scheduler_defaults.weights).items()
            }
        ),
    )
    config = ProjectConfig(simulation=simulation, predictor=predictor, scheduler=scheduler)
    config.validate()
    return config
