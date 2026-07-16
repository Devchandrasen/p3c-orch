"""Shared policy, regime, weather, and metric identifiers."""

from __future__ import annotations

from dataclasses import dataclass

BASELINE_METHODS = frozenset({"random", "nearest", "rate-max", "mucco-like"})
P3C_METHODS = frozenset({"reactive-3c", "p3c-lr", "p3c-sr", "et-p3c"})
ABLATION_METHODS = frozenset(
    {
        "no-ann-prediction",
        "no-cache-value",
        "no-dwell",
        "no-risk-calibration",
        "no-neighbor-cache",
    }
)
SUPPORTED_METHODS = BASELINE_METHODS | P3C_METHODS | ABLATION_METHODS
PRACTICAL_METHODS = (
    "random",
    "nearest",
    "rate-max",
    "mucco-like",
    "reactive-3c",
    "p3c-lr",
    "p3c-sr",
    "et-p3c",
)
DEFAULT_METHODS = PRACTICAL_METHODS
PROTOCOL_BASE_BURST_PROBABILITY = 0.10
PROTOCOL_BASE_BURST_MULTIPLIER = 3.0

WEATHER_ATTENUATION_DB = {"clear": 0.0, "rainy": 3.0, "rain_hot": 5.0}
WEATHER_SHADOWING_STD_DB = {"clear": 2.0, "rainy": 3.0, "rain_hot": 3.8}
WEATHER_PREDICTOR_DRIFT_DB = {"clear": 0.0, "rainy": -0.7, "rain_hot": -1.1}
SUPPORTED_WEATHER_STATES = frozenset(WEATHER_ATTENUATION_DB)


@dataclass(frozen=True)
class Regime:
    """Operating-regime values used by the full evaluation protocol."""

    regime_id: str
    name: str
    arrival_multiplier: float
    burst_probability: float
    burst_multiplier: float
    mobility_multiplier: float
    capacity_multiplier: float
    cache_pressure: float
    weather_probabilities: dict[str, float]


REGIMES = {
    "r1-normal": Regime(
        "R1",
        "normal load",
        0.55,
        0.04,
        2.0,
        0.75,
        1.20,
        0.85,
        {"clear": 0.80, "rainy": 0.15, "rain_hot": 0.05},
    ),
    "r2-near-saturation": Regime(
        "R2",
        "near saturation",
        0.85,
        0.08,
        2.5,
        1.00,
        1.00,
        1.00,
        {"clear": 0.70, "rainy": 0.20, "rain_hot": 0.10},
    ),
    "r3-overload": Regime(
        "R3",
        "overload",
        1.20,
        0.12,
        3.0,
        1.10,
        0.85,
        1.10,
        {"clear": 0.65, "rainy": 0.25, "rain_hot": 0.10},
    ),
    "r4-severe-burst": Regime(
        "R4",
        "severe burst traffic",
        0.85,
        0.28,
        5.0,
        1.00,
        0.95,
        1.15,
        {"clear": 0.65, "rainy": 0.25, "rain_hot": 0.10},
    ),
    "r5-harsh-weather": Regime(
        "R5",
        "harsh weather",
        0.75,
        0.10,
        2.5,
        1.00,
        1.00,
        1.00,
        {"clear": 0.15, "rainy": 0.35, "rain_hot": 0.50},
    ),
    "r6-high-mobility": Regime(
        "R6",
        "high mobility",
        0.80,
        0.10,
        2.5,
        1.90,
        0.95,
        1.00,
        {"clear": 0.60, "rainy": 0.25, "rain_hot": 0.15},
    ),
    "r7-combined-stress": Regime(
        "R7",
        "combined stress",
        1.00,
        0.22,
        4.0,
        1.70,
        0.90,
        1.25,
        {"clear": 0.25, "rainy": 0.35, "rain_hot": 0.40},
    ),
}
DEFAULT_REGIMES = tuple(REGIMES)

OBJECTIVE_WEIGHTS = {
    "average_delay": 0.18,
    "p95_delay": 0.16,
    "energy_kj": 0.16,
    "handovers_per_100": 0.14,
    "outage_probability": 0.12,
    "drop_rate": 0.10,
    "load_imbalance": 0.08,
    "useful_cache_hit_ratio": -0.08,
    "throughput": -0.08,
    "jain_fairness": -0.04,
}


def weather_feature_values(weather: str) -> tuple[float, float]:
    """Return the learned-model indicator values for a supported weather state."""

    if weather not in SUPPORTED_WEATHER_STATES:
        raise ValueError(f"unsupported weather state: {weather}")
    return float(weather == "rainy"), float(weather == "rain_hot")
