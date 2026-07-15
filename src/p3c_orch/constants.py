"""Shared policy and weather identifiers."""

BASELINE_METHODS = frozenset({"random", "nearest", "rate-max"})
P3C_METHODS = frozenset({"reactive-3c", "p3c-lr", "p3c-sr"})
SUPPORTED_METHODS = BASELINE_METHODS | P3C_METHODS
DEFAULT_METHODS = ("nearest", "rate-max", "reactive-3c", "p3c-lr", "p3c-sr")

WEATHER_ATTENUATION_DB = {"clear": 0.0, "rainy": 3.0, "rain_hot": 5.0}
WEATHER_PREDICTOR_DRIFT_DB = {"clear": 0.0, "rainy": -0.7, "rain_hot": -1.1}
SUPPORTED_WEATHER_STATES = frozenset(WEATHER_ATTENUATION_DB)


def weather_feature_values(weather: str) -> tuple[float, float]:
    """Return the learned-model indicator values for a supported weather state."""

    if weather not in SUPPORTED_WEATHER_STATES:
        raise ValueError(f"unsupported weather state: {weather}")
    return float(weather == "rainy"), float(weather == "rain_hot")
