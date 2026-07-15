"""Link-margin predictors and weather-calibrated outage risk."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .constants import WEATHER_PREDICTOR_DRIFT_DB
from .models import PairObservation, RiskEstimate

FEATURE_NAMES = (
    "distance_m",
    "elevation_deg",
    "relative_speed_mps",
    "bandwidth_mhz",
    "transmit_power_dbm",
    "current_margin_db",
    "previous_outage",
    "local_cache_hit",
    "residual_energy_kj",
    "load",
    "time_fraction",
    "weather_rainy",
    "weather_rain_hot",
)


def _feature_matrix(observations: Sequence[PairObservation]) -> np.ndarray:
    rows = []
    for observation in observations:
        features = observation.predictor_features()
        rows.append([features[name] for name in FEATURE_NAMES])
    return np.asarray(rows, dtype=float).reshape(len(rows), len(FEATURE_NAMES))


class MarginPredictor(Protocol):
    def predict_many(self, observations: Sequence[PairObservation]) -> np.ndarray:
        """Predict next-slot link margins in dB."""


class HeuristicMarginPredictor:
    """Deterministic baseline used when no learned artifact is configured."""

    def predict(self, observation: PairObservation) -> float:
        return float(self.predict_many([observation])[0])

    def predict_many(self, observations: Sequence[PairObservation]) -> np.ndarray:
        predictions = []
        for observation in observations:
            mobility_drift = -0.015 * observation.relative_speed_mps
            outage_drift = -0.35 if observation.previous_outage else 0.0
            predictions.append(
                observation.current_margin_db
                + WEATHER_PREDICTOR_DRIFT_DB[observation.weather]
                + mobility_drift
                + outage_drift
            )
        return np.asarray(predictions, dtype=float)


class CurrentMarginPredictor:
    """Reactive estimator that treats the current margin as the next margin."""

    def predict(self, observation: PairObservation) -> float:
        return observation.current_margin_db

    def predict_many(self, observations: Sequence[PairObservation]) -> np.ndarray:
        return np.asarray(
            [observation.current_margin_db for observation in observations],
            dtype=float,
        )


class LearnedMarginPredictor:
    """Load a numeric-only MLP artifact saved by :func:`train_predictor`."""

    def __init__(self, model_path: str | Path) -> None:
        try:
            artifact = np.load(Path(model_path), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid predictor artifact: {model_path}") from exc
        if not hasattr(artifact, "files"):
            raise ValueError("predictor artifact must be an NPZ archive")

        with artifact:
            required = {"feature_names", "scaler_mean", "scaler_scale", "layer_count"}
            missing = required - set(artifact.files)
            if missing:
                raise ValueError(
                    "predictor artifact is missing arrays: " + ", ".join(sorted(missing))
                )
            feature_names = tuple(str(name) for name in artifact["feature_names"].tolist())
            if feature_names != FEATURE_NAMES:
                raise ValueError("model feature order does not match this project version")

            layer_count_array = np.asarray(artifact["layer_count"])
            if layer_count_array.size != 1:
                raise ValueError("predictor layer count must be a scalar")
            layer_count_value = float(layer_count_array.item())
            if (
                not math.isfinite(layer_count_value)
                or not layer_count_value.is_integer()
                or not 1 <= layer_count_value <= 32
            ):
                raise ValueError("predictor layer count must be an integer from 1 to 32")
            layer_count = int(layer_count_value)

            expected_layers = {
                name
                for index in range(layer_count)
                for name in (f"coef_{index}", f"intercept_{index}")
            }
            missing_layers = expected_layers - set(artifact.files)
            if missing_layers:
                raise ValueError(
                    "predictor artifact is missing arrays: "
                    + ", ".join(sorted(missing_layers))
                )
            self.scaler_mean = np.asarray(artifact["scaler_mean"], dtype=float)
            self.scaler_scale = np.asarray(artifact["scaler_scale"], dtype=float)
            self.coefficients = tuple(
                np.asarray(artifact[f"coef_{index}"], dtype=float)
                for index in range(layer_count)
            )
            self.intercepts = tuple(
                np.asarray(artifact[f"intercept_{index}"], dtype=float)
                for index in range(layer_count)
            )
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        feature_shape = (len(FEATURE_NAMES),)
        if self.scaler_mean.shape != feature_shape or self.scaler_scale.shape != feature_shape:
            raise ValueError("predictor scaler shape does not match the feature count")
        arrays = (
            self.scaler_mean,
            self.scaler_scale,
            *self.coefficients,
            *self.intercepts,
        )
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("predictor artifact contains non-finite values")
        if np.any(self.scaler_scale <= 0):
            raise ValueError("predictor scaler values must be positive")

        input_width = len(FEATURE_NAMES)
        for index, (coefficient, intercept) in enumerate(
            zip(self.coefficients, self.intercepts, strict=True)
        ):
            if coefficient.ndim != 2 or coefficient.shape[0] != input_width:
                raise ValueError(f"predictor coefficient {index} has an invalid shape")
            if intercept.shape != (coefficient.shape[1],):
                raise ValueError(f"predictor intercept {index} has an invalid shape")
            input_width = coefficient.shape[1]
        if input_width != 1:
            raise ValueError("predictor output layer must have exactly one value")

    def predict(self, observation: PairObservation) -> float:
        return float(self.predict_many([observation])[0])

    def predict_many(self, observations: Sequence[PairObservation]) -> np.ndarray:
        if not observations:
            return np.empty(0, dtype=float)
        activations = (_feature_matrix(observations) - self.scaler_mean) / self.scaler_scale
        for index, (coefficient, intercept) in enumerate(
            zip(self.coefficients, self.intercepts, strict=True)
        ):
            activations = activations @ coefficient + intercept
            if index < len(self.coefficients) - 1:
                activations = np.maximum(activations, 0.0)
        return np.asarray(activations[:, 0], dtype=float)


class CalibratedRiskEstimator:
    def __init__(
        self,
        margin_predictor: MarginPredictor,
        *,
        margin_threshold_db: float,
        residual_scale_db: dict[str, float],
    ) -> None:
        if not math.isfinite(margin_threshold_db):
            raise ValueError("margin threshold must be finite")
        self.margin_predictor = margin_predictor
        self.margin_threshold_db = margin_threshold_db
        self.residual_scale_db = residual_scale_db

    def estimate(self, observation: PairObservation) -> RiskEstimate:
        return self.estimate_many([observation])[0]

    def estimate_many(
        self, observations: Sequence[PairObservation]
    ) -> list[RiskEstimate]:
        if not observations:
            return []
        margins = np.asarray(
            self.margin_predictor.predict_many(observations), dtype=float
        ).reshape(-1)
        if margins.shape != (len(observations),):
            raise ValueError("margin predictor returned an unexpected number of values")
        if not np.isfinite(margins).all():
            raise ValueError("margin predictor returned a non-finite value")

        scales = np.asarray(
            [
                self.residual_scale_db.get(observation.weather, math.nan)
                for observation in observations
            ],
            dtype=float,
        )
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            invalid_weather = next(
                observation.weather
                for observation, scale in zip(observations, scales, strict=True)
                if not math.isfinite(float(scale)) or scale <= 0
            )
            raise ValueError(f"missing positive residual scale for {invalid_weather!r}")
        logits = np.clip((self.margin_threshold_db - margins) / scales, -60.0, 60.0)
        risks = 1.0 / (1.0 + np.exp(-logits))
        return [
            RiskEstimate(predicted_margin_db=float(margin), outage_probability=float(risk))
            for margin, risk in zip(margins, risks, strict=True)
        ]


def train_predictor(
    csv_path: str | Path, output_path: str | Path, *, random_state: int = 42
) -> dict[str, float]:
    """Train an MLP and persist its numeric parameters in a safe NPZ artifact."""

    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError('Install the ML extra with: pip install -e ".[ml]"') from exc

    rows: list[list[float]] = []
    targets: list[float] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(FEATURE_NAMES) | {"target_next_margin_db"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"training CSV is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            rows.append([float(row[name]) for name in FEATURE_NAMES])
            targets.append(float(row["target_next_margin_db"]))
    if len(rows) < 20:
        raise ValueError("at least 20 training rows are required")

    x = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("training data must contain only finite numeric values")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state
    )
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 64, 64),
            activation="relu",
            random_state=random_state,
            max_iter=500,
            early_stopping=True,
        ),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    scaler = model.named_steps["standardscaler"]
    regressor = model.named_steps["mlpregressor"]
    artifact: dict[str, np.ndarray] = {
        "feature_names": np.asarray(FEATURE_NAMES),
        "scaler_mean": np.asarray(scaler.mean_, dtype=float),
        "scaler_scale": np.asarray(scaler.scale_, dtype=float),
        "layer_count": np.asarray(len(regressor.coefs_), dtype=np.int64),
    }
    for index, (coefficient, intercept) in enumerate(
        zip(regressor.coefs_, regressor.intercepts_, strict=True)
    ):
        artifact[f"coef_{index}"] = np.asarray(coefficient, dtype=float)
        artifact[f"intercept_{index}"] = np.asarray(intercept, dtype=float)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(handle, **artifact)
    return {
        "samples": float(len(rows)),
        "mae_db": float(mean_absolute_error(y_test, predictions)),
        "rmse_db": float(math.sqrt(mean_squared_error(y_test, predictions))),
    }
