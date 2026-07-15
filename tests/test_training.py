import csv
import math

import numpy as np
import pytest

from p3c_orch.predictor import FEATURE_NAMES, LearnedMarginPredictor, train_predictor

from .factories import make_observation, write_training_csv

pytest.importorskip("sklearn")


def test_optional_mlp_training_and_loading(tmp_path) -> None:
    csv_path = tmp_path / "samples.csv"
    write_training_csv(csv_path)
    model_path = tmp_path / "predictor.npz"

    metrics = train_predictor(csv_path, model_path, random_state=7)
    predictor = LearnedMarginPredictor(model_path)

    assert model_path.exists()
    assert metrics["samples"] == 80.0
    assert math.isfinite(predictor.predict(make_observation(time_fraction=0.5)))
    predictions = predictor.predict_many(
        [make_observation(current_margin_db=-2.0), make_observation(current_margin_db=2.0)]
    )
    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()


def test_training_rejects_missing_columns(tmp_path) -> None:
    csv_path = tmp_path / "missing.csv"
    csv_path.write_text("distance_m,target_next_margin_db\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        train_predictor(csv_path, tmp_path / "predictor.npz")


def test_training_rejects_too_few_rows(tmp_path) -> None:
    csv_path = tmp_path / "small.csv"
    write_training_csv(csv_path, rows=19)
    with pytest.raises(ValueError, match="at least 20"):
        train_predictor(csv_path, tmp_path / "predictor.npz")


def test_training_rejects_non_finite_values(tmp_path) -> None:
    csv_path = tmp_path / "nonfinite.csv"
    write_training_csv(csv_path, rows=20)
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[*FEATURE_NAMES, "target_next_margin_db"]
        )
        row = {name: 0.0 for name in FEATURE_NAMES}
        row["distance_m"] = float("nan")
        row["target_next_margin_db"] = 1.0
        writer.writerow(row)
    with pytest.raises(ValueError, match="finite numeric"):
        train_predictor(csv_path, tmp_path / "predictor.npz")


def test_artifact_feature_order_is_validated(tmp_path) -> None:
    model_path = tmp_path / "bad_features.npz"
    np.savez_compressed(
        model_path,
        feature_names=np.asarray(tuple(reversed(FEATURE_NAMES))),
        scaler_mean=np.zeros(len(FEATURE_NAMES)),
        scaler_scale=np.ones(len(FEATURE_NAMES)),
        layer_count=np.asarray(1),
        coef_0=np.zeros((len(FEATURE_NAMES), 1)),
        intercept_0=np.zeros(1),
    )
    with pytest.raises(ValueError, match="feature order"):
        LearnedMarginPredictor(model_path)


def test_artifact_shapes_are_validated(tmp_path) -> None:
    model_path = tmp_path / "bad_shape.npz"
    np.savez_compressed(
        model_path,
        feature_names=np.asarray(FEATURE_NAMES),
        scaler_mean=np.zeros(len(FEATURE_NAMES)),
        scaler_scale=np.ones(len(FEATURE_NAMES)),
        layer_count=np.asarray(1),
        coef_0=np.zeros((3, 1)),
        intercept_0=np.zeros(1),
    )
    with pytest.raises(ValueError, match="coefficient"):
        LearnedMarginPredictor(model_path)


def test_plain_npy_file_is_not_accepted_as_an_artifact(tmp_path) -> None:
    model_path = tmp_path / "array.npy"
    np.save(model_path, np.zeros(3))
    with pytest.raises(ValueError, match="NPZ archive"):
        LearnedMarginPredictor(model_path)
