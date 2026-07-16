import csv

import pytest

from p3c_orch.dataset import generate_predictor_dataset
from p3c_orch.predictor import FEATURE_NAMES

from .test_simulation import small_config


def test_predictor_dataset_has_exact_features_and_is_deterministic(tmp_path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    generate_predictor_dataset(small_config(), first, samples=25, seed=7)
    generate_predictor_dataset(small_config(), second, samples=25, seed=7)
    assert first.read_bytes() == second.read_bytes()
    with first.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == [*FEATURE_NAMES, "target_next_margin_db"]
    assert len(rows) == 25
    assert {row["weather_rainy"] for row in rows} <= {"0.0", "1.0"}


def test_predictor_dataset_rejects_too_few_samples(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 20"):
        generate_predictor_dataset(small_config(), tmp_path / "bad.csv", samples=19)
