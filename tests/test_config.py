from pathlib import Path

import pytest

from p3c_orch.config import PredictorConfig, ProjectConfig, SimulationConfig, load_config
from p3c_orch.constants import DEFAULT_METHODS, DEFAULT_REGIMES


def test_default_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "default.yaml"
    config = load_config(path)
    assert config.simulation.swarms == 4
    assert config.simulation.methods == DEFAULT_METHODS
    assert config.simulation.regimes == DEFAULT_REGIMES
    assert sum(config.simulation.weather_probabilities.values()) == pytest.approx(1.0)


def test_full_protocol_has_eighty_paired_seeds_and_two_hundred_slots() -> None:
    path = Path(__file__).parents[1] / "configs" / "full_protocol.yaml"
    config = load_config(path)
    assert config.simulation.slots == 200
    assert config.simulation.seeds == tuple(range(4001, 4081))
    assert config.simulation.methods == DEFAULT_METHODS
    assert config.simulation.regimes == DEFAULT_REGIMES


def test_unknown_method_is_rejected() -> None:
    config = SimulationConfig(methods=("not-a-policy",))
    with pytest.raises(ValueError, match="unsupported methods"):
        config.validate()


@pytest.mark.parametrize(
    "config, message",
    [
        (SimulationConfig(methods=()), "methods must be non-empty"),
        (SimulationConfig(seeds=(7, 7)), "seeds must be unique"),
        (SimulationConfig(area_size_m=float("nan")), "finite and positive"),
        (SimulationConfig(slots=1.5), "must be a positive integer"),
        (SimulationConfig(seeds=(1.5,)), "seeds must be non-empty integers"),
    ],
)
def test_invalid_simulation_boundaries_are_rejected(
    config: SimulationConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_unknown_yaml_key_is_rejected(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("simulation:\n  slotz: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown simulation keys: slotz"):
        load_config(path)


def test_missing_predictor_model_is_rejected(tmp_path) -> None:
    config = PredictorConfig(model_path=tmp_path / "missing.npz")
    with pytest.raises(ValueError, match="predictor model does not exist"):
        config.validate()


def test_unsupported_weather_is_rejected() -> None:
    config = SimulationConfig(weather_probabilities={"snow": 1.0})
    with pytest.raises(ValueError, match="unsupported weather states"):
        config.validate()


@pytest.mark.parametrize(
    "payload, message",
    [
        ("simulation:\n  weather_probabilities: clear\n", "must be a mapping"),
        ("predictor:\n  residual_scale_db: 4.0\n", "must be a mapping"),
    ],
)
def test_nested_sections_must_be_mappings(tmp_path, payload: str, message: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_empty_yaml_uses_canonical_defaults(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == ProjectConfig()


def test_quoted_numeric_scalars_are_coerced(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "simulation:\n  slots: '3'\nchannel:\n  required_snr_db: '2.5'\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.simulation.slots == 3
    assert config.channel.required_snr_db == 2.5
