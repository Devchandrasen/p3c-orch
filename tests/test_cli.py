import json
import subprocess
import sys
from pathlib import Path

import pytest

from p3c_orch import cli

ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def test_inspect_config_command(capsys) -> None:
    assert cli.main(["inspect-config", "--config", str(DEFAULT_CONFIG)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["simulation"]["swarms"] == 4


def test_simulate_command_writes_outputs(tmp_path, capsys) -> None:
    output = tmp_path / "results"
    result = cli.main(
        [
            "simulate",
            "--config",
            str(DEFAULT_CONFIG),
            "--output",
            str(output),
            "--slots",
            "1",
            "--seeds",
            "7",
            "--methods",
            "p3c-lr",
            "et-p3c",
            "--regimes",
            "r7-combined-stress",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert set(payload) == {"run_metrics", "summary", "metadata"}
    assert all(Path(path).exists() for path in payload.values())


def test_train_predictor_command_routes_arguments(tmp_path, monkeypatch, capsys) -> None:
    captured = {}

    def fake_train(csv_path, output_path, *, random_state):
        captured.update(
            csv_path=csv_path, output_path=output_path, random_state=random_state
        )
        return {"samples": 20.0, "mae_db": 1.0, "rmse_db": 1.2}

    monkeypatch.setattr(cli, "train_predictor", fake_train)
    csv_path = tmp_path / "samples.csv"
    output_path = tmp_path / "model.npz"
    assert (
        cli.main(
            [
                "train-predictor",
                "--csv",
                str(csv_path),
                "--output",
                str(output_path),
                "--random-state",
                "9",
            ]
        )
        == 0
    )
    assert captured == {
        "csv_path": csv_path,
        "output_path": output_path,
        "random_state": 9,
    }
    assert json.loads(capsys.readouterr().out)["samples"] == 20.0


def test_generate_predictor_data_command(tmp_path, capsys) -> None:
    output = tmp_path / "samples.csv"
    assert (
        cli.main(
            [
                "generate-predictor-data",
                "--config",
                str(DEFAULT_CONFIG),
                "--output",
                str(output),
                "--samples",
                "20",
                "--seed",
                "7",
            ]
        )
        == 0
    )
    assert Path(json.loads(capsys.readouterr().out)["dataset"]) == output
    assert output.exists()


def test_ablation_command_selects_combined_stress(tmp_path, monkeypatch, capsys) -> None:
    captured = {}

    def fake_run(config, output):
        captured["config"] = config
        path = Path(output) / "placeholder"
        return {"run_metrics": path, "summary": path, "metadata": path}

    monkeypatch.setattr(cli, "run_experiment", fake_run)
    assert (
        cli.main(
            [
                "ablate",
                "--config",
                str(DEFAULT_CONFIG),
                "--output",
                str(tmp_path),
                "--slots",
                "1",
                "--seeds",
                "7",
            ]
        )
        == 0
    )
    capsys.readouterr()
    config = captured["config"]
    assert config.simulation.regimes == ("r7-combined-stress",)
    assert "no-ann-prediction" in config.simulation.methods
    assert "et-p3c" in config.simulation.methods


def test_module_entrypoint_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "p3c_orch", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Predictive 3C orchestration" in result.stdout


def test_missing_command_is_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.main([])
