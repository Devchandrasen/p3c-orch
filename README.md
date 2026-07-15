# P3C-Orch

P3C-Orch is a reference implementation of predictive communication, computation,
and caching orchestration for UAV-swarm IoT services. It combines calibrated
link-risk estimation, cache-aware utility, load and energy costs, and a dwell guard
that suppresses unnecessary handovers.

The repository is organized as an executable software project. It provides a
command-line simulator, configurable P3C-LR and P3C-SR schedulers, deterministic
experiment seeds, CSV outputs, tests, and continuous integration.

## Features

- P3C-LR predictive, cache-aware swarm-to-cluster assignment
- P3C-SR backlog- and load-aware stability regularization
- Reactive-3C, nearest-swarm, rate-max, and random reference policies
- Weather-calibrated conversion from predicted link margin to outage risk
- Greedy capacity-constrained scheduling with a configurable dwell guard
- Reproducible synthetic UAV-swarm scenarios and machine-readable metrics
- Optional scikit-learn MLP loading and training support

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
p3c-orch simulate --config configs/default.yaml --output results/example
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
p3c-orch simulate --config configs/default.yaml --output results/example
```

The command creates:

- `results/example/run_metrics.csv`: one row per method and seed
- `results/example/summary.csv`: mean and 95% confidence interval by method
- `results/example/run_metadata.json`: resolved configuration and run metadata

## Configuration

The default experiment is defined in [`configs/default.yaml`](configs/default.yaml).
It controls scenario size, seeds, weather probabilities, scheduler weights, risk
calibration, and evaluated methods.

Validate a configuration without running an experiment:

```bash
p3c-orch inspect-config --config configs/default.yaml
```

Run a smaller smoke experiment:

```bash
p3c-orch simulate --config configs/default.yaml --output results/smoke --slots 5 --seeds 7
```

## Optional learned predictor

The default simulator uses a deterministic heuristic margin predictor so the project
runs without a model artifact. To train an MLP from a CSV dataset, install the ML
extra and provide the documented feature columns plus `target_next_margin_db`:

```bash
python -m pip install -e ".[ml]"
p3c-orch train-predictor --csv data/link_samples.csv --output models/link_predictor.npz
```

Set `predictor.model_path` in the YAML configuration to use the saved model.
Relative model paths are resolved from the YAML file's directory. Predictor artifacts
use a numeric-only NumPy format and are loaded with object deserialization disabled.
For example, `configs/default.yaml` would reference the command's output as
`model_path: ../models/link_predictor.npz`.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
```

See [`docs/architecture.md`](docs/architecture.md) for the data flow and extension
points, and [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidance.

## Scope

The bundled environment is a deterministic synthetic reference scenario intended to
exercise and test the orchestration pipeline. Its generated metrics are not presented
as field measurements or as a reproduction of unavailable archived experiment logs.
