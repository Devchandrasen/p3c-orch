# Contributing

Contributions should keep the simulator deterministic for a fixed seed and preserve
the public scheduler interfaces.

## Setup

```bash
python -m venv .venv
python -m pip install -e ".[dev,ml]"
```

## Before opening a pull request

```bash
ruff check .
pytest
p3c-orch simulate --config configs/default.yaml --output results/smoke --slots 3 --seeds 7
```

Include tests for scheduling behavior, configuration validation, and metric changes.
Do not commit generated `results/` data or trained model artifacts.
