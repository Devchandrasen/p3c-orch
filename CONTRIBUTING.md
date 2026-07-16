# Contributing

Contributions must preserve deterministic paired runs, cumulative resource safety,
and the implementation-only repository boundary.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,full]"
```

## Required checks

```powershell
ruff check .
pytest
p3c-orch simulate --config configs/default.yaml --output results/smoke `
  --slots 2 --seeds 7 --regimes r7-combined-stress
python -m build
```

Add tests for configuration boundaries, channel behavior, scheduling invariants,
deterministic simulation, analysis, and command routing as appropriate. Do not
commit generated results, datasets, trained models, plots, or publication sources.
