# P3C-Orch

P3C-Orch is a complete executable implementation of predictive communication,
computation, and caching orchestration for UAV-swarm IoT services. The project
contains the simulator, channel and mobility models, ANN link predictor, scheduling
policies, baselines, ablations, statistical analysis, tests, and automation needed to
run the full evaluation from source.

This repository is implementation-only. Generated datasets, trained models, result
CSVs, plots, and publication files are not committed.

## What is implemented

- Probabilistic-LoS air-to-ground channel with weather attenuation, shadowing,
  mobility mismatch, SNR margin, and capped Shannon service rate
- Persistent clear, rainy, and rain-hot weather process
- Four heterogeneous UAV swarms with configurable UAV count, altitude, radio,
  service capacity, energy, and TTL/size-bounded caches
- Sixty Gauss-Markov mobile users, persistent centroid clustering, Zipf content
  demand, burst traffic, deadlines, and queue evolution
- A 13-feature next-slot link-margin predictor with a `64 x 64 x 64` ReLU MLP,
  safe numeric NPZ serialization, and weather-specific residual calibration
- P3C-LR, stability-regularized P3C-SR, event-triggered ET-P3C, and Reactive-3C
- Random, nearest, rate-max, and MUCCO-like comparison policies
- No-prediction, no-cache-value, no-dwell, no-risk-calibration, and
  no-neighbor-cache ablations
- Seven operating regimes from normal load through combined stress
- Per-run metrics, means with 95% confidence intervals, normalized aggregate
  objective, paired Wilcoxon tests with Holm correction, and generated plots

## Install

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,full]"
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

## Quick run

The default configuration exercises every practical method and all seven regimes
with one seed and ten slots:

```powershell
p3c-orch simulate --config configs/default.yaml --output results/smoke
```

For a minimal command-line smoke test:

```powershell
p3c-orch simulate --config configs/default.yaml --output results/tiny `
  --slots 2 --seeds 7 --regimes r7-combined-stress
```

The output directory contains:

- `run_metrics.csv`: one row per regime, method, and seed
- `summary.csv`: metric means and 95% confidence intervals
- `run_metadata.json`: resolved configuration and provenance for the run

## Full evaluation workflow

First generate current/next-slot channel samples and train the MLP:

```powershell
p3c-orch prepare-predictor `
  --config configs/full_protocol.yaml `
  --output artifacts/predictor `
  --samples 6000 --seed 2026
```

Then run 200 slots for all eight methods, seven regimes, and 80 paired seeds, and
create the statistical outputs and plots:

```powershell
p3c-orch run-protocol `
  --config configs/full_protocol.yaml `
  --model artifacts/predictor/link_predictor.npz `
  --output results/full `
  --analyze
```

This is intentionally a large run: `7 x 8 x 80 = 4,480` independent simulations.
Use `--slots`, `--seeds`, `--methods`, or `--regimes` to validate a smaller subset
before starting it.

Run the component ablations under combined stress:

```powershell
p3c-orch ablate `
  --config configs/full_protocol.yaml `
  --model artifacts/predictor/link_predictor.npz `
  --output results/ablations
```

Analyze an existing fresh run separately:

```powershell
p3c-orch analyze --runs results/full/run_metrics.csv --output results/full/analysis
```

## Commands

```text
simulate                 Run configured methods, regimes, and seeds
run-protocol             Run the complete configured protocol
ablate                   Run all component ablations under combined stress
inspect-config           Validate and print a resolved YAML configuration
generate-predictor-data  Generate ANN current/next-slot training samples
prepare-predictor        Generate samples, train the MLP, and save calibration
train-predictor          Train the MLP from an existing compatible CSV
analyze                  Compute objective, paired tests, and plots
```

Learned predictor paths in YAML are resolved relative to the YAML file. Predictor
artifacts contain numeric NumPy arrays only and are loaded with object
deserialization disabled.

## Development

```powershell
ruff check .
pytest
python -m build
```

See [architecture](docs/architecture.md), [protocol](docs/protocol.md), and
[contributing](CONTRIBUTING.md) for implementation details.
