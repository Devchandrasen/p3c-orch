# Evaluation protocol

`configs/full_protocol.yaml` is the complete evaluation entry point.

## Scenario

- 1,000 m by 1,000 m service area
- Four heterogeneous swarms with 6-10 UAVs each
- UAV altitude sampled from 100-200 m
- Sixty mobile users represented by ten persistent service clusters
- One hundred files sized 1-10 MB with Zipf-distributed requests
- Size- and count-bounded TTL caches with local and neighbor lookup
- Gauss-Markov user motion with reflecting boundaries
- Persistent clear, rainy, and rain-hot weather
- Two hundred time slots per run
- Eighty paired seeds, 4001 through 4080

## Methods

The practical comparison set is Random, Nearest, RateMax, MUCCO-like,
Reactive-3C, P3C-LR, P3C-SR, and ET-P3C. Component ablations disable prediction,
cache value, dwell control, risk calibration, or neighbor caching one at a time.

## Regimes

- R1 normal load
- R2 near saturation
- R3 overload
- R4 severe burst traffic
- R5 harsh weather
- R6 high mobility
- R7 combined stress

Each regime modifies arrival intensity, burst behavior, mobility, service capacity,
cache pressure, and weather mixture through the typed registry in `constants.py`.

## Metrics and analysis

Primary metrics are average and p95 delay, energy, handovers per 100 active
cluster-slots, useful cache-hit ratio, outage probability, throughput, and drop
rate. Secondary metrics include local/neighbor cache hits, weather-specific outage,
average backlog, SLA violations, load imbalance, Jain fairness, and deterministic
controller-operation count.

`summary.csv` reports means and normal-approximation 95% confidence intervals.
The analysis command computes a lower-is-better min-max normalized objective with
weights:

```text
+0.18 average delay       +0.16 p95 delay
+0.16 energy              +0.14 handovers
+0.12 outage              +0.10 drop rate
+0.08 load imbalance      -0.08 useful cache hits
-0.08 throughput          -0.04 Jain fairness
```

Paired two-sided Wilcoxon tests use matching regime/seed runs. Holm correction is
applied across objective metrics within each regime.

## Recommended execution

1. Run a two-slot, one-seed smoke test.
2. Generate and train the MLP predictor.
3. Run one regime with several paired seeds.
4. Start the full protocol only after those checks pass.
5. Run analysis against the newly generated `run_metrics.csv`.
