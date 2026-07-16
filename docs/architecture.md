# Architecture

P3C-Orch separates exogenous scenario generation, link prediction, risk
calibration, pair scoring, constrained assignment, state updates, and analysis.

```text
users + swarms + weather + content demand
                    |
                    v
       air-to-ground pair observations
                    |
                    v
     next-slot margin -> calibrated outage risk
                    |
                    v
 cache + delay + energy + switch + load terms
                    |
                    v
        P3C-LR / P3C-SR / ET-P3C score
                    |
                    v
 urgency order + capacity/energy + dwell guard
                    |
                    v
 queue + cache + energy + assignment state update
                    |
                    v
      per-run metrics -> paired analysis + plots
```

## Modules

- `config.py` loads YAML, applies defaults, resolves model paths, and rejects
  invalid or unknown values.
- `models.py` defines users, swarms, clusters, content, pair observations,
  assignments, and run metrics.
- `channel.py` implements probabilistic LoS, path loss, weather effects,
  shadowing, SNR margin, service rate, and Markov weather.
- `dataset.py` generates paired current/next-slot channel samples.
- `predictor.py` provides reactive, heuristic, and learned margin predictors plus
  weather-calibrated outage estimation.
- `scheduler.py` implements the P3C score, LR/SR/event variants, ablations,
  urgency ordering, dwell control, and cumulative resource constraints.
- `simulation.py` implements user and swarm mobility, clustering, demand, queues,
  caches, all comparison policies, state transitions, and metric aggregation.
- `analysis.py` computes the normalized objective, paired Holm-Wilcoxon tests,
  and result plots.
- `cli.py` exposes every workflow as a command.

## Determinism and pairing

Each run uses separate random streams for initialization, weather, per-slot
exogenous state, and policy-specific random choices. For a fixed seed and regime,
all methods receive the same user motion, swarm motion, traffic randomness,
weather sequence, and channel randomness. Random-policy choices cannot perturb the
environment stream. This preserves paired comparisons while allowing caches,
queues, energy, and assignments to evolve according to each method.

## Scheduling data flow

For each feasible swarm-cluster pair, the scheduler predicts next-slot link margin
and converts it to outage risk:

```text
risk = sigmoid((margin_threshold - predicted_margin) / weather_residual_scale)
```

The cache term rewards fresh local and neighbor hits, discounts them by link
reliability, and penalizes stale or fetched data. The delay term combines deadline,
queue, cache-miss, and outage risk. Switching includes a fixed reassignment cost and
a distance-dependent cost. P3C-LR combines normalized positive and penalty terms;
P3C-SR adds urgency-weighted service minus load-regularized service. ET-P3C applies
a bounded stability impulse only when calibrated outage risk crosses its trigger.

Clusters are processed in urgency/backlog order. Assignment reserves service
capacity and energy cumulatively. The dwell guard keeps a feasible previous swarm
when the score improvement is below its configured threshold.

## Artifact boundary

Configuration and source code are versioned. Predictor datasets, model NPZ files,
run CSVs, metadata, and plots are generated into ignored directories. No fixed
metric table is used by the simulator or analysis pipeline.
