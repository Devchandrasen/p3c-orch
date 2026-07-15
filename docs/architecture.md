# Architecture

P3C-Orch separates environment generation, risk estimation, pair scoring, assignment,
and metric aggregation so each component can be replaced independently.

```text
scenario state
    |
    v
pair observations --> link predictor --> calibrated outage risk
    |                                      |
    +------------------+-------------------+
                       v
              normalized 3C pair score
                       |
                       v
         urgency ordering + dwell guard
                       |
                       v
             capacity-safe assignments
                       |
                       v
         queue/cache/energy state updates
```

## Package layout

- `config.py` loads and validates YAML configuration.
- `models.py` defines scheduler and simulation data structures.
- `predictor.py` provides heuristic and optional learned margin predictors.
- `scheduler.py` implements P3C-LR, P3C-SR, and Reactive-3C assignment.
- `simulation.py` runs deterministic synthetic scenarios and writes CSV outputs.
- `cli.py` exposes project commands.

Relative paths in a configuration file are resolved from that file's directory.
Learned predictor artifacts contain only numeric NumPy arrays and are loaded with
object deserialization disabled.

## P3C score

Every feasible swarm-cluster pair receives normalized positive terms for predicted
margin, useful cache value, fairness, and dwell continuity. Delay, energy, outage,
switching, and load are normalized penalty terms. P3C-SR adds a normalized urgency
and load regularizer before greedy assignment.

Clusters are processed by deadline-aware urgency and backlog. If the previous swarm
is still feasible, the dwell guard retains it when the best alternative improves the
score by no more than `scheduler.dwell_threshold`.

## Extension points

- Implement `MarginPredictor.predict_many()` for a new forecasting model.
- Construct `PairObservation` objects from a trace-driven environment.
- Add a policy in `simulation.py` and include its name in configuration validation.
- Add metrics without changing the scheduler by extending `SimulationMetrics`.
