from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from p3c_orch.models import PairObservation
from p3c_orch.predictor import FEATURE_NAMES


def make_observation(**overrides: Any) -> PairObservation:
    values: dict[str, Any] = {
        "slot": 0,
        "swarm_id": 0,
        "cluster_id": 0,
        "weather": "clear",
        "distance_m": 500.0,
        "elevation_deg": 20.0,
        "relative_speed_mps": 5.0,
        "bandwidth_mhz": 20.0,
        "transmit_power_dbm": 30.0,
        "current_margin_db": 2.0,
        "previous_outage": False,
        "local_cache_hit": False,
        "neighbor_cache_hit": False,
        "residual_energy_kj": 100.0,
        "load": 0.0,
        "time_fraction": 0.0,
        "raw_rate": 5.0,
        "energy_cost": 0.5,
        "delay_cost": 1.0,
    }
    values.update(overrides)
    return PairObservation(**values)


def write_training_csv(path: Path, *, rows: int = 80) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[*FEATURE_NAMES, "target_next_margin_db"],
        )
        writer.writeheader()
        for index in range(rows):
            current_margin = -8.0 + 0.25 * index
            row = {name: 0.0 for name in FEATURE_NAMES}
            row.update(
                {
                    "distance_m": 200.0 + 5.0 * index,
                    "elevation_deg": 25.0,
                    "relative_speed_mps": 3.0,
                    "bandwidth_mhz": 20.0,
                    "transmit_power_dbm": 30.0,
                    "current_margin_db": current_margin,
                    "residual_energy_kj": 100.0,
                    "time_fraction": index / max(rows - 1, 1),
                    "target_next_margin_db": current_margin - 0.2,
                }
            )
            writer.writerow(row)
