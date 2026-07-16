"""Synthetic channel-trace generation for the ANN link predictor."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .channel import AirToGroundChannel
from .config import ProjectConfig
from .models import PairObservation
from .predictor import FEATURE_NAMES


def generate_predictor_dataset(
    config: ProjectConfig,
    output_path: str | Path,
    *,
    samples: int = 6000,
    seed: int = 2026,
) -> Path:
    """Generate paired current/next-slot channel observations as training CSV."""

    if samples < 20:
        raise ValueError("at least 20 predictor samples are required")
    config.validate()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    channel = AirToGroundChannel(config.channel)
    weather_names = tuple(config.simulation.weather_probabilities)
    weather_probabilities = tuple(config.simulation.weather_probabilities.values())
    rows: list[dict[str, float]] = []
    for index in range(samples):
        weather = str(rng.choice(weather_names, p=weather_probabilities))
        next_weather = weather
        if rng.random() >= config.simulation.weather_persistence:
            next_weather = str(rng.choice(weather_names, p=weather_probabilities))
        horizontal_distance = float(
            rng.uniform(20.0, config.channel.maximum_link_distance_m)
        )
        altitude = float(rng.uniform(100.0, 200.0))
        bandwidth = float(rng.uniform(15.0, 25.0))
        transmit_power = float(rng.uniform(28.0, 32.0))
        relative_speed = abs(
            float(rng.normal(0.0, config.simulation.user_speed_std_mps))
        )
        current = channel.sample(
            horizontal_distance_m=horizontal_distance,
            altitude_m=altitude,
            bandwidth_mhz=bandwidth,
            transmit_power_dbm=transmit_power,
            relative_speed_mps=relative_speed,
            weather=weather,
            rng=rng,
        )
        radial_motion = relative_speed * float(rng.uniform(-1.0, 1.0))
        next_distance = min(
            max(horizontal_distance + radial_motion, 1.0),
            config.channel.maximum_link_distance_m * 1.1,
        )
        next_sample = channel.sample(
            horizontal_distance_m=next_distance,
            altitude_m=altitude,
            bandwidth_mhz=bandwidth,
            transmit_power_dbm=transmit_power,
            relative_speed_mps=max(0.0, relative_speed + float(rng.normal(0.0, 0.5))),
            weather=next_weather,
            rng=rng,
        )
        local_cache_hit = bool(rng.random() < 0.30)
        observation = PairObservation(
            slot=index,
            swarm_id=0,
            cluster_id=0,
            weather=weather,
            distance_m=current.distance_m,
            elevation_deg=current.elevation_deg,
            relative_speed_mps=relative_speed,
            bandwidth_mhz=bandwidth,
            transmit_power_dbm=transmit_power,
            current_margin_db=current.margin_db,
            previous_outage=(
                current.margin_db < config.predictor.margin_threshold_db
            ),
            local_cache_hit=local_cache_hit,
            neighbor_cache_hit=False,
            residual_energy_kj=float(rng.uniform(20.0, 280.0)),
            load=float(rng.uniform(0.0, 1.0)),
            time_fraction=(index % config.simulation.slots)
            / max(config.simulation.slots - 1, 1),
            raw_rate=current.rate_mb_per_slot,
            energy_cost=0.0,
            delay_cost=0.0,
        )
        row = observation.predictor_features()
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError("generated predictor features must be finite")
        row["target_next_margin_db"] = next_sample.margin_db
        rows.append(row)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[*FEATURE_NAMES, "target_next_margin_db"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return output
