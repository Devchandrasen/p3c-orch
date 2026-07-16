import math

import numpy as np

from p3c_orch.channel import AirToGroundChannel, MarkovWeather
from p3c_orch.config import ChannelConfig


def test_los_probability_increases_with_elevation() -> None:
    channel = AirToGroundChannel(ChannelConfig())
    assert 0.0 < channel.los_probability(10.0) < channel.los_probability(80.0) < 1.0


def test_channel_sample_is_finite_and_bounded() -> None:
    channel = AirToGroundChannel(ChannelConfig())
    sample = channel.sample(
        horizontal_distance_m=500.0,
        altitude_m=150.0,
        bandwidth_mhz=20.0,
        transmit_power_dbm=30.0,
        relative_speed_mps=5.0,
        weather="rainy",
        rng=np.random.default_rng(7),
    )
    assert all(
        math.isfinite(value)
        for value in (
            sample.distance_m,
            sample.elevation_deg,
            sample.los_probability,
            sample.path_loss_db,
            sample.margin_db,
            sample.rate_mb_per_slot,
        )
    )
    assert sample.distance_m > 500.0
    assert 0.0 <= sample.los_probability <= 1.0
    assert 0.0 <= sample.rate_mb_per_slot <= 20.0


def test_markov_weather_is_deterministic_for_a_seed() -> None:
    probabilities = {"clear": 0.6, "rainy": 0.25, "rain_hot": 0.15}
    first = MarkovWeather(probabilities, 0.8, np.random.default_rng(9))
    second = MarkovWeather(probabilities, 0.8, np.random.default_rng(9))
    assert [first.advance() for _ in range(20)] == [second.advance() for _ in range(20)]
