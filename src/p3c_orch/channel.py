"""Air-to-ground channel and weather-state dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import ChannelConfig
from .constants import WEATHER_ATTENUATION_DB, WEATHER_SHADOWING_STD_DB

SPEED_OF_LIGHT_MPS = 299_792_458.0


@dataclass(frozen=True)
class ChannelSample:
    distance_m: float
    elevation_deg: float
    los_probability: float
    path_loss_db: float
    margin_db: float
    rate_mb_per_slot: float


class AirToGroundChannel:
    """Probabilistic-LoS channel with weather, mobility, and shadowing loss."""

    def __init__(self, config: ChannelConfig) -> None:
        config.validate()
        self.config = config

    def sample(
        self,
        *,
        horizontal_distance_m: float,
        altitude_m: float,
        bandwidth_mhz: float,
        transmit_power_dbm: float,
        relative_speed_mps: float,
        weather: str,
        rng: np.random.Generator,
    ) -> ChannelSample:
        distance_m = math.hypot(horizontal_distance_m, altitude_m)
        elevation_deg = math.degrees(
            math.atan2(altitude_m, max(horizontal_distance_m, 1e-9))
        )
        los_probability = self.los_probability(elevation_deg)
        free_space_loss = 20.0 * math.log10(
            4.0
            * math.pi
            * max(distance_m, 1.0)
            * self.config.carrier_frequency_hz
            / SPEED_OF_LIGHT_MPS
        )
        excess_loss = (
            los_probability * self.config.los_excess_loss_db
            + (1.0 - los_probability) * self.config.nlos_excess_loss_db
        )
        weather_loss = WEATHER_ATTENUATION_DB[weather] * distance_m / 1000.0
        mobility_mismatch_loss = 0.025 * max(relative_speed_mps, 0.0)
        shadowing = float(rng.normal(0.0, WEATHER_SHADOWING_STD_DB[weather]))
        path_loss_db = (
            free_space_loss
            + excess_loss
            + weather_loss
            + mobility_mismatch_loss
            + shadowing
        )
        noise_dbm = (
            self.config.noise_density_dbm_hz
            + 10.0 * math.log10(max(bandwidth_mhz, 1e-9) * 1e6)
            + self.config.receiver_noise_figure_db
        )
        snr_db = transmit_power_dbm - path_loss_db - noise_dbm
        margin_db = snr_db - self.config.required_snr_db
        snr_linear = 10.0 ** (snr_db / 10.0)
        spectral_efficiency = min(
            math.log2(1.0 + max(snr_linear, 0.0)),
            self.config.spectral_efficiency_cap,
        )
        rate_mb_per_slot = max(0.0, bandwidth_mhz * spectral_efficiency / 8.0)
        return ChannelSample(
            distance_m=distance_m,
            elevation_deg=elevation_deg,
            los_probability=los_probability,
            path_loss_db=path_loss_db,
            margin_db=margin_db,
            rate_mb_per_slot=rate_mb_per_slot,
        )

    def los_probability(self, elevation_deg: float) -> float:
        exponent = -self.config.los_environment_b * (
            elevation_deg - self.config.los_environment_a
        )
        return 1.0 / (1.0 + self.config.los_environment_a * math.exp(exponent))


class MarkovWeather:
    """Persistent weather process with configurable stationary target weights."""

    def __init__(
        self,
        probabilities: dict[str, float],
        persistence: float,
        rng: np.random.Generator,
    ) -> None:
        self.names = tuple(probabilities)
        self.probabilities = np.asarray(tuple(probabilities.values()), dtype=float)
        self.persistence = persistence
        self.rng = rng
        self.state = str(rng.choice(self.names, p=self.probabilities))

    def advance(self) -> str:
        if self.rng.random() >= self.persistence:
            self.state = str(self.rng.choice(self.names, p=self.probabilities))
        return self.state
