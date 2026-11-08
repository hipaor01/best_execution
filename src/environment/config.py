"""Configuración del entorno de best execution."""

from dataclasses import dataclass
from numbers import Real
from typing import Optional


@dataclass(frozen=True)
class EnvironmentConfig:
    """Parámetros usados para generar los episodios del entorno.

    Los límites de duración son inclusivos. Para fijar un valor en lugar de
    generar uno aleatorio, se puede usar el mismo mínimo y máximo.
    """

    look_back: int = 60
    min_episode_bins: int = 400
    max_episode_bins: int = 600
    min_care_fraction: float = 0.075
    max_care_fraction: float = 0.125
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._validate_integer("look_back", self.look_back, minimum=0)
        self._validate_integer(
            "min_episode_bins", self.min_episode_bins, minimum=1
        )
        self._validate_integer(
            "max_episode_bins", self.max_episode_bins, minimum=1
        )

        if self.min_episode_bins > self.max_episode_bins:
            raise ValueError(
                "min_episode_bins must be less than or equal to "
                "max_episode_bins"
            )

        self._validate_fraction("min_care_fraction", self.min_care_fraction)
        self._validate_fraction("max_care_fraction", self.max_care_fraction)

        if self.min_care_fraction > self.max_care_fraction:
            raise ValueError(
                "min_care_fraction must be less than or equal to "
                "max_care_fraction"
            )

        if self.min_episode_bins * self.min_care_fraction < 1:
            raise ValueError(
                "min_episode_bins and min_care_fraction must produce a "
                "CARE volume of at least 1"
            )

        if self.seed is not None:
            self._validate_integer("seed", self.seed, minimum=0)

    @staticmethod
    def _validate_integer(name: str, value: int, minimum: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be greater than or equal to {minimum}")

    @staticmethod
    def _validate_fraction(name: str, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be greater than 0 and at most 1")
