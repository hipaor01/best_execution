"""Estrategia de recompensa basada en VWAP."""
from .base import RewardContext, RewardStrategy


class VwapRewardStrategy(RewardStrategy):
    """Calcula una recompensa tomando VWAP como referencia."""

    def compute(self, context: RewardContext) -> float:
        """Calcula la recompensa de la acción descrita por ``context``."""
        if context.episode_vwap <= 0:
            raise ValueError("episode_vwap must be positive")

        if context.execution_volume == 0.0:
            return 0.0

        return (
            context.episode_vwap - context.execution_price
        ) / context.episode_vwap

    

