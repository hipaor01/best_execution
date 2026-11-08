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

    def compute_terminal(self, context: RewardContext) -> float:
        """Penaliza la fracción de volumen CARE pendiente.

        Devuelve ``0.0`` cuando la orden se ha completado. Si termina
        incompleta, devuelve un valor entre ``-1.0`` y ``0.0`` proporcional
        a la fracción de la orden que queda por ejecutar.

        Raises:
            ValueError: Si el volumen total de la orden CARE no es positivo.
        """
        if context.care_volume <= 0:
            raise ValueError("care_volume must be positive")

        remaining_volume = max(context.care_volume - context.executed_volume, 0.0)
        return -(remaining_volume / context.care_volume)

