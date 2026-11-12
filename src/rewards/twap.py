"""Recompensa por seguir una trayectoria de ejecución uniforme."""

from .base import RewardContext, RewardStrategy


class TwapRewardStrategy(RewardStrategy):
    """Penaliza la distancia a la pauta y el volumen pendiente.

    El horizonte virtual evita que terminar pronto elimine las penalizaciones
    futuras. Su descuento debe coincidir con el gamma del agente.
    """

    def __init__(self, gamma: float = 0.99, power: int = 1):
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if power not in (1, 2):
            raise ValueError("power must be 1 or 2")
        self.gamma = gamma
        self.power = power

    def _error(self, context: RewardContext, step: int) -> float:
        return abs(context.executed_volume / context.care_volume - step / context.episode_bins) ** self.power

    def compute(self, context: RewardContext) -> float:
        return -self._error(context, context.state_pos + 1)

    def compute_terminal(self, context: RewardContext) -> float:
        step = context.state_pos + 1
        future_error = sum(
            self.gamma ** (future_step - step) * self._error(context, future_step)
            for future_step in range(step + 1, context.episode_bins + 1)
        )
        pending = max(context.care_volume - context.executed_volume, 0) / context.care_volume
        return -future_error - pending
