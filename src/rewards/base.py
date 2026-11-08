"""Contratos compartidos por las estrategias de recompensa."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardContext:
    """Estado necesario para calcular la recompensa de una acción.

    Attributes:
        execution_price:
            Precio al que se ejecutó la acción actual. Su valor es ``0.0``
            cuando la acción no ejecuta volumen.
        execution_volume:
            Volumen ejecutado exclusivamente por la acción actual. Puede ser
            ``0.0`` para una acción de espera.
        episode_vwap:
            VWAP de mercado calculado sobre el episodio completo y utilizado
            como referencia de ejecución.
        state_pos:
            Posición, empezando en cero, del step en el que se ejecutó la
            acción. Corresponde al estado anterior al incremento efectuado
            por ``BestExecutionEnv.step()``.
        episode_bins:
            Número total de steps disponibles en el episodio.
        executed_volume:
            Volumen acumulado ejecutado por el algoritmo después de aplicar
            la acción actual. Por tanto, incluye ``execution_volume``.
        care_volume:
            Volumen total de la orden CARE que debe completarse durante el
            episodio.
    """

    execution_price: float
    execution_volume: float
    episode_vwap: float
    state_pos: int
    episode_bins: int
    executed_volume: float
    care_volume: float


class RewardStrategy(ABC):
    """Interfaz para calcular recompensas sin acoplarlas al entorno."""

    @abstractmethod
    def compute(self, context: RewardContext) -> float:
        """Calcula la recompensa asociada a una acción."""
        raise NotImplementedError

    def compute_terminal(self, context: RewardContext) -> float:
        """Calcula la recompensa adicional al finalizar el episodio."""
        return 0.0
