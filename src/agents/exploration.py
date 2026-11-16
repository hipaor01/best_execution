"""Políticas de exploración para el agente DDQN.

El agente consulta ``policy.epsilon`` en cada acción. Las políticas reciben
eventos del entrenamiento (pasos, episodios y evaluaciones) y deciden si ese
evento debe modificar epsilon:

* ``LinearDecayEpsilon`` reproduce la agenda de los ejercicios 1 y 2: epsilon
  depende únicamente del contador de pasos. Se conserva como referencia.
* ``PerformanceEpsilon`` (ejercicio 3) ignora los contadores. Solo cambia
  epsilon cuando una evaluación sin exploración muestra que la política ha
  mejorado de forma sostenida.
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Mapping


class ExplorationPolicy(ABC):
    """Interfaz común. Los eventos no usados son no-ops."""

    @property
    @abstractmethod
    def epsilon(self) -> float:
        """Probabilidad de elegir una acción aleatoria."""

    def on_step(self, step: int) -> None:
        """Se llama tras cada interacción de entrenamiento."""

    def on_episode_end(self, episode: int) -> None:
        """Se llama al terminar cada episodio de entrenamiento."""

    def on_evaluation(self, metrics: Mapping[str, float]) -> None:
        """Se llama tras cada evaluación sin exploración."""

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Estado serializable en JSON para los checkpoints."""

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restaura el estado guardado por ``state_dict``."""


def _check_bounds(epsilon_min: float, epsilon_max: float) -> None:
    if not 0 <= epsilon_min <= epsilon_max <= 1:
        raise ValueError("bounds must satisfy 0 <= epsilon_min <= epsilon_max <= 1")


class LinearDecayEpsilon(ExplorationPolicy):
    """Agenda lineal por pasos usada en los ejercicios 1 y 2 (referencia)."""

    def __init__(self, epsilon_max: float = 1.0, epsilon_min: float = 0.05, decay_steps: int = 50_000):
        _check_bounds(epsilon_min, epsilon_max)
        if decay_steps <= 0:
            raise ValueError("decay_steps must be positive")
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay_steps = decay_steps
        self._epsilon = epsilon_max

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def on_step(self, step: int) -> None:
        fraction = min(step / self.decay_steps, 1.0)
        self._epsilon = max(self.epsilon_min, self.epsilon_max - (self.epsilon_max - self.epsilon_min) * fraction)

    def state_dict(self) -> Dict[str, Any]:
        return {"epsilon": self._epsilon}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._epsilon = float(state["epsilon"])


class PerformanceEpsilon(ExplorationPolicy):
    """Epsilon guiado por lo que aprende el agente, no por el contador.

    Señal: ``metrics['score']`` en [0, 1], producida por una evaluación sin
    exploración (ver ``src.training.evaluation``): 0 significa que la política
    greedy no completa órdenes ni sigue la pauta; 1, que completa todas con un
    error comparable al TWAP determinista.

    Dos vías hacen bajar epsilon, y ninguna depende de pasos ni episodios:

    1. **Mejora de resultados.** ``progreso`` es la mejor media móvil de
       ``window`` evaluaciones y fija un techo::

           techo_resultados = epsilon_min + (epsilon_max - epsilon_min) * (1 - progreso)

    2. **Exploración agotada.** Si la media móvil no mejora en ``patience``
       evaluaciones seguidas, la exploración actual ya no aporta información
       nueva: el agente conoce lo que puede descubrir a ese nivel. El techo de
       saturación se multiplica por ``stall_factor`` y el contador se reinicia.
       Sin esta vía, una política que sigue la pauta pero no remata la orden
       (puntuación ~0,5) dejaría epsilon fijo en ~0,5 y no vería con
       suficiente frecuencia sus propios estados finales.

    ``epsilon = max(epsilon_min, min(techo_resultados, techo_saturación))``.

    Protecciones frente al ruido:

    * media móvil: hasta tener ``window`` evaluaciones epsilon no se mueve y
      una evaluación aislada no decide sola;
    * tolerancia ``min_delta``: solo cuenta como mejora superar el mejor
      valor previo en más de ese margen;
    * trinquete: una evaluación mala no devuelve exploración ya descartada;
    * paso máximo: epsilon no baja más de ``max_decrease`` por evaluación.
    """

    def __init__(
        self,
        epsilon_max: float = 1.0,
        epsilon_min: float = 0.05,
        window: int = 3,
        min_delta: float = 0.02,
        patience: int = 4,
        stall_factor: float = 0.7,
        max_decrease: float = 0.25,
    ):
        _check_bounds(epsilon_min, epsilon_max)
        if window < 1:
            raise ValueError("window must be at least 1")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if not 0 < stall_factor <= 1:
            raise ValueError("stall_factor must be in (0, 1]")
        if not 0 < max_decrease <= 1:
            raise ValueError("max_decrease must be in (0, 1]")
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.window = window
        self.min_delta = min_delta
        self.patience = patience
        self.stall_factor = stall_factor
        self.max_decrease = max_decrease
        self._scores = deque(maxlen=window)
        self._progress = 0.0
        self._stall_cap = epsilon_max
        self._since_improvement = 0
        self._epsilon = epsilon_max
        self.last_reason = None

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def progress(self) -> float:
        return self._progress

    def on_evaluation(self, metrics: Mapping[str, float]) -> None:
        score = float(metrics["score"])
        if not 0 <= score <= 1:
            raise ValueError("score must be in [0, 1]")
        self._scores.append(score)
        self.last_reason = None
        if len(self._scores) < self.window:
            return
        smoothed = sum(self._scores) / len(self._scores)
        if smoothed > self._progress + self.min_delta:
            self._progress = smoothed
            self._since_improvement = 0
            self.last_reason = "mejora"
        else:
            self._since_improvement += 1
            if self._since_improvement >= self.patience:
                self._stall_cap = max(self.epsilon_min, self._stall_cap * self.stall_factor)
                self._since_improvement = 0
                self.last_reason = "estancamiento"
        result_cap = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * (1 - self._progress)
        target = max(self.epsilon_min, min(result_cap, self._stall_cap))
        self._epsilon = min(self._epsilon, max(target, self._epsilon - self.max_decrease))

    def state_dict(self) -> Dict[str, Any]:
        return {"epsilon": self._epsilon, "progress": self._progress, "scores": list(self._scores),
                "stall_cap": self._stall_cap, "since_improvement": self._since_improvement}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._epsilon = float(state["epsilon"])
        self._progress = float(state["progress"])
        self._scores = deque(state["scores"], maxlen=self.window)
        self._stall_cap = float(state["stall_cap"])
        self._since_improvement = int(state["since_improvement"])
