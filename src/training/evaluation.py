"""Evaluación sin exploración de un agente de ejecución TWAP.

Se usa con dos fines distintos:

* ``probe_cases``: escenarios fijos con los que el entrenador mide la mejora
  durante el entrenamiento (señal del ejercicio 3).
* ``workshop_cases``: los 27 escenarios del ejercicio 2, reservados para
  informar del resultado final y comparar con ese ejercicio.

Las dos colecciones no comparten ninguna ventana (día y posición), de modo
que la señal que guía epsilon y la parada no se calcula sobre los escenarios
con los que se informa el resultado. Todos proceden de ``train``.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.agents.twap import twap_action
from src.environment.best_execution import BestExecutionEnv
from src.environment.config import EnvironmentConfig
from src.rewards import VwapRewardStrategy

HORIZONS = (400, 500, 600)
CARE_FRACTIONS = (0.075, 0.10, 0.125)


@dataclass(frozen=True)
class Case:
    """Una orden CARE concreta: día, inicio, horizonte y volumen fijos."""

    day: object
    start: int
    horizon: int
    volume: int
    window: pd.DataFrame
    look_back: int = 60


def usable_days(data: Mapping[object, pd.DataFrame]) -> Dict[object, pd.DataFrame]:
    """Excluye los días en los que ``cumvol`` desciende (el entorno los rechaza)."""
    return {day: frame for day, frame in data.items() if not (frame.cumvol.diff() < 0).any()}


def _case(data, day, horizon, fraction, position_fraction, look_back=60) -> Case:
    frame = data[day]
    volume = int(horizon * fraction)
    start = look_back + int((len(frame) - horizon - look_back) * position_fraction)
    window = frame.iloc[start - look_back:start + horizon].copy()
    return Case(day, start, horizon, volume, window, look_back)


def workshop_cases(data: Mapping[object, pd.DataFrame]) -> List[Case]:
    """Los 27 escenarios del ejercicio 2 (mismo orden y construcción)."""
    days = list(data)
    cases = []
    for horizon in HORIZONS:
        for fraction in CARE_FRACTIONS:
            for day_index, position in ((0, .1), (len(days) // 2, .5), (-1, .9)):
                cases.append(_case(data, days[day_index], horizon, fraction, position))
    return cases


def probe_cases(data: Mapping[object, pd.DataFrame]) -> List[Case]:
    """9 escenarios de control: una orden por horizonte y tamaño CARE.

    Días en los cuartiles 1 y 3 de la lista y posiciones 0,3 / 0,7 / 0,5, que
    no coinciden con ninguna ventana de ``workshop_cases``.
    """
    days = list(data)
    slots = ((len(days) // 4, .3), (3 * len(days) // 4, .7), (len(days) // 4 + 1, .5))
    cases = []
    for i, horizon in enumerate(HORIZONS):
        for j, fraction in enumerate(CARE_FRACTIONS):
            day_index, position = slots[(i + j) % 3]
            cases.append(_case(data, days[day_index], horizon, fraction, position))
    return cases


def _make_env(case: Case) -> BestExecutionEnv:
    # nextafter evita perder una unidad por redondeo en int(fracción * horizonte).
    fraction = float(np.nextafter(case.volume / case.horizon, 1.))
    config = EnvironmentConfig(look_back=case.look_back, min_episode_bins=case.horizon,
                               max_episode_bins=case.horizon, min_care_fraction=fraction,
                               max_care_fraction=fraction, seed=7)
    env = BestExecutionEnv({case.day: case.window}, config, VwapRewardStrategy())
    return env


def _summarise(case: Case, env: BestExecutionEnv) -> Dict[str, object]:
    path = np.pad(np.cumsum(env.vol_hist), (0, case.horizon - env.state_pos), mode="edge") / case.volume
    ideal = np.arange(1, case.horizon + 1) / case.horizon
    return {"day": case.day, "start": case.start, "horizon": case.horizon, "volume": case.volume,
            "mae": float(np.mean(np.abs(path - ideal))),
            "complete": bool(env.exec_vol == case.volume),
            "finish_fraction": env.state_pos / case.horizon,
            "algo_vwap": env.algo_vwap_hist[-1], "market_vwap": env.episode_vwap,
            "path": path, "actions": np.array(env.actions_hist)}


def run_policy(cases: Sequence[Case], q_model=None,
               action_fn: Optional[Callable[[BestExecutionEnv], int]] = None) -> List[Dict[str, object]]:
    """Ejecuta una política greedy en todos los casos a la vez.

    Con ``q_model`` la acción es el argmax de sus valores Q y los estados de
    todos los casos activos se evalúan en una única llamada por paso. Con
    ``action_fn`` se usa una política explícita (por ejemplo, el TWAP
    determinista).
    """
    if (q_model is None) == (action_fn is None):
        raise ValueError("pass exactly one of q_model or action_fn")
    envs = [_make_env(case) for case in cases]
    states = [env.reset() for env in envs]
    for case, env in zip(cases, envs):
        assert env.vol_care == case.volume
    active = list(range(len(envs)))
    while active:
        if q_model is not None:
            batch = np.stack([states[i] for i in active])
            actions = np.argmax(q_model(batch, training=False).numpy(), axis=1)
        else:
            actions = [action_fn(envs[i]) for i in active]
        still_active = []
        for i, action in zip(active, actions):
            state, _, done, _ = envs[i].step(int(action))
            states[i] = state
            if not done:
                still_active.append(i)
        active = still_active
    return [_summarise(case, env) for case, env in zip(cases, envs)]


class TwapEvaluator:
    """Convierte la ejecución greedy en métricas y en una puntuación [0, 1].

    Por escenario:

    * ``complete``: 1 si la orden CARE se completa;
    * ``tracking``: 1 si el error frente a la pauta (MAE) iguala al del TWAP
      determinista y 0 si lo supera en ``mae_tolerance`` o más; lineal entre
      ambos. 0,05 es el margen que el ejercicio 2 fijó como objetivo.

    ``score`` es la media de ``tracking * (0,5 + 0,5 * complete)``:

    * comprar en todos los pasos completa la orden pero no sigue la pauta: 0;
    * no comprar nunca: 0;
    * seguir la pauta sin terminar la orden: como máximo 0,5;
    * seguir la pauta y completar la orden: 1.
    """

    def __init__(self, cases: Sequence[Case], mae_tolerance: float = 0.05):
        if mae_tolerance <= 0:
            raise ValueError("mae_tolerance must be positive")
        self.cases = list(cases)
        self.mae_tolerance = mae_tolerance
        self.reference = run_policy(self.cases, action_fn=twap_action)
        self.reference_mae = np.array([row["mae"] for row in self.reference])

    def details(self, q_model) -> pd.DataFrame:
        rows = run_policy(self.cases, q_model=q_model)
        frame = pd.DataFrame([{k: v for k, v in row.items() if k not in ("path", "actions")} for row in rows])
        frame["reference_mae"] = self.reference_mae
        excess = (frame["mae"] - frame["reference_mae"]).clip(lower=0)
        frame["tracking"] = (1 - excess / self.mae_tolerance).clip(lower=0, upper=1)
        frame["case_score"] = frame["tracking"] * (0.5 + 0.5 * frame["complete"].astype(float))
        return frame

    def evaluate(self, q_model) -> Dict[str, float]:
        frame = self.details(q_model)
        return {"score": float(frame["case_score"].mean()),
                "completion": float(frame["complete"].mean()),
                "median_mae": float(frame["mae"].median()),
                "worst_mae": float(frame["mae"].max()),
                "tracking": float(frame["tracking"].mean())}

    def meets_workshop_target(self, metrics: Mapping[str, float]) -> bool:
        """Criterio del ejercicio 2: >= 90 % completadas y MAE mediano <= referencia + 0,05."""
        return (metrics["completion"] >= 0.9
                and metrics["median_mae"] <= float(np.median(self.reference_mae)) + self.mae_tolerance)
