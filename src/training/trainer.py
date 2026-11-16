"""Bucle de entrenamiento del DDQN con evaluación, parada y checkpoints.

Orden de cada interacción (igual que en los ejercicios 1 y 2):

    acción -> step -> experiencia al buffer -> aprendizaje cada ``learn_every``
    -> copia a la red objetivo cada ``target_update_every`` -> evento de paso

Cada ``eval_every`` pasos se evalúa la política greedy con el evaluador. La
métrica alimenta a la política de exploración, a la parada temprana y al
checkpoint ``best``. Ningún componente modifica epsilon por episodios.
"""

import json
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf


@dataclass
class TrainerConfig:
    max_steps: int = 100_000
    warmup_steps: int = 256
    learn_every: int = 16
    target_update_every: int = 128
    eval_every: int = 2_000
    checkpoint_every: Optional[int] = 10_000
    checkpoint_dir: Optional[str] = None

    def __post_init__(self):
        for name in ("max_steps", "learn_every", "target_update_every", "eval_every"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.checkpoint_every is not None and self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive or None")


@dataclass
class EarlyStopping:
    """Condiciones de parada basadas en el aprendizaje (idea 4C).

    * ``target``: la puntuación de evaluación alcanza ``target_score`` en
      ``target_patience`` evaluaciones consecutivas. Una sola evaluación buena
      puede ser casualidad; dos seguidas indican una política estable.
    * ``plateau``: la mejor media móvil de ``smoothing`` evaluaciones no
      mejora en más de ``min_delta`` durante ``plateau_patience``
      evaluaciones. Seguir entrenando gasta cómputo sin evidencia de
      progreso. Se usa la media móvil para que un pico aislado (por ejemplo,
      una política inicial que compra muy deprisa) no fije una marca que
      luego dispare la parada.
    * Ninguna de las dos se aplica antes de ``min_steps``: con epsilon alto
      la política greedy todavía no es representativa.
    * Si no se cumple ninguna, se para en ``max_steps``.

    Con ``restore_best`` el agente termina con los pesos de la mejor
    evaluación y no con los del último paso.
    """

    target_score: float = 0.95
    target_patience: int = 2
    plateau_patience: int = 15
    min_delta: float = 0.01
    smoothing: int = 3
    min_steps: int = 20_000
    restore_best: bool = True
    # estado interno
    hits: int = field(default=0, init=False)
    since_improvement: int = field(default=0, init=False)
    best_smoothed: float = field(default=float("-inf"), init=False)
    recent: list = field(default_factory=list, init=False)

    def update(self, score: float, step: int) -> Optional[str]:
        self.hits = self.hits + 1 if score >= self.target_score else 0
        self.recent = (self.recent + [score])[-self.smoothing:]
        if len(self.recent) == self.smoothing:
            smoothed = sum(self.recent) / self.smoothing
            if smoothed > self.best_smoothed + self.min_delta:
                self.best_smoothed, self.since_improvement = smoothed, 0
            else:
                self.since_improvement += 1
        if step < self.min_steps:
            return None
        if self.hits >= self.target_patience:
            return "target"
        if self.since_improvement >= self.plateau_patience:
            return "plateau"
        return None


def _json_default(value):
    """Convierte tipos de NumPy y rutas al guardar ``state.json``."""
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return str(value)


class Trainer:
    def __init__(self, agent, env, evaluator, config: TrainerConfig,
                 early_stopping: Optional[EarlyStopping] = None):
        if agent.exploration_policy is None:
            raise ValueError("Trainer requires an agent with an exploration_policy")
        self.agent = agent
        self.env = env
        self.evaluator = evaluator
        self.config = config
        self.early_stopping = early_stopping
        self.step = 0
        self.episode = 0
        self.best_score = -np.inf
        self.best_step = None
        self._best_weights = None
        self.stop_reason = None
        self.episode_records: List[Dict] = []
        self.eval_records: List[Dict] = []
        self._state = None
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._warmed_up = False
        self._elapsed = 0.0

    # ------------------------------------------------------------------ bucle
    def _warmup(self):
        state = self.env.reset()
        for _ in range(self.config.warmup_steps):
            action = self.env.action_sample()
            next_state, reward, done, _ = self.env.step(action)
            self.agent.experience(state, action, reward, next_state, done)
            state = self.env.reset() if done else next_state
        self._state = self.env.reset()
        self._warmed_up = True

    def train(self) -> "Trainer":
        """Entrena hasta una condición de parada. Puede llamarse tras ``resume``."""
        cfg, agent, policy = self.config, self.agent, self.agent.exploration_policy
        agent.set_trainable(True)
        if not self._warmed_up:
            self._warmup()
        started = time.monotonic()
        while self.step < cfg.max_steps and self.stop_reason is None:
            action = agent.act(self._state)
            next_state, reward, done, _ = self.env.step(action)
            agent.experience(self._state, action, reward, next_state, done)
            self.step += 1
            self._episode_reward += reward
            self._episode_steps += 1
            if self.step % cfg.learn_every == 0:
                agent.learn()
            if self.step % cfg.target_update_every == 0:
                agent.update_target()
            policy.on_step(self.step)
            if done:
                self.episode += 1
                policy.on_episode_end(self.episode)
                self.episode_records.append({"step": self.step, "episode": self.episode,
                                             "reward": self._episode_reward,
                                             "episode_steps": self._episode_steps,
                                             "epsilon": agent.epsilon})
                self._state, self._episode_reward, self._episode_steps = self.env.reset(), 0.0, 0
            else:
                self._state = next_state
            if self.step % cfg.eval_every == 0:
                self._elapsed += time.monotonic() - started
                started = time.monotonic()
                self._evaluate()
            if cfg.checkpoint_dir and cfg.checkpoint_every and self.step % cfg.checkpoint_every == 0:
                self.save(Path(cfg.checkpoint_dir) / "last")
        self._elapsed += time.monotonic() - started
        if self.stop_reason is None:
            self.stop_reason = "max_steps"
        if self.config.checkpoint_dir:
            self.save(Path(self.config.checkpoint_dir) / "last")
        if self.early_stopping and self.early_stopping.restore_best and self._best_weights is not None:
            agent.primary_model.set_weights(self._best_weights)
            agent.update_target()
        agent.set_trainable(False)
        return self

    def _evaluate(self):
        metrics = self.evaluator.evaluate(self.agent.primary_model)
        policy = self.agent.exploration_policy
        epsilon_before = policy.epsilon
        policy.on_evaluation(metrics)
        if metrics["score"] > self.best_score:
            self.best_score, self.best_step = metrics["score"], self.step
            self._best_weights = self.agent.primary_model.get_weights()
            if self.config.checkpoint_dir:
                self.save(Path(self.config.checkpoint_dir) / "best")
        if self.early_stopping:
            self.stop_reason = self.early_stopping.update(metrics["score"], self.step)
        self.eval_records.append(dict(metrics, step=self.step, episode=self.episode,
                                      epsilon_before=epsilon_before, epsilon=policy.epsilon,
                                      seconds=self._elapsed))

    # ------------------------------------------------------------- resultados
    @property
    def episode_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.episode_records)

    @property
    def eval_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.eval_records)

    # ------------------------------------------------------------ checkpoints
    def save(self, path) -> Path:
        """Guarda todo lo necesario para continuar el entrenamiento.

        * ``primary.keras``: red principal con el estado del optimizador Adam;
        * ``target.weights.h5``: red objetivo;
        * ``buffer.pkl``: replay buffer y estados aleatorios (NumPy y entorno);
        * ``state.json``: contadores, epsilon, parada y métricas.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.agent.primary_model.save(path / "primary.keras")
        self.agent.target_model.save_weights(path / "target.weights.h5")
        with (path / "buffer.pkl").open("wb") as file:
            pickle.dump({"buffer": list(self.agent.buffer),
                         "numpy_random": np.random.get_state(),
                         "env_rng": self.env._rng.bit_generator.state,
                         "env_snapshot": self._env_snapshot(),
                         "state": self._state}, file)
        state = {"step": self.step, "episode": self.episode,
                 "best_score": self.best_score, "best_step": self.best_step,
                 "stop_reason": self.stop_reason,
                 "episode_reward": self._episode_reward, "episode_steps": self._episode_steps,
                 "elapsed_seconds": self._elapsed,
                 "policy": self.agent.exploration_policy.state_dict(),
                 "early_stopping": ({"hits": self.early_stopping.hits,
                                     "since_improvement": self.early_stopping.since_improvement,
                                     "best_smoothed": self.early_stopping.best_smoothed,
                                     "recent": self.early_stopping.recent}
                                    if self.early_stopping else None),
                 "config": asdict(self.config),
                 "eval_records": self.eval_records, "episode_records": self.episode_records}
        (path / "state.json").write_text(json.dumps(state, default=_json_default, indent=1))
        return path

    def _env_snapshot(self):
        names = ("episode_bins", "episode_full_len", "vol_care", "episode", "episode_full",
                 "episode_vwap", "market_ep_vol", "state_pos", "exec_vol", "algo_vwap",
                 "actions_hist", "algo_vwap_hist", "market_vwap_hist", "reward_hist",
                 "price_hist", "vol_hist")
        return {name: getattr(self.env, name) for name in names if hasattr(self.env, name)}

    def resume(self, path, continue_training: bool = True) -> "Trainer":
        """Restaura un checkpoint en este entrenador (mismo agente y entorno).

        Con ``continue_training`` se borra el motivo de parada guardado para
        poder seguir entrenando, por ejemplo con un ``max_steps`` mayor.
        """
        path = Path(path)
        state = json.loads((path / "state.json").read_text())
        self.agent.primary_model = tf.keras.models.load_model(path / "primary.keras")
        self.agent.target_model.load_weights(path / "target.weights.h5")
        with (path / "buffer.pkl").open("rb") as file:
            saved = pickle.load(file)
        self.agent.buffer.clear()
        self.agent.buffer.extend(saved["buffer"])
        np.random.set_state(saved["numpy_random"])
        self.env._rng.bit_generator.state = saved["env_rng"]
        for name, value in saved["env_snapshot"].items():
            setattr(self.env, name, value)
        self._state = saved["state"]
        self._warmed_up = True
        self.step, self.episode = state["step"], state["episode"]
        self.best_score, self.best_step = state["best_score"], state["best_step"]
        self._best_weights = self.agent.primary_model.get_weights() if state["best_step"] == state["step"] else None
        best_dir = path.parent / "best"
        if self._best_weights is None and (best_dir / "primary.keras").exists():
            self._best_weights = tf.keras.models.load_model(best_dir / "primary.keras").get_weights()
        self._episode_reward, self._episode_steps = state["episode_reward"], state["episode_steps"]
        self._elapsed = state["elapsed_seconds"]
        self.agent.exploration_policy.load_state_dict(state["policy"])
        if self.early_stopping and state["early_stopping"]:
            for key, value in state["early_stopping"].items():
                setattr(self.early_stopping, key, value)
        self.eval_records, self.episode_records = state["eval_records"], state["episode_records"]
        self.stop_reason = None if continue_training else state["stop_reason"]
        return self


def load_for_evaluation(agent, path) -> None:
    """Carga los pesos de un checkpoint (p. ej. ``best``) para evaluar el agente."""
    path = Path(path)
    agent.primary_model = tf.keras.models.load_model(path / "primary.keras")
    agent.update_target()
    agent.set_trainable(False)
