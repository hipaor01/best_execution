"""Tests de la exploración adaptativa, del entrenador y de los checkpoints."""
import numpy as np
import pandas as pd
import pytest

from src.agents.exploration import LinearDecayEpsilon, PerformanceEpsilon


def feed(policy, scores):
    values = []
    for score in scores:
        policy.on_evaluation({"score": score})
        values.append(policy.epsilon)
    return values


# ------------------------------------------------------------- PerformanceEpsilon
def test_epsilon_ignores_steps_and_episodes():
    policy = PerformanceEpsilon()
    for i in range(1, 10_000):
        policy.on_step(i)
        policy.on_episode_end(i)
    assert policy.epsilon == 1.0


def test_epsilon_waits_for_a_full_window():
    policy = PerformanceEpsilon(window=3)
    assert feed(policy, [0.9, 0.9]) == [1.0, 1.0]
    policy.on_evaluation({"score": 0.9})
    assert policy.epsilon < 1.0


def test_epsilon_decreases_with_improvement_and_respects_bounds():
    policy = PerformanceEpsilon(epsilon_min=0.05, window=1, max_decrease=1.0)
    values = feed(policy, [0.2, 0.5, 0.8, 1.0])
    assert values == sorted(values, reverse=True)
    assert values[-1] == pytest.approx(0.05)
    assert all(0.05 <= v <= 1.0 for v in values)


def test_single_bad_evaluation_does_not_raise_epsilon():
    policy = PerformanceEpsilon(window=1, max_decrease=1.0)
    feed(policy, [0.8])
    before = policy.epsilon
    feed(policy, [0.0])
    assert policy.epsilon == before


def test_noise_within_tolerance_is_not_improvement():
    policy = PerformanceEpsilon(window=1, min_delta=0.05, patience=100, max_decrease=1.0)
    feed(policy, [0.5])
    before = policy.epsilon
    feed(policy, [0.53, 0.51, 0.54])
    assert policy.epsilon == before


def test_max_decrease_limits_each_update():
    policy = PerformanceEpsilon(window=1, max_decrease=0.1)
    assert feed(policy, [1.0]) == [pytest.approx(0.9)]


def test_stagnation_lowers_the_cap():
    policy = PerformanceEpsilon(window=1, patience=2, stall_factor=0.5, max_decrease=1.0, min_delta=0.0)
    feed(policy, [0.0, 0.0])
    assert policy.epsilon == pytest.approx(0.5)
    feed(policy, [0.0, 0.0])
    assert policy.epsilon == pytest.approx(0.25)
    feed(policy, [0.0] * 20)
    assert policy.epsilon == pytest.approx(0.05)


def test_state_round_trip():
    policy = PerformanceEpsilon(window=2, patience=2)
    feed(policy, [0.1, 0.3, 0.3, 0.3])
    clone = PerformanceEpsilon(window=2, patience=2)
    clone.load_state_dict(policy.state_dict())
    assert feed(clone, [0.6, 0.2]) == feed(policy, [0.6, 0.2])


@pytest.mark.parametrize("kwargs", [{"epsilon_min": 0.5, "epsilon_max": 0.4}, {"window": 0},
                                    {"patience": 0}, {"stall_factor": 0}, {"max_decrease": 0}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        PerformanceEpsilon(**kwargs)


def test_score_outside_range_is_rejected():
    with pytest.raises(ValueError):
        PerformanceEpsilon().on_evaluation({"score": 1.5})


def test_linear_decay_reproduces_previous_schedule():
    policy = LinearDecayEpsilon(1.0, 0.05, 100)
    policy.on_step(50)
    assert policy.epsilon == pytest.approx(0.525)
    policy.on_step(500)
    assert policy.epsilon == pytest.approx(0.05)


# --------------------------------------------------------------- agente y trainer
tf = pytest.importorskip("tensorflow")


def _data(days=3, rows=60):
    frames = {}
    for d in range(days):
        frames[f"d{d}"] = pd.DataFrame({"ask1": 101.0 + np.arange(rows) * 0.01,
                                        "bid1": 100.0 + np.arange(rows) * 0.01,
                                        "cumvol": np.arange(rows, dtype=float) * 10})
    return frames


def _setup(seed=3):
    from src.agents.dqn import DDQNAgent
    from src.environment.best_execution import BestExecutionEnv
    from src.environment.config import EnvironmentConfig
    from src.rewards import TwapRewardStrategy
    tf.keras.utils.set_random_seed(seed)
    config = EnvironmentConfig(look_back=2, min_episode_bins=20, max_episode_bins=30,
                               min_care_fraction=0.1, max_care_fraction=0.2, seed=seed)
    env = BestExecutionEnv(_data(), config, TwapRewardStrategy(power=2))
    agent = DDQNAgent(env, batch_size=8, buffer_size=200, hidden_neurons=4,
                      exploration_policy=PerformanceEpsilon(window=1))
    return env, agent


class _FixedEvaluator:
    def __init__(self, scores):
        self.scores = iter(scores)

    def evaluate(self, q_model):
        return {"score": next(self.scores), "completion": 0.0, "median_mae": 0.0}


def test_agent_epsilon_comes_from_policy_and_is_read_only():
    _, agent = _setup()
    assert agent.epsilon == 1.0
    with pytest.raises(AttributeError):
        agent.epsilon = 0.3


def test_agent_without_policy_keeps_editable_epsilon():
    from src.agents.dqn import DDQNAgent
    env, _ = _setup()
    agent = DDQNAgent(env, hidden_neurons=4)
    agent.epsilon = 0.3
    assert agent.epsilon == 0.3


def test_trainer_stops_on_target_after_min_steps():
    from src.training.trainer import EarlyStopping, Trainer, TrainerConfig
    env, agent = _setup()
    trainer = Trainer(agent, env, _FixedEvaluator([1.0] * 50),
                      TrainerConfig(max_steps=1_000, warmup_steps=16, eval_every=50, checkpoint_every=None),
                      EarlyStopping(target_patience=2, min_steps=200))
    trainer.train()
    assert trainer.stop_reason == "target"
    assert trainer.step == 200
    assert trainer.eval_history.epsilon.iloc[-1] < 1.0


def test_trainer_stops_on_plateau():
    from src.training.trainer import EarlyStopping, Trainer, TrainerConfig
    env, agent = _setup()
    trainer = Trainer(agent, env, _FixedEvaluator([0.3] * 50),
                      TrainerConfig(max_steps=2_000, warmup_steps=16, eval_every=50, checkpoint_every=None),
                      EarlyStopping(plateau_patience=3, smoothing=1, min_steps=0))
    trainer.train()
    assert trainer.stop_reason == "plateau"
    assert trainer.step == 200


def test_checkpoint_resume_continues_identically(tmp_path):
    from src.training.trainer import Trainer, TrainerConfig
    scores = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7]

    env, agent = _setup()
    full = Trainer(agent, env, _FixedEvaluator(scores),
                   TrainerConfig(max_steps=300, warmup_steps=16, eval_every=50, checkpoint_every=None)).train()

    env, agent = _setup()
    Trainer(agent, env, _FixedEvaluator(scores[:3]),
            TrainerConfig(max_steps=150, warmup_steps=16, eval_every=50, checkpoint_every=150,
                          checkpoint_dir=tmp_path)).train()
    env, agent = _setup(seed=99)  # proceso nuevo: otra inicialización
    resumed = Trainer(agent, env, _FixedEvaluator(scores[3:]),
                      TrainerConfig(max_steps=300, warmup_steps=16, eval_every=50, checkpoint_every=None))
    resumed.resume(tmp_path / "last").train()

    assert resumed.step == full.step == 300
    assert resumed.agent.epsilon == pytest.approx(full.agent.epsilon)
    pd.testing.assert_frame_equal(resumed.episode_history, full.episode_history)
    for a, b in zip(resumed.agent.primary_model.get_weights(), full.agent.primary_model.get_weights()):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
