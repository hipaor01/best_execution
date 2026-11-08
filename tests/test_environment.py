"""Tests del contrato público de ``BestExecutionEnv``."""

from typing import Mapping, Optional

import numpy as np
import pandas as pd

from src.environment.best_execution import BestExecutionEnv
from src.environment.config import EnvironmentConfig
from src.rewards import VwapRewardStrategy


def _make_market_data(
    num_rows: int = 7,
    price_offset: float = 0.0,
) -> pd.DataFrame:
    """Construye datos de mercado sintéticos con volumen creciente."""
    return pd.DataFrame(
        {
            "ask1": np.arange(101.0, 101.0 + num_rows) + price_offset,
            "bid1": np.arange(100.0, 100.0 + num_rows) + price_offset,
            "cumvol": np.arange(num_rows, dtype=float),
        }
    )


def _make_environment(
    config: Optional[EnvironmentConfig] = None,
    data: Optional[Mapping[str, pd.DataFrame]] = None,
) -> BestExecutionEnv:
    """Construye un entorno pequeño y determinista para los tests."""
    if config is None:
        config = EnvironmentConfig(
            look_back=2,
            min_episode_bins=5,
            max_episode_bins=5,
            min_care_fraction=0.4,
            max_care_fraction=0.4,
            seed=7,
        )
    if data is None:
        data = {"test_day": _make_market_data()}

    return BestExecutionEnv(
        data=data,
        config=config,
        reward_strategy=VwapRewardStrategy(),
    )


def _make_stochastic_environment(seed: int) -> BestExecutionEnv:
    """Construye un entorno con varios episodios posibles."""
    config = EnvironmentConfig(
        look_back=2,
        min_episode_bins=5,
        max_episode_bins=8,
        min_care_fraction=0.2,
        max_care_fraction=0.4,
        seed=seed,
    )
    data = {
        "first_day": _make_market_data(num_rows=30),
        "second_day": _make_market_data(num_rows=30, price_offset=100.0),
    }

    return _make_environment(config=config, data=data)


def test_reset_observation_shape_matches_environment_features() -> None:
    env = _make_environment()

    first_observation = env.reset()
    second_observation = env.reset()

    expected_shape = (env.n_features,)
    assert first_observation.shape == expected_shape
    assert second_observation.shape == expected_shape


def test_observation_shape_remains_stable_during_episode() -> None:
    env = _make_environment()
    observation = env.reset()
    expected_shape = observation.shape

    for _ in range(env.episode_bins - 1):
        observation, _, done, _ = env.step(action=0)

        assert not done
        assert observation.shape == expected_shape


def test_episode_finishes_when_time_is_exhausted() -> None:
    env = _make_environment()
    env.reset()

    for _ in range(env.episode_bins - 1):
        _, _, done, _ = env.step(action=0)
        assert not done

    observation, _, done, _ = env.step(action=0)

    assert done
    assert observation is None
    assert env.state_pos == env.episode_bins
    assert env.exec_vol < env.vol_care


def test_episode_finishes_when_care_order_is_completed() -> None:
    env = _make_environment()
    env.reset()

    for _ in range(env.vol_care - 1):
        _, _, done, _ = env.step(action=1)
        assert not done

    observation, _, done, _ = env.step(action=1)

    assert done
    assert observation is None
    assert env.exec_vol == env.vol_care
    assert env.state_pos < env.episode_bins


def test_care_volume_stays_within_configured_limits() -> None:
    config = EnvironmentConfig(
        look_back=2,
        min_episode_bins=5,
        max_episode_bins=5,
        min_care_fraction=0.2,
        max_care_fraction=0.8,
        seed=11,
    )
    env = _make_environment(config=config)

    for _ in range(50):
        env.reset()

        min_care_volume = int(env.episode_bins * config.min_care_fraction)
        max_care_volume = int(env.episode_bins * config.max_care_fraction)

        assert min_care_volume <= env.vol_care <= max_care_volume


def test_episode_generation_is_reproducible_with_same_seed() -> None:
    first_env = _make_stochastic_environment(seed=23)
    second_env = _make_stochastic_environment(seed=23)

    for _ in range(5):
        first_observation = first_env.reset()
        second_observation = second_env.reset()

        np.testing.assert_array_equal(first_observation, second_observation)
        assert first_env.episode_bins == second_env.episode_bins
        assert first_env.vol_care == second_env.vol_care
        pd.testing.assert_frame_equal(first_env.episode, second_env.episode)


def test_action_sampling_is_reproducible_with_same_seed() -> None:
    first_env = _make_stochastic_environment(seed=41)
    second_env = _make_stochastic_environment(seed=41)
    first_env.reset()
    second_env.reset()

    first_actions = [first_env.action_sample() for _ in range(50)]
    second_actions = [second_env.action_sample() for _ in range(50)]

    assert first_actions == second_actions


def test_complete_episode_keeps_state_and_histories_consistent() -> None:
    env = _make_stochastic_environment(seed=53)
    observation = env.reset()
    expected_observation_shape = observation.shape
    steps = 0
    done = False

    while not done:
        action = env.action_sample()
        observation, reward, done, info = env.step(action)
        steps += 1

        assert np.isfinite(reward)
        assert isinstance(info, dict)
        if not done:
            assert observation.shape == expected_observation_shape

    assert observation is None
    assert steps == env.state_pos
    assert steps <= env.episode_bins
    assert env.exec_vol <= env.vol_care
    assert env.state_pos == env.episode_bins or env.exec_vol == env.vol_care

    histories = {
        "actions": env.actions_hist,
        "algorithm VWAP": env.algo_vwap_hist,
        "market VWAP": env.market_vwap_hist,
        "rewards": env.reward_hist,
        "prices": env.price_hist,
        "volumes": env.vol_hist,
    }
    for history_name, history in histories.items():
        assert len(history) == steps, history_name
