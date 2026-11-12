"""Comprobaciones de la pauta TWAP y su recompensa."""
from types import SimpleNamespace

import pytest

from src.agents.twap import twap_action
from src.rewards import RewardContext, TwapRewardStrategy


def context(step, executed, horizon=10, volume=3):
    return RewardContext(0, 0, 100, step - 1, horizon, executed, volume)


def test_reference_completes_and_stays_close_to_target():
    env = SimpleNamespace(vol_care=3, episode_bins=10, state_pos=0, exec_vol=0)
    actions = []
    for step in range(10):
        env.state_pos = step
        action = twap_action(env)
        actions.append(action)
        env.exec_vol += action
        assert 0 <= (step + 1) * 3 / 10 - env.exec_vol < 1
    assert actions == [0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    assert env.exec_vol == 3


def test_reward_is_based_on_post_action_progress():
    reward = TwapRewardStrategy()
    assert reward.compute(context(4, 1)) == pytest.approx(-abs(1 / 3 - 4 / 10))
    assert TwapRewardStrategy(power=2).compute(context(4, 1)) == pytest.approx(-(1 / 3 - 4 / 10) ** 2)


def test_terminal_penalizes_early_completion_with_discount():
    reward = TwapRewardStrategy(gamma=0.5)
    c = context(2, 1, horizon=4, volume=1)
    assert reward.compute_terminal(c) == pytest.approx(-0.5 * 0.25)


def test_terminal_penalizes_incomplete_order():
    assert TwapRewardStrategy().compute_terminal(context(10, 1)) == pytest.approx(-2 / 3)


@pytest.mark.parametrize('kwargs', [{'gamma': -1}, {'gamma': 1.1}, {'power': 3}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        TwapRewardStrategy(**kwargs)


@pytest.mark.parametrize(('horizon', 'volume'), [(400, 30), (500, 50), (600, 75)])
def test_reference_handles_workshop_order_sizes(horizon, volume):
    env = SimpleNamespace(vol_care=volume, episode_bins=horizon, state_pos=0, exec_vol=0)
    for step in range(horizon):
        env.state_pos = step
        env.exec_vol += twap_action(env)
        assert env.exec_vol <= volume
        assert abs(env.exec_vol - volume * (step + 1) / horizon) < 1
    assert env.exec_vol == volume


def test_ddqn_does_not_force_purchase_on_first_step():
    pytest.importorskip('tensorflow')
    from src.agents.dqn import DDQNAgent

    agent = object.__new__(DDQNAgent)
    agent.env = SimpleNamespace(state_pos=0)
    agent._trainable = False
    agent._act_no_explore = lambda state: 0
    assert agent.act(None) == 0
