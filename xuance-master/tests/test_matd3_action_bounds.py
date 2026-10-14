"""Replay must receive the bounded command that is sent to the environment."""
from argparse import Namespace
import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from xuance.torch.agents.multi_agent_rl.iddpg_agents import IDDPG_Agents
from xuance.torch.agents.multi_agent_rl.matd3_agents import MATD3_Agents


@pytest.mark.parametrize('test_mode', [False, True])
def test_matd3_bounds_actions_before_returning_them_to_replay(monkeypatch, test_mode):
    original = {'hidden_state': {'keep': 1}, 'actions': [
        {'uav_0': np.array([1.3, -1.2, .4], np.float32)},
        {'uav_0': np.array([-.5, .1, -2.], np.float32)}]}
    monkeypatch.setattr(IDDPG_Agents, 'action', lambda self, *args, **kwargs: original)
    agent = object.__new__(MATD3_Agents)
    agent.config = Namespace(matd3_clip_actions=True)
    agent.action_space = {'uav_0': Box(-1., 1., (3,), np.float32)}
    result = agent.action([{}, {}], test_mode=test_mode)
    np.testing.assert_array_equal(result['actions'][0]['uav_0'], np.array([1., -1., .4], np.float32))
    np.testing.assert_array_equal(result['actions'][1]['uav_0'], np.array([-.5, .1, -1.], np.float32))
    assert result['hidden_state'] == {'keep': 1}


def test_historical_action_behavior_remains_opt_in(monkeypatch):
    original = {'hidden_state': None, 'actions': [{'uav_0': np.array([1.3, 0., 0.])}]}
    monkeypatch.setattr(IDDPG_Agents, 'action', lambda self, *args, **kwargs: original)
    agent = object.__new__(MATD3_Agents)
    agent.config = Namespace()
    result = agent.action([{}])
    assert result is original and result['actions'][0]['uav_0'][0] == 1.3


def test_real_inherited_exploration_is_clipped(monkeypatch):
    agent = object.__new__(MATD3_Agents)
    agent.config = Namespace(matd3_clip_actions=True)
    agent.use_parameter_sharing = True
    agent.n_agents = 1
    agent.agent_keys = ['uav_0']
    agent.action_space = {'uav_0': Box(-1., 1., (3,), np.float32)}
    agent.e_greedy = None
    agent.noise_scale = .5
    agent._build_inputs = lambda obs: ({}, None, None)
    agent.policy = lambda **kw: (None, {'uav_0': torch.tensor([[.95, -.95, .2]])})
    monkeypatch.setattr(np.random, 'normal', lambda *args, **kw: np.array([[[.5, -.5, .1]]]))
    commands = agent.action([{}], test_mode=False)['actions'][0]
    np.testing.assert_allclose(commands['uav_0'], [1., -1., .3], rtol=1e-6)
