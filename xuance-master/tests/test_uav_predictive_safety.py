"""Execution safety: dynamics, swept contact, brake tail and replay semantics."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv
from xuance.environment.multi_agent_env.uav_kinematics_3d import advance_translation
from xuance.environment.multi_agent_env.uav_predictive_safety_3d import (
    PredictiveSafetyFilter3D, swept_pair_distance)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / 'experiments/baseline_exp014.yaml').read_text())


def scene(**options):
    config = deepcopy(CONFIG)
    config.update(safety_filter_enabled=True, safety_margin=2., safety_horizon_seconds=8.)
    config.update(options)
    config['buildings_dir'] = str(ROOT / 'xuance/environment/multi_agent_env/buildings')
    env = UAVPursuitCoverage3DEnv(Namespace(**config))
    env.reset()
    env.buildings = np.empty((0, 5), np.float32)
    env.uav_positions = np.array([[400., 350., 200.], [400., 450., 200.],
                                  [400., 550., 200.], [400., 650., 200.]], np.float32)
    env.uav_velocities.fill(0.)
    env.uav_accelerations.fill(0.)
    env.target_position = np.array([800., 500., 200.], np.float32)
    env.target_velocity.fill(0.)
    env.target_acceleration.fill(0.)
    env._evader_control = lambda *args, **kwargs: np.zeros(3, np.float32)
    return env


def test_safe_proposal_passes_and_predictor_matches_actual_lag_jerk_dynamics():
    env = scene()
    env.uav_velocities[:] = [4., .2, -.1]
    env.uav_accelerations[:] = [.4, -.1, .2]
    proposal = np.tile([1., -.2, .1], (4, 1))
    result, certificate = env.safety_filter.filter(proposal)
    np.testing.assert_allclose(result, proposal)
    assert certificate['certified'] and not any(certificate['intervened'])
    paths, _ = env.safety_filter._rollout(proposal[:, None])
    for agent in range(4):
        p, v, a = (value[agent].copy() for value in (
            env.uav_positions, env.uav_velocities, env.uav_accelerations))
        for step in range(env.safety_filter.steps):
            command = proposal[agent] if step == 0 else env.safety_filter.brake(
                v.astype(np.float64), a.astype(np.float64))
            p, v, a = advance_translation(p, v, a, command, .1, .5, 3., 6., 11.)
            np.testing.assert_array_equal(paths[agent, 0, step + 1], p)


@pytest.mark.parametrize('hazard', ['building', 'boundary', 'roof'])
def test_persistent_unsafe_commands_are_filtered_before_contact(hazard):
    env = scene()
    if hazard == 'building':
        env.buildings = np.array([[500., 550., 300., 700., 350.]], np.float32)
        env.uav_positions[:, 0] = 450.
        velocity, command = [8., 0., 0.], [1., 0., 0.]
    elif hazard == 'boundary':
        env.uav_positions[:, 0] = 950.
        velocity, command = [8., 0., 0.], [1., 0., 0.]
    else:
        env.buildings = np.array([[350., 450., 300., 700., 200.]], np.float32)
        env.uav_positions[:, 2] = 235.
        velocity, command = [0., 0., -6.], [0., 0., -1.]
    env.uav_velocities[:] = velocity
    actions = {name: np.array(command, np.float32) for name in env.agents}
    intervention = False
    for _ in range(45):
        _, _, terminated, truncated, info = env.step(actions)
        assert not info['is_collision']
        assert not any(terminated.values()) and not truncated
        assert info['safety_filter']['certified']
        intervention |= any(info['safety_filter']['intervened_agents'])
    assert intervention


def test_building_segments_are_checked_between_endpoints():
    env = scene()
    env.buildings = np.array([[499., 501., 300., 700., 350.]], np.float32)
    path = np.repeat(env.uav_positions[:, None, None], 2, axis=2)
    path[:, 0, 0, 0], path[:, 0, 1, 0] = 490., 510.
    safe = env.safety_filter._individual_safe(path, np.zeros((4, 1)))
    assert not np.any(safe)


def test_terminal_ball_rejects_a_short_horizon_that_only_looks_clear():
    env = scene(safety_horizon_seconds=.1)
    env.uav_positions[:, 0] = 997.1
    proposal = np.tile([3., 0., 0.], (4, 1))
    paths, tail = env.safety_filter._rollout(proposal[:, None])
    assert np.all(paths[..., -1, 0] < 997.5)  # next position alone is clear
    assert not np.any(env.safety_filter._individual_safe(paths, tail))


def test_terminal_ball_encloses_continued_braking_after_prediction_horizon():
    env = scene(safety_horizon_seconds=1.)
    env.uav_velocities[:] = [.3, .1, .05]
    env.uav_accelerations[:] = [.02, .01, 0.]
    proposal = env.safety_filter.brake(env.uav_velocities.astype(float),
                                     env.uav_accelerations.astype(float))
    path, radius = env.safety_filter._rollout(proposal[:, None])
    assert np.all(np.isfinite(radius))
    p, v, a = (value[0].copy() for value in (
        env.uav_positions, env.uav_velocities, env.uav_accelerations))
    for index in range(300):
        command = env.safety_filter.brake(v.astype(float), a.astype(float))
        p, v, a = advance_translation(p, v, a, command, .1, .5, 3., 6., 11.)
        if index >= env.safety_filter.steps:
            assert np.linalg.norm(p - path[0, 0, -1]) <= radius[0, 0]


def test_head_on_pair_is_filtered_even_with_legacy_pair_collision_disabled():
    env = scene(uav_collision_enabled=False)
    env.uav_positions[:2] = [[450., 400., 200.], [500., 400., 200.]]
    env.uav_velocities[:2] = [[7., 0., 0.], [-7., 0., 0.]]
    actions = {name: np.zeros(3, np.float32) for name in env.agents}
    actions['uav_0'][0], actions['uav_1'][0] = 1., -1.
    intervention = False
    for _ in range(35):
        _, _, _, _, info = env.step(actions)
        assert info['safety_filter']['certified']
        assert not info['is_collision']
        assert info['min_uav_gap'] > env.uav_collision_distance
        intervention |= any(info['safety_filter']['intervened_agents'])
    assert intervention


def test_swept_pair_check_catches_crossing_with_clear_endpoints():
    first = np.array([[-2., 0., 0.], [2., 0., 0.]])
    second = -first
    assert swept_pair_distance(first, second) == pytest.approx(0.)


def test_infeasible_state_is_disclosed_and_contact_ends_even_video_mode():
    env = scene(terminate_on_building_collision=False, terminate_on_boundary_collision=False,
                terminate_on_collision=False)
    env.buildings = np.array([[500., 550., 300., 700., 350.]], np.float32)
    env.uav_positions[0] = [499., 350., 200.]
    env.uav_velocities[0] = [11., 0., 0.]
    _, rewards, done, truncated, info = env.step({name: np.zeros(3) for name in env.agents})
    assert not info['safety_filter']['certified']
    assert info['is_collision'] and not info['is_success']
    assert info['damaged_agents'] == ['uav_0']
    assert all(done.values()) and not truncated
    assert info['infos']['episode_elapsed_seconds'] == pytest.approx(.1)
    assert len(info['safety_filter']['executed_commands']) == 1
    assert all(info['episode_sub_rewards'][name]['r_failure'] == -30. for name in env.agents)
    with pytest.raises(RuntimeError, match='reset'):
        env.step({name: np.zeros(3) for name in env.agents})
    env.reset()
    assert not env._safety_episode_done


def test_filter_never_mutates_policy_actions_and_keeps_observation_compatibility():
    env = scene()
    env.uav_positions[:, 0] = 960.
    env.uav_velocities[:] = [8., 0., 0.]
    actions = {name: np.array([1., 0., 0.], np.float32) for name in env.agents}
    original = {name: action.copy() for name, action in actions.items()}
    obs, _, _, _, info = env.step(actions)
    for name in env.agents:
        np.testing.assert_array_equal(actions[name], original[name])
        assert obs[name].shape == (79,)
    assert info['safety_filter']['action_semantics'] == 'network_proposal'
    assert len(info['safety_filter']['executed_commands']) == 5
    assert info['infos']['safety_checks_episode'] == 20


@pytest.mark.parametrize('margin,horizon', [(0., 8.), (2., 0.), (np.nan, 8.)])
def test_invalid_safety_configuration_is_rejected(margin, horizon):
    with pytest.raises(ValueError):
        scene(safety_margin=margin, safety_horizon_seconds=horizon)


def test_training_csv_retains_safety_accounting(tmp_path):
    import csv
    from xuance.engine.uav_baseline_logging import BaselineTrainingLog
    from xuance.environment.utils.wrapper import XuanCeMultiAgentEnvWrapper
    wrapped = XuanCeMultiAgentEnvWrapper(scene(episode_length=1))
    wrapped.reset()
    _, _, _, truncated, info = wrapped.step({name: np.zeros(3) for name in wrapped.agents})
    assert truncated
    log = BaselineTrainingLog(tmp_path, lambda *args: None)
    try:
        log.record_episode(4, 0, info)
    finally:
        log.close()
        wrapped.close()
    row = next(csv.DictReader((tmp_path / 'training_episodes.csv').open()))
    assert int(row['safety_checks']) == 20
    assert int(row['safety_interventions']) == info['infos']['safety_interventions_episode']
    assert int(row['safety_uncertified_substeps']) == 0
    assert float(row['safety_mean_correction']) == pytest.approx(
        info['infos']['safety_mean_correction_episode'])
