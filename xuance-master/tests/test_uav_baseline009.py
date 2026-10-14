"""Behavior gates for the hold-only, stronger-evader pursuit baseline."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "experiments/baseline_exp009.yaml").read_text())


def environment(**overrides):
    config = deepcopy(CONFIG)
    config.update(overrides)
    config["buildings_dir"] = str(ROOT / "xuance/environment/multi_agent_env/buildings")
    env = UAVPursuitCoverage3DEnv(Namespace(**config))
    env.reset()
    return env


def stationary_scene(env, distance=4.):
    env.buildings = np.empty((0, 5), np.float32)
    env.target_position = np.array([500., 500., 200.], np.float32)
    env.target_velocity.fill(0.)
    env.target_acceleration.fill(0.)
    env.uav_positions = np.array([[500.+distance, 500., 200.], [600., 500., 200.],
                                  [650., 500., 200.], [700., 500., 200.]], np.float32)
    env.uav_velocities.fill(0.)
    env.uav_accelerations.fill(0.)
    env._evader_control = lambda: np.zeros(3, np.float32)
    return {agent: np.zeros(3, np.float32) for agent in env.agents}


def test_no_direct_capture_and_same_agent_holds_for_two_real_seconds():
    env = environment()
    try:
        actions = stationary_scene(env)
        for index in range(4):
            _, _, done, truncated, info = env.step(actions)
            assert not truncated
            assert info["is_success"] == (index == 3)
            assert all(done.values()) == (index == 3)
        assert info["capture_type"] == "sustained"
        assert info["capture_time"] == pytest.approx(2.)
        assert info["infos"]["episode_elapsed_seconds"] == pytest.approx(2.)
        assert env.episode_sub_rewards[env.agents[0]]["r_progress"] == 0.
        assert env.episode_sub_rewards[env.agents[0]]["r_terminal"] == 30.
    finally:
        env.close()


def test_capture_does_not_require_collision_free_teammates():
    env = environment(terminate_on_boundary_collision=False, terminate_on_building_collision=False)
    try:
        actions = stationary_scene(env)
        env.uav_positions[1] = [env.uav_radius, 600., 200.]
        env.uav_velocities[1] = [-1., 0., 0.]
        for _ in range(4):
            _, _, _, _, info = env.step(actions)
        assert info["is_success"] and info["capture_type"] == "sustained"
        assert info["infos"]["boundary_contact_events_episode"] > 0
        assert env.episode_sub_rewards[env.agents[0]]["r_terminal"] == 30.
    finally:
        env.close()


def test_hold_times_cannot_be_shared_across_agents():
    env = environment()
    try:
        stationary_scene(env)
        env.capture_hold_times[:] = [1.5, 1.5, 0., 0.]
        old = np.array([[100., 0., 0.], [4., 0., 0.], [150., 0., 0.], [200., 0., 0.]])
        assert env._capture_event(old, old, .1) is None
        assert env.capture_hold_times[0] == 0.
        assert env.capture_hold_times[1] == pytest.approx(1.6)
    finally:
        env.close()


@pytest.mark.parametrize("level,speed", list(enumerate([3., 5., 7., 8., 12.])))
def test_all_curriculum_levels_apply_stronger_dynamics_and_keep_observations(level, speed):
    env = environment()
    try:
        env.set_curriculum_level(level)
        obs, _ = env.reset()
        assert env.target_max_speed == speed and env.target_accel == 4. and env.evader_sense == 100.
        assert len(env.buildings) == 20 * (level + 1)
        assert env.catch_radius == [50., 35., 25., 20., 15.][level]
        assert all(value.shape == (79,) and np.isfinite(value).all() for value in obs.values())
        assert all(env.observation_space[agent].contains(value) for agent, value in obs.items())
        assert env.decision_dt == .5 and env.physics_dt == .1 and env.max_episode_steps == 800
        assert not env.direct_intercept_enabled and not env.use_obstacle_gat
        assert env.discount_gamma ** 2 == pytest.approx(.99)
        assert not env.uav_collision_enabled and env.baseline_teammate_weight == 0
    finally:
        env.close()


@pytest.mark.parametrize("closing", [[1., 2., 3., 4.], [2., 2., 2., -4.], [-1., -2., -3., -4.]])
def test_closing_speed_mean_plus_min_and_team_sharing(closing):
    env = environment()
    try:
        stationary_scene(env, 100.)
        env.target_velocity = np.array([2., 0., 0.], np.float32)
        env.uav_velocities[:, 0] = 2. - np.asarray(closing)
        pursuit, formation, diagnostics = env._pursuit_reward_terms()
        expected = np.mean(closing) + min(closing)
        assert pursuit == pytest.approx(expected)
        assert diagnostics["closing_speeds"] == pytest.approx(closing)
        distances = np.linalg.norm(env.uav_positions - env.target_position, axis=1)
        env._baseline_rewards(distances, [1000.] * 4, [0.] * 4, False, .5)
        for components in env.episode_sub_rewards.values():
            assert components["r_purs"] == pytest.approx(.01 * expected * .5)
            assert components["r_form"] == pytest.approx(.01 * formation * .5)
    finally:
        env.close()


def test_nearest_teammate_angle_matches_formula_and_is_translation_invariant():
    env = environment()
    try:
        stationary_scene(env)
        env.uav_positions = np.array([[450., 500., 200.], [450., 520., 200.],
                                      [520., 550., 200.], [550., 450., 200.]], np.float32)
        _, angle, diagnostics = env._pursuit_reward_terms()
        expected = []
        for i in range(4):
            gaps = np.linalg.norm(env.uav_positions - env.uav_positions[i], axis=1)
            gaps[i] = np.inf
            j = np.argmin(gaps)
            mate = env.uav_positions[j] - env.uav_positions[i]
            target = env.target_position - env.uav_positions[i]
            expected.append(np.arccos(np.clip(np.dot(mate, target) / np.linalg.norm(mate) / np.linalg.norm(target), -1, 1)))
            assert diagnostics["nearest_teammates"][i] == j
        assert angle == pytest.approx(np.mean(expected))
        env.uav_positions += [20., 10., -10.]
        env.target_position += [20., 10., -10.]
        assert env._pursuit_reward_terms()[1] == pytest.approx(angle)
    finally:
        env.close()


def test_coincident_positions_and_target_remain_finite():
    env = environment()
    try:
        stationary_scene(env, 0.)
        env.uav_positions[:] = env.target_position
        pursuit, formation, diagnostics = env._pursuit_reward_terms()
        assert pursuit == formation == 0.
        assert np.isfinite(diagnostics["closing_speeds"]).all()
        assert np.isfinite(diagnostics["angles_radians"]).all()
    finally:
        env.close()


def test_new_dense_rewards_scale_with_elapsed_time_and_reset():
    env = environment()
    try:
        stationary_scene(env, 100.)
        env.uav_velocities[:, 0] = -3.
        env._baseline_rewards(np.full(4, 100.), [1000.] * 4, [0.] * 4, False, .25)
        quarter = env.episode_sub_rewards[env.agents[0]]["r_purs"]
        env._baseline_rewards(np.full(4, 100.), [1000.] * 4, [0.] * 4, False, .5)
        assert env.episode_sub_rewards[env.agents[0]]["r_purs"] == pytest.approx(3 * quarter)
        env.reset()
        assert env.episode_elapsed_seconds == env.episode_pursuit_integral == env.episode_formation_integral == 0.
        assert env.pursuit_diagnostics == {}
    finally:
        env.close()


def test_evader_detects_pursuer_at_90m_and_accelerates_away():
    env = environment(curriculum_start_level=4)
    try:
        stationary_scene(env, 90.)
        env.uav_positions[:] = [[590., 500., 200.], [700., 500., 200.],
                                [750., 500., 200.], [800., 500., 200.]]
        # Use the real APF rather than the stationary-scene stub.
        command = UAVPursuitCoverage3DEnv._evader_control(
            env, env.target_position, env.target_velocity, env.uav_positions, deterministic=True)
        assert command[0] < 0
        assert np.linalg.norm(command) == pytest.approx(4.)
    finally:
        env.close()


def test_evader_can_reach_12mps_when_the_threat_stays_in_sensing_range():
    env = environment(curriculum_start_level=4)
    try:
        stationary_scene(env, 90.)
        for _ in range(80):
            env.uav_positions[:] = env.target_position + np.array([90., 0., 0.])
            command = UAVPursuitCoverage3DEnv._evader_control(env, deterministic=True)
            env._advance_evader_substep(command, .1)
            assert np.linalg.norm(env.target_acceleration) <= 4.00001
            assert np.linalg.norm(env.target_velocity) <= 12.00001
            assert not env._is_in_building(env.target_position, env.target_radius)
        assert env.target_speed == pytest.approx(12., abs=.05)
    finally:
        env.close()
