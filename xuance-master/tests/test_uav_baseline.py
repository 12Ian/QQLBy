"""Behavior tests for sparse capture, fixed-map curriculum and free-space flight."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv
from xuance.engine.uav_curriculum import PerformanceCurriculum


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "xuance/configs/maddpg/uav_pursuit_baseline_3d.yaml").read_text())


def test_tensorboard_launcher_keeps_seed_root_instead_of_selecting_single_metric(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("baseline_tensorboard_launcher", ROOT / "examples/run_tensorboard.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    seed = tmp_path / "seed_1"
    child = seed / "Train-SubRewards-uav_3_r_failure_env-8"
    child.mkdir(parents=True)
    (seed / "events.out.tfevents.main").touch()
    (child / "events.out.tfevents.single_metric").touch()
    assert Path(launcher.get_latest_log_dir(seed)) == seed
    assert Path(launcher.get_latest_log_dir(tmp_path)) == seed


def make_env(**overrides):
    config = deepcopy(CONFIG)
    config.update(overrides)
    config["buildings_dir"] = str(ROOT / "xuance/environment/multi_agent_env/buildings")
    return UAVPursuitCoverage3DEnv(Namespace(**config))


def clear_scene(env, distances):
    env.buildings = np.empty((0, 5), dtype=np.float32)
    env.target_position = np.array([500, 500, 200], dtype=np.float32)
    env.target_velocity.fill(0)
    env.target_acceleration.fill(0)
    env.uav_positions = np.array([[500 + distance, 500, 200] for distance in distances], dtype=np.float32)
    env.uav_velocities.fill(0)
    env.uav_accelerations.fill(0)
    env._evader_control = lambda *args, **kwargs: np.zeros(3, dtype=np.float32)
    return {agent: np.zeros(3) for agent in env.agents}


def test_progress_only_profile_ignores_teammate_contacts_in_training_outcome(tmp_path):
    import csv
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from xuance.engine.uav_baseline_logging import BaselineTrainingLog
    from xuance.environment.utils.wrapper import XuanCeMultiAgentEnvWrapper
    from xuance.torch.agents.base.agents_marl import MARLAgents

    config = yaml.safe_load((ROOT / "experiments/baseline_exp008.yaml").read_text())
    config["buildings_dir"] = str(ROOT / "xuance/environment/multi_agent_env/buildings")
    env = UAVPursuitCoverage3DEnv(Namespace(**config))
    wrapped = XuanCeMultiAgentEnvWrapper(env)
    wrapped.reset()
    actions = clear_scene(env, [4., 4.5, 100., 200.])
    _, rewards, done, truncated, info = wrapped.step(actions)
    assert all(done.values()) and not truncated and info["is_success"]
    assert not info["is_collision"] and not info["is_uav_collision"]
    assert not info["uav_conflict"] and info["episode_uav_conflicts"] == 0
    assert info["infos"]["uav_contact_events_episode"] == 0
    for agent in env.agents:
        parts = info["episode_sub_rewards"][agent]
        assert parts["r_distance"] == parts["r_teammate"] == parts["r_collision"] == parts["r_failure"] == 0
        assert parts["r_progress"] > 0 and parts["r_terminal"] == 30
        assert rewards[agent] == pytest.approx(sum(parts.values()))
    writer = SummaryWriter(str(tmp_path / "tensorboard"))
    logger_agent = Namespace(use_wandb=False, writer=writer, config=Namespace(reward_mode="baseline"))
    log = BaselineTrainingLog(tmp_path, lambda values, step: MARLAgents.log_infos(logger_agent, values, step))
    try:
        log.record_episode(1, 0, info)
        MARLAgents.log_infos(logger_agent, {"Train-SubRewards-uav_3/r_progress": {"env-0": 7.}}, 1)
    finally:
        log.close()
        writer.close()
        env.close()
    with (tmp_path / "training_episodes.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["safe_success"] == "1" and row["collision"] == "0"
    accumulator = EventAccumulator(str(tmp_path / "tensorboard"))
    accumulator.Reload()
    for agent in env.agents:
        for name, value in info["episode_sub_rewards"][agent].items():
            assert accumulator.Scalars(f"Train-Rewards/{agent}/{name}")[-1].value == pytest.approx(value)
            assert accumulator.Scalars(f"Train-Rewards-Per-Step/{agent}/{name}")[-1].value == pytest.approx(value)
        assert accumulator.Scalars(f"Train-Returns/{agent}")[-1].value == pytest.approx(rewards[agent])
    assert accumulator.Scalars("Train-SubRewards-uav_3/r_progress/env-0")[-1].value == 7.
    assert not any(p.is_dir() for p in (tmp_path / "tensorboard").iterdir())


def test_far_side_profile_spawns_safe_and_separated_on_all_fixed_cities():
    config = yaml.safe_load((ROOT / "experiments/baseline_exp008.yaml").read_text())
    config["buildings_dir"] = str(ROOT / "xuance/environment/multi_agent_env/buildings")
    env = UAVPursuitCoverage3DEnv(Namespace(**config))
    box = np.asarray(config["target_spawn_box"])
    try:
        for level in range(5):
            env.set_curriculum_level(level)
            for seed in range(10):
                env.seed(seed)
                _, info = env.reset()
                assert np.all((env.target_position >= box[0]) & (env.target_position <= box[1]))
                assert np.all(env.uav_positions[:, 0] < 200.)
                assert np.all(np.linalg.norm(env.uav_positions - env.target_position, axis=1) >= 450.)
                assert not env._is_in_building(env.target_position, env.target_radius)
                env._validate_pursuer_spawns(env.uav_positions)
                toward = env.target_position - env.uav_positions
                assert np.all(np.sum(toward * env.uav_velocities, axis=1) > 0.)
                assert info["infos"]["initial_target_distance_min"] >= 450.
                _, _, _, _, step_info = env.step({agent: np.zeros(3) for agent in env.agents})
                assert not step_info["is_collision"]
    finally:
        env.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_uav_collision_switch_controls_contact_failure_and_conflict_statistics(enabled):
    env = make_env(uav_collision_enabled=enabled, terminate_on_collision=True,
                   baseline_teammate_weight=0, baseline_distance_weight=0,
                   baseline_progress_weight=1., baseline_failure_penalty=300.)
    env.reset()
    actions = clear_scene(env, [100., 100.5, 200., 300.])
    _, _, done, truncated, info = env.step(actions)
    assert not truncated and not info["is_success"]
    assert all(done.values()) == enabled
    assert info["is_collision"] == info["is_uav_collision"] == info["uav_conflict"] == enabled
    assert (info["episode_uav_conflicts"] > 0) == enabled
    assert (info["infos"]["uav_contact_events_episode"] > 0) == enabled
    for agent in env.agents:
        assert info["episode_sub_rewards"][agent]["r_failure"] == (-300. if enabled else 0.)
    env.close()


def test_braking_safety_penalizes_approach_before_contact():
    env = make_env(baseline_braking_safety=True, baseline_braking_reaction_time=1.5,
                   baseline_obstacle_weight=3., baseline_boundary_weight=3.,
                   obstacle_avoid_range=50., boundary_warning_distance=50.)
    env.reset()
    clear_scene(env, [100, 150, 200, 250])
    env.buildings = np.array([[100, 140, 480, 520, 150]], dtype=np.float32)
    env.uav_positions[0] = [60, 500, 100]
    agent = env.agents[0]

    def reward_part(name, velocity, building_min):
        before = env.episode_sub_rewards[agent][name]
        env.uav_velocities[0] = velocity
        env._baseline_rewards(np.full(4, 100.), building_min, [0.] * 4, False, 1.)
        return env.episode_sub_rewards[agent][name] - before

    stationary = reward_part('r_obstacle', [0, 0, 0], [40.] * 4)
    approach = reward_part('r_obstacle', [10, 0, 0], [40.] * 4)
    retreat = reward_part('r_obstacle', [-10, 0, 0], [40.] * 4)
    effective = 39.5 - 10 * 1.5 - 10 ** 2 / (2 * env.max_accel)
    assert approach == pytest.approx(-3. * (1. - effective / 50.) ** 2)
    assert approach < stationary and retreat == pytest.approx(stationary)
    env.buildings = np.empty((0, 5), dtype=np.float32)
    env.uav_positions[0] = [40, 500, 200]
    stationary = reward_part('r_boundary', [0, 0, 0], [1000.] * 4)
    approach = reward_part('r_boundary', [-10, 0, 0], [1000.] * 4)
    retreat = reward_part('r_boundary', [10, 0, 0], [1000.] * 4)
    assert approach == pytest.approx(-3. * (1. - effective / 50.) ** 2)
    assert approach < stationary and retreat == pytest.approx(stationary)


@pytest.mark.parametrize("deficit_penalty", [False, True])
def test_braking_reward_keeps_original_collision_physics(deficit_penalty):
    env = make_env(baseline_braking_safety=True, baseline_braking_deficit_penalty=deficit_penalty,
                   terminate_on_boundary_collision=True)
    env.reset()
    actions = clear_scene(env, [100, 150, 200, 250])
    env.uav_positions[0] = [1, 500, 200]
    env.uav_velocities[0] = [-10, 0, 0]
    actions[env.agents[0]] = np.array([-1, 0, 0])
    _, _, terminated, _, info = env.step(actions)
    assert all(terminated.values()) and info['is_boundary_collision']


@pytest.mark.parametrize("kind", ["building", "boundary"])
def test_braking_deficit_cost_keeps_growing_inside_braking_distance(kind):
    values = []
    for enabled in [False, True]:
        env = make_env(baseline_braking_safety=True, baseline_braking_deficit_penalty=enabled,
                       baseline_braking_reaction_time=1.5, baseline_reward_scale=.1,
                       baseline_obstacle_weight=3., baseline_boundary_weight=3.,
                       obstacle_avoid_range=50., boundary_warning_distance=50.)
        env.reset()
        clear_scene(env, [100, 150, 200, 250])
        if kind == "building":
            env.buildings = np.array([[100, 140, 480, 520, 150]], dtype=np.float32)
            env.uav_positions[0] = [90, 500, 100]
        else:
            env.uav_positions[0] = [10, 500, 200]
        name = "r_obstacle" if kind == "building" else "r_boundary"
        rewards = []
        for speed in [8., 10.]:
            env.uav_velocities[0] = [speed if kind == "building" else -speed, 0, 0]
            before = env.episode_sub_rewards[env.agents[0]][name]
            env._baseline_rewards(np.full(4, 100.), [10.] * 4, [0.] * 4, False, 1.)
            rewards.append(env.episode_sub_rewards[env.agents[0]][name] - before)
        values.append(rewards)
        env.close()
    old, new = values
    assert old[0] == pytest.approx(old[1])  # Both risks were already saturated.
    assert new[1] < new[0] < old[0]
    assert np.isfinite(new).all()


def test_braking_deficit_penalty_requires_braking_safety():
    with pytest.raises(ValueError, match="requires braking safety"):
        make_env(baseline_braking_safety=False, baseline_braking_deficit_penalty=True)


def test_full_checkpoint_initialization_loads_all_weights_and_validates(tmp_path):
    import torch
    from xuance.engine.run_marl import initialize_baseline_policy

    policy = torch.nn.Module()
    for group in ['actor', 'target_actor', 'critic_A', 'critic_B', 'target_critic_A', 'target_critic_B']:
        setattr(policy, group, torch.nn.ModuleDict({'uav_0': torch.nn.Linear(4, 1)}))
    state = {key: torch.full_like(value, .125) for key, value in policy.state_dict().items()}
    path = tmp_path / 'continuation.pth'
    torch.save({'policy': state}, path)
    agent = Namespace(policy=policy, device='cpu')
    config = Namespace(baseline_init_checkpoint=str(path), baseline_init_actor=None)
    audit = initialize_baseline_policy(agent, config)
    assert audit['kind'] == 'full_policy' and audit['tensors'] == len(state)
    assert all(torch.equal(value, state[key]) for key, value in policy.state_dict().items())
    with pytest.raises(ValueError, match='exclusive'):
        initialize_baseline_policy(agent, Namespace(baseline_init_checkpoint=str(path), baseline_init_actor=str(path)))
    state['critic_A.uav_0.weight'][0, 0] = float('nan')
    torch.save({'policy': state}, path)
    with pytest.raises(ValueError, match='non-finite'):
        initialize_baseline_policy(agent, config)


def test_fixed_maps_quotas_and_geometry():
    manifest = json.loads((ROOT / "xuance/environment/multi_agent_env/buildings/uniform_map_manifest.json").read_text())
    details = manifest["building_details"]
    specs = {1: ((20, 40), (10, 60)), 2: ((40, 60), (60, 150)), 3: ((60, 80), (150, 350))}
    for b in details:
        sides, heights = specs[b["type"]]
        assert b["width"] in sides and b["length"] in sides
        assert heights[0] <= b["top"] <= heights[1]
    maps = []
    for count in (20, 40, 60, 80, 100):
        maps.append(json.loads((ROOT / f"xuance/environment/multi_agent_env/buildings/buildings_uniform_{count}.json").read_text()))
    for count, geometry in zip((20, 40, 60, 80, 100), maps):
        assert len(geometry) == count
        assert geometry == maps[-1][:count]
        assert geometry == [[b[k] for k in ("xmin", "xmax", "ymin", "ymax", "top")] for b in details[:count]]
        assert [sum(b["type"] == kind for b in details[:count]) for kind in specs] == [count // 2, count * 2 // 5, count // 10]
        assert np.min(np.asarray(geometry)[:, 4]) < 50
        for axis in (0, 2):
            centers = [(b[axis] + b[axis + 1]) / 2 for b in geometry]
            assert all(centers.count(x) == count // 10 for x in range(50, 1000, 100))


@pytest.mark.parametrize("level,count,speed,radius", [(0,20,5,30),(1,40,6,25),(2,60,7,20),(3,80,8,18),(4,100,9,15)])
def test_level_applies_only_at_reset_and_maps_stay_fixed(level, count, speed, radius):
    env = make_env()
    env.reset()
    before = env.buildings.copy()
    env.schedule_curriculum_level(level)
    assert np.array_equal(env.buildings, before)
    obs, _ = env.reset()
    assert len(env.buildings) == count
    assert env.target_max_speed == speed and env.catch_radius == radius
    after = env.buildings.copy()
    for _ in range(3):
        obs, _ = env.reset()
        assert np.array_equal(env.buildings, after)
        assert not env._is_in_building(env.target_position, env.target_radius)
        assert all(not env._is_in_building(p, env.uav_radius) for p in env.uav_positions)
    assert all(obs[a].shape == env.observation_space[a].shape for a in env.agents)
    assert env.state().shape == env.state_space.shape
    assert np.isclose(env._curriculum_context()[2], speed / 11)


def test_distance_reward_is_individual_and_coverage_is_not_computed(monkeypatch):
    import xuance.environment.multi_agent_env.uav_pursuit_coverage_3d as module
    def forbidden(*args, **kwargs):
        raise AssertionError("Baseline must not compute coverage forecasts")
    monkeypatch.setattr(module, "forecast_target", forbidden)
    env = make_env()
    env.reset()
    actions = clear_scene(env, [100, 150, 200, 250])
    _, rewards, _, _, _ = env.step(actions)
    assert np.allclose([rewards[a] for a in env.agents], [-.1, -.15, -.2, -.25])
    assert set(env.episode_sub_rewards[env.agents[0]]) == {
        "r_distance", "r_obstacle", "r_boundary", "r_teammate", "r_collision", "r_terminal",
        "r_progress", "r_failure"}


def test_environment_seed_reproduces_spawns_and_control_without_global_rng():
    a, b = make_env(env_seed=123), make_env(env_seed=123)
    for _ in range(3):
        a.reset()
        np.random.uniform(size=100)
        b.reset()
        assert np.array_equal(a.target_position, b.target_position)
        assert np.array_equal(a.uav_positions, b.uav_positions)
        assert np.array_equal(a.target_velocity, b.target_velocity)
    a.seed()
    a.reset()
    c = make_env(env_seed=123)
    c.reset()
    assert np.array_equal(a.target_position, c.target_position)
    d = make_env(env_seed=124)
    d.reset()
    assert not np.array_equal(c.target_position, d.target_position)


def test_potential_shaping_has_equal_discounted_total_for_different_capture_paths():
    env = make_env(baseline_progress_weight=1.0, baseline_progress_scale=10.0)
    env.reset()
    clear_scene(env, [100, 150, 200, 250])
    totals = []
    for path in ([100, 90, 0], [100, 120, 80, 0]):
        total = 0.0
        for step, (previous, current) in enumerate(zip(path, path[1:])):
            env.episode_sub_rewards = {a: {k: 0.0 for k in env.episode_sub_rewards[a]} for a in env.agents}
            env._baseline_rewards(np.full(4, current), [1000] * 4, [0] * 4,
                                  step == len(path) - 2, 1.0,
                                  previous_distances=np.full(4, previous))
            total += env.discount_gamma ** step * env.episode_sub_rewards[env.agents[0]]["r_progress"]
        totals.append(total)
    assert totals == pytest.approx([10.0, 10.0])


def test_collision_failure_is_shared_and_reward_scaling_applies_to_all_terms():
    env = make_env(baseline_failure_penalty=300.0, baseline_reward_scale=0.1,
                   terminate_on_boundary_collision=True)
    env.reset()
    actions = clear_scene(env, [100, 150, 200, 250])
    env.uav_positions = np.array([[998.5,500,200], [300,500,200],
                                  [500,700,200], [500,300,200]], np.float32)
    env.uav_velocities[0] = [10, 0, 0]
    _, rewards, done, truncated, info = env.step(actions)
    assert all(done.values()) and not truncated and not info["is_success"]
    assert info["is_boundary_collision"]
    for agent in env.agents:
        parts = env.episode_sub_rewards[agent]
        assert parts["r_failure"] == -30.0 and parts["r_terminal"] == 0.0
        assert rewards[agent] == pytest.approx(sum(parts.values()))
    assert env.episode_sub_rewards[env.agents[0]]["r_collision"] < -10.0 + 1e-5


def test_curriculum_spawn_changes_on_reset_and_final_task_keeps_original_difficulty():
    schedule = deepcopy(CONFIG["curriculum_schedule"])
    for stage, radius, vertical in zip(schedule, (100,125,150,175,200), (30,40,50,60,70)):
        stage.update(spawn_radius=radius, spawn_vertical_radius=vertical)
    env = make_env(curriculum_schedule=schedule)
    env.reset()
    assert env.spawn_radius == 100 and env.spawn_vertical_radius == 30
    env.schedule_curriculum_level(4)
    assert env.spawn_radius == 100
    env.reset()
    assert env.spawn_radius == 200 and env.spawn_vertical_radius == 70
    assert len(env.buildings) == 100 and env.target_max_speed == 9 and env.catch_radius == 15


def test_clearance_rewards_include_roofs_boundaries_and_teammates():
    env = make_env()
    env.reset()
    env.uav_positions = np.array([[430,430,116], [436,430,116],
                                  [600,600,200], [700,700,200]], np.float32)
    env.buildings = np.array([[400,460,400,460,100]], np.float32)
    distances = np.full(4, 200.)
    building_min = [np.min(env._building_distances_3d(p)) for p in env.uav_positions]
    rewards = env._baseline_rewards(distances, building_min, [-80, 0, 0, 0], False, .5)
    parts = env.episode_sub_rewards[env.agents[0]]
    assert parts["r_distance"] == pytest.approx(-.1)
    assert parts["r_obstacle"] == pytest.approx(-.5 * (1 - 15.5 / 30) ** 2)
    assert parts["r_teammate"] == pytest.approx(-.125)
    assert parts["r_boundary"] == 0 and parts["r_collision"] == -80
    assert rewards[env.agents[0]] == pytest.approx(sum(parts.values()))
    env.uav_positions[2, 2] = 65
    env._baseline_rewards(distances, [1000] * 4, [0] * 4, False, 1)
    assert env.episode_sub_rewards[env.agents[2]]["r_boundary"] == pytest.approx(-.25)


def test_direct_capture_rewards_whole_team_once():
    env = make_env()
    env.reset()
    actions = clear_scene(env, [4, 100, 200, 300])
    _, rewards, done, truncated, info = env.step(actions)
    assert all(done.values()) and not truncated
    assert info["capture_type"] == "direct"
    assert all(env.episode_sub_rewards[a]["r_terminal"] == 300 for a in env.agents)
    assert np.allclose(list(rewards.values()), 300)


def test_sustained_capture_requires_same_agent_for_two_seconds():
    env = make_env()
    env.reset()
    actions = clear_scene(env, [20, 100, 200, 300])
    _, _, done, _, _ = env.step(actions)
    assert not any(done.values())
    env.uav_positions[0, 0] = 550
    env.step(actions)
    assert env.capture_hold_times[0] == 0
    env.uav_positions[0, 0] = 520
    _, _, done, _, _ = env.step(actions)
    assert not any(done.values())
    _, _, done, _, info = env.step(actions)
    assert all(done.values()) and info["capture_type"] == "sustained"
    assert all(env.episode_sub_rewards[a]["r_terminal"] == 300 for a in env.agents)


@pytest.mark.parametrize("start,delta", [
    ([430,430,100.6], [0,0,-.2]),  # roof descent regression
    ([430,430,101.5], [0,0,-1]),   # contact exactly at the roof endpoint
    ([350,430,80], [150,0,0]),     # crossed prism despite a free endpoint
    ([398,400,80], [4,-.6,0]),    # corner crossing
    ([399.4,430,80], [.2,0,0]),   # ordinary horizontal collision
])
def test_swept_building_collision_keeps_position_free(start, delta):
    env = make_env()
    env.buildings = np.array([[400,460,400,460,100]], np.float32)
    start = np.array(start, np.float32)
    assert not env._is_in_building(start, .5)
    result, hit = env._move_velocity_with_clip_3d(start, np.array(delta), .5)
    assert hit and not env._is_in_building(result, .5)
    assert "building:0" in env._prospective_collision_sources_velocity(start, np.array(delta), .5)


def test_roof_overflight_and_motion_away_from_wall_are_allowed():
    env = make_env()
    env.buildings = np.array([[400,460,400,460,100]], np.float32)
    result, hit = env._move_velocity_with_clip_3d(np.array([350,430,101]), np.array([150,0,0]), .5)
    assert not hit and np.allclose(result, [500,430,101])
    result, hit = env._move_velocity_with_clip_3d(np.array([399.5,430,80]), np.array([-1,0,0]), .5)
    assert not hit and result[0] < 399.5


def test_center_bias_and_configured_repulsion_actually_affect_commands():
    env = make_env(evader_center_pull=0, evader_rep_coef=7, evader_obstacle_range=35)
    assert env.evader_rep_coef == 7 and env.evader_obstacle_range == 35
    env.buildings = np.empty((0,5), np.float32)
    pos, vel = np.array([99,600,200]), np.array([0,3,0])
    no_pursuers = np.empty((0,3))
    before = env._evader_control(pos, vel, no_pursuers)
    env.evader_center_pull = 1.5
    after = env._evader_control(pos, vel, no_pursuers)
    assert not np.allclose(before, after)
    env.evader_center_pull = 0
    env.evader_rep_coef = 7
    before = env._evader_control(np.array([12,500,200]), vel, np.array([[22,490,200]]))
    env.evader_rep_coef = 70
    after = env._evader_control(np.array([12,500,200]), vel, np.array([[22,490,200]]))
    assert not np.allclose(before, after)


def test_course_gate_minimum_steps_consecutive_passes_and_safety():
    gate = PerformanceCurriculum()
    good = dict(episodes=100, safe_success_rate=.8, collision_rate=.1)
    assert gate.observe(50000, good) is None
    assert gate.observe(100000, good) is None
    assert gate.observe(100000, good) is None  # cannot count the same evaluation twice
    assert gate.observe(150000, dict(good, collision_rate=.2)) is None
    assert gate.observe(200000, good) is None
    assert gate.observe(250000, good) == 1
    assert gate.observe(300000, good) is None
    assert gate.observe(350000, dict(good, episodes=99)) is None
    assert gate.observe(400000, good) is None
    assert gate.observe(450000, good) == 2


def test_course_gate_stops_at_final_level_and_rejects_nonfinite_metrics():
    gate = PerformanceCurriculum(min_steps=0, required_passes=1)
    good = dict(episodes=100, safe_success_rate=1., collision_rate=0.)
    for step in range(1,5):
        assert gate.observe(step, good) == step
    assert gate.observe(5, good) is None and gate.level == 4
    gate = PerformanceCurriculum(min_steps=0)
    with pytest.raises(ValueError):
        gate.observe(1, dict(good, safe_success_rate=float("nan")))


def test_subprocess_workers_apply_map_and_radius_on_reset():
    from xuance.common import get_arguments
    from xuance.environment import make_envs
    overrides = Namespace(parallels=2, seed=21, env_seed=21, device="cpu")
    config = get_arguments("maddpg", "uav_pursuit_coverage_3d", "baseline_worker_test",
                           str(ROOT / "xuance/configs/maddpg/uav_pursuit_baseline_3d.yaml"), overrides)
    envs = make_envs(config)
    try:
        envs.reset()
        assert all(info["building_count"] == 20 for info in envs.env_method("get_curriculum_info"))
        envs.env_method("schedule_curriculum_level", 1)
        assert all(info["building_count"] == 20 for info in envs.env_method("get_curriculum_info"))
        envs.reset()
        assert all(info["building_count"] == 40 and info["capture_radius"] == 25
                   for info in envs.env_method("get_curriculum_info"))
    finally:
        envs.close()


def test_delayed_actor_updates_critic_each_step_without_updating_actor_early(tmp_path, monkeypatch):
    import torch
    from xuance import get_runner

    monkeypatch.chdir(tmp_path)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    runner = None
    try:
        overrides = Namespace(
            env_id="baseline_delay_test", device="cpu", parallels=1, running_steps=64,
            episode_length=16, seed=1, env_seed=1, actor_hidden_size=[32, 32],
            critic_hidden_size=[32, 32], batch_size=8, buffer_size=128,
            start_training=32, training_frequency=1, maddpg_actor_update_delay=2)
        runner = get_runner("maddpg", "uav_pursuit_coverage_3d", "baseline_delay_test",
                            str(ROOT / "xuance/configs/maddpg/uav_pursuit_baseline_3d.yaml"), overrides)
        agent = runner.agent
        agent.train(train_steps=16)
        initial = {k: v.clone() for k, v in agent.policy.state_dict().items()}
        first = agent.learner.update(agent.memory.sample())
        state = agent.policy.state_dict()
        actor_keys = [k for k in initial if k.startswith("actor.")]
        critic_keys = [k for k in initial if k.startswith("critic.")]
        assert all(torch.equal(initial[k], state[k]) for k in actor_keys)
        assert any(not torch.equal(initial[k], state[k]) for k in critic_keys)
        assert not any(k.endswith("/loss_actor") for k in first)
        assert any(k.endswith("/loss_critic") for k in first)
        second = agent.learner.update(agent.memory.sample())
        assert any(not torch.equal(initial[k], agent.policy.state_dict()[k]) for k in actor_keys)
        assert any(k.endswith("/loss_actor") for k in second)
    finally:
        if runner is not None:
            runner.agent.train_envs.close()
            runner.agent.writer.close()
        torch.set_num_threads(previous_threads)


def test_update_monitor_counts_only_actual_actor_updates(tmp_path):
    import csv
    from xuance.engine.uav_baseline_logging import BaselineTrainingLog

    monitor = BaselineTrainingLog(tmp_path, lambda *_: None)
    try:
        monitor.record_update(16, {"uav_0/loss_critic": 2., "uav_0/predictQ": 3., "noise_scale": .05})
        monitor.record_update(32, {"uav_0/loss_actor": -4., "uav_0/loss_critic": 6.,
                                   "uav_0/predictQ": 7., "noise_scale": .05})
    finally:
        monitor.close()
    with (tmp_path / "training_updates.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert int(row["window_updates"]) == 2
    assert int(row["window_actor_updates"]) == 1
    assert int(row["actor_updates"]) == 1
    assert float(row["loss_actor_mean"]) == -4.
    assert float(row["loss_critic_mean"]) == 4.
    assert float(row["predictQ_mean"]) == 5.


def test_matd3_target_smoothing_respects_noise_and_action_bounds():
    import torch
    from gymnasium.spaces import Box
    from xuance.torch.learners.multi_agent_rl.matd3_learner import MATD3_Learner

    learner = object.__new__(MATD3_Learner)
    learner.target_policy_noise, learner.target_noise_clip = 10., .2
    learner.policy = Namespace(action_space={"uav_0": Box(-1., 1., shape=(3,))})
    actions = {"uav_0": torch.tensor([[.99, -.99, .0]]).repeat(16, 1)}
    before = actions["uav_0"].clone()
    smoothed = learner._smooth_target_actions(actions)["uav_0"]
    assert torch.all(smoothed.abs() <= 1.)
    assert torch.max((smoothed - before).abs()) <= .200001
    assert not torch.equal(smoothed, before)
    assert torch.equal(actions["uav_0"], before)


def test_matd3_real_update_uses_two_critics_smoothing_and_delayed_actor(tmp_path, monkeypatch):
    import torch
    from xuance import get_runner

    monkeypatch.chdir(tmp_path)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    runner = None
    try:
        overrides = Namespace(
            agent="MATD3", learner="MATD3_Learner", policy="MATD3_Policy", actor_update_delay=2,
            target_policy_noise=.2, target_noise_clip=.5, td3_actor_use_first_critic=True,
            vectorize="DummyVecMultiAgentEnv",
            env_id="baseline_matd3_test", device="cpu", parallels=1, running_steps=64,
            episode_length=16, seed=1, env_seed=1, actor_hidden_size=[32, 32],
            critic_hidden_size=[32, 32], batch_size=8, buffer_size=128,
            start_training=32, training_frequency=1)
        runner = get_runner("matd3", "uav_pursuit_coverage_3d", "baseline_matd3_test",
                            str(ROOT / "xuance/configs/maddpg/uav_pursuit_baseline_3d.yaml"), overrides)
        agent = runner.agent
        agent.train(train_steps=16)
        key = agent.model_keys[0]
        policy = agent.policy
        assert not torch.equal(policy.critic_A[key].model[0].weight, policy.critic_B[key].model[0].weight)
        with torch.no_grad():
            for critic, constant in ((policy.target_critic_A[key], 2.), (policy.target_critic_B[key], 5.)):
                for parameter in critic.parameters():
                    parameter.zero_()
                critic.model[-1].bias.fill_(constant)
        joint_action_size = sum(space.shape[0] for space in policy.action_space.values())
        obs_size = sum(space.shape[0] for space in agent.observation_space.values())
        minimum = policy.Qtarget(torch.zeros(1, obs_size), torch.zeros(1, joint_action_size), torch.eye(4))
        assert torch.all(minimum[key] == 2.)
        original_target, original_qtarget, original_qpolicy = policy.Atarget, policy.Qtarget, policy.Qpolicy
        observed = {}

        def capture_target_actions(*args, **kwargs):
            hidden, actions = original_target(*args, **kwargs)
            observed["raw"] = actions[key].detach().clone()
            return hidden, actions

        def capture_smoothed_actions(*args, **kwargs):
            observed["smoothed"] = kwargs["joint_actions"].detach().clone()
            return original_qtarget(*args, **kwargs)

        def capture_actor_objective(*args, **kwargs):
            qa, qb, average = original_qpolicy(*args, **kwargs)
            if kwargs.get("agent_key") is not None:
                observed["actor_q1_mean"] = qa[key].detach().mean().item()
            return qa, qb, average

        monkeypatch.setattr(policy, "Atarget", capture_target_actions)
        monkeypatch.setattr(policy, "Qtarget", capture_smoothed_actions)
        monkeypatch.setattr(policy, "Qpolicy", capture_actor_objective)
        initial = {k: v.clone() for k, v in policy.state_dict().items()}
        first = agent.learner.update(agent.memory.sample())
        actor_keys = [k for k in initial if k.startswith("actor.")]
        assert all(torch.equal(initial[k], policy.state_dict()[k]) for k in actor_keys)
        for group in ("critic_A.", "critic_B."):
            assert any(not torch.equal(initial[k], policy.state_dict()[k]) for k in initial if k.startswith(group))
        assert not any(k.endswith("/loss_actor") for k in first)
        smoothed = observed["smoothed"].reshape_as(observed["raw"])
        assert torch.all(smoothed.abs() <= 1.)
        assert torch.max((smoothed - observed["raw"]).abs()) <= .500001
        assert not torch.equal(smoothed, observed["raw"])
        assert first[f"{key}/predictQ"] <= min(first[f"{key}/predictQ_A"], first[f"{key}/predictQ_B"]) + 1e-5
        second = agent.learner.update(agent.memory.sample())
        assert second[f"{key}/loss_actor"] == pytest.approx(-observed["actor_q1_mean"])
        assert any(not torch.equal(initial[k], policy.state_dict()[k]) for k in actor_keys)
        assert any(k.endswith("/loss_actor") for k in second)
    finally:
        if runner is not None:
            runner.agent.train_envs.close()
            runner.agent.writer.close()
        torch.set_num_threads(previous_threads)


def test_runner_trains_networks_and_applies_curriculum(tmp_path, monkeypatch):
    """Use permissive gates only here to exercise a real runner promotion."""
    import csv
    import math
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from xuance import get_runner

    monkeypatch.chdir(tmp_path)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    overrides = Namespace(
        env_id="baseline_runner_test", device="cpu", parallels=2, running_steps=512, eval_interval=256,
        test_episode=3, baseline_eval_parallels=2, episode_length=16, seed=1, env_seed=1,
        actor_hidden_size=[32, 32], critic_hidden_size=[32, 32],
        batch_size=16, buffer_size=1024, start_training=16, training_frequency=1,
        curriculum_min_steps_per_level=128, curriculum_success_threshold=0.,
        curriculum_collision_threshold=1., curriculum_required_passes=1,
        noise_decay_steps=64, baseline_progress_weight=1.0, baseline_progress_scale=10.0,
        baseline_reward_scale=0.1, baseline_failure_penalty=300.0,
        baseline_evaluate_full_episode=True, baseline_final_eval_interval=512,
        baseline_evaluate_previous_levels=True)
    try:
        runner = get_runner("maddpg", "uav_pursuit_coverage_3d", "baseline_runner_test",
                            str(ROOT / "xuance/configs/maddpg/uav_pursuit_baseline_3d.yaml"), overrides)
        initial = {k: v.clone() for k, v in runner.agent.policy.state_dict().items()}
        log_dir = runner.agent.log_dir
        observed_workers = []
        original_train = runner.agent.train

        def train_and_observe(*args, **kwargs):
            result = original_train(*args, **kwargs)
            observed_workers.append(runner.agent.train_envs.env_method("get_curriculum_info"))
            return result

        runner.agent.train = train_and_observe
        runner.run(mode="benchmark")
        final = runner.agent.policy.state_dict()
        for group in ("actor", "critic"):
            keys = [k for k in initial if group in k.lower() and "target" not in k.lower()]
            assert keys and any(not torch.equal(initial[k], final[k]) for k in keys)
        assert all(torch.isfinite(v).all() for v in final.values())
        assert runner.agent.noise_scale == runner.agent.end_noise
        assert all(info["curriculum_level"] == 1 for info in observed_workers[-1])
        result_dir = tmp_path / "outputs/results/maddpg/baseline_runner_test"
        summary = json.loads((result_dir / "baseline_summary.json").read_text())
        assert summary["environment_steps"] == 512
        assert summary["transitions"] == [{"step": 256, "level": 1}]
        assert summary["best_final_task"]["level"] == 4
        assert (result_dir / "best_model/best_model.pth").is_file()
        with (result_dir / "curriculum_eval.csv").open() as stream:
            history = list(csv.DictReader(stream))
        assert all(row["episodes"] == "3" for row in history)
        assert [row["step"] for row in history if row["phase"] == "final_task"] == ["0", "512"]
        assert any(row["phase"] == "current" and row["level"] == "1" for row in history)
        assert any(row["phase"] == "retention" and row["level"] == "0" and row["step"] == "512"
                   for row in history)
        assert json.loads((result_dir / "config.json").read_text())["evaluation_env_seed"] == 10001
        with (result_dir / "training_episodes.csv").open() as stream:
            training = list(csv.DictReader(stream))
        assert training and all(row["success"] in ("0", "1") for row in training)
        assert any(row["building_count"] == "40" for row in training)
        assert (result_dir / "training_status.json").is_file()
        assert (result_dir / "training_updates.csv").stat().st_size > 100
        metrics = {}
        for event in Path(log_dir).rglob("events.out.tfevents.*"):
            reader = EventAccumulator(str(event), size_guidance={"scalars": 0})
            reader.Reload()
            for tag in reader.Tags()["scalars"]:
                if tag.endswith(("loss_actor", "loss_critic", "predictQ")):
                    metrics[tag] = [v.value for v in reader.Scalars(tag)]
        assert len(metrics) == 3
        assert all(values and all(math.isfinite(v) for v in values) for values in metrics.values())
    finally:
        torch.set_num_threads(previous_threads)


def test_existing_coverage_mode_still_steps_with_prediction_observations():
    env = make_env(reward_mode="coverage", use_target_prediction_obs=True,
                   curriculum_context_obs=False)
    obs, _ = env.reset()
    assert env.target_forecast is not None
    obs, rewards, _, _, _ = env.step({a: np.zeros(3) for a in env.agents})
    assert all(np.isfinite(rewards[a]) for a in env.agents)
    assert all(obs[a].shape == env.observation_space[a].shape for a in env.agents)


@pytest.mark.parametrize("level", range(5))
def test_random_actions_and_evader_remain_outside_buildings(level):
    np.random.seed(20 + level)
    env = make_env(curriculum_start_level=level)
    obs, _ = env.reset()
    for _ in range(40):
        actions = {a: np.random.uniform(-1,1,3) for a in env.agents}
        obs, rewards, done, truncated, info = env.step(actions)
        assert not env._is_in_building(env.target_position, env.target_radius)
        assert all(not env._is_in_building(p, env.uav_radius) for p in env.uav_positions)
        assert all(np.all(np.isfinite(obs[a])) and np.isfinite(rewards[a]) for a in env.agents)
        if any(done.values()) or truncated:
            obs, _ = env.reset()
