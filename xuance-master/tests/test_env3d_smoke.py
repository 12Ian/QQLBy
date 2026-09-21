import numpy as np
import matplotlib
matplotlib.use("Agg")
from xuance.environment.multi_agent_env.uav_pursuit_apollonius_3d import UAVPursuitApollonius3DEnv


class Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def make_env(n=6):
    return UAVPursuitApollonius3DEnv(Cfg(num_agents=n, env_seed=0,
                                        building_mode="empty",
                                        curriculum_enabled=False))


# ---- Task 6: scaffold ----
def test_dims_match_spaces():
    env = make_env(6)
    assert env.obs_dim == 17 + 3 * (6 - 1) + 16          # 48
    assert env.state_dim == 9 * 6 + 6                     # 60
    obs, info = env.reset()
    a0 = env.agents[0]
    assert obs[a0].shape == (env.obs_dim,)
    assert env.state().shape == (env.state_dim,)
    assert env.action_space[a0].shape == (3,)


def test_reset_positions_in_bounds():
    env = make_env(6)
    env.reset()
    assert np.all(env.uav_positions[:, 2] >= env.z_min - 1e-3)
    assert np.all(env.uav_positions[:, 2] <= env.z_max + 1e-3)
    assert env.z_min - 1e-3 <= env.target_position[2] <= env.z_max + 1e-3


# ---- Task 7: dynamics ----
def test_step_100_random_no_nan_in_bounds():
    env = make_env(6)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(100):
        acts = {a: rng.uniform(-1, 1, size=3).astype(np.float32) for a in env.agents}
        obs, rew, term, trunc, info = env.step(acts)
        for a in env.agents:
            assert np.all(np.isfinite(obs[a]))
            assert np.isfinite(rew[a])
        assert np.all(env.uav_positions[:, 0] >= -1e-3) and np.all(env.uav_positions[:, 0] <= env.map_size + 1e-3)
        assert np.all(env.uav_positions[:, 2] >= env.z_min - 1e-3) and np.all(env.uav_positions[:, 2] <= env.z_max + 1e-3)
        if term[env.agents[0]] or trunc:
            env.reset()


# ---- Task 8: obs/state ----
def test_obs_state_ranges_and_layout():
    env = make_env(6)
    obs, _ = env.reset()
    o = obs[env.agents[0]]
    assert o.shape == (48,)
    assert np.all(o >= -1.5) and np.all(o <= 1.5)
    s = env.state()
    assert s.shape == (60,)


# ---- Task 9: guide assignment ----
def test_guide_points_valid():
    env = make_env(6)
    env.reset()
    gp = env._assign_target_points()
    assert len(gp) == env.num_agents
    for a, p in gp.items():
        assert p.shape == (3,)
        assert np.linalg.norm(p - env.target_position) > env.catch_radius
        assert env.z_min - 1e-3 <= p[2] <= env.z_max + 1e-3


def test_n4_distribution_target_is_regular_tetrahedron():
    env = make_env(4)
    ideal = env._ideal_spherical_nn_angle(4)
    assert np.isclose(ideal, np.arccos(-1.0 / 3.0))
    tetra_score = np.exp(-abs(ideal - ideal))
    clustered_score = np.exp(-abs((np.pi / 3.0) - ideal))
    assert tetra_score > clustered_score


def test_guide_candidate_supplement_keeps_dangerous_directions():
    dangerous = np.array([False, True, False, False, False, False, True, False])
    free = np.array([8.0, 1.0, 7.0, 6.0, 5.0, 4.0, 0.5, 3.0])
    selected = UAVPursuitApollonius3DEnv._guide_candidate_indices(dangerous, free, 4)
    assert len(selected) == 4
    assert {1, 6}.issubset(set(selected.tolist()))


# ---- Task 10: reward ----
def test_reward_terms_present_and_finite():
    env = make_env(6)
    env.reset()
    acts = {a: np.zeros(3, np.float32) for a in env.agents}
    _, rew, _, _, info = env.step(acts)
    for a in env.agents:
        assert np.isfinite(rew[a])
        sr = info["episode_sub_rewards"][a]
        for k in ["r_near", "r_safe", "r_pos", "r_gap", "r_finish"]:
            assert k in sr


def test_separation_shaping_is_opt_in_and_has_early_warning():
    legacy = UAVPursuitApollonius3DEnv(Cfg(num_agents=2, building_mode="empty",
                                           curriculum_enabled=False))
    shaped = UAVPursuitApollonius3DEnv(Cfg(num_agents=2, building_mode="empty",
                                           curriculum_enabled=False,
                                           separation_weight=4.0,
                                           separation_distance=20.0))
    positions = np.array([[300.0, 300.0, 200.0],
                          [310.0, 300.0, 200.0]], dtype=np.float32)
    legacy.uav_positions = positions.copy()
    shaped.uav_positions = positions.copy()

    assert legacy._teammate_separation_penalty(0) == 0.0
    # At 10 m inside a 20 m barrier: -(1 - 10/20)^2 = -0.25.
    assert np.isclose(shaped._teammate_separation_penalty(0), -0.25)

    shaped.uav_positions[1, 0] = 325.0
    assert shaped._teammate_separation_penalty(0) == 0.0


def test_soft_collision_keeps_episode_alive_but_reports_hit():
    def wall_step(terminate_on_collision):
        env = UAVPursuitApollonius3DEnv(Cfg(num_agents=2, building_mode="empty",
                                            curriculum_enabled=False,
                                            terminate_on_collision=terminate_on_collision))
        env.reset()
        env.uav_positions[:] = np.array([[0.6, 300.0, 200.0],
                                         [300.0, 800.0, 200.0]], dtype=np.float32)
        env.uav_yaws[:] = np.array([np.pi, 0.0], dtype=np.float32)
        env.uav_pitches[:] = 0.0
        env.uav_speeds[:] = env.uav_min_speed
        env.target_position[:] = np.array([800.0, 500.0, 200.0], dtype=np.float32)
        env.last_distances = np.linalg.norm(
            env.uav_positions - env.target_position[None, :], axis=1)
        env._update_radar_cache()
        actions = {a: np.zeros(3, np.float32) for a in env.agents}
        return env, env.step(actions)

    hard, (_, _, hard_term, _, hard_info) = wall_step(True)
    soft, (_, _, soft_term, _, soft_info) = wall_step(False)

    assert hard_info["is_collision"] is True
    assert soft_info["is_collision"] is True
    assert hard_term[hard.agents[0]] is True
    assert soft_term[soft.agents[0]] is False


def test_capture_gives_finish_reward():
    env = make_env(6)
    env.reset()
    # place a pursuer exactly on the target; after one step max separation is
    # uav_speed (~9) + target_speed (~4.5) < catch_radius*... -> within 15.
    env.uav_positions[0] = env.target_position.copy()
    acts = {a: np.zeros(3, np.float32) for a in env.agents}
    _, rew, term, _, info = env.step(acts)
    assert info["is_success"] is True
    assert term[env.agents[0]] is True


# ---- Task 11: curriculum ----
def test_curriculum_set_get_level():
    env = UAVPursuitApollonius3DEnv(Cfg(num_agents=6, curriculum_enabled=True))
    env.set_curriculum_level(2)
    info = env.get_curriculum_info()
    assert info["curriculum_level"] == 2
    assert "reward_weights" in info


# ---- Task 12: render ----
def test_render_returns_rgb():
    env = make_env(6)
    env.reset()
    img = env.render(mode="rgb_array")
    assert img.ndim == 3 and img.shape[2] == 3


# ---- Task 13: registration ----
def test_registry_contains_3d_env():
    from xuance.environment.multi_agent_env import REGISTRY_MULTI_AGENT_ENV as R
    assert "uav_pursuit_apollonius_3d" in R
    assert not isinstance(R["uav_pursuit_apollonius_3d"], str)  # import succeeded


# ---- Increment 2a: criterion_mode wiring ----
def test_criterion_mode_changes_escape_angle_with_buildings():
    # same geometry, euclidean vs visibility env -> visibility never seals more
    e_euc = UAVPursuitApollonius3DEnv(Cfg(num_agents=6, building_mode="medium",
                                          curriculum_enabled=False, criterion_mode="euclidean"))
    e_vis = UAVPursuitApollonius3DEnv(Cfg(num_agents=6, building_mode="medium",
                                          curriculum_enabled=False, criterion_mode="visibility"))
    assert e_euc.criterion_mode == "euclidean"
    assert e_vis.criterion_mode == "visibility"
    np.random.seed(3)
    e_euc.reset()
    # copy the SAME geometry into the visibility env
    e_vis.uav_positions = e_euc.uav_positions.copy()
    e_vis.uav_yaws = e_euc.uav_yaws.copy()
    e_vis.uav_pitches = e_euc.uav_pitches.copy()
    e_vis.uav_speeds = e_euc.uav_speeds.copy()
    e_vis.target_position = e_euc.target_position.copy()
    e_vis.target_speed = e_euc.target_speed
    for e in (e_euc, e_vis):
        e.apollonius_margin_ema = None
        e.apollonius_dangerous_prev = None
        e._assign_target_points()
    om_euc = e_euc.apollonius_escape["escape_solid_angle"]
    om_vis = e_vis.apollonius_escape["escape_solid_angle"]
    assert om_vis >= om_euc - 1e-6   # visibility voids occluded pursuers; never seals more


def test_default_criterion_mode_is_euclidean():
    env = make_env(6)
    assert env.criterion_mode == "euclidean"


# ---- P2: reward-term ablation ----
def test_reward_disable_zeros_named_term():
    env = UAVPursuitApollonius3DEnv(Cfg(num_agents=6, building_mode="empty",
                                        curriculum_enabled=False, reward_disable=["r_pos"]))
    assert env.reward_disable == {"r_pos"}
    env.reset()
    acts = {a: np.zeros(3, np.float32) for a in env.agents}
    _, _, _, _, info = env.step(acts)
    for a in env.agents:
        assert info["episode_sub_rewards"][a]["r_pos"] == 0.0


def test_reward_disable_default_empty():
    env = make_env(6)
    assert env.reward_disable == set()


# ---- P3: clean finish/closure split (gate separation must preserve the full reward) ----
def _step_zero_reward(disable):
    np.random.seed(11)   # identical reset geometry + step RNG across variants
    e = UAVPursuitApollonius3DEnv(Cfg(num_agents=6, building_mode="medium",
                                      curriculum_enabled=False, closure_weight=8.0,
                                      reward_disable=disable))
    e.reset()
    acts = {a: np.zeros(3, np.float32) for a in e.agents}
    _, rew, _, _, _ = e.step(acts)
    return np.array([rew[a] for a in e.agents], dtype=np.float64)


def test_finish_closure_split_is_additive():
    # full reward must equal the sum of the two independent removals (closure is carried on
    # the finish channel; disabling r_finish and r_closure separately must partition it cleanly,
    # leaving the neither-disabled reward byte-identical to the pre-split method).
    full = _step_zero_reward([])
    nofin = _step_zero_reward(["r_finish"])
    noclo = _step_zero_reward(["r_closure"])
    noboth = _step_zero_reward(["r_finish", "r_closure"])
    assert np.allclose(full - noboth, (full - nofin) + (full - noclo), atol=1e-6)


def test_reward_disable_closure_keeps_capture_finish():
    # disabling the closure bonus must NOT remove the terminal capture reward
    env = UAVPursuitApollonius3DEnv(Cfg(num_agents=6, building_mode="empty",
                                        curriculum_enabled=False, reward_disable=["r_closure"]))
    assert env.reward_disable == {"r_closure"}
    env.reset()
    env.uav_positions[0] = env.target_position.copy()
    acts = {a: np.zeros(3, np.float32) for a in env.agents}
    _, rew, term, _, info = env.step(acts)
    assert info["is_success"] is True
    assert info["episode_sub_rewards"][env.agents[0]]["r_finish"] > 0.0


def test_configurable_target_speed_when_curriculum_off():
    env = UAVPursuitApollonius3DEnv(Cfg(num_agents=6, building_mode="open",
                                        curriculum_enabled=False, target_max_speed=8.0))
    assert env.target_max_speed == 8.0


def test_domain_randomization_varies_density():
    env = UAVPursuitApollonius3DEnv(Cfg(num_agents=6, curriculum_enabled=False,
                                        randomize_density=["empty", "medium"]))
    np.random.seed(0)
    counts = set()
    for _ in range(25):
        env.reset()
        counts.add(len(env.buildings))
    assert 0 in counts             # empty (0 buildings) appeared
    assert 16 in counts            # medium (16 buildings) appeared


def test_guide_collapse_flag_keeps_guides_far():
    from xuance.environment.multi_agent_env import geometry3d as g

    def mean_guide_dist(collapse):
        e = UAVPursuitApollonius3DEnv(Cfg(num_agents=6, building_mode="empty",
                                          curriculum_enabled=False, guide_collapse=collapse))
        np.random.seed(1)
        e.reset()
        e.uav_positions = (g.fibonacci_sphere(6) * 30.0 + e.target_position).astype(np.float32)
        e.apollonius_margin_ema = None
        e.apollonius_dangerous_prev = None
        gp = e._assign_target_points()
        return np.mean([np.linalg.norm(gp[a] - e.target_position) for a in e.agents])

    assert mean_guide_dist(False) > mean_guide_dist(True)   # no-collapse keeps guides farther


def test_reward_weights_and_closure_configurable():
    env = UAVPursuitApollonius3DEnv(Cfg(num_agents=6, building_mode="open",
                                        curriculum_enabled=False, w_pos=0.6, w_gap=0.8,
                                        closure_weight=8.0))
    assert env.reward_weights["w_pos"] == 0.6
    assert env.reward_weights["w_gap"] == 0.8
    assert env.closure_weight == 8.0


def test_guide_radius_collapses_when_encircled():
    env = make_env(6)
    rf = 120.0
    free_k = 400.0
    open_r = env._encircle_guide_radius(0.6 * 4 * np.pi, rf, free_k)   # fully open
    enc_r = env._encircle_guide_radius(0.0, rf, free_k)               # fully encircled
    assert enc_r < open_r                                  # encircled -> guides closer
    assert abs(enc_r - env.catch_radius * 1.3) < 1e-6      # encircled -> near catch radius
    assert enc_r > env.catch_radius                        # still outside capture


def test_closure_bonus_is_progress_based_no_hover_farm():
    env = make_env(6)
    mrs = env.uav_max_speed + env.target_max_speed
    enc_closing = env._closure_bonus(0.0, 100.0, 90.0, mrs)        # encircled, closing
    enc_static = env._closure_bonus(0.0, 100.0, 100.0, mrs)        # encircled, no progress
    open_closing = env._closure_bonus(0.6 * 4 * np.pi, 100.0, 90.0, mrs)  # open, closing
    assert enc_closing > enc_static                # closing rewarded when encircled
    assert abs(enc_static) < 1e-9                  # no progress -> 0 (cannot hover-farm)
    assert enc_closing > open_closing              # only meaningful when encircled


# ---- Increment 2b: obstacle node features ----
def test_obstacle_features_nearest_and_padding():
    env = UAVPursuitApollonius3DEnv(Cfg(num_agents=6, building_mode="medium",
                                        curriculum_enabled=False, use_obstacle_gat=True))
    assert env.use_obstacle_gat is True
    assert env.obs_dim == 17 + 3 * (6 - 1) + 16 + env.obstacle_gat_k * 4   # 64
    env.reset()
    feats = env._nearest_obstacle_features(env.uav_positions[0], env.obstacle_gat_k)
    assert feats.shape == (env.obstacle_gat_k, 4)
    assert np.all(feats[:, 3] == 1.0)                     # medium has >= k buildings -> all valid
    assert np.all(np.abs(feats[:, :2]) <= 1.0 + 1e-6)     # normalized rel
    d = np.hypot(feats[:, 0], feats[:, 1])
    assert np.all(np.diff(d) >= -1e-6)                    # nearest first


def test_obstacle_features_empty_map_all_padding():
    env = UAVPursuitApollonius3DEnv(Cfg(num_agents=6, building_mode="empty",
                                        curriculum_enabled=False, use_obstacle_gat=True))
    env.reset()
    feats = env._nearest_obstacle_features(env.uav_positions[0], env.obstacle_gat_k)
    assert np.all(feats == 0.0)            # no buildings -> all padding, valid_flag 0


def test_obstacle_gat_off_keeps_obs_48():
    env = make_env(6)                       # default cfg -> use_obstacle_gat absent -> False
    assert env.use_obstacle_gat is False
    assert env.obs_dim == 48
    obs, _ = env.reset()
    assert obs[env.agents[0]].shape == (48,)


# ---- Task 14: real buildings map loads + steps cleanly ----
def test_medium_map_loads_and_steps():
    # default buildings_dir is the env-local 'buildings/' folder (populated on
    # the server). 'medium' map has 16 prism footprints.
    env = UAVPursuitApollonius3DEnv(Cfg(num_agents=6, building_mode="medium",
                                        curriculum_enabled=False))
    assert len(env.buildings) == 16
    env.reset()
    rng = np.random.default_rng(1)
    for _ in range(30):
        acts = {a: rng.uniform(-1, 1, size=3).astype(np.float32) for a in env.agents}
        obs, rew, term, trunc, info = env.step(acts)
        for a in env.agents:
            assert np.all(np.isfinite(obs[a])) and np.isfinite(rew[a])
        if term[env.agents[0]] or trunc:
            env.reset()
