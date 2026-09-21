import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from xuance.environment.multi_agent_env.uav_pursuit_apollonius_3d import (
    UAVPursuitApollonius3DEnv,
)
from xuance.environment.multi_agent_env.eval_metrics import summarize_eval


class Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def make_env(n=4, mode="medium", **kwargs):
    return UAVPursuitApollonius3DEnv(Cfg(
        num_agents=n,
        building_mode=mode,
        curriculum_enabled=False,
        **kwargs,
    ))


def test_dynamic_clearance_and_legacy_reproduce_n4_bug():
    env = make_env(spawn_safety_enabled=False)
    env.reset()
    assert env.spawn_clearance == pytest.approx(12.0)
    assert env._spawn_clearance_xy(env.uav_positions[2]) < 0.0


def test_legacy_n4_collision_reports_agent_and_building():
    env = make_env(spawn_safety_enabled=False)
    env.reset()
    actions = {agent: np.zeros(3, np.float32) for agent in env.agents}
    _, _, _, _, info = env.step(actions)
    assert "uav_2" in info["collision_agents"]
    assert "building:8" in info["collision_sources"]["uav_2"]


def test_collision_rate_counts_simultaneous_capture_and_collision():
    summary = summarize_eval([{
        "caught": True,
        "crashed": True,
        "steps": 1,
        "min_omega_euc": 1.0,
        "min_omega_vis": 1.0,
    }])
    assert summary["success_rate"] == 1.0
    assert summary["collision_rate"] == 1.0
    assert summary["timeout_rate"] == 0.0


@pytest.mark.parametrize(("config_name", "value"), [
    ("spawn_safety_margin", np.nan),
    ("spawn_safety_margin", np.inf),
    ("spawn_search_resolution", np.nan),
    ("spawn_search_resolution", np.inf),
])
def test_spawn_safety_config_rejects_nonfinite_values(config_name, value):
    with pytest.raises(ValueError, match=config_name):
        make_env(**{config_name: value})


@pytest.mark.parametrize("n", range(3, 9))
@pytest.mark.parametrize("mode", ["empty", "open", "medium", "complex"])
def test_all_n_and_maps_satisfy_reset_invariants(n, mode):
    env = make_env(n, mode)
    _, info = env.reset()
    assert np.all(np.isfinite(env.uav_positions))
    assert np.all((env.uav_positions[:, 2] >= env.z_min) &
                  (env.uav_positions[:, 2] <= env.z_max))
    assert all(env._spawn_clearance_xy(p) >= env.spawn_clearance - 1e-5
               for p in env.uav_positions)
    assert all(np.linalg.norm(env.uav_positions[i] - env.uav_positions[j])
               >= env.spawn_min_pair_distance - 1e-5
               for i, j in itertools.combinations(range(n), 2))
    assert info["infos"]["spawn_clearance"] == pytest.approx(12.0)


@pytest.mark.parametrize("mode,count", [
    ("empty", 0), ("open", 4), ("medium", 16), ("complex", 26),
])
def test_assets_present(mode, count):
    assert len(make_env(4, mode).buildings) == count


def test_valid_empty_line_spawn_is_unchanged():
    env = make_env(8, "empty")
    env.reset()
    np.testing.assert_array_equal(env.uav_positions, env.init_uav_positions)


def test_n4_medium_allocation_is_deterministic():
    first, second = make_env(), make_env()
    first.reset()
    second.reset()
    np.testing.assert_array_equal(first.uav_positions, second.uav_positions)
    assert first.get_curriculum_info()["spawn_adjusted_count"] == 2


def test_adjusted_positions_are_nearest_valid_lattice_candidates():
    env = make_env()
    env.reset()
    assigned = []
    for origin, chosen, offset in zip(
            env.init_uav_positions, env.uav_positions, env.spawn_offsets):
        radius = int(np.ceil(float(offset) / env.spawn_search_resolution))
        for ix in range(-radius, radius + 1):
            for iy in range(-radius, radius + 1):
                candidate = origin.copy()
                candidate[:2] += env.spawn_search_resolution * np.array([ix, iy])
                candidate_offset = float(np.linalg.norm(candidate - origin))
                if candidate_offset < float(offset) - 1e-5:
                    assert not env._is_valid_pursuer_spawn(candidate, assigned)
        assigned.append(chosen)


def test_density_cache_is_keyed_by_actual_buildings(monkeypatch):
    env = make_env(randomize_density=["empty", "medium"])
    modes = iter(["empty", "medium"])
    monkeypatch.setattr(np.random, "choice", lambda _: next(modes))
    env.reset()
    empty_positions = env.uav_positions.copy()
    env.reset()
    medium_positions = env.uav_positions.copy()
    np.testing.assert_array_equal(empty_positions, env.init_uav_positions)
    assert not np.array_equal(medium_positions, empty_positions)
    assert all(env._spawn_clearance_xy(p) >= env.spawn_clearance - 1e-5
               for p in medium_positions)


def test_safe_allocation_does_not_consume_rng_or_change_paired_episode_draws():
    legacy, safe = make_env(spawn_safety_enabled=False), make_env()
    np.random.seed(7)
    legacy.reset()
    np.random.seed(7)
    safe.reset()
    np.testing.assert_array_equal(safe.target_position, legacy.target_position)
    np.testing.assert_array_equal(safe.uav_yaws, legacy.uav_yaws)
    assert safe.target_yaw == legacy.target_yaw


def test_every_first_step_heading_is_collision_free():
    env = make_env()
    env.reset()
    for pos in env.uav_positions:
        for yaw in np.linspace(0, 2 * np.pi, 72, endpoint=False):
            for pitch in (-env.pitch_max, 0.0, env.pitch_max):
                _, hit = env._move_with_clip_3d(
                    pos, yaw, pitch, env.initial_step_reach, env.uav_radius)
                assert not hit


def test_missing_nonempty_asset_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="buildings_medium"):
        UAVPursuitApollonius3DEnv(Cfg(
            num_agents=4,
            building_mode="medium",
            curriculum_enabled=False,
            buildings_dir=str(tmp_path),
        ))


@pytest.mark.parametrize("payload", [
    "{",
    "[]",
    json.dumps([[0, 1, 2]]),
    json.dumps([[2, 1, 0, 1]]),
    json.dumps([[-1, 1, 0, 1]]),
    "[[0, 1, NaN, 2]]",
])
def test_invalid_nonempty_asset_fails_with_path(tmp_path, payload):
    asset = tmp_path / "buildings_medium.json"
    asset.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="buildings_medium"):
        make_env(buildings_dir=str(tmp_path))


def test_reset_consumers_use_adjusted_positions(monkeypatch):
    env = make_env()
    seen = {}
    original_radar = env._update_radar_cache
    original_guides = env._assign_target_points

    def record_radar():
        seen["radar"] = env.uav_positions.copy()
        return original_radar()

    def record_guides():
        seen["guides"] = env.uav_positions.copy()
        return original_guides()

    monkeypatch.setattr(env, "_update_radar_cache", record_radar)
    monkeypatch.setattr(env, "_assign_target_points", record_guides)
    obs, _ = env.reset()
    np.testing.assert_array_equal(seen["radar"], env.uav_positions)
    np.testing.assert_array_equal(seen["guides"], env.uav_positions)
    for index, agent in enumerate(env.agents):
        expected = env.uav_positions[index] / env.map_size * 2.0 - 1.0
        expected[2] = ((env.uav_positions[index, 2] - env.z_min)
                       / (env.z_max - env.z_min) * 2.0 - 1.0)
        np.testing.assert_allclose(obs[agent][:3], expected, atol=1e-6)
        assert env.last_distances[index] == pytest.approx(
            np.linalg.norm(env.uav_positions[index] - env.target_position))


def _evaluation_runtime_root():
    source = Path(
        sys.modules[UAVPursuitApollonius3DEnv.__module__].__file__
    ).resolve()
    for parent in source.parents:
        if (parent / "examples" / "evaluate_3d.py").is_file():
            return parent
    raise AssertionError(f"Cannot find evaluator above {source}")


def _greedy_pair_trace(tmp_path, safety):
    root = _evaluation_runtime_root()
    summary = tmp_path / f"pair_{safety}.json"
    trace = tmp_path / f"pair_{safety}.jsonl"
    runtime_env = os.environ.copy()
    runtime_env["PYTHONPATH"] = str(root)
    command = [
        sys.executable, str(root / "examples" / "evaluate_3d.py"),
        "--policy", "greedy", "--seed", "7",
        "--building-mode", "medium", "--num-agents", "3",
        "--target-max-speed", "6", "--k", "2",
        "--spawn-safety", safety, "--collision-mode", "hard",
        "--out", str(summary), "--trace-out", str(trace),
    ]
    completed = subprocess.run(
        command, cwd=root, env=runtime_env, capture_output=True,
        text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]


def test_evaluator_pairs_each_episode_across_spawn_safety(tmp_path):
    off = _greedy_pair_trace(tmp_path, "off")
    on = _greedy_pair_trace(tmp_path, "on")
    assert [record["episode"] for record in off] == [0, 1]
    assert [record["episode"] for record in on] == [0, 1]
    assert off[0]["steps"] != on[0]["steps"]
    paired_keys = (
        "initial_target_position", "initial_uav_yaws", "initial_uav_pitches",
    )
    assert [[record[key] for key in paired_keys] for record in off] == [
        [record[key] for key in paired_keys] for record in on
    ]
    assert any(
        before["initial_positions"] != after["initial_positions"]
        for before, after in zip(off, on)
    )
