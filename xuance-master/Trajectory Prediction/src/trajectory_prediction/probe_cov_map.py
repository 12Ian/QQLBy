"""Controlled geometry probe using the cov environment's configured building map."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .cov_map import CovMap
from .prediction import TrajectoryPredictor, primitive_candidates


def probe_cov_map(cov_config_path: Path, buildings_dir: Path,
                  prediction_config_path: Path, output: Path,
                  device_name: str = "cpu", live_resets: int = 10,
                  seed: int = 1) -> dict:
    cov_config = yaml.safe_load(cov_config_path.read_text(encoding="utf-8"))
    cov_map = CovMap.from_cov_config(cov_config_path, buildings_dir)
    model_config = yaml.safe_load(prediction_config_path.read_text(encoding="utf-8"))["model"]
    model_config = dict(model_config)
    model_config["candidate_acceleration_g"] = float(cov_config.get("target_accel", 1.0)) / 9.81
    model_config["min_altitude_m"] = cov_map.z_min
    model_config["threat_range_m"] = min(model_config["threat_range_m"], 100.0)
    model_config["threat_gate_scale_m"] = 20.0
    speed = float(cov_config.get("target_min_speed", 3.0))
    if cov_map.buildings.size:
        xmin, xmax, ymin, ymax, top = cov_map.buildings[0]
        approach = max(3.0, speed * model_config["future_steps"] * model_config["dt"] * 0.5)
        if xmin - approach > 40:
            start_x, direction = float(xmin - approach), 1.0
        else:
            start_x, direction = float(xmax + approach), -1.0
        start = np.array([start_x, (ymin + ymax) / 2,
                          max(cov_map.z_min, min(cov_map.z_max, top - 1.0))],
                         np.float32)
    else:
        start = np.array([cov_map.map_size / 2, cov_map.map_size / 2,
                          (cov_map.z_min + cov_map.z_max) / 2], np.float32)
        direction = 1.0
    defender = start + np.array([0.0, -30.0, 0.0], np.float32)
    target_history = np.zeros((1, 80, 6), np.float32)
    defender_history = np.zeros_like(target_history)
    target_history[0, :, :3] = start
    target_history[0, :, 3] = direction * speed
    defender_history[0, :, :3] = defender
    defender_history[0, :, 4] = float(cov_config.get("uav_initial_speed", 1.0))
    torch.manual_seed(1)
    device = torch.device(device_name)
    model = TrajectoryPredictor(**model_config).to(device).eval()
    with torch.no_grad():
        result = model(torch.from_numpy(target_history).to(device),
                       torch.from_numpy(defender_history).to(device),
                       torch.full((1, 9), 1 / 9, device=device), cov_map=cov_map)
    candidates = result["candidate_xyz"].detach().cpu()
    causes = cov_map.classify_candidates(candidates)
    output.mkdir(parents=True, exist_ok=True)
    report = {"purpose": "cov map geometry probe; predictor weights are untrained",
              "map": cov_map.describe(), "target_start_xyz_m": start.tolist(),
              "defender_start_xyz_m": defender.tolist(),
              "target_speed_mps": speed,
              "candidate_acceleration_mps2": float(cov_config.get("target_accel", 1.0)),
              "candidate_valid": result["candidate_valid"][0].cpu().tolist(),
              "candidate_kept": result["candidate_kept"][0].cpu().tolist(),
              "building_collision": causes["building"][0].cpu().tolist(),
              "boundary_collision": causes["boundary"][0].cpu().tolist(),
              "fallback": bool(result["candidate_fallback"][0].cpu()),
              "valid_count": int(result["candidate_valid"][0].sum().cpu())}
    if live_resets:
        report["live_reset_probe"] = _probe_env_resets(
            cov_config_path, cov_config, cov_map, model_config, live_resets, seed)
    (output / "cov_map_probe.json").write_text(json.dumps(report, indent=2),
                                                  encoding="utf-8")
    _plot_probe(cov_map, candidates[0].numpy(), start, report["candidate_valid"],
                output / "cov_map_probe.png")
    return report


def _probe_env_resets(config_path: Path, cov_config: dict, static_map: CovMap,
                      model_config: dict, count: int, seed: int) -> dict:
    if count < 0:
        raise ValueError("live_resets must be nonnegative")
    repo_root = config_path.resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import (
        UAVPursuitCoverage3DEnv)

    np.random.seed(seed)
    env = UAVPursuitCoverage3DEnv(SimpleNamespace(**cov_config))
    valid_counts = []
    snapshot_matches = []
    for _ in range(count):
        env.reset()
        snapshot = CovMap.from_env(env)
        snapshot_matches.append(np.array_equal(snapshot.buildings, static_map.buildings))
        candidates = primitive_candidates(
            torch.from_numpy(env.target_position[None].copy()),
            torch.from_numpy(env.target_velocity[None].copy()),
            model_config["future_steps"], model_config["dt"],
            float(cov_config.get("target_accel", 1.0)))
        valid_counts.append(int(snapshot.candidate_valid(candidates)[0].sum()))
    return {"resets": count, "valid_candidates_per_reset": valid_counts,
            "all_snapshots_match_config_map": all(snapshot_matches),
            "fully_feasible_resets": sum(value == 9 for value in valid_counts),
            "history_probe": _probe_real_history(cov_config, model_config, seed)}


def _probe_real_history(cov_config: dict, model_config: dict, seed: int) -> dict:
    """Run cov physics at 0.1 s and exercise the predictor's live map input."""
    from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import (
        UAVPursuitCoverage3DEnv)

    settings = dict(cov_config)
    settings["decision_dt"] = model_config["dt"]
    settings["physics_dt"] = model_config["dt"]
    settings["episode_length"] = max(int(settings.get("episode_length", 400)), 100)
    np.random.seed(seed)
    env = UAVPursuitCoverage3DEnv(SimpleNamespace(**settings))
    env.reset()
    cov_map = CovMap.from_env(env)
    target_states, defender_states = [], []
    for step in range(80):
        target_states.append(np.concatenate((env.target_position, env.target_velocity)))
        defender_states.append(np.concatenate((env.uav_positions[0], env.uav_velocities[0])))
        if step == 79:
            break
        _, _, terminated, truncated, _ = env.step(
            {agent: np.zeros(3, np.float32) for agent in env.agents})
        if truncated or any(terminated.values()):
            return {"observed_steps": len(target_states), "terminated_early": True}
    target = torch.from_numpy(np.asarray(target_states, np.float32)[None])
    defender = torch.from_numpy(np.asarray(defender_states, np.float32)[None])
    torch.manual_seed(seed)
    model = TrajectoryPredictor(**model_config).eval()
    with torch.no_grad():
        result = model(target, defender, torch.full((1, 9), 1 / 9), cov_map=cov_map)
    return {"observed_steps": 80, "terminated_early": False,
            "dt_s": model_config["dt"],
            "action_policy": "zero acceleration commands",
            "decision_dt_override_s": model_config["dt"],
            "target_last_xyz_m": target[0, -1, :3].tolist(),
            "valid_candidates": int(result["candidate_valid"][0].sum()),
            "fallback": bool(result["candidate_fallback"][0]),
            "note": "real cov history, untrained predictor, geometry check only"}


def _plot_probe(cov_map: CovMap, candidates: np.ndarray, start: np.ndarray,
                valid: list[bool], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(12, 5))
    for xmin, xmax, ymin, ymax, top in cov_map.buildings:
        ax_xy.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                  facecolor="0.65", edgecolor="0.25", alpha=0.7))
        if ymin <= start[1] <= ymax:
            ax_xz.add_patch(Rectangle((xmin, cov_map.z_min), xmax - xmin,
                                      top - cov_map.z_min, facecolor="0.65",
                                      edgecolor="0.25", alpha=0.7))
    for index, path in enumerate(candidates):
        points = np.vstack((start, path))
        color = "green" if valid[index] else "crimson"
        ax_xy.plot(points[:, 0], points[:, 1], color=color,
                   linewidth=2 if index == 0 else 1, alpha=0.85,
                   label=f"{index}: {'valid' if valid[index] else 'blocked'}")
        ax_xz.plot(points[:, 0], points[:, 2], color=color,
                   linewidth=2 if index == 0 else 1, alpha=0.85)
    ax_xy.scatter([start[0]], [start[1]], marker="*", s=130, color="navy", zorder=5)
    ax_xz.scatter([start[0]], [start[2]], marker="*", s=130, color="navy", zorder=5)
    for ax in (ax_xy, ax_xz):
        ax.set_xlim(max(0, start[0] - 50), min(cov_map.map_size, start[0] + 70))
        ax.set_xlabel("x (m)")
    ax_xy.set_ylim(max(0, start[1] - 50), min(cov_map.map_size, start[1] + 50))
    ax_xy.set_aspect("equal")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("Top view")
    ax_xy.legend(fontsize=7, ncol=2, loc="upper right")
    ax_xz.set_ylim(max(cov_map.z_min, start[2] - 30),
                   min(cov_map.z_max, start[2] + 35))
    ax_xz.set_ylabel("altitude z (m)")
    ax_xz.set_title("Side view at probe y")
    fig.suptitle(f"cov {cov_map.mode} map: green valid, red blocked")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
