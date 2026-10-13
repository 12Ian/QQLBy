"""按固定随机种子评估当前 Coverage 三维围捕模型。"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from xuance import get_runner
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _terminal_guidance_action(env, index):
    """仅供对照评估：近距按目标速度和预瞄位置跟踪。"""
    position = env.uav_positions[index]
    velocity = env.uav_velocities[index]
    distance = float(np.linalg.norm(env.target_position - position))
    lead_time = min(10.0, distance / max(env.uav_max_speed - env.target_speed, 1.0))
    goal = env.target_position + lead_time * env.target_velocity
    desired_velocity = (goal - position) * (0.5 if distance < 25.0 else 0.25)
    if distance < 25.0:
        desired_velocity += env.target_velocity
    desired_velocity /= max(1.0, float(np.linalg.norm(desired_velocity)) / env.uav_max_speed)
    command = (desired_velocity - velocity) / env.acceleration_lag
    return np.clip(command / env.max_accel, -1.0, 1.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-json")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--building-mode", default="open")
    parser.add_argument("--target-min-speed", type=float, default=3.0)
    parser.add_argument("--target-max-speed", type=float, default=5.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--legacy-obs89", action="store_true",
                        help="将旧版 89 维 Actor 的权重映射到新增保持进度的 90 维观测")
    parser.add_argument("--terminal-guidance-radius", type=float, default=0.0,
                        help="对照实验：最近追捕机进入此距离后切换速度跟踪；0 表示纯策略")
    parser.add_argument("--action-noise", type=float, default=0.0,
                        help="对照实验：按训练方式给每个动作分量添加高斯噪声")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.terminal_guidance_radius < 0:
        parser.error("--terminal-guidance-radius must be nonnegative")
    if args.action_noise < 0:
        parser.error("--action-noise must be nonnegative")

    model_path = Path(args.model_path).resolve()
    model_sha256 = _sha256(model_path)
    config_path = Path(args.config_json).resolve() if args.config_json else model_path.parent.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    config.update({
        "algo": "maddpg", "env": "uav_pursuit_coverage_3d", "env_id": "coverage_3d",
        "device": args.device, "parallels": 1, "render": False,
        "curriculum_enabled": False, "surround_spawn": True,
        "num_agents": args.num_agents, "building_mode": args.building_mode,
        "target_min_speed": args.target_min_speed, "target_max_speed": args.target_max_speed,
    })
    runner = get_runner(algo="maddpg", env="uav_pursuit_coverage_3d",
                        env_id="coverage_3d", parser_args=argparse.Namespace(**config))
    agent = runner.agent
    if args.legacy_obs89:
        old_policy = torch.load(model_path, map_location="cpu", weights_only=False)["policy"]
        new_policy = agent.policy.state_dict()
        for name, value in old_policy.items():
            if name not in new_policy:
                continue
            if name.startswith(("actor.", "target_actor.")) and name.endswith("model.0.weight"):
                value = torch.cat((value[:, :10], torch.zeros_like(value[:, :1]), value[:, 10:]), dim=1)
            elif name.endswith("obstacle_gat.q.weight"):
                value = torch.cat((value[:, :10], torch.zeros_like(value[:, :1]), value[:, 10:]), dim=1)
            if value.shape == new_policy[name].shape:
                new_policy[name] = value
        agent.policy.load_state_dict(new_policy)
    else:
        agent.load_model(str(model_path))
    env = UAVPursuitCoverage3DEnv(runner.config)
    records = []
    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        np.random.seed(episode_seed)
        obs, _ = env.reset()
        collision = building_collision = boundary_collision = uav_conflict = False
        min_distance = float("inf")
        relative_speed_at_closest = float("nan")
        max_hold_seconds = 0.0
        target_speeds = []
        terminal_guidance_engaged = False
        done = False
        while not done:
            actions = agent.action([obs], test_mode=True)["actions"][0]
            if args.action_noise > 0.0:
                for name in env.agents:
                    actions[name] = np.asarray(actions[name]) + np.random.normal(
                        0.0, args.action_noise, size=env.action_space[name].shape)
            if args.terminal_guidance_radius > 0.0:
                distances = np.linalg.norm(env.uav_positions - env.target_position[None, :], axis=1)
                nearest = int(np.argmin(distances))
                if distances[nearest] < args.terminal_guidance_radius:
                    terminal_guidance_engaged = True
                    actions[env.agents[nearest]] = _terminal_guidance_action(env, nearest)
            obs, _, terminated, truncated, info = env.step(actions)
            target_speeds.append(float(np.linalg.norm(env.target_velocity)))
            step_distance = float(info["min_target_distance"])
            if step_distance < min_distance:
                min_distance = step_distance
                closest = int(np.argmin(np.linalg.norm(
                    env.uav_positions - env.target_position[None, :], axis=1)))
                relative_speed_at_closest = float(np.linalg.norm(
                    env.uav_velocities[closest] - env.target_velocity))
            max_hold_seconds = max(max_hold_seconds, float(np.max(env.capture_hold_times)))
            collision |= bool(info["is_collision"])
            building_collision |= bool(info["is_building_collision"])
            boundary_collision |= bool(info["is_boundary_collision"])
            uav_conflict |= bool(info["is_uav_collision"])
            done = bool(terminated[env.agents[0]]) or bool(truncated)
        diagnostics = info["infos"]
        records.append({
            "seed": episode_seed, "success": bool(info["is_success"]),
            "capture_type": info["capture_type"], "capture_agent": info["capture_agent"],
            "steps": int(env._episode_step), "min_target_distance": min_distance,
            "terminal_guidance_engaged": terminal_guidance_engaged,
            "relative_speed_at_closest": relative_speed_at_closest,
            "max_hold_seconds": max_hold_seconds,
            "collision": collision, "building_collision": building_collision,
            "boundary_collision": boundary_collision, "uav_conflict": uav_conflict,
            "target_speed_mean": float(np.mean(target_speeds)),
            "coverage_mean": float(diagnostics["coverage_mean_episode"]),
            "preparation_mean": float(diagnostics["preparation_mean_episode"]),
        })

    def rate(key):
        return float(np.mean([record[key] for record in records]))

    summary = {
        "episodes": args.episodes, "seed_start": args.seed,
        "success_rate": rate("success"),
        "direct_capture_rate": float(np.mean([r["capture_type"] == "direct" for r in records])),
        "sustained_capture_rate": float(np.mean([r["capture_type"] == "sustained" for r in records])),
        "collision_rate": rate("collision"), "building_collision_rate": rate("building_collision"),
        "boundary_collision_rate": rate("boundary_collision"),
        "uav_conflict_rate": rate("uav_conflict"),
        "min_target_distance_mean": float(np.mean([r["min_target_distance"] for r in records])),
        "near_25m_rate": float(np.mean([r["min_target_distance"] <= 25.0 for r in records])),
        "near_15m_rate": float(np.mean([r["min_target_distance"] <= 15.0 for r in records])),
        "relative_speed_at_closest_mean": float(np.mean([
            r["relative_speed_at_closest"] for r in records])),
        "max_hold_seconds_mean": float(np.mean([r["max_hold_seconds"] for r in records])),
        "terminal_guidance_radius": args.terminal_guidance_radius,
        "terminal_guidance_engaged_rate": rate("terminal_guidance_engaged"),
        "action_noise_scale": args.action_noise,
        "target_speed_mean": float(np.mean([r["target_speed_mean"] for r in records])),
        "coverage_mean": float(np.mean([r["coverage_mean"] for r in records])),
        "model_sha256": model_sha256,
        "legacy_obs89_adapter": args.legacy_obs89,
        "environment_sha256": _sha256(inspect.getfile(UAVPursuitCoverage3DEnv)),
    }
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "episodes": records}, indent=2,
                                 ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
