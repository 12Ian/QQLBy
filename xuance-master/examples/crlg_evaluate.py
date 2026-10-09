"""Evaluate trained midcourse MADDPG with the final 5 s flown by PNG."""

import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
from xuance import get_runner
from xuance.environment.multi_agent_env.crlg_3d import CRLG3DEnv


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--team-size", type=int, choices=(3, 4, 5, 6))
    parser.add_argument("--target-mode", choices=("CV", "CA", "Singer"))
    parser.add_argument("--output", default="outputs/results/crlg_3d/evaluation.json")
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--save-projection-geometry", action="store_true")
    parser.add_argument("--episode-start", type=int, default=0)
    args = parser.parse_args()
    overrides = argparse.Namespace(
        algo="maddpg", env="crlg_3d", env_id="crlg_3d",
        device=args.device, parallels=1, seed=args.seed, env_seed=args.seed,
        evaluation_png=True, evaluation_noise_scale=args.noise_scale,
        team_size=args.team_size, target_mode=args.target_mode)
    runner = get_runner(algo="maddpg", env="crlg_3d", env_id="crlg_3d",
                        parser_args=overrides)
    env = None
    try:
        model_path = Path(args.model).resolve()
        runner.agent.load_model(str(model_path))
        env = CRLG3DEnv(runner.config)
        records = []
        for episode in range(args.episode_start, args.episode_start + args.episodes):
            env.rng = np.random.default_rng(args.seed + episode)
            obs, _ = env.reset()
            initial_target_position = env.target.position.tolist()
            initial_pursuer_positions = [state.position.tolist()
                                         for state in env.pursuers[:env.active_count]]
            done = False
            reward_sum = 0.
            phases = []
            coverages = []
            area_ratios = []
            trajectory = []
            while not done:
                action = runner.agent.action([obs], test_mode=True)["actions"][0]
                obs, rewards, terminated, truncated, info = env.step(action)
                if args.save_trajectories:
                    step_record = {
                        "time_s": float(env.elapsed),
                        "target_position_m": env.target.position.tolist(),
                        "target_yaw_rad": float(env.target.yaw),
                        "pursuer_positions_m": [state.position.tolist()
                                                for state in env.pursuers[:env.active_count]],
                        "pursuer_yaws_rad": [float(state.yaw)
                                             for state in env.pursuers[:env.active_count]],
                        "actions_2d": {key: np.asarray(action[key]).tolist()
                                       for key in env.agents[:env.active_count]},
                        "coverage_probability": info["coverage_probability"],
                        "predicted_projected_miss_m": info["predicted_projected_miss_m"],
                        "actual_miss_m": info["terminal_miss_m"],
                        "guidance_phase": info["guidance_phase"],
                        "estimated_encounter_time_s": info["encounter_time"],
                    }
                    if args.save_projection_geometry:
                        step_record["projection_geometry"] = env.projection_geometry()
                    trajectory.append(step_record)
                reward_sum += sum(rewards.values()) / env.active_count
                phases.append(info["guidance_phase"])
                coverages.append(info["coverage_probability"])
                ideal = info["ideal_radius_m"]
                if ideal > 1e-6:
                    area_ratios.append(float(np.mean(np.square(
                        info["calibrated_radius_m"])) / ideal ** 2))
                done = bool(terminated[env.agents[0]] or truncated)
            valid = not info["invalid_episode"]
            miss = min(info["terminal_miss_m"])
            closest = min(info["closest_approach_m"])
            records.append({
                "episode": episode, "valid": valid,
                "invalid_reason": info["invalid_reason"],
                "termination_reason": info["termination_reason"],
                "episode_steps": info["episode_step"],
                "active_agents": info["active_agents"],
                "target_mode": info["target_mode"],
                "initial_target_position_m": initial_target_position,
                "initial_pursuer_positions_m": initial_pursuer_positions,
                "initial_pursuer_speed_mps": info["initial_pursuer_speed_mps"],
                "initial_target_speed_mps": info["initial_target_speed_mps"],
                "encounter_time_s": info["encounter_time"],
                "initial_encounter_time_s": info["initial_encounter_time"],
                "terminal_min_miss_m": miss,
                "closest_approach_m": closest,
                "terminal_each_miss_m": info["terminal_miss_m"],
                "mean_agent_return": reward_sum,
                "command_energy": info["episode_command_energy"],
                "reward_components": info["episode_sub_rewards"],
                "png_steps": phases.count("PNG"),
                "terminal_phase_locked": info["terminal_phase_locked"],
                "coverage_mean": float(np.mean(coverages)),
                "coverage_saturated_fraction": float(np.mean(np.asarray(coverages) > 0.99)),
                "calibrated_to_ideal_area_ratio_mean": (
                    float(np.mean(area_ratios)) if area_ratios else None),
            })
            if args.save_trajectories:
                records[-1]["trajectory"] = trajectory
        valid = [r for r in records if r["valid"]]
        thresholds = (1., 2., 5., 10.)
        strata = {}
        for size in (3, 4, 5, 6):
            for mode in ("CV", "CA", "Singer"):
                subset = [r for r in records if r["active_agents"] == size and
                          r["target_mode"] == mode]
                valid_subset = [r for r in subset if r["valid"]]
                strata[f"{size}_{mode}"] = {
                    "episodes": len(subset),
                    "valid_episodes": len(valid_subset),
                    "median_terminal_miss_m": (
                        float(np.median([r["terminal_min_miss_m"]
                                         for r in valid_subset])) if valid_subset else None),
                    "success_rate_5m_valid": (
                        sum(r["terminal_min_miss_m"] <= 5. for r in valid_subset) /
                        len(valid_subset) if valid_subset else None),
                }
        summary = {
            "episodes": len(records), "valid_episodes": len(valid),
            "invalid_episodes": len(records) - len(valid),
            "invalid_reasons": {reason: sum(r["invalid_reason"] == reason
                                    for r in records)
                                for reason in ("pursuer_boundary", "target_boundary")},
            "termination_reasons": {reason: sum(r["termination_reason"] == reason
                                         for r in records)
                                    for reason in ("encounter", "max_steps",
                                                   "pursuer_boundary", "target_boundary")},
            "thresholds_m": list(thresholds),
            "valid_success_rate": {
                str(radius): (sum(r["terminal_min_miss_m"] <= radius
                                  for r in valid) / len(valid) if valid else None)
                for radius in thresholds},
            "all_episode_success_rate": {
                str(radius): sum(r["valid"] and r["terminal_min_miss_m"] <= radius
                                 for r in records) / len(records)
                for radius in thresholds},
            "valid_median_terminal_miss_m": (
                float(np.median([r["terminal_min_miss_m"] for r in valid]))
                if valid else None),
            "by_team_size_and_target_mode": strata,
            "evaluation_noise_scale": args.noise_scale,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        repository = Path(__file__).resolve().parents[1]
        metadata = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "model_path": str(model_path),
            "model_sha256": file_sha256(model_path),
            "environment_sha256": file_sha256(repository /
                "xuance/environment/multi_agent_env/crlg_3d/env.py"),
            "evaluation_script_sha256": file_sha256(__file__),
            "config_sha256": file_sha256(repository /
                "xuance/configs/maddpg/crlg_3d.yaml"),
            "settings": {name: getattr(runner.config, name) for name in (
                "decision_dt", "physics_dt", "actuator_lag", "max_speed",
                "mid_accel", "terminal_accel", "pursuer_jerk_max",
                "terminal_jerk_max", "target_jerk_max", "yaw_rate_max",
                "terminal_seconds", "episode_length",
                "navigation_constant",
                "pursuer_initial_speed_min", "pursuer_initial_speed_max",
                "target_initial_speed_ratio", "target_accel", "singer_sigma",
                "position_sigma", "velocity_sigma", "prediction_accel_sigma",
                "reward_coverage", "reward_prediction", "reward_energy",
                "reward_terminal", "coverage_samples")},
        }
        output.write_text(json.dumps({"metadata": metadata,
                                      "summary": summary, "episodes": records},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if env is not None:
            env.close()
        if getattr(runner, "envs", None) is not None:
            runner.envs.close()


if __name__ == "__main__":
    main()
