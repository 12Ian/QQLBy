"""Evaluate a frozen baseline checkpoint on explicitly chosen held-out environment seeds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml
from xuance import get_runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--env-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--parallels", type=int, default=8)
    parser.add_argument("--level", type=int, choices=range(5), default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.episodes, args.parallels) < 1 or args.env_seed < 0:
        parser.error("Episode count and parallels must be positive; seed must be nonnegative")
    if args.output.exists():
        parser.error("Evaluation output already exists")
    config = yaml.safe_load(args.config.read_text())
    method = config["agent"].lower()
    name = f"holdout_{method}_{args.env_seed}_{args.level}_{os.getpid()}"
    overrides = argparse.Namespace(
        env_id=name, device=args.device, seed=1, env_seed=args.env_seed,
        parallels=min(args.parallels, args.episodes), test_episode=args.episodes,
        curriculum_start_level=args.level, test_curriculum_level=args.level,
        baseline_init_actor=None, terminate_on_boundary_collision=False,
        terminate_on_collision=False, terminate_on_building_collision=False)
    runner = get_runner(method, "uav_pursuit_coverage_3d", name,
                        config_path=str(args.config.resolve()), parser_args=overrides)
    envs = runner.agent.train_envs
    started = time.time()
    try:
        checkpoint = torch.load(args.checkpoint, map_location=runner.agent.device, weights_only=False)
        runner.agent.policy.load_state_dict(checkpoint["policy"])
        assert all(torch.isfinite(value).all().item() for value in checkpoint["policy"].values())
        runner.agent.policy.eval()
        envs.env_method("seed")
        scores = runner.agent.run_episodes(n_episodes=args.episodes, run_envs=envs,
                                          test_mode=True, close_envs=False, curriculum_level=args.level)
        metrics = dict(runner.agent.last_test_metrics)
        assert metrics["episodes"] == args.episodes
        environment = envs.env_method("get_curriculum_info")
        assert all(item["curriculum_level"] == args.level for item in environment)
        report = dict(checkpoint=str(args.checkpoint.resolve()),
                      checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      config=str(args.config.resolve()), env_seed=args.env_seed,
                      level=args.level, exploration_noise=False, collision_termination=False,
                      metrics=metrics, mean_return=float(np.mean(scores)), environment=environment,
                      elapsed_seconds=round(time.time() - started, 3))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print(json.dumps({"output": str(args.output), "metrics": metrics}, ensure_ascii=False), flush=True)
    finally:
        envs.close()
        if not runner.agent.use_wandb:
            runner.agent.writer.close()


if __name__ == "__main__":
    main()
