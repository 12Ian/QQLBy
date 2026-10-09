"""Train the 3D CRL-G midcourse policy with MADDPG."""

import argparse
import json
from copy import copy
from pathlib import Path

import numpy as np
from xuance import get_runner
from xuance.environment.multi_agent_env.crlg_3d import CRLG3DEnv


def evaluate_success(agent, config, episodes, seed):
    eval_config = copy(config)
    eval_config.evaluation_png = True
    env = CRLG3DEnv(eval_config)
    misses = []
    valid = 0
    successes = 0
    invalid = 0
    timeouts = 0
    try:
        for episode in range(episodes):
            env.rng = np.random.default_rng(seed + episode)
            obs, _ = env.reset()
            while True:
                actions = agent.action([obs], test_mode=True)["actions"][0]
                obs, _, terminated, truncated, info = env.step(actions)
                if terminated[env.agents[0]] or truncated:
                    break
            if info["invalid_episode"]:
                invalid += 1
                continue
            valid += 1
            miss = min(info["terminal_miss_m"])
            misses.append(miss)
            successes += miss <= 5.
            timeouts += info["termination_reason"] == "max_steps"
    finally:
        env.close()
    return {
        "episodes": episodes,
        "valid_episodes": valid,
        "invalid_episodes": invalid,
        "successes_5m": successes,
        "success_rate_5m_all": successes / episodes,
        "success_rate_5m_valid": successes / valid if valid else None,
        "median_terminal_miss_m_valid": float(np.median(misses)) if misses else None,
        "max_steps_valid": timeouts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--parallels", type=int, default=8)
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--eval-interval", type=int, default=0,
                        help="Environment steps between PNG success-rate evaluations; 0 disables them")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-output")
    args = parser.parse_args()
    overrides = argparse.Namespace(
        algo="maddpg", env="crlg_3d", env_id=f"crlg_3d_{args.run_name}",
        device=args.device, running_steps=args.steps, seed=args.seed,
        env_seed=args.seed, parallels=args.parallels, evaluation_png=False)
    runner = get_runner(algo="maddpg", env="crlg_3d", env_id="crlg_3d",
                        parser_args=overrides)
    try:
        if args.eval_interval <= 0:
            runner.run(mode="train")
        else:
            if args.steps % runner.n_envs or args.eval_interval % runner.n_envs:
                raise ValueError("steps and eval-interval must be divisible by parallels")
            if args.eval_episodes <= 0 or not args.eval_output:
                raise ValueError("periodic evaluation needs positive episodes and --eval-output")
            output = Path(args.eval_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            remaining = args.steps // runner.n_envs
            chunk_size = args.eval_interval // runner.n_envs
            while remaining:
                chunk = min(remaining, chunk_size)
                runner.agent.train(train_steps=chunk)
                remaining -= chunk
                step = runner.agent.current_step
                runner.agent.save_model(model_name=f"step_{step}.pth")
                result = evaluate_success(runner.agent, runner.config,
                                          args.eval_episodes, seed=2026)
                result["step"] = step
                with output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                runner.agent.log_infos({
                    "Eval-CRLG/success_rate_5m_all": result["success_rate_5m_all"],
                    "Eval-CRLG/invalid_episode_rate": result["invalid_episodes"] /
                    result["episodes"],
                }, step)
                print("Periodic PNG evaluation:", json.dumps(result, ensure_ascii=False),
                      flush=True)
            runner.agent.save_model(model_name="final_train_model.pth")
    finally:
        if args.eval_interval > 0:
            runner._finalize()


if __name__ == "__main__":
    main()
