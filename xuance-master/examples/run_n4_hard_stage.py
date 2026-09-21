import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from xuance import get_runner


def checkpoint_key(summary):
    validate_summary(summary)
    return (float(summary["success_rate"]), -float(summary["collision_rate"]),
            float(summary["mean_episode_return"]))


def validate_summary(summary):
    for key in ("success_rate", "collision_rate", "mean_episode_return"):
        if key not in summary:
            raise KeyError(key)
        if not math.isfinite(float(summary[key])):
            raise FloatingPointError(f"Nonfinite evaluator summary: {key}")
    for key in ("success_rate", "collision_rate"):
        if not 0.0 <= float(summary[key]) <= 1.0:
            raise ValueError(f"Evaluator summary out of range: {key}")


def assert_finite_metrics(value, path="train"):
    if isinstance(value, dict):
        for key, child in value.items(): assert_finite_metrics(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value): assert_finite_metrics(child, f"{path}[{index}]")
    elif torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()): raise FloatingPointError(f"Nonfinite training metric: {path}")
    elif isinstance(value, (int, float, np.number)) and not np.isfinite(value):
        raise FloatingPointError(f"Nonfinite training metric: {path}")


def claim_output(out_dir):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    claim = out_dir / ".n4_hard_stage.claim"
    if claim.exists():
        raise FileExistsError(f"Output run claim already active: {out_dir}")
    if any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty out-dir without resume semantics: {out_dir}")
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f"Output run claim already active: {out_dir}") from exc
    os.close(fd)
    return claim


def current_step(agent):
    return int(agent.current_step)


def main(args):
    if args.steps <= 0 or args.eval_every <= 0 or args.parallels <= 0:
        raise ValueError("steps, eval-every and parallels must all be positive")
    if args.steps % args.eval_every or args.eval_every % args.parallels:
        raise ValueError("steps must be divisible by eval-every, and eval-every by parallels")
    claim = claim_output(args.out_dir)
    out_dir = claim.parent
    try:
        config = Namespace(algo="maddpg", env="uav_pursuit_apollonius_3d",
        env_id=f"apollonius_3d_{args.name}", device=args.device, num_agents=4,
        seed=args.seed, parallels=args.parallels, running_steps=args.steps,
        curriculum_enabled=False, target_max_speed=6.0, closure_weight=8.0,
        w_pos=0.6, w_gap=0.8, separation_weight=4.0, separation_distance=20.0,
        terminate_on_collision=True, spawn_safety_enabled=True,
        start_noise=.10, end_noise=.01)
        if args.randomize_density: config.randomize_density = args.randomize_density.split(",")
        else: config.building_mode = args.building_mode
        (out_dir / "training_config.json").write_text(json.dumps({**vars(config),
        "load_from": str(Path(args.load_from).resolve()), "eval_every": args.eval_every,
        "eval_mode": args.eval_mode, "eval_seed": args.eval_seed, "eval_k": args.eval_k},
            ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    except BaseException:
        if claim.exists(): claim.unlink()
        raise
    runner = None
    primary_error = None
    try:
        runner = get_runner(algo=config.algo, env=config.env, env_id=config.env_id, parser_args=config)
        checkpoint = torch.load(args.load_from, map_location=args.device, weights_only=True)
        runner.agent.policy.load_state_dict(checkpoint["policy"], strict=False)
        best_key = None
        metrics_path = out_dir / "checkpoint_metrics.jsonl"
        evaluator = Path(__file__).with_name("evaluate_3d.py")
        for environment_step in range(args.eval_every, args.steps + 1, args.eval_every):
            step_before = current_step(runner.agent)
            train_info = runner.agent.train(train_steps=args.eval_every // args.parallels)
            assert_finite_metrics(train_info)
            step_after = current_step(runner.agent)
            if step_after - step_before != args.eval_every:
                raise RuntimeError(f"Interval step delta {step_after - step_before} != {args.eval_every}")
            snapshot = out_dir / f"snapshot_{environment_step}.pth"
            runner.agent.save_model(snapshot.name, model_path=str(out_dir))
            if not snapshot.is_file():
                raise FileNotFoundError(f"Missing snapshot artifact: {snapshot}")
            result_path = out_dir / f"eval_{environment_step}.json"
            trace_path = out_dir / f"eval_{environment_step}.jsonl"
            command = [sys.executable, str(evaluator), "--policy", "model", "--model-path", str(snapshot),
                "--seed", str(args.eval_seed), "--building-mode", args.eval_mode, "--num-agents", "4",
                "--target-max-speed", "6", "--k", str(args.eval_k), "--closure-weight", "8",
                "--w-pos", "0.6", "--w-gap", "0.8", "--separation-weight", "4",
                "--separation-distance", "20", "--spawn-safety", "on", "--collision-mode", "hard",
                "--out", str(result_path), "--trace-out", str(trace_path)]
            subprocess.run(command, cwd=os.getcwd(), check=True)
            if not result_path.is_file():
                raise FileNotFoundError(f"Missing evaluator result artifact: {result_path}")
            if not trace_path.is_file():
                raise FileNotFoundError(f"Missing evaluator trace artifact: {trace_path}")
            summary = json.loads(result_path.read_text(encoding="utf-8"))
            validate_summary(summary)
            row = {"environment_step": environment_step, "step_before": step_before,
                "step_after": step_after, "step_delta": step_after-step_before,
                "snapshot": str(snapshot), "summary": summary}
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            key = checkpoint_key(summary)
            if best_key is None or key > best_key:
                best_key = key
                shutil.copy2(snapshot, out_dir / "selected_model.pth")
                (out_dir / "selected_summary.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_error = None
        if runner is not None and getattr(runner, "envs", None) is not None:
            try: runner.envs.close()
            except BaseException as exc: cleanup_error = exc
        if runner is not None and getattr(runner, "agent", None) is not None:
            try: runner.agent.finish()
            except BaseException as exc: cleanup_error = cleanup_error or exc
        if claim.exists():
            try: claim.unlink()
            except BaseException as exc: cleanup_error = cleanup_error or exc
        if primary_error is None and cleanup_error is not None: raise cleanup_error
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True); parser.add_argument("--seed", type=int, required=True)
    density = parser.add_mutually_exclusive_group(required=True)
    density.add_argument("--building-mode"); density.add_argument("--randomize-density")
    parser.add_argument("--steps", type=int, required=True); parser.add_argument("--load-from", required=True)
    parser.add_argument("--parallels", type=int, default=8); parser.add_argument("--eval-every", type=int, default=250000)
    parser.add_argument("--eval-mode", required=True); parser.add_argument("--eval-seed", type=int, default=17)
    parser.add_argument("--eval-k", type=int, default=20); parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
