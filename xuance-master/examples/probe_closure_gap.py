"""Measure how close Chapter 4 pursuers actually get to their targets.

Re-scoring Chapter 4 under Chapter 3's rule (physical reach within catch_radius) returned
per_target_capture_rate = 0.0 across 30 episodes and 2 targets -- not one target was ever
reached. That is either a scoring bug or the encircle-but-never-close failure Chapter 3 hit
before its closure weight was raised from 8 to 16, and the two are easy to tell apart: if the
closest approach piles up just outside the catch radius, the pursuers are parking on the
encirclement shell rather than closing, and nothing is wrong with the scoring.

Reports the distribution of closest approach per target, in units of the catch radius.
"""
import argparse
import json
import os

import numpy as np

from xuance import get_runner
from xuance.environment.multi_agent_env.uav_pursuit_apollonius_multitarget_3d import (
    UAVPursuitApolloniusMultiTarget3DEnv,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--algo", default="maddpg")
    ap.add_argument("--num-agents", type=int, default=6)
    ap.add_argument("--num-targets", type=int, default=2)
    ap.add_argument("--building-mode", default="medium")
    ap.add_argument("--target-max-speed", type=float, default=10.0)
    ap.add_argument("--target-min-speed", type=float, default=8.0)
    ap.add_argument("--apollonius-alloc", action="store_true")
    ap.add_argument("--no-dynamic-alloc", action="store_true")
    ap.add_argument("--strict-capture", action="store_true")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="closure_gap.json")
    args = ap.parse_args()

    env_name = "uav_pursuit_apollonius_multitarget_3d"
    p = argparse.Namespace(algo=args.algo, env=env_name,
                           env_id="apollonius_mt_3d", device=args.device)
    p.parallels = 1
    p.curriculum_enabled = False
    p.building_mode = args.building_mode
    p.num_agents = args.num_agents
    p.num_targets = args.num_targets
    p.target_max_speed = args.target_max_speed
    p.target_min_speed = args.target_min_speed
    if args.apollonius_alloc:
        p.apollonius_alloc = True
    if args.no_dynamic_alloc:
        p.dynamic_alloc = False
    if args.strict_capture:
        p.strict_capture = True

    runner = get_runner(algo=args.algo, env=env_name, env_id="apollonius_mt_3d", parser_args=p)
    agent = runner.agent
    agent.load_model(os.path.abspath(args.model_path))
    env = UAVPursuitApolloniusMultiTarget3DEnv(runner.config)
    cr = float(env.catch_radius)

    # Also record the success this rollout observes. If it disagrees with evaluate_mt.py on the
    # same model, the fault is in this probe rather than in the policy.
    closest, successes, steps = [], 0, []
    for ep in range(args.k):
        np.random.seed(args.seed + ep)
        obs, _ = env.reset()
        best = np.full(args.num_targets, np.inf)
        done, n, ok = False, 0, False
        while not done:
            d = np.linalg.norm(env.uav_positions[:, None, :] - env.target_positions[None, :, :],
                               axis=-1).min(axis=0)
            best = np.minimum(best, d)
            acts = agent.action([obs], test_mode=True)["actions"][0]
            obs, _, term, trunc, info = env.step(acts)
            n += 1
            inf0 = info[env.agents[0]] if isinstance(info, dict) and env.agents[0] in info else info
            if isinstance(inf0, dict) and inf0.get("is_success"):
                ok = True
            done = bool(term[env.agents[0]]) or bool(trunc[env.agents[0]])
        d = np.linalg.norm(env.uav_positions[:, None, :] - env.target_positions[None, :, :],
                           axis=-1).min(axis=0)
        best = np.minimum(best, d)          # the final step can be the capturing one
        closest.append(best.copy())
        successes += int(ok)
        steps.append(n)

    arr = np.concatenate(closest)
    ratio = arr / cr
    out = {
        "catch_radius": cr,
        "n_target_instances": int(arr.size),
        "closest_approach_m": {
            "min": float(arr.min()), "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)), "p75": float(np.percentile(arr, 75)),
            "max": float(arr.max()), "mean": float(arr.mean()),
        },
        "frac_within_catch_radius": float((arr <= cr).mean()),
        "frac_within_1_5x": float((arr <= 1.5 * cr).mean()),
        "frac_within_2x": float((arr <= 2.0 * cr).mean()),
        "median_in_catch_radii": float(np.median(ratio)),
        "probe_success_rate": float(successes / max(args.k, 1)),
        "mean_episode_steps": float(np.mean(steps)),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
