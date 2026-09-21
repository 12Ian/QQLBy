"""Measure pursuer-pursuer proximity on a trained policy, and re-evaluate across obstacle
densities. Both are pure evaluation -- no training.

Why the proximity part exists: the crash test only ever sees one uav's own state, so contact
between pursuers has never been terminal and has never entered collision_rate. Whether that
omission matters is an empirical question, and this answers it with the distribution of the
smallest pursuer-pursuer gap per episode rather than an assumption. A uav has radius 0.5 m, so
physical contact means centres within 1 m.

Usage:
  python examples/measure_uav_gap.py --model-path <pth> --num-agents 3 --target-max-speed 11 \
      --surround-spawn --densities medium,empty,open,complex --k 30 --out out.json
"""
import os
import sys
import json
import math
import argparse
import numpy as np

from xuance import get_runner
from xuance.environment.multi_agent_env.uav_pursuit_apollonius_3d import UAVPursuitApollonius3DEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_density(agent, cfg_proto, density, k, seed, surround, num_agents, vmax, vmin):
    """One density: returns capture stats plus the pursuer-gap distribution."""
    p = argparse.Namespace(**vars(cfg_proto))
    p.building_mode = density
    np.random.seed(seed)
    env = UAVPursuitApollonius3DEnv(p)

    succ = coll = to = 0
    ep_min_gaps = []
    contact_eps = 0          # episodes where two pursuers came within uav_collision_distance
    near5_eps = 0            # episodes where they came within 5 m
    for ep in range(k):
        np.random.seed((int(seed) + ep) % (2 ** 32))
        obs, _ = env.reset()
        done = False
        caught = crashed = False
        while not done:
            acts = agent.action([obs], test_mode=True)["actions"][0]
            obs, _, term, trunc, info = env.step(acts)
            caught = caught or bool(info.get("is_success", False))
            crashed = crashed or bool(info.get("is_collision", False))
            done = bool(term[env.agents[0]]) or bool(trunc)
        g = float(info.get("episode_min_uav_gap", float("inf")))
        if math.isfinite(g):
            ep_min_gaps.append(g)
            if g <= env.uav_collision_distance:
                contact_eps += 1
            if g <= 5.0:
                near5_eps += 1
        if caught:
            succ += 1
        elif crashed:
            coll += 1
        else:
            to += 1

    gaps = np.asarray(ep_min_gaps, dtype=float) if ep_min_gaps else np.asarray([np.nan])
    return {
        "density": density,
        "episodes": k,
        "success_rate": succ / k,
        "collision_rate": coll / k,
        "timeout_rate": to / k,
        "uav_gap_min": float(np.nanmin(gaps)),
        "uav_gap_p05": float(np.nanpercentile(gaps, 5)),
        "uav_gap_median": float(np.nanmedian(gaps)),
        "contact_threshold_m": env.uav_collision_distance,
        "episodes_with_contact": contact_eps,
        "contact_episode_rate": contact_eps / k,
        "episodes_within_5m": near5_eps,
        "within_5m_episode_rate": near5_eps / k,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--algo", default="maddpg")
    ap.add_argument("--num-agents", type=int, default=3)
    ap.add_argument("--target-max-speed", type=float, default=11.0)
    ap.add_argument("--target-min-speed", type=float, default=8.0)
    ap.add_argument("--surround-spawn", action="store_true")
    ap.add_argument("--densities", default="medium")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = argparse.Namespace(algo=args.algo, env="uav_pursuit_apollonius_3d",
                           env_id="apollonius_3d", device=args.device)
    p.parallels = 1
    p.seed = args.seed
    p.curriculum_enabled = False
    p.building_mode = "medium"
    p.num_agents = args.num_agents
    p.target_max_speed = args.target_max_speed
    p.target_min_speed = args.target_min_speed
    if args.surround_spawn:
        p.surround_spawn = True

    np.random.seed(args.seed)
    runner = get_runner(algo=args.algo, env="uav_pursuit_apollonius_3d",
                        env_id="apollonius_3d", parser_args=p)
    agent = runner.agent
    agent.load_model(os.path.abspath(args.model_path))
    cfg = runner.config

    rows = [run_density(agent, cfg, d.strip(), args.k, args.seed,
                        args.surround_spawn, args.num_agents,
                        args.target_max_speed, args.target_min_speed)
            for d in args.densities.split(",") if d.strip()]

    print(f"{'density':>9} {'succ':>7} {'coll':>7} {'to':>7} "
          f"{'gap_min':>8} {'gap_p05':>8} {'gap_med':>8} {'contact%':>9} {'<5m%':>7}")
    for r in rows:
        print(f"{r['density']:>9} {r['success_rate']*100:>6.1f}% {r['collision_rate']*100:>6.1f}% "
              f"{r['timeout_rate']*100:>6.1f}% {r['uav_gap_min']:>8.2f} {r['uav_gap_p05']:>8.2f} "
              f"{r['uav_gap_median']:>8.2f} {r['contact_episode_rate']*100:>8.1f}% "
              f"{r['within_5m_episode_rate']*100:>6.1f}%")
    print(f"(gaps in metres; contact threshold = {rows[0]['contact_threshold_m']:.1f} m)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"model_path": args.model_path, "num_agents": args.num_agents,
                       "target_max_speed": args.target_max_speed,
                       "surround_spawn": bool(args.surround_spawn),
                       "k": args.k, "seed": args.seed, "results": rows}, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
