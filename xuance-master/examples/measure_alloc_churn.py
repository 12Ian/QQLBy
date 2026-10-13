"""Measure pursuer->target assignment churn, for the Chapter 4 dynamic-vs-static comparison.

Dynamic allocation re-optimises the assignment every step; static computes the same optimal
assignment once at reset and only reassigns pursuers whose target has died. The two measured
indistinguishably (65.6% vs 72.2%, p=0.334), so the published "dynamic ~2x static" claim does
not hold. If dynamic gains nothing, the natural mechanism is churn: re-optimising every step
lets a pursuer flip between targets as relative distances fluctuate, spending its effort on
switching instead of committing to one encirclement. This counts those switches directly.

Usage:
  python examples/measure_alloc_churn.py --model-path <pth> --num-agents 6 --num-targets 2 \
      [--no-dynamic-alloc] --k 30 --out churn.json
"""
import os, sys, json, argparse
import numpy as np

from xuance import get_runner
from xuance.environment.multi_agent_env.uav_pursuit_apollonius_multitarget_3d import (
    UAVPursuitApolloniusMultiTarget3DEnv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--num-agents", type=int, default=6)
    ap.add_argument("--num-targets", type=int, default=2)
    ap.add_argument("--building-mode", default="medium")
    ap.add_argument("--target-max-speed", type=float, default=10.0)
    ap.add_argument("--target-min-speed", type=float, default=8.0)
    ap.add_argument("--no-dynamic-alloc", action="store_true")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = argparse.Namespace(algo="maddpg", env="uav_pursuit_apollonius_multitarget_3d",
                           env_id="apollonius_mt_3d", device="cuda:0")
    p.parallels = 1
    p.seed = args.seed
    p.curriculum_enabled = False
    p.building_mode = args.building_mode
    p.num_agents = args.num_agents
    p.num_targets = args.num_targets
    p.target_max_speed = args.target_max_speed
    if args.no_dynamic_alloc:
        p.dynamic_alloc = False

    np.random.seed(args.seed)
    runner = get_runner(algo="maddpg", env="uav_pursuit_apollonius_multitarget_3d",
                        env_id="apollonius_mt_3d", parser_args=p)
    agent = runner.agent
    agent.load_model(os.path.abspath(args.model_path))
    np.random.seed(args.seed)
    env = UAVPursuitApolloniusMultiTarget3DEnv(runner.config)

    per_ep_switches, per_ep_rate, steps_all, succ = [], [], [], 0
    for ep in range(args.k):
        obs, _ = env.reset()
        prev = np.array(env.assign, copy=True)
        switches, steps, done = 0, 0, False
        while not done:
            acts = agent.action([obs], test_mode=True)["actions"][0]
            obs, _, term, trunc, info = env.step(acts)
            cur = np.array(env.assign, copy=True)
            # count only genuine target-to-target flips: a pursuer whose target died is forced
            # to move and that is not churn, so ignore transitions out of a dead assignment.
            alive = np.asarray(env.target_alive, dtype=bool)
            for i in range(env.num_agents):
                a, b = int(prev[i]), int(cur[i])
                if a >= 0 and b >= 0 and a != b and a < len(alive) and alive[a]:
                    switches += 1
            prev = cur
            steps += 1
            # multitarget returns truncated as a PER-AGENT DICT (the single-target env returns a
            # scalar), and bool() on a non-empty dict is always True -- which ended every
            # episode after one step and zeroed both success and switch counts.
            done = bool(term[env.agents[0]]) or bool(trunc[env.agents[0]])
        per_ep_switches.append(switches)
        per_ep_rate.append(switches / max(steps, 1))
        steps_all.append(steps)
        if bool(info.get("is_success", False)):
            succ += 1

    sw = np.asarray(per_ep_switches, float)
    rt = np.asarray(per_ep_rate, float)
    res = {
        "model_path": args.model_path, "dynamic_alloc": not args.no_dynamic_alloc,
        "num_agents": args.num_agents, "num_targets": args.num_targets,
        "episodes": args.k, "success_rate": succ / args.k,
        "mean_steps": float(np.mean(steps_all)),
        "switches_per_episode_mean": float(sw.mean()),
        "switches_per_episode_max": int(sw.max()),
        "switches_per_step_mean": float(rt.mean()),
        "episodes_with_any_switch": int((sw > 0).sum()),
        "switch_episode_rate": float((sw > 0).mean()),
    }
    print(f"{'dynamic' if res['dynamic_alloc'] else 'static ':8} "
          f"succ={res['success_rate']*100:5.1f}%  "
          f"switches/ep={res['switches_per_episode_mean']:7.2f}  "
          f"switches/step={res['switches_per_step_mean']:6.3f}  "
          f"eps_with_switch={res['switch_episode_rate']*100:5.1f}%")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
