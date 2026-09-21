"""Record what actually closes on the evader during a capture.

The question this answers is why an evader that is no slower than its pursuers does not simply
pick a gap and run. Per step it logs the escape solid angle, how many of the 200 sampled
directions are still open, the evader's commanded speed, and the closest pursuer, so the claim
"the escape set is bounded and shrinking" can be read off measurements instead of asserted.

At a speed ratio of 1 the set of points the evader reaches before a given pursuer is the
half-space on its side of the perpendicular bisector, so the evader's safe region is the
intersection of those half-spaces -- its Voronoi cell against the pursuer set. Surrounded, that
cell is bounded, and no direction leaves it. Above ratio 1 the per-pursuer region is unbounded
and real gaps exist, which is the regime where the measured success rate drops.
"""
import argparse
import json
import os

import numpy as np

from xuance import get_runner
from xuance.environment.multi_agent_env.uav_pursuit_apollonius_3d import UAVPursuitApollonius3DEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--algo", default="maddpg")
    ap.add_argument("--num-agents", type=int, default=3)
    ap.add_argument("--building-mode", default="medium")
    ap.add_argument("--target-max-speed", type=float, default=11.0)
    ap.add_argument("--target-min-speed", type=float, default=8.0)
    ap.add_argument("--surround-spawn", action="store_true")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="mechanism.json")
    args = ap.parse_args()

    p = argparse.Namespace(algo=args.algo, env="uav_pursuit_apollonius_3d",
                           env_id="apollonius_3d", device=args.device)
    p.parallels = 1
    p.curriculum_enabled = False
    p.building_mode = args.building_mode
    p.num_agents = args.num_agents
    p.target_max_speed = args.target_max_speed
    p.target_min_speed = args.target_min_speed
    if args.surround_spawn:
        p.surround_spawn = True
    np.random.seed(args.seed)

    runner = get_runner(algo=args.algo, env="uav_pursuit_apollonius_3d",
                        env_id="apollonius_3d", parser_args=p)
    agent = runner.agent
    agent.load_model(args.model_path)
    env = UAVPursuitApollonius3DEnv(runner.config)
    full = 4.0 * np.pi

    episodes = []
    for _ in range(args.k):
        obs, _ = env.reset()
        tr, done = [], False
        while not done:
            f = env.apollonius_escape if isinstance(getattr(env, "apollonius_escape", None), dict) else {}
            esa = float(f.get("escape_solid_angle", np.nan))
            openv = f.get("geometry_open")
            dang = f.get("dangerous")
            n_open = int(np.sum(np.asarray(openv) & ~np.asarray(dang))) if (
                openv is not None and dang is not None) else -1
            tr.append({
                "esa": esa,
                "esa_frac": esa / full if np.isfinite(esa) else float("nan"),
                "n_open_dirs": n_open,
                "evader_speed": float(env.target_speed),
                "min_dist": float(np.min(np.linalg.norm(
                    env.uav_positions - env.target_position, axis=1))),
            })
            acts = agent.action([obs], test_mode=True)["actions"][0]
            obs, _, term, trunc, info = env.step(acts)
            done = bool(term[env.agents[0]]) or bool(trunc)
        episodes.append({"caught": bool(info.get("is_success", False)),
                         "steps": len(tr), "trace": tr})

    won = [e for e in episodes if e["caught"]]
    out = {
        "n_episodes": len(episodes), "n_captured": len(won),
        "evader_max_speed": args.target_max_speed,
        "pursuer_max_speed": float(env.uav_max_speed),
        "speed_ratio": args.target_max_speed / float(env.uav_max_speed),
        "catch_radius": float(env.catch_radius),
    }
    if won:
        e = max(won, key=lambda x: x["steps"])       # the most detailed capture
        tr = e["trace"]
        q = [0, len(tr) // 4, len(tr) // 2, 3 * len(tr) // 4, len(tr) - 1]
        out["representative_capture"] = {
            "steps": e["steps"],
            "timeline": [{"step": i, **{k: round(tr[i][k], 3) for k in tr[i]}} for i in q],
        }
        sp = np.concatenate([[s["evader_speed"] for s in x["trace"]] for x in won])
        out["evader_speed_stats"] = {
            "mean": float(sp.mean()), "min": float(sp.min()), "max": float(sp.max()),
            "frac_at_max": float((sp >= args.target_max_speed - 1e-6).mean()),
        }
        first = np.array([x["trace"][0]["esa_frac"] for x in won])
        last = np.array([x["trace"][-1]["esa_frac"] for x in won])
        out["escape_fraction"] = {"at_start": float(np.nanmean(first)),
                                  "at_capture": float(np.nanmean(last))}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("MECH " + json.dumps(out))


if __name__ == "__main__":
    main()
