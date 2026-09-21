"""Log what Chapter 4 actually instructs its pursuers to do, step by step.

Five separate attempts at the strict capture rule have returned exactly 0.0 -- raising the
closure weight, enabling the rule, adding the surround spawn, porting the missing reward terms,
and warm-starting from a policy that already encircles at 96.7%. A failure that survives all of
those is unlikely to be a training problem, so this stops guessing at the policy and reads the
instructions instead: per step it records the escape solid angle, the formation radius r_f, the
guide radius derived from them, how far each pursuer is from its assigned guide, and the true
distance to the target.

The specific suspicion is r_f. _compute_r_f is inherited from the single-target environment and
reads self.target_position, which the multi-target reset writes once and then never updates --
the live targets live in self.target_positions. If that attribute is stale, every quantity
derived from r_f is computed against a phantom target, including the probe range handed to the
escape-field routine and the outer bound on the guide radius.
"""
import argparse
import json

import numpy as np

from xuance import get_runner
from xuance.environment.multi_agent_env.uav_pursuit_apollonius_multitarget_3d import (
    UAVPursuitApolloniusMultiTarget3DEnv,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--num-agents", type=int, default=6)
    ap.add_argument("--num-targets", type=int, default=2)
    ap.add_argument("--building-mode", default="medium")
    ap.add_argument("--target-max-speed", type=float, default=10.0)
    ap.add_argument("--surround-spawn", action="store_true")
    ap.add_argument("--strict-capture", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="ch4_guides.json")
    args = ap.parse_args()

    np.random.seed(args.seed)
    p = argparse.Namespace(algo="maddpg", env="uav_pursuit_apollonius_multitarget_3d",
                           env_id="apollonius_mt_3d", device="cuda:0")
    p.parallels = 1
    p.num_agents = args.num_agents
    p.num_targets = args.num_targets
    p.curriculum_enabled = False
    p.building_mode = args.building_mode
    p.target_max_speed = args.target_max_speed
    if args.surround_spawn:
        p.surround_spawn = True
    if args.strict_capture:
        p.strict_capture = True

    runner = get_runner(algo="maddpg", env="uav_pursuit_apollonius_multitarget_3d",
                        env_id="apollonius_mt_3d", parser_args=p)
    agent = runner.agent
    agent.load_model(args.model_path)
    env = UAVPursuitApolloniusMultiTarget3DEnv(runner.config)
    cr = float(env.catch_radius)

    episodes = []
    for _ in range(args.k):
        obs, _ = env.reset()
        tr, done = [], False
        while not done:
            # distance from the stale single-target attribute to each live target: if this is
            # large, r_f is being computed against a phantom
            stale = float(np.linalg.norm(env.target_position - env.target_positions[0]))
            r_f = float(env._compute_r_f())
            gd = [float(np.linalg.norm(env.uav_positions[i] - env.current_guide_points[a]))
                  for i, a in enumerate(env.agents) if a in env.current_guide_points]
            gr = [float(np.linalg.norm(env.current_guide_points[a] - env.target_positions[env.assign[i]]))
                  for i, a in enumerate(env.agents)
                  if a in env.current_guide_points and 0 <= env.assign[i] < env.num_targets]
            esa = float(env.sub_fields[0].get("escape_solid_angle", np.nan)) \
                if getattr(env, "sub_fields", None) and isinstance(env.sub_fields[0], dict) else float("nan")
            d = np.linalg.norm(env.uav_positions[:, None, :] - env.target_positions[None, :, :],
                               axis=-1).min(axis=0)
            tr.append({"esa": esa, "r_f": r_f, "stale_offset": stale,
                       "guide_radius_mean": float(np.mean(gr)) if gr else float("nan"),
                       "guide_dist_mean": float(np.mean(gd)) if gd else float("nan"),
                       "min_dist_t0": float(d[0]), "min_dist_t1": float(d[1]) if len(d) > 1 else float("nan")})
            acts = agent.action([obs], test_mode=True)["actions"][0]
            obs, _, term, trunc, info = env.step(acts)
            done = bool(term[env.agents[0]]) or bool(trunc[env.agents[0]])
        episodes.append(tr)

    tr = max(episodes, key=len)
    idx = [0, len(tr) // 4, len(tr) // 2, 3 * len(tr) // 4, len(tr) - 1]
    out = {
        "catch_radius": cr, "steps": len(tr),
        "timeline": [{"step": i, **{k: (round(tr[i][k], 2) if np.isfinite(tr[i][k]) else None)
                                    for k in tr[i]}} for i in idx],
    }
    allr = np.array([s["guide_radius_mean"] for t in episodes for s in t], float)
    alle = np.array([s["esa"] for t in episodes for s in t], float)
    alls = np.array([s["stale_offset"] for t in episodes for s in t], float)
    out["summary"] = {
        "guide_radius": {"min": float(np.nanmin(allr)), "median": float(np.nanmedian(allr)),
                         "max": float(np.nanmax(allr))},
        "esa": {"min": float(np.nanmin(alle)), "median": float(np.nanmedian(alle))},
        "stale_target_offset": {"median": float(np.nanmedian(alls)), "max": float(np.nanmax(alls))},
        "frac_guides_inside_2x_catch": float(np.nanmean(allr <= 2 * cr)),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("GUIDES " + json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
