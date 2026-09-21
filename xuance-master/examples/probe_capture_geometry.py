"""Where do captures actually happen -- in open space, or against the arena boundary?

A capture that only works because the evader ran out of room is a much weaker claim than one
that closes in open air, and at a speed ratio above 1 it is the obvious thing to suspect: the
walls are doing the work the pursuers could not. This measures it directly. For each captured
episode it records the capture point's distance to the nearest arena wall, to the nearest
building footprint, and to the arena centre, plus how much room the evader still had along its
own heading, and classifies the capture as cornered or open-field.

"Cornered" is defined generously against us: within one evader-second (its top speed) of a wall
or a building. If captures are legitimate that threshold should still classify most of them as
open-field.
"""
import argparse
import json

import numpy as np

from xuance import get_runner
from xuance.environment.multi_agent_env import geometry3d as g3
from xuance.environment.multi_agent_env.uav_pursuit_apollonius_3d import UAVPursuitApollonius3DEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--algo", default="maddpg")
    ap.add_argument("--num-agents", type=int, default=3)
    ap.add_argument("--building-mode", default="medium")
    ap.add_argument("--target-max-speed", type=float, default=12.0)
    ap.add_argument("--target-min-speed", type=float, default=8.0)
    ap.add_argument("--surround-spawn", action="store_true")
    ap.add_argument("--spawn-radius", type=float, default=None)
    ap.add_argument("--cbf-safety", action="store_true",
                    help="must match how the model was trained and evaluated")
    ap.add_argument("--cbf-margin", type=float, default=None)
    ap.add_argument("--cbf-eta", type=float, default=None)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--evader-center-pull", type=float, default=None,
                    help="APF pull toward airspace centre; keeps captures off the walls")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="capture_geometry.json")
    args = ap.parse_args()

    p = argparse.Namespace(algo=args.algo, env="uav_pursuit_apollonius_3d",
                           env_id="apollonius_3d", device=args.device)
    p.parallels = 1
    p.curriculum_enabled = False
    p.building_mode = args.building_mode
    if args.evader_center_pull is not None:
        p.evader_center_pull = args.evader_center_pull
    p.num_agents = args.num_agents
    p.target_max_speed = args.target_max_speed
    p.target_min_speed = args.target_min_speed
    if args.cbf_safety:
        p.cbf_enabled = True
    if args.cbf_margin is not None:
        p.cbf_margin = args.cbf_margin
    if args.cbf_eta is not None:
        p.cbf_eta = args.cbf_eta
    if args.spawn_radius is not None:
        p.spawn_radius = args.spawn_radius
    if args.surround_spawn:
        p.surround_spawn = True
    np.random.seed(args.seed)

    runner = get_runner(algo=args.algo, env="uav_pursuit_apollonius_3d",
                        env_id="apollonius_3d", parser_args=p)
    agent = runner.agent
    agent.load_model(args.model_path)
    env = UAVPursuitApollonius3DEnv(runner.config)
    M = float(env.map_size)
    near = float(args.target_max_speed)          # one evader-second of room

    zlo, zhi = float(env.z_min), float(env.z_max)

    def wall_dist(q):                       # the four vertical walls only
        return float(min(q[0], M - q[0], q[1], M - q[1]))

    def z_dist(q):                          # floor and ceiling; the arena is a box, not a column
        return float(min(q[2] - zlo, zhi - q[2]))

    def n_buildings_near(q, radius=150.0):
        """How much built-up structure surrounds the capture point."""
        if len(env.buildings) == 0:
            return 0
        b = np.asarray(env.buildings, float)
        cx = np.clip(q[0], b[:, 0], b[:, 1])
        cy = np.clip(q[1], b[:, 2], b[:, 3])
        return int(np.sum(np.hypot(q[0] - cx, q[1] - cy) < radius))

    def blocked_by_buildings(q):
        """Fraction of the 200 sampled escape directions that terminate on a building rather
        than staying open. This is the quantity the pursuers can raise by herding the evader
        into built-up ground, so it separates 'trapped by structure' from 'trapped by pursuers'."""
        free = g3.ray_free_distance(q.astype(np.float64), env.escape_dirs, env.map_size,
                                    env.z_min, env.z_max, env.buildings)
        thresh = max(env.catch_radius * 1.2, 1.0)
        return float(np.mean(free <= thresh * 4))

    def bld_dist(q):
        if len(env.buildings) == 0:
            return float("inf")
        b = np.asarray(env.buildings, float)
        cx = np.clip(q[0], b[:, 0], b[:, 1])
        cy = np.clip(q[1], b[:, 2], b[:, 3])
        return float(np.min(np.hypot(q[0] - cx, q[1] - cy)))

    caps = []
    for ep in range(args.k):
        obs, _ = env.reset()
        done = False
        while not done:
            acts = agent.action([obs], test_mode=True)["actions"][0]
            obs, _, term, trunc, info = env.step(acts)
            done = bool(term[env.agents[0]]) or bool(trunc)
        if not info.get("is_success", False):
            continue
        q = env.target_position.copy()
        dw, db, dz = wall_dist(q), bld_dist(q), z_dist(q)
        nb, bb = n_buildings_near(q), blocked_by_buildings(q)
        caps.append({
            "steps": int(env.current_step) if hasattr(env, "current_step") else -1,
            "pos": [round(float(v), 1) for v in q],
            "dist_to_wall": round(dw, 1),
            "dist_to_ceiling_or_floor": round(dz, 1),
            "dist_to_any_boundary": round(min(dw, dz), 1),
            "z": round(float(q[2]), 1),
            "dist_to_building": round(db, 1),
            "buildings_within_150m": nb,
            "frac_dirs_blocked_by_structure": round(bb, 3),
            "dist_to_centre": round(float(np.hypot(q[0] - M / 2, q[1] - M / 2)), 1),
            "cornered": bool(min(dw, dz, db) < near),
        })

    n = len(caps)
    out = {"model": args.model_path, "n_episodes": args.k, "n_captured": n,
           "evader_max_speed": args.target_max_speed,
           "cornered_threshold_m": near, "z_range": [zlo, zhi], "captures": caps}
    if n:
        dw = np.array([c["dist_to_wall"] for c in caps])
        db = np.array([c["dist_to_building"] for c in caps])
        dz = np.array([c["dist_to_ceiling_or_floor"] for c in caps])
        da = np.array([c["dist_to_any_boundary"] for c in caps])
        zz = np.array([c["z"] for c in caps])
        nb = np.array([c["buildings_within_150m"] for c in caps], float)
        bb = np.array([c["frac_dirs_blocked_by_structure"] for c in caps])
        out["summary"] = {
            "frac_cornered": float(np.mean([c["cornered"] for c in caps])),
            "wall_dist": {"min": float(dw.min()), "median": float(np.median(dw)),
                          "mean": float(dw.mean())},
            "building_dist": {"min": float(db.min()), "median": float(np.median(db)),
                              "mean": float(db.mean())},
            "z_dist": {"min": float(dz.min()), "median": float(np.median(dz))},
            "any_boundary_dist": {"min": float(da.min()), "median": float(np.median(da))},
            "z_value": {"min": float(zz.min()), "median": float(np.median(zz)),
                        "max": float(zz.max()), "arena_mid": (zlo + zhi) / 2},
            "frac_wall_within_50m": float(np.mean(dw < 50)),
            "frac_z_within_50m": float(np.mean(dz < 50)),
            "frac_any_boundary_within_50m": float(np.mean(da < 50)),
            "frac_building_within_50m": float(np.mean(db < 50)),
            "buildings_within_150m": {"median": float(np.median(nb)), "mean": float(nb.mean())},
            "dirs_blocked_by_structure": {"median": float(np.median(bb)), "mean": float(bb.mean())},
            "frac_captures_in_builtup": float(np.mean(nb >= 2)),
        }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("CAPGEO " + json.dumps(out.get("summary", {})))


if __name__ == "__main__":
    main()
