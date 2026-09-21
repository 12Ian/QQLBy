"""Escape-solid-angle sampling convergence study (justifies the 200-direction choice).

The capture criterion measures the evader's escape solid angle by counting Fibonacci
directions classified as "dangerous" -- i.e. it estimates the area of a spherical region by
which lattice points fall inside it. Gonzalez (Math. Geosci. 42:49-64, 2010) shows a Fibonacci
lattice is the right instrument for exactly that measurement; this script fixes the remaining
free parameter by showing the estimate has converged at n = 200.

States are taken from real rollouts (model or greedy policy), so the configurations are the
ones the criterion actually sees. At each sampled step the escape solid angle is recomputed for
every n against a fine reference (default n = 1600), with the EMA/hysteresis state disabled
(margin_ema=None, dangerous_prev=None) so the comparison is a pure quadrature question.

Reports, per n: RMS / max error relative to the full sphere, and -- what actually matters --
how often the n-direction field agrees with the reference on the ENCIRCLED decision.

Usage (server):
  python xuance-master/examples/sampling_convergence.py --policy greedy --num-agents 3 \
      --building-mode medium --target-max-speed 10 --k 15 --out p3_results/sampling_conv.json
"""
import os
import sys
import json
import math
import argparse
import numpy as np

from xuance import get_runner
from xuance.environment.multi_agent_env.uav_pursuit_apollonius_3d import UAVPursuitApollonius3DEnv
from xuance.environment.multi_agent_env import geometry3d as g3
from xuance.environment.multi_agent_env import apollonius3d as ap3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_3d import greedy_actions          # noqa: E402  (module guards main())


def field_at(env, dirs, mode):
    """Instantaneous escape solid angle for one direction set (no EMA / no hysteresis)."""
    free = g3.ray_free_distance(env.target_position, dirs, env.map_size,
                                env.z_min, env.z_max, env.buildings)
    f = ap3.compute_escape_field_3d(env.target_position, env.target_speed,
                                    env.uav_positions, env.uav_max_speed, dirs, free,
                                    env.catch_radius, env._compute_r_f(),
                                    margin_ema=None, dangerous_prev=None,
                                    criterion_mode=mode, buildings=env.buildings)
    return float(f["escape_solid_angle"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="greedy", choices=["model", "greedy", "random"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--algo", default="maddpg")
    ap.add_argument("--num-agents", type=int, default=3)
    ap.add_argument("--building-mode", default="medium")
    ap.add_argument("--target-max-speed", type=float, default=10.0)
    ap.add_argument("--target-min-speed", type=float, default=8.0)
    ap.add_argument("--criterion", default="euclidean", choices=["euclidean", "visibility"])
    ap.add_argument("--surround-spawn", action="store_true")
    ap.add_argument("--k", type=int, default=15, help="episodes")
    ap.add_argument("--stride", type=int, default=5, help="sample every Nth step")
    ap.add_argument("--levels", default="50,100,200,400,800")
    ap.add_argument("--reference", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    p = argparse.Namespace(algo=args.algo, env="uav_pursuit_apollonius_3d",
                           env_id="apollonius_3d", device=args.device)
    p.parallels = 1
    p.seed = args.seed
    p.curriculum_enabled = False
    p.building_mode = args.building_mode
    p.num_agents = args.num_agents
    p.target_max_speed = args.target_max_speed
    p.target_min_speed = args.target_min_speed
    p.criterion_mode = args.criterion
    if args.surround_spawn:
        p.surround_spawn = True

    np.random.seed(args.seed)
    runner = get_runner(algo=args.algo, env="uav_pursuit_apollonius_3d",
                        env_id="apollonius_3d", parser_args=p)
    cfg = runner.config
    agent = None
    if args.policy == "model":
        agent = runner.agent
        agent.load_model(os.path.abspath(args.model_path))

    np.random.seed(args.seed)
    env = UAVPursuitApollonius3DEnv(cfg)
    dirsets = {n: g3.fibonacci_sphere(n) for n in levels + [args.reference]}
    rng = np.random.default_rng(args.seed)

    samples = {n: [] for n in levels}
    ref_vals = []
    for ep in range(args.k):
        np.random.seed((int(args.seed) + ep) % (2 ** 32))
        obs, _ = env.reset()
        done, steps = False, 0
        while not done:
            if steps % args.stride == 0:
                ref = field_at(env, dirsets[args.reference], args.criterion)
                ref_vals.append(ref)
                for n in levels:
                    samples[n].append(field_at(env, dirsets[n], args.criterion))
            if args.policy == "model":
                acts = agent.action([obs], test_mode=True)["actions"][0]
            elif args.policy == "greedy":
                acts = greedy_actions(env)
            else:
                acts = {a: rng.uniform(-1, 1, 3).astype(np.float32) for a in env.agents}
            obs, _, term, trunc, _ = env.step(acts)
            steps += 1
            done = bool(term[env.agents[0]]) or bool(trunc)
        print(f"  episode {ep + 1}/{args.k}: {steps} steps, {len(ref_vals)} snapshots", flush=True)

    full = 4.0 * math.pi
    ref = np.asarray(ref_vals)
    # "Encircled" is the decision the criterion actually drives: escape solid angle collapsed.
    enc_thresh = 0.05 * full
    ref_enc = ref <= enc_thresh

    rows = []
    for n in levels:
        v = np.asarray(samples[n])
        err = (v - ref) / full                       # error as a fraction of the full sphere
        agree = float(np.mean((v <= enc_thresh) == ref_enc))
        rows.append({
            "n": n,
            "angular_resolution_deg": math.degrees(math.sqrt(full / n)),
            "rms_error_frac_sphere": float(np.sqrt(np.mean(err ** 2))),
            "max_abs_error_frac_sphere": float(np.max(np.abs(err))),
            "mean_bias_frac_sphere": float(np.mean(err)),
            "encircled_decision_agreement": agree,
        })

    result = {
        "snapshots": int(ref.size),
        "episodes": args.k,
        "reference_n": args.reference,
        "criterion": args.criterion,
        "num_agents": args.num_agents,
        "building_mode": args.building_mode,
        "policy": args.policy,
        "encircled_threshold_frac_sphere": 0.05,
        "levels": rows,
    }
    print("\n=== escape solid angle sampling convergence "
          f"({ref.size} snapshots, reference n={args.reference}) ===")
    print(f"{'n':>6} {'ang.res':>9} {'RMS err':>10} {'max err':>10} {'encircled agree':>16}")
    for r in rows:
        print(f"{r['n']:>6} {r['angular_resolution_deg']:>8.1f}° "
              f"{r['rms_error_frac_sphere']:>10.4f} {r['max_abs_error_frac_sphere']:>10.4f} "
              f"{r['encircled_decision_agreement'] * 100:>15.1f}%")
    print("(errors are fractions of the full sphere 4pi)")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
