"""Evaluate a multi-target (Chapter 4) pursuit model: success = ALL targets captured
within the episode. Reports success rate + per-target capture rate + Wilson 95% CI."""
import argparse
import json
import numpy as np
from xuance import get_runner
from xuance.environment.multi_agent_env.uav_pursuit_apollonius_multitarget_3d import (
    UAVPursuitApolloniusMultiTarget3DEnv)
from xuance.environment.multi_agent_env.eval_metrics import wilson_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--num-agents", type=int, default=6)
    ap.add_argument("--num-targets", type=int, default=2)
    ap.add_argument("--building-mode", default="medium")
    ap.add_argument("--target-max-speed", type=float, default=None)
    ap.add_argument("--no-dynamic-alloc", action="store_true")
    ap.add_argument("--apollonius-alloc", action="store_true")
    ap.add_argument("--strict-capture", action="store_true",
                    help="score a target captured only on physical reach, matching Ch3")
    ap.add_argument("--surround-spawn", action="store_true",
                    help="ring each sub-team around its own target at reset")
    ap.add_argument("--spawn-radius", type=float, default=None)
    ap.add_argument("--evader-center-pull", type=float, default=None)
    ap.add_argument("--cbf-safety", action="store_true")
    ap.add_argument("--cbf-margin", type=float, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    np.random.seed(args.seed)
    p = argparse.Namespace(algo="maddpg", env="uav_pursuit_apollonius_multitarget_3d",
                           env_id="apollonius_mt_3d", device="cuda:0")
    p.parallels = 1
    if args.strict_capture:
        p.strict_capture = True
    if args.surround_spawn:
        p.surround_spawn = True
    if args.spawn_radius is not None:
        p.spawn_radius = args.spawn_radius
    if args.evader_center_pull is not None:
        p.evader_center_pull = args.evader_center_pull
    if args.cbf_safety:
        p.cbf_enabled = True
    if args.cbf_margin is not None:
        p.cbf_margin = args.cbf_margin
    p.num_agents = args.num_agents
    p.num_targets = args.num_targets
    p.curriculum_enabled = False
    p.building_mode = args.building_mode
    if args.target_max_speed is not None:
        p.target_max_speed = args.target_max_speed
    if args.apollonius_alloc:
        p.apollonius_alloc = True
    if args.no_dynamic_alloc:
        p.dynamic_alloc = False
    runner = get_runner(algo="maddpg", env="uav_pursuit_apollonius_multitarget_3d",
                        env_id="apollonius_mt_3d", parser_args=p)
    agent = runner.agent
    agent.load_model(args.model_path)
    env = UAVPursuitApolloniusMultiTarget3DEnv(runner.config)

    succ, caught_total, steps_list = 0, 0, []
    # Closest approach per target, recorded in the same rollout that produces the score. Under
    # Chapter 3's strict rule this environment captured nothing at all, and the distribution of
    # closest approach says which failure it is: bunched just outside the catch radius means the
    # team encircles without closing, far outside means it never gets there.
    closest_all = []
    for _ in range(args.k):
        obs, info = env.reset()
        done, steps = False, 0
        best = np.full(args.num_targets, np.inf)
        while not done:
            d = np.linalg.norm(env.uav_positions[:, None, :] - env.target_positions[None, :, :],
                               axis=-1).min(axis=0)
            best = np.minimum(best, d[:args.num_targets])
            acts = agent.action([obs], test_mode=True)["actions"][0]
            obs, rew, term, trunc, info = env.step(acts)
            steps += 1
            done = bool(term[env.agents[0]]) or bool(trunc[env.agents[0]])
        d = np.linalg.norm(env.uav_positions[:, None, :] - env.target_positions[None, :, :],
                           axis=-1).min(axis=0)
        closest_all.append(np.minimum(best, d[:args.num_targets]))
        if info["is_success"]:
            succ += 1
            steps_list.append(steps)
        caught_total += info["num_caught"]

    n = args.k
    result = {
        "n_episodes": n, "num_agents": args.num_agents, "num_targets": args.num_targets,
        "success_rate": succ / n, "success_ci": list(wilson_ci(succ, n)),
        "per_target_capture_rate": caught_total / (n * args.num_targets),
        "mean_full_capture_steps": float(np.mean(steps_list)) if steps_list else None,
    }
    if closest_all:
        ca = np.concatenate(closest_all)
        result["closest_approach"] = {
            "catch_radius": float(env.catch_radius),
            "min": float(ca.min()), "median": float(np.median(ca)), "mean": float(ca.mean()),
            "frac_within_catch_radius": float((ca <= env.catch_radius).mean()),
            "frac_within_2x": float((ca <= 2 * env.catch_radius).mean()),
        }
    print("RESULT " + json.dumps(result))
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)
        print("saved", args.out)
    runner.envs.close()


if __name__ == "__main__":
    main()
