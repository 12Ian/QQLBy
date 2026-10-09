"""Unified evaluator: roll out a policy (model|random|greedy) for K episodes at a
curriculum level, compute the full metric suite (euclidean + visibility judges),
print and save JSON. Guard with __main__ (SubprocVecMultiAgentEnv uses spawn)."""
import os
import sys
import json
import argparse
import math
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
from xuance import get_runner
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv
from xuance.environment.multi_agent_env.Apollonius import geometry3d as g3
from xuance.environment.multi_agent_env.Apollonius import apollonius3d as ap3
from xuance.environment.multi_agent_env.Apollonius.eval_metrics import summarize_eval


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def escape_angle(env, mode):
    free = g3.ray_free_distance(env.target_position, env.escape_dirs, env.map_size,
                                env.z_min, env.z_max, env.buildings)
    f = ap3.compute_escape_field_3d(env.target_position, env.target_speed,
            env.uav_positions, env.uav_max_speed, env.escape_dirs, free,
            env.catch_radius, env._compute_r_f(), margin_ema=None, dangerous_prev=None,
            criterion_mode=mode, buildings=env.buildings)
    return f["escape_solid_angle"]


def greedy_actions(env):
    """按目标方向生成三维加速度基线动作。"""
    acts = {}
    for i, agent in enumerate(env.agents):
        displacement = env.target_position - env.uav_positions[i]
        distance = float(np.linalg.norm(displacement))
        direction = displacement / max(distance, 1e-6)
        desired_velocity = direction * env.uav_max_speed
        desired_acceleration = (desired_velocity - env.uav_velocities[i]) / max(
            env.decision_dt, 1e-6)
        acts[agent] = np.clip(desired_acceleration / max(env.max_accel, 1e-6),
                              -1.0, 1.0).astype(np.float32)
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="model", choices=["model", "random", "greedy"])
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--evader-center-pull", type=float, default=None,
                    help="APF pull toward airspace centre; keeps captures off the walls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--algo", default="maddpg")   # must match the trained model's algorithm
    ap.add_argument("--out", default=None)
    # match the trained model's architecture/env (GAT / criterion / building mode)
    ap.add_argument("--use-obstacle-gat", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--obstacle-gat-k", type=int, default=5)
    ap.add_argument("--use-graph-module", action="store_true")
    ap.add_argument("--criterion", default="euclidean", choices=["euclidean", "visibility", "detour"])
    ap.add_argument("--building-mode", default=None)   # if set, curriculum off + fixed density
    ap.add_argument("--target-max-speed", type=float, default=None)
    ap.add_argument("--target-min-speed", type=float, default=None)
    ap.add_argument("--pincer-weight", type=float, default=None)
    ap.add_argument("--num-agents", type=int, default=None)  # MUST match trained model's N
    ap.add_argument("--surround-spawn", action="store_true")  # match trained model's spawn
    ap.add_argument("--spawn-radius", type=float, default=None)
    ap.add_argument("--spawn-safety", choices=["on", "off"], default="on")
    ap.add_argument("--collision-mode", choices=["hard", "soft"], default="hard",
                    help="whether teammate UAV collisions terminate; boundary hits continue by default")
    ap.add_argument("--terminate-on-boundary-collision", action="store_true",
                    help="end an episode when a pursuer hits the arena boundary")
    ap.add_argument("--boundary-collision-penalty", type=float, default=None)
    ap.add_argument("--boundary-warning-distance", type=float, default=None)
    ap.add_argument("--trace-out", default=None)
    ap.add_argument("--closure-weight", type=float, default=None)
    ap.add_argument("--w-pos", type=float, default=None)
    ap.add_argument("--w-gap", type=float, default=None)
    ap.add_argument("--separation-weight", type=float, default=None)
    ap.add_argument("--separation-distance", type=float, default=None)
    # Evaluation must be able to run on the same device the policy was trained on; a hard-wired
    # cuda:0 also makes this script unusable on CPU-only machines.
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    p = argparse.Namespace(algo=args.algo, env="uav_pursuit_coverage_3d",
                           env_id="coverage_3d", device=args.device)
    p.parallels = 1
    p.seed = args.seed
    p.spawn_safety_enabled = args.spawn_safety == "on"
    p.terminate_on_collision = args.collision_mode == "hard"
    p.terminate_on_boundary_collision = args.terminate_on_boundary_collision
    for name in ("closure_weight", "w_pos", "w_gap",
                 "separation_weight", "separation_distance", "pincer_weight",
                 "boundary_collision_penalty", "boundary_warning_distance"):
        value = getattr(args, name)
        if value is not None:
            setattr(p, name, value)
    p.use_obstacle_gat = args.use_obstacle_gat
    if args.obstacle_gat_k is not None:
        p.obstacle_gat_k = args.obstacle_gat_k
        p.nearest_building_k = args.obstacle_gat_k
    p.use_graph_module = args.use_graph_module
    p.criterion_mode = args.criterion
    if args.building_mode:
        p.curriculum_enabled = False
        p.building_mode = args.building_mode
    if args.target_max_speed is not None:
        p.target_max_speed = args.target_max_speed
    if args.target_min_speed is not None:
        p.target_min_speed = args.target_min_speed
    if args.evader_center_pull is not None:
        p.evader_center_pull = args.evader_center_pull
    if args.num_agents is not None:
        p.num_agents = args.num_agents
    if args.surround_spawn:
        p.surround_spawn = True
    if args.spawn_radius is not None:
        p.spawn_radius = args.spawn_radius
    # Seed the GLOBAL numpy RNG that drives env.reset (target spawn, initial velocity, evader
    # noise). Without this every eval point would see a different irreproducible set of
    # episodes; seeding makes all sweep/anchor points share the SAME episode draws (paired).
    np.random.seed(args.seed)
    runner = None
    try:
        runner = get_runner(algo=args.algo, env="uav_pursuit_coverage_3d",
                            env_id="coverage_3d", parser_args=p)
        cfg = runner.config
        agent = None
        resolved_model_path = None
        if args.policy == "model":
            agent = runner.agent
            mp = args.model_path or os.path.join(
                os.getcwd(), "outputs", "results", "maddpg", "coverage_3d",
                "best_model", "best_model.pth")
            resolved_model_path = os.path.abspath(mp)
            agent.load_model(resolved_model_path)

        np.random.seed(args.seed)
        env = UAVPursuitCoverage3DEnv(cfg)
        if getattr(env, "curriculum_enabled", False):
            env.set_curriculum_level(args.level)
        rng = np.random.default_rng(args.seed)
        source_path = os.path.abspath(sys.modules[UAVPursuitCoverage3DEnv.__module__].__file__)
        environment_sha256 = sha256_file(source_path)
        evaluator_sha256 = sha256_file(os.path.abspath(__file__))
        model_sha256 = sha256_file(resolved_model_path) if resolved_model_path else None
        trace_records = []
        spawn_adjusted = []
        spawn_max_offsets = []
        records = []

        for ep in range(args.k):
            np.random.seed((int(args.seed) + ep) % (2 ** 32))
            obs, info = env.reset()
            spawn_info = info.get("infos", {})
            initial_positions = env.uav_positions.astype(float).tolist()
            initial_target_position = env.target_position.astype(float).tolist()
            initial_uav_velocities = env.uav_velocities.astype(float).tolist()
            episode_return = 0.0
            collided = False
            first_collision_step = None
            first_collision_agents = []
            first_collision_sources = {}
            done, steps = False, 0
            oe, ov = math.inf, math.inf
            caught = False
            while not done:
                oe = min(oe, escape_angle(env, "euclidean"))
                ov = min(ov, escape_angle(env, "visibility"))
                if args.policy == "model":
                    acts = agent.action([obs], test_mode=True)["actions"][0]
                elif args.policy == "greedy":
                    acts = greedy_actions(env)
                else:
                    acts = {a: rng.uniform(-1, 1, 3).astype(np.float32) for a in env.agents}
                obs, rew, term, trunc, info = env.step(acts)
                steps += 1
                episode_return += float(np.mean(list(rew.values())))
                collided = collided or bool(info.get("is_collision", False))
                if info.get("is_collision", False) and first_collision_step is None:
                    first_collision_step = steps
                    first_collision_agents = list(info.get("collision_agents", []))
                    first_collision_sources = dict(info.get("collision_sources", {}))
                caught = caught or bool(info.get("is_success", False))
                done = bool(term[env.agents[0]]) or bool(trunc)
            record = {
                "episode": ep,
                "caught": caught,
                "crashed": collided,
                "steps": steps,
                "episode_return": episode_return,
                "initial_positions": initial_positions,
                "initial_target_position": initial_target_position,
                "initial_uav_velocities": initial_uav_velocities,
                "spawn_offsets": list(spawn_info.get("spawn_offsets", [])),
                "spawn_adjusted_count": int(spawn_info.get("spawn_adjusted_count", 0)),
                "spawn_max_offset": float(spawn_info.get("spawn_max_offset", 0.0)),
                "first_collision_step": first_collision_step,
                "first_collision_agents": first_collision_agents,
                "first_collision_sources": first_collision_sources,
                "first_collision_buildings": sorted({
                    source
                    for sources in first_collision_sources.values()
                    for source in sources
                    if source.startswith("building:")
                }),
                "environment_sha256": environment_sha256,
                "evaluator_sha256": evaluator_sha256,
                "model_sha256": model_sha256,
                "min_omega_euc": oe,
                "min_omega_vis": ov,
            }
            trace_records.append(record)
            spawn_adjusted.append(record["spawn_adjusted_count"])
            spawn_max_offsets.append(record["spawn_max_offset"])
            records.append(record)

        summary = summarize_eval(records)
        summary.update({
            "policy": args.policy,
            "level": args.level,
            "num_agents": env.num_agents,
            "mean_episode_return": float(np.mean([r["episode_return"] for r in trace_records])),
            "mean_spawn_adjusted_count": float(np.mean(spawn_adjusted)),
            "max_spawn_offset": float(np.max(spawn_max_offsets)),
            "environment_sha256": environment_sha256,
            "evaluator_sha256": evaluator_sha256,
            "model_sha256": model_sha256,
            "spawn_safety": args.spawn_safety,
            "collision_mode": args.collision_mode,
            "evaluation_config": {
                "policy": args.policy,
                "algo": args.algo,
                "seed": args.seed,
                "episodes": args.k,
                "level": args.level,
                "building_mode": env.building_mode,
                "num_agents": env.num_agents,
                "target_min_speed": float(env.target_min_speed),
                "target_max_speed": float(env.target_max_speed),
                "evader_center_pull": float(env.evader_center_pull),
                "pincer_weight": float(env.pincer_weight),
                "criterion": args.criterion,
                "terminate_on_collision": bool(env.terminate_on_collision),
                "terminate_on_boundary_collision": bool(env.terminate_on_boundary_collision),
                "boundary_collision_penalty": float(env.boundary_collision_penalty),
                "boundary_warning_distance": float(env.boundary_warning_distance),
                "spawn_safety_enabled": bool(env.spawn_safety_enabled),
                "spawn_safety_margin": float(env.spawn_safety_margin),
                "spawn_search_resolution": float(env.spawn_search_resolution),
                "closure_weight": float(env.closure_weight),
                "w_pos": float(env.reward_weights["w_pos"]),
                "w_gap": float(env.reward_weights["w_gap"]),
                "separation_weight": float(env.separation_weight),
                "separation_distance": float(env.separation_distance),
            },
        })
        print("RESULT " + json.dumps(summary))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2)
            print("saved", args.out)
        if args.trace_out:
            with open(args.trace_out, "w", encoding="utf-8") as stream:
                for record in trace_records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        cleanup_error = None
        active_exception = sys.exc_info()[0] is not None
        if getattr(runner, "envs", None) is not None:
            try:
                runner.envs.close()
            except Exception as exc:
                cleanup_error = exc
        if getattr(runner, "agent", None) is not None:
            try:
                runner.agent.finish()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if cleanup_error is not None and not active_exception:
            raise cleanup_error


if __name__ == "__main__":
    main()
