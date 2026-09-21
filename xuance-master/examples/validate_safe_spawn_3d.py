import argparse
import itertools
import json
import os
import sys
import numpy as np

from xuance.environment.multi_agent_env.uav_pursuit_apollonius_3d import (
    UAVPursuitApollonius3DEnv,
)

class Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def csv_ints(value):
    return [int(item) for item in value.split(",") if item]

def csv_strings(value):
    return [item for item in value.split(",") if item]

def seed_range(value):
    start, end = (int(item) for item in value.split(":", 1))
    return range(start, end + 1)

def action_probes(n, seed, random_count):
    yield "zero", np.zeros((n, 3), dtype=np.float32)
    for index, values in enumerate(itertools.product((-1.0, 0.0, 1.0), repeat=3)):
        yield f"grid_{index}", np.tile(np.asarray(values, np.float32), (n, 1))
    rng = np.random.default_rng(seed + 1009 * n)
    for index in range(random_count):
        yield f"random_{index}", rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-agents", default="3,4,5,6,7,8")
    parser.add_argument("--building-modes", default="empty,open,medium,complex")
    parser.add_argument("--seeds", default="0:49")
    parser.add_argument("--random-actions", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.random_actions < 0:
        parser.error("--random-actions must be >= 0")
    result = {
        "invalid_spawn_count": 0,
        "spawn_caused_first_step_collision_count": 0,
        "cases": [],
    }
    for n, mode in itertools.product(csv_ints(args.num_agents), csv_strings(args.building_modes)):
        env = UAVPursuitApollonius3DEnv(Cfg(
            num_agents=n, building_mode=mode, curriculum_enabled=False,
            spawn_safety_enabled=True,
        ))
        case = {"num_agents": n, "mode": mode, "resets": 0,
                "adjusted_total": 0, "max_offset": 0.0,
                "invalid_spawns": 0, "geometric_first_step_collisions": 0,
                "action_probe_count": 0, "action_first_step_collisions": 0,
                "first_step_collisions": 0}
        for seed in seed_range(args.seeds):
            np.random.seed(seed)
            _, info = env.reset()
            case["resets"] += 1
            metrics = info["infos"]
            case["adjusted_total"] += metrics["spawn_adjusted_count"]
            case["max_offset"] = max(case["max_offset"], metrics["spawn_max_offset"])
            for index, pos in enumerate(env.uav_positions):
                if (not np.all(np.isfinite(pos)) or
                        not env.z_min <= float(pos[2]) <= env.z_max or
                        env._spawn_clearance_xy(pos) < env.spawn_clearance - 1e-5):
                    case["invalid_spawns"] += 1
                for yaw in np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False):
                    for pitch in (-env.pitch_max, 0.0, env.pitch_max):
                        _, hit = env._move_with_clip_3d(
                            pos, yaw, pitch, env.initial_step_reach, env.uav_radius)
                        case["geometric_first_step_collisions"] += int(hit)
            for i, j in itertools.combinations(range(n), 2):
                if np.linalg.norm(env.uav_positions[i] - env.uav_positions[j]) \
                        < env.spawn_min_pair_distance - 1e-5:
                    case["invalid_spawns"] += 1
            for _, action_matrix in action_probes(n, seed, args.random_actions):
                np.random.seed(seed)
                env.reset()
                actions = {
                    agent: action_matrix[index]
                    for index, agent in enumerate(env.agents)
                }
                _, _, _, _, step_info = env.step(actions)
                case["action_probe_count"] += 1
                case["action_first_step_collisions"] += int(
                    bool(step_info.get("is_collision", False)))
        case["first_step_collisions"] = (
            case["geometric_first_step_collisions"] +
            case["action_first_step_collisions"])
        result["invalid_spawn_count"] += case["invalid_spawns"]
        result["spawn_caused_first_step_collision_count"] += case["first_step_collisions"]
        result["cases"].append(case)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
    print("RESULT " + json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["invalid_spawn_count"] == 0 and \
        result["spawn_caused_first_step_collision_count"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
