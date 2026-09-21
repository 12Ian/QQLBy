"""False-capture-rate eval: per step compute BOTH euclidean and visibility escape
solid angles for the current state; report each judge's false-capture rate
(judged-encircled but target actually escapes) plus success rate.

Usage (server, in ~/UAV_Mine):  python xuance-master/examples/falsecapture_eval_3d.py [K]
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")

from xuance import get_runner
from xuance.environment.multi_agent_env.uav_pursuit_apollonius_3d import UAVPursuitApollonius3DEnv
from xuance.environment.multi_agent_env import geometry3d as g3
from xuance.environment.multi_agent_env import apollonius3d as ap3


def escape_angle(env, mode):
    free = g3.ray_free_distance(env.target_position, env.escape_dirs, env.map_size,
                                env.z_min, env.z_max, env.buildings)
    r_f = env._compute_r_f()
    f = ap3.compute_escape_field_3d(env.target_position, env.target_speed,
            env.uav_positions, env.uav_max_speed, env.escape_dirs, free,
            env.catch_radius, r_f, margin_ema=None, dangerous_prev=None,
            criterion_mode=mode, buildings=env.buildings)
    return f["escape_solid_angle"]


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    eps = 0.1 * 4 * np.pi
    p = argparse.Namespace(algo="maddpg", env="uav_pursuit_apollonius_3d",
                           env_id="apollonius_3d", device="cuda:0")
    p.parallels = 1
    runner = get_runner(algo="maddpg", env="uav_pursuit_apollonius_3d",
                        env_id="apollonius_3d", parser_args=p)
    agent = runner.agent
    best = os.path.join(os.getcwd(), "results", "maddpg", "apollonius_3d",
                        "best_model", "best_model.pth")
    agent.load_model(best)
    cfg = runner.config
    env = UAVPursuitApollonius3DEnv(cfg)
    if getattr(env, "curriculum_enabled", False):
        env.set_curriculum_level(int(getattr(cfg, "test_curriculum_level", 4)))

    enc_euc = enc_vis = fc_euc = fc_vis = caught = 0
    for ep in range(K):
        obs, info = env.reset()
        done = False
        min_euc, min_vis = np.inf, np.inf
        while not done:
            min_euc = min(min_euc, escape_angle(env, "euclidean"))
            min_vis = min(min_vis, escape_angle(env, "visibility"))
            res = agent.action([obs], test_mode=True)
            obs, rew, term, trunc, info = env.step(res["actions"][0])
            done = bool(term[env.agents[0]]) or bool(trunc)
        success = bool(info.get("is_success", False))
        caught += int(success)
        je, jv = (min_euc < eps), (min_vis < eps)
        enc_euc += int(je)
        enc_vis += int(jv)
        if je and not success:
            fc_euc += 1
        if jv and not success:
            fc_vis += 1

    print(f"RESULT K={K} success_rate={caught / K:.3f}")
    print(f"  euclidean : judged_encircled={enc_euc} false_capture_rate="
          f"{(fc_euc / enc_euc) if enc_euc else float('nan'):.3f}")
    print(f"  visibility: judged_encircled={enc_vis} false_capture_rate="
          f"{(fc_vis / enc_vis) if enc_vis else float('nan'):.3f}")


if __name__ == "__main__":
    main()
