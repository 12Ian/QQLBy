"""Evaluate the trained 3D pursuit policy and render a 3D trajectory figure.

Builds the MADDPG agent via get_runner, loads the benchmark best model, then
rolls out K episodes at the test curriculum level, reports success rate / mean
steps, and saves a 3D trajectory PNG of a representative (successful) episode.

Usage (on server, in ~/UAV_Mine):
    python xuance-master/examples/render_eval_3d.py [K]
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")

from xuance import get_runner
from xuance.environment.multi_agent_env.uav_pursuit_apollonius_3d import UAVPursuitApollonius3DEnv


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    p = argparse.Namespace(algo="maddpg", env="uav_pursuit_apollonius_3d",
                           env_id="apollonius_3d", device="cuda:0")
    p.parallels = 1            # minimal: we only need the policy, not training
    runner = get_runner(algo="maddpg", env="uav_pursuit_apollonius_3d",
                        env_id="apollonius_3d", parser_args=p)
    agent = runner.agent

    best_file = os.path.join(os.getcwd(), "results", "maddpg", "apollonius_3d",
                             "best_model", "best_model.pth")
    agent.load_model(best_file)   # load_model accepts a direct .pth file path
    print("loaded best model from", best_file)

    cfg = runner.config
    env = UAVPursuitApollonius3DEnv(cfg)
    level = int(getattr(cfg, "test_curriculum_level", 4))
    if getattr(env, "curriculum_enabled", False):
        env.set_curriculum_level(level)
    print("eval curriculum level:", env.curriculum_level,
          "building_mode:", env.building_mode, "n_buildings:", len(env.buildings))

    out_dir = os.path.join(os.getcwd(), "results", "maddpg", "apollonius_3d")
    successes, steps_list = 0, []
    saved_success = False

    for ep in range(K):
        obs, info = env.reset()
        done, steps = False, 0
        while not done:
            res = agent.action([obs], test_mode=True)
            acts = res["actions"][0]
            obs, rew, term, trunc, info = env.step(acts)
            steps += 1
            done = bool(term[env.agents[0]]) or bool(trunc)
        succ = bool(info.get("is_success", False))
        successes += int(succ)
        steps_list.append(steps)
        if succ and not saved_success:
            env.render(save_path=os.path.join(out_dir, "traj3d_success.png"))
            saved_success = True
            print(f"  saved successful-episode trajectory (ep {ep}, {steps} steps)")

    if not saved_success:
        env.render(save_path=os.path.join(out_dir, "traj3d_lastep.png"))
        print("  no success in K episodes; saved last-episode trajectory")

    print(f"RESULT eval_episodes={K} success_rate={successes / K:.3f} "
          f"mean_steps={np.mean(steps_list):.1f}")


if __name__ == "__main__":
    main()
