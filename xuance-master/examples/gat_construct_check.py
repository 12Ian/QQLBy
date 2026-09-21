"""Build the MADDPG runner with use_obstacle_gat=True, run one deterministic
action() on a reset obs, and confirm the policy has GAT-O params. Quick wiring
check (no training, parallels=1, no benchmark writes).

Must guard with __main__ because SubprocVecMultiAgentEnv uses spawn."""
import argparse
import numpy as np
from xuance import get_runner


def main():
    p = argparse.Namespace(algo="maddpg", env="uav_pursuit_apollonius_3d",
                           env_id="apollonius_3d", device="cuda:0")
    p.parallels = 1
    p.use_obstacle_gat = True
    p.use_graph_module = True
    runner = get_runner(algo="maddpg", env="uav_pursuit_apollonius_3d",
                        env_id="apollonius_3d", parser_args=p)
    agent = runner.agent
    has_gat = any("obstacle_gat" in n for n, _ in agent.policy.named_parameters())
    has_graph = any("graph_comm" in n for n, _ in agent.policy.named_parameters())
    obs, _ = runner.envs.reset()
    out = agent.action(obs, test_mode=True)
    acts = out["actions"][0]
    ok = all(np.all(np.isfinite(v)) for v in acts.values())
    print(f"RESULT obstacle_gat_params={has_gat} graph_comm_params={has_graph} "
          f"action_finite={ok} n_actions={len(acts)} "
          f"act_dim={len(next(iter(acts.values())))}")
    runner.envs.close()


if __name__ == "__main__":
    main()
