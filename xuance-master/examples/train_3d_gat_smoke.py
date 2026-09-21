"""Smoke run with GAT-O + GAT-T enabled (run after the full training frees the GPU)."""
import argparse
from xuance import get_runner


def parse_args():
    p = argparse.ArgumentParser("3D UAV pursuit GAT smoke run.")
    p.add_argument("--algo", type=str, default="maddpg")
    p.add_argument("--env", type=str, default="uav_pursuit_apollonius_3d")
    p.add_argument("--env-id", type=str, default="apollonius_3d")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


if __name__ == '__main__':
    parser = parse_args()
    parser.use_obstacle_gat = True      # GAT-O
    parser.use_graph_module = True       # GAT-T
    parser.parallels = 4
    parser.running_steps = 150000
    parser.eval_interval = 30000
    parser.test_episode = 20
    parser.start_training = 1000
    parser.curriculum_min_steps_per_level = 1000000
    runner = get_runner(algo=parser.algo,
                        env=parser.env,
                        env_id=parser.env_id,
                        parser_args=parser)
    runner.run(mode='benchmark')
