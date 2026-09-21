"""Smoke run with the visibility-constrained criterion driving r_gap."""
import argparse
from xuance import get_runner


def parse_args():
    p = argparse.ArgumentParser("3D UAV pursuit visibility smoke run.")
    p.add_argument("--algo", type=str, default="maddpg")
    p.add_argument("--env", type=str, default="uav_pursuit_apollonius_3d")
    p.add_argument("--env-id", type=str, default="apollonius_3d")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


if __name__ == '__main__':
    parser = parse_args()
    parser.criterion_mode = "visibility"
    parser.parallels = 4
    parser.running_steps = 150000
    parser.eval_interval = 30000
    parser.test_episode = 20
    parser.start_training = 1000
    parser.curriculum_min_steps_per_level = 1000000  # stay on level 0 during smoke
    runner = get_runner(algo=parser.algo,
                        env=parser.env,
                        env_id=parser.env_id,
                        parser_args=parser)
    runner.run(mode='benchmark')
