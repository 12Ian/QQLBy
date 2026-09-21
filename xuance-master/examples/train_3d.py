import argparse
from xuance import get_runner


def parse_args():
    parser = argparse.ArgumentParser("Run a 3D UAV cooperative pursuit MARL demo.")
    parser.add_argument("--algo", type=str, default="maddpg")
    parser.add_argument("--env", type=str, default="uav_pursuit_apollonius_3d")
    parser.add_argument("--env-id", type=str, default="apollonius_3d")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


if __name__ == '__main__':
    parser = parse_args()
    runner = get_runner(algo=parser.algo,
                        env=parser.env,
                        env_id=parser.env_id,
                        parser_args=parser)
    runner.run(mode='benchmark')
