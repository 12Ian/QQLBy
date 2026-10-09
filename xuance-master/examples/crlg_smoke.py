"""Small dynamic integration check for the CRL-G environment and MADDPG path."""

from argparse import Namespace
import numpy as np
from xuance import get_runner


def main():
    overrides = Namespace(
        algo="maddpg", env="crlg_3d", env_id="crlg_3d_smoke",
        device="cuda:0", parallels=1, vectorize="DummyVecMultiAgentEnv",
        running_steps=2, buffer_size=64, batch_size=8,
        start_training=0, training_frequency=1,
        coverage_samples=32, seed=123, env_seed=123,
        model_dir="outputs/runs/crlg_smoke/models/",
        log_dir="outputs/runs/crlg_smoke/logs/")
    runner = get_runner(algo="maddpg", env="crlg_3d", env_id="crlg_3d",
                        parser_args=overrides)
    try:
        observations, _ = runner.envs.reset()
        actions = runner.agent.action(observations, test_mode=False)["actions"]
        for key, action in actions[0].items():
            assert np.linalg.norm(action) <= 1.00001, (key, action)
            if observations[0][key][-1] == 0:
                assert np.allclose(action, 0), (key, action)
        print("action_projection_ok")
        result = runner.agent.train(2)
        print("learner_update_ok", sorted(result))
    finally:
        runner._finalize()


if __name__ == "__main__":
    main()
