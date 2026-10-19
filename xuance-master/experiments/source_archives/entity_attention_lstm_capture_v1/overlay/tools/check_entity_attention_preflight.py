"""Exercise production-sized MATD3 updates before starting the long run."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch import nn
import yaml
from gymnasium.spaces import Box
from xuance.common import MultiAgentBaseCallback
from xuance.torch.learners.multi_agent_rl.matd3_learner import MATD3_Learner
from xuance.torch.policies.uav_attention_matd3 import UAV_EntityAttention_MATD3_Policy


def main(run):
    run = Path(run)
    cfg = yaml.safe_load((run / "configs/attention_lstm_seed_1.yaml").read_text())
    cfg.update(distributed_training=False, n_agents=4, device="cuda:0")
    torch.set_num_threads(2)
    torch.manual_seed(1)
    keys = [f"uav_{i}" for i in range(4)]
    policy = UAV_EntityAttention_MATD3_Policy({k: Box(-1., 1., (3,)) for k in keys}, [keys[0]],
        actor_hidden_size=cfg["actor_hidden_size"], critic_hidden_size=cfg["critic_hidden_size"],
        initialize=nn.init.orthogonal_, activation=nn.LeakyReLU,
        prediction_sha256=cfg["uav_prediction_sha256"], device="cuda:0")
    learner = MATD3_Learner(Namespace(**cfg), [keys[0]], keys, policy, MultiAgentBaseCallback())
    batch = cfg["batch_size"]
    rng = np.random.default_rng(1)
    obs = rng.uniform(-.1, .1, (batch, 4, 88)).astype(np.float32)
    obs[:, :, 47] = 12 / 11
    obs[:, :, 63::6] = 1
    sample = dict(batch_size=batch, obs={k: obs[:, i] for i, k in enumerate(keys)},
        obs_next={k: obs[:, i] * .9 for i, k in enumerate(keys)},
        actions={k: rng.uniform(-1, 1, (batch, 3)).astype(np.float32) for k in keys},
        rewards={k: rng.uniform(-1, 1, batch).astype(np.float32) for k in keys},
        terminals={k: np.zeros(batch, np.float32) for k in keys},
        agent_mask={k: np.ones(batch, np.float32) for k in keys})
    torch.cuda.reset_peak_memory_stats()
    for _ in range(4):
        learner.update(sample)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(10):
        metrics = learner.update(sample)
        assert all(np.isfinite(v) for v in metrics.values())
        assert all(torch.isfinite(v.grad).all() for v in policy.parameters() if v.grad is not None)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    result = dict(passed=True, batch_size=batch, hidden_sizes=cfg["actor_hidden_size"],
        timed_updates=10, seconds_per_learner_update=seconds / 10,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        trainable_parameters=sum(v.numel() for v in policy.parameters() if v.requires_grad),
        last_metrics=metrics)
    (run / "preflight/production_batch.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run)
