"""Smoke tests for the multi-target 3D pursuit env. Standalone script (not pytest) to
avoid the tensorboard/tf import issue during pytest collection.
Run: python tests/test_multitarget_smoke.py"""
# stub tensorflow: importing xuance pulls torch.utils.tensorboard, whose lazy tf probe
# intermittently fails ('tensorflow' has no attribute 'io'). The env needs no tf; a minimal
# concrete stub (just what tensorboard probes) makes the import deterministic for testing.
import sys as _sys
import types as _types

_tf = _types.ModuleType("tensorflow")
_tf.__file__ = "<tf-stub>"
_tf.__version__ = "2.0.0"
_io = _types.ModuleType("tensorflow.io")
_io.gfile = _types.SimpleNamespace(join=lambda *a, **k: "", exists=lambda *a, **k: False,
                                   makedirs=lambda *a, **k: None, isdir=lambda *a, **k: False)
_tf.io = _io
_sys.modules["tensorflow"] = _tf
_sys.modules["tensorflow.io"] = _io

import numpy as np
from xuance.environment.multi_agent_env.uav_pursuit_apollonius_multitarget_3d import (
    UAVPursuitApolloniusMultiTarget3DEnv)


class Cfg:
    def __init__(self, **k):
        self.__dict__.update(k)


def make(M, N=6, mode="medium"):
    return UAVPursuitApolloniusMultiTarget3DEnv(
        Cfg(num_agents=N, num_targets=M, env_seed=0, building_mode=mode,
            curriculum_enabled=False))


def test_dims_and_step():
    N = 6
    for M in [1, 2, 3]:
        e = make(M)
        assert e.obs_dim == 17 + 3 * (N - 1) + 16 + 3 * (M - 1), (M, e.obs_dim)
        assert e.state_dim == 9 * N + 6 * M, (M, e.state_dim)
        obs, info = e.reset()
        a0 = e.agents[0]
        assert obs[a0].shape == (e.obs_dim,), (M, obs[a0].shape)
        assert e.state().shape == (e.state_dim,), (M, e.state().shape)
        assert set(e.assign.tolist()) <= set(range(M))
        acts = {a: np.zeros(3, np.float32) for a in e.agents}
        obs, rew, term, trunc, info = e.step(acts)
        assert all(np.isfinite(rew[a]) for a in e.agents), M
        assert all(np.all(np.isfinite(obs[a])) for a in e.agents), M
        assert info["num_targets"] == M
        print(f"M={M} OK  obs_dim={e.obs_dim} state_dim={e.state_dim} caught={info['num_caught']}")


def test_100_random_steps_no_nan():
    e = make(2)
    e.reset()
    for t in range(100):
        acts = {a: np.random.uniform(-1, 1, 3).astype(np.float32) for a in e.agents}
        obs, rew, term, trunc, info = e.step(acts)
        assert all(np.all(np.isfinite(obs[a])) for a in e.agents), t
        assert all(np.isfinite(rew[a]) for a in e.agents), t
        if info["is_success"] or all(trunc.values()):
            break
    print("100-step M=2 OK")


def test_capture_flips_alive():
    # place all pursuers on target 0 -> it should get captured (alive False) after a step
    e = make(2)
    e.reset()
    e.uav_positions[:] = e.target_positions[0]  # all pursuers exactly on target 0
    e.assign = np.zeros(e.num_agents, dtype=int)
    acts = {a: np.zeros(3, np.float32) for a in e.agents}
    _, _, _, _, info = e.step(acts)
    assert e.target_alive[0] == False, e.target_alive
    assert info["num_caught"] >= 1
    print("capture flips alive OK")


if __name__ == "__main__":
    test_dims_and_step()
    test_100_random_steps_no_nan()
    test_capture_flips_alive()
    print("ALL PASS")
