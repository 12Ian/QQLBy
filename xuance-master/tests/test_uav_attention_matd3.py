"""Verify the real MATD3 learner, twin independence and complete target updates."""
from argparse import Namespace
from copy import deepcopy
import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from xuance.common import MultiAgentBaseCallback
from xuance.torch.learners.multi_agent_rl.matd3_learner import MATD3_Learner
from xuance.torch.policies.uav_attention_matd3 import UAV_EntityAttention_MATD3_Policy
from test_uav_entity_attention import observations, ids

KEYS = [f"agent_{i}" for i in range(4)]


def policy(device="cpu"):
    torch.manual_seed(51)
    return UAV_EntityAttention_MATD3_Policy({k: Box(-1., 1., (3,)) for k in KEYS}, [KEYS[0]],
        actor_hidden_size=[32, 32], critic_hidden_size=[32, 32], device=device)


def learner(p):
    config = Namespace(distributed_training=False, n_agents=4, use_parameter_sharing=True,
        episode_length=600, device="cpu", use_grad_clip=True, grad_clip_norm=.5,
        running_steps=1024, start_training=32, training_frequency=1, parallels=2,
        model_dir="unused", learning_rate_actor=1e-3, learning_rate_critic=1e-3,
        gamma=.99, tau=.1, end_factor_lr_decay=1., actor_update_delay=2,
        target_policy_noise=.2, target_noise_clip=.5, td3_actor_use_first_critic=True)
    return MATD3_Learner(config, [KEYS[0]], KEYS, p, MultiAgentBaseCallback())


def sample():
    obs = observations().reshape(3, 4, 88).numpy()
    return dict(batch_size=3, obs={k: obs[:, i] for i, k in enumerate(KEYS)},
        obs_next={k: obs[:, i] * .9 for i, k in enumerate(KEYS)},
        actions={k: np.full((3, 3), .1, np.float32) for k in KEYS},
        rewards={k: np.ones(3, np.float32) for k in KEYS},
        terminals={k: np.zeros(3, np.float32) for k in KEYS},
        agent_mask={k: np.ones(3, np.float32) for k in KEYS})


def test_all_parameter_groups_independent_and_targets_frozen():
    p = policy()
    actor, critic = p.parameters_actor[KEYS[0]], p.parameters_critic[KEYS[0]]
    assert len(actor + critic) == len(set(map(id, actor + critic)))
    assert set(map(id, actor + critic)) == {id(v) for v in p.parameters() if v.requires_grad}
    assert not torch.equal(p.critic_A_representation[KEYS[0]].self_mlp[0].weight,
                           p.critic_B_representation[KEYS[0]].self_mlp[0].weight)
    for name in p._online_modules:
        for a, b in zip(getattr(p, name).parameters(), getattr(p, "target_" + name).parameters()):
            assert a is not b and not b.requires_grad
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("tau", [0., .1, 1.])
def test_soft_update_every_parameter_by_name(tau):
    p = policy()
    before = {n: v.clone() for n, v in p.named_parameters() if n.startswith("target_")}
    with torch.no_grad():
        for name, value in p.named_parameters():
            if not name.startswith("target_"):
                value.add_(.5)
    p.soft_update(tau)
    named = dict(p.named_parameters())
    for name in before:
        torch.testing.assert_close(named[name], before[name] * (1 - tau) + named[name[7:]] * tau)


def test_real_learner_delays_actor_updates_and_keeps_finite_gradients():
    p = policy()
    learn = learner(p)
    before = {n: v.clone() for n, v in p.named_parameters()}
    first = learn.update(sample())
    assert f"{KEYS[0]}/loss_actor" not in first
    for name, value in p.named_parameters():
        if name.startswith(("actor.", "actor_representation.", "target_")):
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    second = learn.update(sample())
    assert f"{KEYS[0]}/loss_actor" in second and all(np.isfinite(v) for v in second.values())
    for name in p._online_modules:
        assert any(not torch.equal(v, before[n]) for n, v in p.named_parameters() if n.startswith(name + "."))
    assert all(v.grad is None for n, v in p.named_parameters() if n.startswith("target_"))
    for name, value in p.named_parameters():
        if name.startswith("target_"):
            torch.testing.assert_close(value, before[name] * .9 + dict(p.named_parameters())[name[7:]] * .1)
    assert all(torch.isfinite(v.grad).all() for v in p.parameters() if v.grad is not None)


def test_action_gradient_reaches_all_agents_but_not_other_scenes():
    p = policy()
    actions = torch.zeros(3, 12, requires_grad=True)
    a, b, mean = p.Qpolicy(observations().reshape(3, -1), actions, ids())
    torch.testing.assert_close(mean[KEYS[0]], (a[KEYS[0]] + b[KEYS[0]]) / 2)
    for q in (a, b):
        gradient = torch.autograd.grad(q[KEYS[0]][0], actions, retain_graph=True)[0]
        assert (gradient[0].reshape(4, 3).abs().sum(-1) > 0).all()
        assert not gradient[1:].any()


def test_joint_order_and_q_query_ids_with_sentinels():
    p = policy()
    class EchoEncoder(torch.nn.Module):
        def forward(self, obs, agent_ids):
            return {"state": obs[:, :1].expand(-1, 128)}
    class EchoHead(torch.nn.Module):
        def forward(self, values):
            return (values[:, :1] + 10 * values[:, 128:129] + 100 * values[:, 256:257] +
                    1000 * values[:, 384:385] + values[:, 512:524].sum(-1, keepdim=True) +
                    (values[:, -4:] * torch.arange(4.)).sum(-1, keepdim=True))
    for branch in ("A", "B"):
        getattr(p, "critic_" + branch + "_representation")[KEYS[0]] = EchoEncoder()
        getattr(p, "critic_" + branch)[KEYS[0]] = EchoHead()
    obs = observations(batch=2)
    obs[:, 0] = torch.arange(8.)
    result = p.Qpolicy(obs.reshape(2, -1), torch.ones(2, 12), ids(2))[0][KEYS[0]].reshape(2, 4)
    torch.testing.assert_close(result, torch.tensor([[3222., 3223., 3224., 3225.],
                                                    [7666., 7667., 7668., 7669.]]))


def test_actor_locality_and_checkpoint_round_trip(tmp_path):
    p = policy()
    obs = observations()
    changed = obs.clone()
    changed[1:, 49:] = 0
    a = p({KEYS[0]: obs}, ids())[1][KEYS[0]]
    b = p({KEYS[0]: changed}, ids())[1][KEYS[0]]
    torch.testing.assert_close(a[0], b[0])
    assert a.shape == (12, 3) and (a.abs() <= 1).all()
    torch.save(p.state_dict(), tmp_path / "state.pt")
    restored = policy()
    restored.load_state_dict(torch.load(tmp_path / "state.pt", weights_only=True), strict=False)
    torch.testing.assert_close(a, restored({KEYS[0]: obs}, ids())[1][KEYS[0]], rtol=0, atol=0)
    broken = deepcopy(p.state_dict())
    broken.pop(next(k for k in broken if "target_critic_B_representation" in k))
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(broken, strict=False)
    with pytest.raises(ValueError, match="mismatch"):
        restored.load_state_dict({})


def test_target_minimum():
    p = policy()
    with torch.no_grad():
        for name, value in (("target_critic_A", 5.), ("target_critic_B", 2.)):
            head = getattr(p, name)[KEYS[0]]
            for parameter in head.parameters():
                parameter.zero_()
            head.model[-1].bias.fill_(value)
    q = p.Qtarget(observations().reshape(3, -1), torch.zeros(3, 12), ids())[KEYS[0]]
    assert (q == 2).all() and not q.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu():
    cpu, gpu = policy(), policy("cuda:0")
    gpu.load_state_dict(cpu.state_dict())
    args = (observations().reshape(3, -1), torch.zeros(3, 12), ids())
    for left, right in zip(cpu.Qpolicy(*args), gpu.Qpolicy(*args)):
        torch.testing.assert_close(left[KEYS[0]], right[KEYS[0]].cpu(), atol=3e-6, rtol=1e-4)
